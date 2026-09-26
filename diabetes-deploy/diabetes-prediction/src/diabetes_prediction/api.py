"""API FastAPI que expoe os pipelines Kedro (exercicio extra 3).

Executar localmente:
    uv run uvicorn diabetes_prediction.api:app --host 0.0.0.0 --port 8000
Documentacao interativa: http://localhost:8000/docs

Endpoints
---------
GET  /health                      - status da API e existencia de modelo treinado
GET  /datasets                    - lista os datasets do catalogo
GET  /datasets/{name}             - expoe um dataset do catalogo como JSON (extra: dataset como API)
GET  /metrics                     - metricas do ultimo treino e da ultima inferencia em lote
POST /train                       - roda data_engineering + training em background
GET  /train/status                - acompanha o job de treino
POST /batch-inference             - roda o pipeline de inferencia sobre data/01_raw/*inference.csv
POST /inference                   - inferencia online: recebe pacientes em JSON, devolve predicoes
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Raiz do projeto: .../diabetes-prediction (dois niveis acima de src/diabetes_prediction)
PROJECT_PATH = Path(__file__).resolve().parents[2]
bootstrap_project(PROJECT_PATH)

app = FastAPI(
    title="Diabetes Prediction API",
    description="Pipelines Kedro (engenharia de dados, treinamento e inferencia) expostos via FastAPI.",
    version="0.1.0",
)

# Estado do job de treino (um por processo). Em producao real: fila + banco.
_train_state: dict[str, Any] = {"status": "idle", "started_at": None, "finished_at": None, "error": None}
_train_lock = threading.Lock()


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class Patient(BaseModel):
    """Uma linha do dataset original (mesmos nomes de coluna do CSV)."""

    Pregnancies: int = Field(ge=0, examples=[6])
    Glucose: float = Field(ge=0, examples=[148])
    BloodPressure: float = Field(ge=0, examples=[72])
    SkinThickness: float = Field(ge=0, examples=[35])
    Insulin: float = Field(ge=0, examples=[0])
    BMI: float = Field(ge=0, examples=[33.6])
    DiabetesPedigreeFunction: float = Field(ge=0, examples=[0.627])
    Age: int = Field(ge=0, examples=[50])


class InferenceRequest(BaseModel):
    patients: list[Patient] = Field(min_length=1)


class Prediction(BaseModel):
    prediction: int
    probability: float | None = None


class InferenceResponse(BaseModel):
    n: int
    predictions: list[Prediction]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _run_pipelines(names: list[str]) -> None:
    """Roda pipelines em sessoes Kedro isoladas (uma por pipeline)."""
    for name in names:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            session.run(pipeline_name=name)


def _load_dataset(name: str):
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog
        if name not in catalog:
            raise HTTPException(status_code=404, detail=f"dataset {name!r} nao existe no catalogo")
        if not catalog.exists(name):
            raise HTTPException(status_code=404, detail=f"dataset {name!r} ainda nao foi gerado - rode /train")
        return catalog.load(name)


def _model_exists() -> bool:
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        return session.load_context().catalog.exists("model")


def _train_job() -> None:
    try:
        _run_pipelines(["data_engineering", "training"])
        _train_state.update(status="completed", error=None)
    except Exception as exc:  # noqa: BLE001 - queremos expor o erro ao cliente
        logger.exception("Treino falhou")
        _train_state.update(status="failed", error=str(exc))
    finally:
        _train_state["finished_at"] = datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_trained": _model_exists(), "project": PROJECT_PATH.name}


@app.get("/datasets")
def list_datasets() -> dict[str, list[str]]:
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = session.load_context().catalog
        names = sorted(n for n in catalog.keys() if not n.startswith("params:") and n != "parameters")
    return {"datasets": names}


@app.get("/datasets/{name}")
def get_dataset(name: str, limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    """Expoe um dataset tabular ou JSON do catalogo (paginado)."""
    data = _load_dataset(name)
    if isinstance(data, pd.DataFrame):
        page = data.iloc[offset : offset + limit]
        return {"name": name, "total_rows": int(len(data)), "offset": offset, "limit": limit,
                "columns": list(data.columns), "rows": page.to_dict(orient="records")}
    if isinstance(data, dict):
        return {"name": name, "data": data}
    raise HTTPException(status_code=415, detail=f"dataset {name!r} ({type(data).__name__}) nao e serializavel em JSON")


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("model_metrics", "best_params", "inference_metrics"):
        try:
            out[name] = _load_dataset(name)
        except HTTPException:
            out[name] = None
    return out


@app.post("/train", status_code=202)
def train() -> dict[str, Any]:
    """Dispara data_engineering + training em background e retorna imediatamente."""
    with _train_lock:
        if _train_state["status"] == "running":
            raise HTTPException(status_code=409, detail="ja existe um treino em execucao")
        _train_state.update(status="running", started_at=datetime.now(timezone.utc).isoformat(), finished_at=None, error=None)
        threading.Thread(target=_train_job, daemon=True).start()
    return {"message": "treino iniciado", **_train_state}


@app.get("/train/status")
def train_status() -> dict[str, Any]:
    return _train_state


@app.post("/batch-inference")
def batch_inference() -> dict[str, Any]:
    """Roda o pipeline de inferencia sobre o CSV do catalogo e devolve as metricas."""
    if not _model_exists():
        raise HTTPException(status_code=409, detail="modelo nao treinado - chame POST /train primeiro")
    _run_pipelines(["inference"])
    return {"message": "inferencia em lote concluida", "metrics": _load_dataset("inference_metrics")}


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest) -> InferenceResponse:
    """Inferencia online: substitui ``raw_inference_data`` por um MemoryDataset com o JSON recebido."""
    if not _model_exists():
        raise HTTPException(status_code=409, detail="modelo nao treinado - chame POST /train primeiro")

    df = pd.DataFrame([p.model_dump() for p in request.patients])

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        catalog = context.catalog
        # Override apenas nesta sessao: entrada vem da memoria, saida fica na memoria
        catalog["raw_inference_data"] = MemoryDataset(df)
        catalog["inference_predictions"] = MemoryDataset(copy_mode="assign")

        from kedro.framework.project import pipelines
        from kedro.runner import SequentialRunner

        # Ate o no "predict": sem avaliacao (nao ha alvo) e sem gravar em disco.
        online_pipeline = pipelines["inference"].to_nodes("predict")
        SequentialRunner().run(online_pipeline, catalog)
        preds: pd.DataFrame = catalog.load("inference_predictions")

    result = [
        Prediction(prediction=int(row["PREDICTION"]), probability=float(row["PROBABILITY"]) if "PROBABILITY" in preds else None)
        for _, row in preds.iterrows()
    ]
    return InferenceResponse(n=len(result), predictions=result)
