"""Nos do pipeline de inferencia."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from diabetes_prediction.pipelines.training.nodes import compute_metrics

logger = logging.getLogger(__name__)


def drop_target_if_present(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove a coluna alvo (se existir) antes de prever. O arquivo de inferencia a contem."""
    return data.drop(columns=[target, target.upper()], errors="ignore")


def predict(model: BaseEstimator, X_inference: pd.DataFrame, raw_inference: pd.DataFrame) -> pd.DataFrame:
    """Gera predicao e probabilidade, anexadas as linhas originais (legiveis)."""
    out = raw_inference.copy().reset_index(drop=True)
    out["PREDICTION"] = model.predict(X_inference).astype(int)
    if hasattr(model, "predict_proba"):
        out["PROBABILITY"] = model.predict_proba(X_inference)[:, 1].round(4)
    logger.info("Inferencia concluida: %d linhas, %d positivos", len(out), int(out["PREDICTION"].sum()))
    return out


def evaluate_inference(predictions: pd.DataFrame, target: str) -> dict[str, Any]:
    """Se o arquivo de inferencia trouxer o alvo real, calcula metricas (backtest)."""
    if target not in predictions.columns:
        return {"note": f"coluna {target!r} ausente - metricas nao calculadas"}
    proba = predictions["PROBABILITY"] if "PROBABILITY" in predictions.columns else None
    metrics = compute_metrics(predictions[target], predictions["PREDICTION"], proba)
    metrics["n_inference"] = int(len(predictions))
    logger.info("Metricas na base de inferencia: %s", metrics)
    return metrics
