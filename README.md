# Prevendo a incidência de Diabetes — Pipelines Kedro

Trabalho da disciplina **Deployment / Production-Ready Data Science** (Insper, Prof. Donald Neumann).
Tradução do notebook `notebooks/diabetes-prediction.ipynb` para três pipelines Kedro
(engenharia de dados, treinamento e inferência), expostos via FastAPI e conteinerizados com Docker.

## Como executar

Pré-requisitos: Python 3.10–3.13 e [uv](https://docs.astral.sh/uv/) (`pip install uv` ou `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone <url-do-repo> && cd diabetes-prediction
uv sync                      # cria .venv e instala tudo a partir do uv.lock
uv run kedro run             # data_engineering -> training -> inference
uv run kedro viz run         # visualização do DAG em http://localhost:4141
```

Pipelines individuais:

```bash
uv run kedro run --pipeline data_engineering
uv run kedro run --pipeline training       # exige data_engineering já executado
uv run kedro run --pipeline inference      # exige modelo treinado
uv sync --extra dev && uv run pytest       # testes unitários dos nós
```

### API (extra)

```bash
uv run uvicorn diabetes_prediction.api:app --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
```

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status e se há modelo treinado |
| GET | `/datasets` | Lista datasets do catálogo |
| GET | `/datasets/{name}?limit=&offset=` | **Expõe um dataset como API** (ex.: `raw_modelling_data`, `inference_predictions`, `model_metrics`) |
| GET | `/metrics` | Métricas de teste, melhores hiperparâmetros e métricas da inferência em lote |
| POST | `/train` | Roda `data_engineering` + `training` em background (`GET /train/status` acompanha) |
| POST | `/batch-inference` | Roda o pipeline de inferência sobre `data/01_raw/diabetes-dataset-inference.csv` |
| POST | `/inference` | Inferência online: JSON com pacientes → predição + probabilidade |

Exemplo:

```bash
curl -X POST localhost:8000/inference -H 'content-type: application/json' -d '{
  "patients": [{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,
                "Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}]}'
```

### Docker (extra)

```bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
curl localhost:8000/health
```

## Estrutura

```
conf/base/catalog.yml        # datasets (I/O) — o código nunca lê/grava arquivos diretamente
conf/base/parameters.yml     # parâmetros: colunas, split, modelo, grid de hiperparâmetros
data/01_raw ... 08_reporting # camadas de dados (raw versionado no git; o resto é gerado)
src/diabetes_prediction/
  pipelines/data_engineering # zeros→NaN, split, imputação KNN, outliers, features, encoding, scaling
  pipelines/training         # GridSearchCV, avaliação no teste, importância das variáveis
  pipelines/inference        # reaplica os artefatos do treino e prevê
  pipeline_registry.py       # __default__ = DE + training + inference
  api.py                     # FastAPI
tests/                       # pytest
Dockerfile                   # container de inferência
```

## Planejamento dos pipelines

```
                 ┌──────────────── data_engineering ─────────────────┐
raw_modelling ─► zeros→NaN ─► split(70/30) ─► fit imputer ─► fit outlier thr ─► fit encoder ─► fit scaler
                                     │ train/test ──── transform ──── transform ──── transform ──── transform ─► X_*, y_*
                                                                                                            │
                 ┌──── training ────┐                                                                       ▼
                 GridSearchCV ─► model ─► evaluate(X_test) ─► model_metrics, best_params, feature_importances

                 ┌──── inference ────┐
raw_inference ─► zeros→NaN ─► imputer ─► outlier thr ─► features ─► encoder ─► scaler ─► predict ─► predictions.csv
                                                                                                └► inference_metrics (se houver Outcome)
```

Todo artefato que aprende algo dos dados (imputer, limites de outlier, encoder, scaler, modelo)
é ajustado **somente no treino** e persistido em `data/06_models`. Treino, teste e inferência
recebem exatamente as mesmas transformações.

## Decisões em relação ao notebook

| Tema | Notebook | Pipeline | Motivo |
|---|---|---|---|
| Ordem fit/split | `fit(all) → split` | `split → fit(train) → transform(all)` | Elimina *data leakage* (slide da disciplina) |
| `NEW_AGE_BMI_NOM` | regras "obese" usavam `BMI > 18.5` (sobrescrevia tudo) | `BMI >= 30` | Bug |
| `NEW_INSULIN_SCORE` | `None`/`"Abnormal"` (descartado no one-hot) | `"Normal"`/`"Abnormal"` | Variável realmente binária |
| Métricas | `recall_score(y_pred, y_test)` (invertido) | `(y_true, y_pred)`; AUC com `predict_proba` | Correção |
| Modelo | 9 modelos avaliados, nenhum escolhido | `training.model_type` em `parameters.yml` (default `random_forest`, grid do notebook) | Config separada do código; troca sem editar `src` |
| Colunas com zero | detectadas por `min()==0` nos dados | lista fixa em `parameters.yml` | Determinístico; não depende do lote de inferência |
| Nomes `GLUCOSE * INSULIN` | com espaço/asterisco | `NEW_GLUCOSE_X_INSULIN` | Compatibilidade XGBoost/LightGBM |
| Split | sem estratificação | `stratify=Outcome` | Classe minoritária ~35% |

Para trocar o modelo, edite `conf/base/parameters.yml`, por exemplo:

```yaml
training:
  model_type: xgboost
  param_grid: {n_estimators: [50, 100, 200], learning_rate: [0.1, 0.5, 1.0]}
```

## Resultados (Random Forest, grid do notebook)

| Conjunto | Accuracy | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| Teste (196 linhas) | 0.719 | 0.623 | 0.597 | 0.610 | 0.811 |
| Inferência (116 linhas) | 0.759 | 0.675 | 0.643 | 0.659 | 0.806 |

Os valores diferem dos do notebook porque lá os encoders/scaler viam o conjunto de teste
antes do split (estimativa otimista) e porque as métricas de recall/precision estavam invertidas.

## Autores

Grupo: 
- Bárbara Prado
- Leticia Rodrigues
- Gabrielle Paschoalino
