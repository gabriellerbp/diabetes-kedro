# =============================================================================
# Container de inferencia via FastAPI (exercicio extra 4)
#   docker build -t diabetes-api .
#   docker run -p 8000:8000 diabetes-api
# O modelo e os artefatos ja treinados (data/06_models) sao copiados para a
# imagem, entao /inference funciona imediatamente. /train tambem funciona
# (re-treina dentro do container; para persistir, monte data/ como volume).
# =============================================================================
FROM python:3.12-slim

# libgomp1: exigido por LightGBM/XGBoost; curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# uv (gerenciador de dependencias) - imagem oficial, sem pip install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) Dependencias primeiro (cache de camadas): so reinstala se pyproject/lock mudarem
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Codigo, configuracao e dados/artefatos
COPY src ./src
COPY conf ./conf
COPY data ./data
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    KEDRO_DISABLE_TELEMETRY=true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "diabetes_prediction.api:app", "--host", "0.0.0.0", "--port", "8000"]
