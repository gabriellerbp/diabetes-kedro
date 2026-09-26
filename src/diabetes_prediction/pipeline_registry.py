"""Registro dos pipelines do projeto."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Ordem logica de execucao do pipeline completo
PIPELINE_ORDER = ["data_engineering", "training", "inference"]


def register_pipelines() -> dict[str, Pipeline]:
    """Descobre os pipelines em ``pipelines/`` e define o ``__default__``.

    ``kedro run``                       -> data_engineering + training + inference
    ``kedro run --pipeline training``   -> apenas treinamento (exige DE ja executado)
    ``kedro run --pipeline inference``  -> apenas inferencia (exige modelo salvo)

    Observacao: o Kedro 1.x carrega pipelines sob demanda (``--pipeline X`` importa
    apenas X), por isso o ``__default__`` soma somente os que foram encontrados.
    """
    pipelines = find_pipelines(raise_errors=True)
    ordered = [pipelines[name] for name in PIPELINE_ORDER if name in pipelines]
    ordered += [p for name, p in pipelines.items() if name not in PIPELINE_ORDER]
    pipelines["__default__"] = sum(ordered, Pipeline([]))
    return pipelines
