"""Pipeline de treinamento: X/y -> modelo + metricas + importancias."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, extract_feature_importances, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(train_model, ["X_train", "y_train", "params:training"], ["model", "best_params"], name="train_model"),
            node(evaluate_model, ["model", "X_test", "y_test"], "model_metrics", name="evaluate_model"),
            node(extract_feature_importances, ["model", "X_train"], "feature_importances", name="feature_importances"),
        ],
        tags="training",
    )
