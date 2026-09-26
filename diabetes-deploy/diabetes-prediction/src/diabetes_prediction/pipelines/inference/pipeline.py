"""Pipeline de inferencia: reaproveita os nos de transformacao com os artefatos do treino."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from diabetes_prediction.pipelines.data_engineering.nodes import (
    cap_outliers,
    encode_features,
    extract_features,
    impute_missing,
    replace_zeros_with_nan,
    scale_features,
)

from .nodes import drop_target_if_present, evaluate_inference, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(drop_target_if_present, ["raw_inference_data", "params:target"], "inference_input", name="drop_target_inference"),
            node(replace_zeros_with_nan, ["inference_input", "params:preprocessing"], "inference_with_nan", name="replace_zeros_inference"),
            node(impute_missing, ["inference_with_nan", "imputer"], "inference_imputed", name="impute_inference"),
            node(cap_outliers, ["inference_imputed", "outlier_thresholds"], "inference_clean", name="cap_outliers_inference"),
            node(extract_features, "inference_clean", "inference_features", name="extract_features_inference"),
            node(encode_features, ["inference_features", "encoder"], "inference_encoded", name="encode_inference"),
            node(scale_features, ["inference_encoded", "scaler"], "X_inference", name="scale_inference"),
            node(predict, ["model", "X_inference", "raw_inference_data"], "inference_predictions", name="predict"),
            node(evaluate_inference, ["inference_predictions", "params:target"], "inference_metrics", name="evaluate_inference"),
        ],
        tags="inference",
    )
