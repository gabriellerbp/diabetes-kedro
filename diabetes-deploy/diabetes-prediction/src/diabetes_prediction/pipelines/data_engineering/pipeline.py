"""Pipeline de engenharia de dados: raw -> X_train/X_test/y_train/y_test + artefatos."""
from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    cap_outliers,
    encode_features,
    extract_features,
    fit_encoder,
    fit_imputer,
    fit_outlier_thresholds,
    fit_scaler,
    impute_missing,
    replace_zeros_with_nan,
    scale_features,
    split_features_target,
    split_train_test,
)


def _transform_chain(prefix: str) -> list:
    """Cadeia de transformacoes (sem fit) aplicada a um conjunto ``prefix`` in {train, test}."""
    return [
        node(impute_missing, [f"{prefix}_raw", "imputer"], f"{prefix}_imputed", name=f"impute_{prefix}"),
        node(cap_outliers, [f"{prefix}_imputed", "outlier_thresholds"], f"{prefix}_clean", name=f"cap_outliers_{prefix}"),
        node(extract_features, f"{prefix}_clean", f"{prefix}_features", name=f"extract_features_{prefix}"),
        node(encode_features, [f"{prefix}_features", "encoder"], f"{prefix}_encoded", name=f"encode_{prefix}"),
        node(scale_features, [f"{prefix}_encoded", "scaler"], f"{prefix}_scaled", name=f"scale_{prefix}"),
        node(split_features_target, [f"{prefix}_scaled", "params:target"], [f"X_{prefix}", f"y_{prefix}"], name=f"split_xy_{prefix}"),
    ]


def create_pipeline(**kwargs) -> Pipeline:
    fit_nodes = [
        node(replace_zeros_with_nan, ["raw_modelling_data", "params:preprocessing"], "modelling_with_nan", name="replace_zeros_with_nan"),
        node(split_train_test, ["modelling_with_nan", "params:split", "params:target"], ["train_raw", "test_raw"], name="split_train_test"),
        node(fit_imputer, ["train_raw", "params:preprocessing"], "imputer", name="fit_imputer"),
        node(fit_outlier_thresholds, ["train_imputed", "params:preprocessing", "params:target"], "outlier_thresholds", name="fit_outlier_thresholds"),
        node(fit_encoder, ["train_features", "params:target"], "encoder", name="fit_encoder"),
        node(fit_scaler, ["train_encoded", "params:preprocessing", "params:target"], "scaler", name="fit_scaler"),
    ]
    return pipeline(fit_nodes + _transform_chain("train") + _transform_chain("test"), tags="data_engineering")
