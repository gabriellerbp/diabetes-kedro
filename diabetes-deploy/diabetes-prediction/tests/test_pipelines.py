"""Testes unitarios dos nos (funcoes puras -> faceis de testar sem Kedro)."""
import numpy as np
import pandas as pd
import pytest

from diabetes_prediction.pipeline_registry import register_pipelines
from diabetes_prediction.pipelines.data_engineering import nodes as de


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n),
            "Glucose": np.where(rng.random(n) < 0.1, 0, rng.integers(70, 200, n)),
            "BloodPressure": rng.integers(50, 100, n),
            "SkinThickness": rng.integers(10, 50, n),
            "Insulin": np.where(rng.random(n) < 0.3, 0, rng.integers(15, 300, n)),
            "BMI": rng.uniform(18, 45, n).round(1),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 2, n).round(3),
            "Age": rng.integers(21, 70, n),
            "Outcome": rng.integers(0, 2, n),
        }
    )


PRE = {
    "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    "knn_imputer_neighbors": 5,
    "outlier_q1": 0.05,
    "outlier_q3": 0.95,
    "outlier_iqr_factor": 1.5,
    "categorical_threshold": 10,
}


def test_replace_zeros_creates_nan(raw):
    out = de.replace_zeros_with_nan(raw, PRE)
    assert out["Glucose"].isna().sum() == (raw["Glucose"] == 0).sum()
    assert (out["Pregnancies"] == 0).sum() == (raw["Pregnancies"] == 0).sum()  # nao mexe em Pregnancies


def test_imputer_removes_all_nan(raw):
    with_nan = de.replace_zeros_with_nan(raw, PRE)
    imputer = de.fit_imputer(with_nan, PRE)
    out = de.impute_missing(with_nan, imputer)
    assert out[PRE["zero_as_missing"]].isna().sum().sum() == 0


def test_extract_features_fixes_notebook_bugs(raw):
    out = de.extract_features(raw)
    assert set(out["NEW_INSULIN_SCORE"].unique()) <= {"Normal", "Abnormal"}
    obese = out.loc[out["BMI"] >= 30, "NEW_AGE_BMI_NOM"]
    assert obese.str.startswith("obese").all()
    healthy = out.loc[(out["BMI"] >= 18.5) & (out["BMI"] < 25), "NEW_AGE_BMI_NOM"]
    assert healthy.str.startswith("healthy").all()


def test_encoder_aligns_columns_between_train_and_new_data(raw):
    feats = de.extract_features(raw)
    encoder = de.fit_encoder(feats, "Outcome")
    # dado novo sem varias categorias -> deve sair com as MESMAS colunas do treino
    small = de.extract_features(raw.head(3).drop(columns="Outcome"))
    encoded = de.encode_features(small, encoder)
    assert list(encoded.columns) == encoder["feature_columns"]


def test_default_pipeline_has_three_stages():
    from pathlib import Path

    from kedro.framework.startup import bootstrap_project

    bootstrap_project(Path(__file__).resolve().parents[1])
    pipes = register_pipelines()
    assert {"data_engineering", "training", "inference"} <= set(pipes)
    assert len(pipes["__default__"].nodes) == sum(len(pipes[n].nodes) for n in ("data_engineering", "training", "inference"))
