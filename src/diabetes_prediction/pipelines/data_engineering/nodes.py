"""Nos do pipeline de engenharia de dados.

Traducao dos capitulos 15-19 do notebook ``diabetes-prediction.ipynb`` para
funcoes puras, com uma diferenca deliberada: TODO artefato que "aprende" algo
dos dados (imputer, limites de outlier, encoder, scaler) e ajustado (``fit_*``)
APENAS no conjunto de treino e depois aplicado (``transform``) a treino, teste e
inferencia. Isso elimina o *data leakage* presente no notebook, que ajustava
tudo no dataset completo antes do split.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. Zeros -> NaN
# ----------------------------------------------------------------------------
def replace_zeros_with_nan(data: pd.DataFrame, preprocessing: dict[str, Any]) -> pd.DataFrame:
    """Marca como missing os zeros fisiologicamente impossiveis (cap. 15)."""
    df = data.copy()
    cols = [c for c in preprocessing["zero_as_missing"] if c in df.columns]
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
    logger.info("Zeros convertidos em NaN em %s. Missing por coluna:\n%s", cols, df[cols].isna().sum().to_dict())
    return df


# ----------------------------------------------------------------------------
# 2. Split treino/teste (ANTES de qualquer fit)
# ----------------------------------------------------------------------------
def split_train_test(data: pd.DataFrame, split: dict[str, Any], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino e teste mantendo a coluna alvo em ambos (cap. 20)."""
    train, test = train_test_split(
        data,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=data[target],
    )
    logger.info("Split: treino=%d linhas, teste=%d linhas", len(train), len(test))
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Imputacao KNN (fit no treino / transform em qualquer conjunto)
# ----------------------------------------------------------------------------
def fit_imputer(train: pd.DataFrame, preprocessing: dict[str, Any]) -> dict[str, Any]:
    """Ajusta RobustScaler + KNNImputer nas colunas com missing (cap. 15, Option 2)."""
    cols = [c for c in preprocessing["zero_as_missing"] if c in train.columns]
    scaler = RobustScaler().fit(train[cols])
    scaled = pd.DataFrame(scaler.transform(train[cols]), columns=cols)
    imputer = KNNImputer(n_neighbors=preprocessing["knn_imputer_neighbors"]).fit(scaled)
    return {"columns": cols, "scaler": scaler, "imputer": imputer}


def impute_missing(data: pd.DataFrame, imputer: dict[str, Any]) -> pd.DataFrame:
    """Aplica o imputer ajustado: escala -> imputa -> desescala."""
    df = data.copy()
    cols = imputer["columns"]
    scaled = pd.DataFrame(imputer["scaler"].transform(df[cols]), columns=cols)
    filled = pd.DataFrame(imputer["imputer"].transform(scaled), columns=cols)
    df[cols] = imputer["scaler"].inverse_transform(filled)
    return df


# ----------------------------------------------------------------------------
# 4. Outliers (limites calculados no treino)
# ----------------------------------------------------------------------------
def fit_outlier_thresholds(train: pd.DataFrame, preprocessing: dict[str, Any], target: str) -> dict[str, list[float]]:
    """Calcula limites inferior/superior por coluna numerica (cap. 16)."""
    q1, q3, k = preprocessing["outlier_q1"], preprocessing["outlier_q3"], preprocessing["outlier_iqr_factor"]
    thresholds: dict[str, list[float]] = {}
    for col in train.columns:
        if col == target or not pd.api.types.is_numeric_dtype(train[col]):
            continue
        lo_q, hi_q = train[col].quantile(q1), train[col].quantile(q3)
        iqr = hi_q - lo_q
        thresholds[col] = [float(lo_q - k * iqr), float(hi_q + k * iqr)]
    return thresholds


def cap_outliers(data: pd.DataFrame, thresholds: dict[str, list[float]]) -> pd.DataFrame:
    """Substitui valores fora dos limites pelos proprios limites (winsorizacao)."""
    df = data.copy()
    n_capped = 0
    for col, (low, up) in thresholds.items():
        if col not in df.columns:
            continue
        mask = (df[col] < low) | (df[col] > up)
        n_capped += int(mask.sum())
        df[col] = df[col].clip(lower=low, upper=up)
    logger.info("Outliers tratados: %d valores ajustados aos limites", n_capped)
    return df


# ----------------------------------------------------------------------------
# 5. Feature extraction (sem fit - regras de negocio fixas, cap. 17)
# ----------------------------------------------------------------------------
def _age_group(age: pd.Series) -> pd.Series:
    return pd.Series(np.where(age >= 50, "senior", "mature"), index=age.index)


def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cria as variaveis derivadas do notebook e converte nomes para maiusculas.

    Correcoes em relacao ao notebook:
    * ``NEW_AGE_BMI_NOM``: as regras de "obese" usavam ``BMI > 18.5`` (bug que
      sobrescrevia as demais categorias); aqui usam ``BMI >= 30``.
    * ``NEW_INSULIN_SCORE``: retorna "Normal" em vez de ``None`` para o intervalo
      16-166, tornando a variavel realmente binaria.
    * Nomes de colunas sem espacos/asteriscos (compatibilidade com XGBoost/LightGBM).
    """
    df = data.copy()

    age_cat = _age_group(df["Age"])
    df["NEW_AGE_CAT"] = age_cat

    df["NEW_BMI"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"]
    ).astype(str)

    df["NEW_GLUCOSE"] = pd.cut(
        df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"]
    ).astype(str)

    bmi_cat = pd.Series(
        np.select(
            [df["BMI"] < 18.5, df["BMI"] < 25, df["BMI"] < 30],
            ["underweight", "healthy", "overweight"],
            default="obese",
        ),
        index=df.index,
    )
    df["NEW_AGE_BMI_NOM"] = bmi_cat + age_cat

    glucose_cat = pd.Series(
        np.select(
            [df["Glucose"] < 70, df["Glucose"] < 100, df["Glucose"] <= 125],
            ["low", "normal", "hidden"],
            default="high",
        ),
        index=df.index,
    )
    df["NEW_AGE_GLUCOSE_NOM"] = glucose_cat + age_cat

    df["NEW_INSULIN_SCORE"] = np.where(df["Insulin"].between(16, 166), "Normal", "Abnormal")
    df["NEW_GLUCOSE_X_INSULIN"] = df["Glucose"] * df["Insulin"]
    df["NEW_GLUCOSE_X_PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

    df.columns = [c.upper() for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# 6. Encoding (fit no treino)
# ----------------------------------------------------------------------------
def _categorical_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Colunas categoricas: object/str (pandas 2 e 3) ou Categorical."""
    return [
        c
        for c in df.columns
        if c != target
        and (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        )
    ]


def fit_encoder(train: pd.DataFrame, target: str) -> dict[str, Any]:
    """Define label encoding para binarias e one-hot (drop_first) para as demais (cap. 18)."""
    target = target.upper()
    cat_cols = _categorical_columns(train, target)
    binary_cols = [c for c in cat_cols if train[c].nunique() == 2]
    onehot_cols = [c for c in cat_cols if c not in binary_cols]

    # LabelEncoder ordena alfabeticamente: classes[0] -> 0, classes[1] -> 1
    binary_maps = {c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique()))} for c in binary_cols}

    encoded = _apply_encoding(train, binary_maps, onehot_cols, feature_columns=None)
    feature_columns = [c for c in encoded.columns if c != target]
    return {
        "target": target,
        "binary_maps": binary_maps,
        "onehot_cols": onehot_cols,
        "feature_columns": feature_columns,
    }


def _apply_encoding(
    df: pd.DataFrame,
    binary_maps: dict[str, dict[str, int]],
    onehot_cols: list[str],
    feature_columns: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in binary_maps.items():
        out[col] = out[col].map(mapping).astype("Int64")
    out = pd.get_dummies(out, columns=onehot_cols, drop_first=True, dtype=int)
    if feature_columns is not None:
        # Garante o MESMO conjunto/ordem de colunas do treino (categorias ausentes -> 0)
        target_cols = [c for c in out.columns if c not in feature_columns]  # ex.: OUTCOME
        out = out.reindex(columns=feature_columns + target_cols, fill_value=0)
    return out


def encode_features(data: pd.DataFrame, encoder: dict[str, Any]) -> pd.DataFrame:
    """Aplica o encoding definido no treino."""
    return _apply_encoding(data, encoder["binary_maps"], encoder["onehot_cols"], encoder["feature_columns"])


# ----------------------------------------------------------------------------
# 7. Padronizacao (fit no treino, cap. 19)
# ----------------------------------------------------------------------------
def fit_scaler(train: pd.DataFrame, preprocessing: dict[str, Any], target: str) -> dict[str, Any]:
    """Ajusta RobustScaler nas colunas numericas (regra grab_col_names: nunique >= cat_th)."""
    target = target.upper()
    th = preprocessing["categorical_threshold"]
    num_cols = [c for c in train.columns if c != target and pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique() >= th]
    scaler = RobustScaler().fit(train[num_cols])
    logger.info("Scaler ajustado em %d colunas numericas", len(num_cols))
    return {"columns": num_cols, "scaler": scaler}


def scale_features(data: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    """Aplica o scaler ajustado no treino."""
    df = data.copy()
    cols = scaler["columns"]
    df[cols] = scaler["scaler"].transform(df[cols])
    return df


# ----------------------------------------------------------------------------
# 8. Separa X e y
# ----------------------------------------------------------------------------
def split_features_target(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa matriz de features e vetor alvo."""
    target = target.upper()
    X = data.drop(columns=[target])
    y = data[[target]]
    return X, y
