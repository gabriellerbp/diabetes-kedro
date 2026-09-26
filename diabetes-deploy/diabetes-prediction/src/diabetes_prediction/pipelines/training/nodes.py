"""Nos do pipeline de treinamento (cap. 20 do notebook)."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def _build_estimator(model_type: str, random_state: int, **params: Any) -> BaseEstimator:
    """Fabrica de modelos - os 9 classificadores do notebook, selecionaveis por parametro."""
    registry = {
        "random_forest": lambda: RandomForestClassifier(random_state=random_state, **params),
        "logistic_regression": lambda: LogisticRegression(random_state=random_state, max_iter=1000, **params),
        "knn": lambda: KNeighborsClassifier(**params),
        "svc": lambda: SVC(random_state=random_state, probability=True, **params),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=random_state, **params),
        "adaboost": lambda: AdaBoostClassifier(random_state=random_state, **params),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=random_state, **params),
    }
    if model_type == "xgboost":
        from xgboost import XGBClassifier  # import tardio: dependencia opcional pesada

        return XGBClassifier(random_state=random_state, **params)
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=random_state, verbose=-1, **params)
    if model_type not in registry:
        raise ValueError(f"model_type desconhecido: {model_type!r}. Opcoes: {sorted(registry) + ['xgboost', 'lightgbm']}")
    return registry[model_type]()


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, training: dict[str, Any]) -> tuple[BaseEstimator, dict[str, Any]]:
    """Treina o modelo, com ou sem GridSearchCV, conforme ``params:training``."""
    y = y_train.iloc[:, 0]
    model_type, rs = training["model_type"], training["random_state"]

    if training.get("hyperparameter_tuning", False):
        base = _build_estimator(model_type, rs)
        grid = GridSearchCV(base, training["param_grid"], cv=training["cv_folds"], n_jobs=-1)
        grid.fit(X_train, y)
        logger.info("GridSearchCV concluido. Melhor CV accuracy=%.4f params=%s", grid.best_score_, grid.best_params_)
        best_params = {"model_type": model_type, "cv_accuracy": round(float(grid.best_score_), 4), **grid.best_params_}
        return grid.best_estimator_, _jsonable(best_params)

    model = _build_estimator(model_type, rs, **training.get("fixed_params", {}))
    model.fit(X_train, y)
    return model, _jsonable({"model_type": model_type, **training.get("fixed_params", {})})


def compute_metrics(y_true: pd.Series, y_pred, y_proba=None) -> dict[str, float]:
    """Accuracy, recall, precision, F1 e AUC.

    Observacao: o notebook chama ``recall_score(y_pred, y_test)`` (argumentos
    invertidos), o que troca recall por precision. Aqui a ordem e a correta:
    ``(y_true, y_pred)``. AUC usa probabilidades quando disponiveis.
    """
    auc_input = y_proba if y_proba is not None else y_pred
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_true, auc_input)), 4),
    }


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict[str, Any]:
    """Avalia no conjunto de teste (nunca visto por nenhum fit)."""
    y = y_test.iloc[:, 0]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y, y_pred, y_proba)
    metrics["n_test"] = int(len(y))
    logger.info("Metricas de teste: %s", metrics)
    return metrics


def extract_feature_importances(model: BaseEstimator, X_train: pd.DataFrame) -> pd.DataFrame:
    """Importancia das variaveis (cap. 20.1 plot_importance) - tabela em vez de grafico."""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = abs(model.coef_).ravel()
    else:
        return pd.DataFrame({"feature": X_train.columns, "importance": float("nan")})
    return (
        pd.DataFrame({"feature": X_train.columns, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
