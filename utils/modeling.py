import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)


# ---------- Training helpers ----------

def split_train_test(X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series, test_size: float = 0.2):
    """Simple chronological train/test split.

    *test_size* is the fraction of data to reserve for testing.  Returns a
    tuple of ``(X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test)``.
    """

    split = int(len(X) * (1 - test_size))
    X_train = X[:split]
    X_test = X[split:]
    y_reg_train = y_reg[:split]
    y_reg_test = y_reg[split:]
    y_clf_train = y_clf[:split]
    y_clf_test = y_clf[split:]
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def train_regressor(X: pd.DataFrame, y: pd.Series, **kwargs) -> XGBRegressor:
    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": 42,
    }
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_classifier(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float = 1.0, **kwargs) -> XGBClassifier:
    params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
    }
    params.update(kwargs)
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def evaluate_regressor(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


def evaluate_classifier(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.55) -> dict:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > threshold).astype(int)
    return {"accuracy": float(accuracy_score(y_test, pred)), "prob_up": float(prob.mean())}


def get_feature_importances(model: XGBClassifier, feature_names: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    })
    return df.sort_values("Importance", ascending=False)
