"""Bộ mô hình (đề bài 2.6–2.8): Logistic (OvR), Multinomial Logistic, Random Forest, XGBoost.

Mọi bộ tiền xử lý (điền thiếu, chuẩn hóa) nằm TRONG Pipeline và chỉ fit trên train -> không rò rỉ.
Mô hình stress được hiệu chỉnh xác suất (isotonic) trên NGÀY TRAIN CUỐI, không dùng dữ liệu val/test.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

DIRECTION_MODELS = ["LogReg-OvR", "MNLogit", "RandomForest", "XGBoost"]
STRESS_MODELS = ["LogReg", "RandomForest", "XGBoost"]


def _linear(clf):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def make_model(name: str, task: str, seed: int = 42, n_trees: int | None = None):
    """task: 'direction' (3 lớp) hoặc 'stress' (nhị phân)."""
    if name in ("LogReg-OvR", "LogReg"):
        lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
        return _linear(OneVsRestClassifier(lr) if task == "direction" else lr)
    if name == "MNLogit":
        return _linear(LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
    if name == "RandomForest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=n_trees or 200, max_depth=10, min_samples_leaf=100,
                                                    max_samples=0.3, max_features="sqrt",
                                                    class_weight="balanced_subsample", n_jobs=-1, random_state=seed))
    if name == "XGBoost":
        kw = dict(n_estimators=n_trees or 400, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0, tree_method="hist",
                  n_jobs=-1, random_state=seed, verbosity=0)
        if task == "direction":
            return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **kw)
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
    raise KeyError(name)


def fit_direction(name: str, X: np.ndarray, y: np.ndarray, seed: int = 42, n_trees: int | None = None):
    """y ∈ {−1, 0, 1} -> mã hóa 0/1/2 = down/flat/up. Cân bằng lớp bằng trọng số."""
    m = make_model(name, "direction", seed, n_trees)
    yc = (np.asarray(y) + 1).astype(int)
    if name == "XGBoost":
        m.fit(X, yc, sample_weight=compute_sample_weight("balanced", yc))
    else:
        m.fit(X, yc)
    return m


def predict_direction(model, X: np.ndarray) -> np.ndarray:
    """Xác suất (n, 3) theo thứ tự [down, flat, up]."""
    return model.predict_proba(X)


def fit_stress(name: str, X_fit, y_fit, X_cal, y_cal, seed: int = 42, n_trees: int | None = None):
    """Fit trên X_fit, hiệu chỉnh isotonic trên X_cal (ngày train cuối). XGB dùng scale_pos_weight."""
    m = make_model(name, "stress", seed, n_trees)
    y_fit = np.asarray(y_fit).astype(int)
    if name == "XGBoost":
        pos = max(y_fit.mean(), 1e-6)
        m.set_params(scale_pos_weight=(1 - pos) / pos)
    m.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic")
    cal.fit(X_cal, np.asarray(y_cal).astype(int))
    return cal


def predict_stress(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def model_complexity(model) -> dict:
    """Số tham số / số cây để so sánh độ phức tạp (bảng latency/complexity)."""
    est = model
    if isinstance(est, CalibratedClassifierCV):
        est = est.calibrated_classifiers_[0].estimator
        est = getattr(est, "estimator", est)
    last = est.steps[-1][1] if hasattr(est, "steps") else est
    if isinstance(last, XGBClassifier):
        df = last.get_booster().trees_to_dataframe()
        return {"type": "trees", "n_trees": int(df["Tree"].nunique()), "n_nodes": int(len(df))}
    if isinstance(last, RandomForestClassifier):
        return {"type": "trees", "n_trees": len(last.estimators_),
                "n_nodes": int(sum(t.tree_.node_count for t in last.estimators_))}
    if isinstance(last, OneVsRestClassifier):
        return {"type": "linear", "n_params": int(sum(e.coef_.size + e.intercept_.size for e in last.estimators_))}
    if isinstance(last, LogisticRegression):
        return {"type": "linear", "n_params": int(last.coef_.size + last.intercept_.size)}
    return {"type": type(last).__name__}
