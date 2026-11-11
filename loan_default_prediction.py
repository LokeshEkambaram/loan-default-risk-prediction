"""Project: Loan Default Risk Prediction.

Author: Lokesh Ekambaram
Dataset: UCI 'Default of Credit Card Clients' via OpenML (fallback: German Credit)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score, average_precision_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Fetching dataset...")
    try:
        data = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    except Exception as e:
        logger.warning(f"Primary dataset fetch failed ({e}). Using fallback dataset...")
        data = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X, y = data.data, data.target
    if y.dtype.name == "category" or y.dtype == object:
        y = y.astype(str).str.lower().isin(["1", "yes", "default", "bad"]).astype(int)
    else:
        y = y.astype(int)
    return X, y


def main():
    X, y = load_data()
    print(f"Total records (N): {len(X)} | Features: {X.shape[1]} | Default rate: {100 * y.mean():.1f}%")
    print(f"Missing values: {int(X.isna().sum().sum())}")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Stratified split keeps the default rate the same in train and test
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

    # class_weight="balanced" handles the class imbalance
    models = {
        "Logistic Regression (L2)": Pipeline([("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=42))]),
        "Hist Gradient Boosting": Pipeline([("pre", preprocessor),
            ("clf", HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42))]),
        "Random Forest Classifier": Pipeline([("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                           random_state=42, n_jobs=-1))]),
    }

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 110)
    print(f"{'Model Classification Evaluation Summary':^110}")
    print("=" * 110)

    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_tr, y_tr, cv=cv_strat, scoring="roc_auc")
        pipeline.fit(X_tr, y_tr)
        probs = pipeline.predict_proba(X_te)[:, 1]
        preds = pipeline.predict(X_te)
        print(f"{name:26s} | CV-AUC: {cv_scores.mean():.3f} | Test-AUC: {roc_auc_score(y_te, probs):.3f}"
              f" | PR-AUC: {average_precision_score(y_te, probs):.3f}"
              f" | Recall: {recall_score(y_te, preds, zero_division=0):.3f}"
              f" | Precision: {precision_score(y_te, preds, zero_division=0):.3f}"
              f" | F1: {f1_score(y_te, preds, zero_division=0):.3f}")
    print("=" * 110)

    # Feature importance from the Random Forest (already fitted above)
    rf = models["Random Forest Classifier"]
    names = rf.named_steps["pre"].get_feature_names_out()
    top = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(5)
    print("\nTop 5 default drivers (Random Forest importance):")
    print(top.to_string())


if __name__ == "__main__":
    main()
