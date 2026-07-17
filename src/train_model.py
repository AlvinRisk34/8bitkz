"""
Обучение модели приоритизации обращений граждан.

Модель: логистическая регрессия на TF-IDF + структурные признаки.
Почему логрег, а не бустинг: коэффициенты линейно интерпретируемы —
для каждого обращения легко показать, какие слова и признаки повысили
или понизили срочность (contribution = coef * feature_value).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split

from features import (
    TARGET_COL,
    add_date_features,
    build_feature_matrix,
    load_raw,
)


def train():
    df = load_raw()
    df = add_date_features(df)

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.25, random_state=42, stratify=df[TARGET_COL]
    )
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]

    X_train, feature_names, vectorizer, encoder = build_feature_matrix(df_train, fit=True)
    X_test, _, _, _ = build_feature_matrix(df_test, vectorizer=vectorizer, encoder=encoder, fit=False)

    y_train, y_test = df_train[TARGET_COL], df_test[TARGET_COL]

    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"ROC-AUC (test): {auc:.3f}")
    print(f"PR-AUC  (test): {ap:.3f}")
    print(classification_report(y_test, proba > 0.5))

    joblib.dump(model, "data/processed/model.pkl")
    joblib.dump(vectorizer, "data/processed/vectorizer.pkl")
    joblib.dump(encoder, "data/processed/encoder.pkl")
    joblib.dump(feature_names, "data/processed/feature_names.pkl")
    df.to_pickle("data/processed/appeals_full.pkl")

    print("Модель и артефакты сохранены в data/processed/")
    return model, vectorizer, encoder, feature_names, df


if __name__ == "__main__":
    train()
