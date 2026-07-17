"""
Признаки для приоритизации обращений:
- текст обращения -> TF-IDF (n-грамы слов, русский язык)
- категория -> one-hot
- repeat_count -> число повторных обращений по похожей проблеме
- день недели / месяц (сезонность потока обращений)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import numpy as np

STRUCTURED_NUMERIC = ["repeat_count", "month", "day_of_week"]
CATEGORY_COL = "category"
TEXT_COL = "text"
TARGET_COL = "is_urgent"


def load_raw(data_dir="data/raw"):
    return pd.read_csv(f"{data_dir}/appeals.csv", parse_dates=["date"])


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def build_feature_matrix(df: pd.DataFrame, vectorizer=None, encoder=None, fit=True):
    """
    Возвращает разреженную матрицу признаков (TF-IDF + категория + числовые),
    а также обученные (или переданные) vectorizer и encoder.
    """
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_text = vectorizer.fit_transform(df[TEXT_COL])

        encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = encoder.fit_transform(df[[CATEGORY_COL]])
    else:
        X_text = vectorizer.transform(df[TEXT_COL])
        X_cat = encoder.transform(df[[CATEGORY_COL]])

    X_num = df[STRUCTURED_NUMERIC].to_numpy()
    X_num_sparse = sp.csr_matrix(X_num)

    X = sp.hstack([X_text, X_cat, X_num_sparse]).tocsr()

    feature_names = (
        list(vectorizer.get_feature_names_out())
        + list(encoder.get_feature_names_out([CATEGORY_COL]))
        + STRUCTURED_NUMERIC
    )

    return X, feature_names, vectorizer, encoder


if __name__ == "__main__":
    df = load_raw()
    df = add_date_features(df)
    X, feature_names, vectorizer, encoder = build_feature_matrix(df, fit=True)
    print("Размер матрицы признаков:", X.shape)
    print("Пример признаков:", feature_names[:10], "...")
