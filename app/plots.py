"""
Переиспользуемые Plotly-графики для дашборда приоритизации обращений.

Все функции возвращают готовый `plotly.graph_objects.Figure` с тёмной
темой, совпадающей с оформлением дашборда. Цвета тиеров фиксированы:
    Срочно     — #E5534B  (красный)
    Средне     — #D9A73B  (жёлтый)
    Не срочно  — #4FAF7D  (зелёный)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# --- палитра ---------------------------------------------------------------
TIER_COLORS = {"Срочно": "#E5534B", "Средне": "#D9A73B", "Не срочно": "#4FAF7D"}
TIER_ORDER = ["Срочно", "Средне", "Не срочно"]

# нейтральные акценты для категорий/районов
SERIES_PALETTE = ["#5B8DEF", "#4FAF7D", "#D9A73B", "#E5534B", "#B069E8", "#3FB6C8"]

BG = "#12151C"        # фон страницы
CARD_BG = "#1B1F28"   # фон карточек
GRID = "#2A303B"      # сетка
TEXT = "#E6E8EB"      # основной текст
MUTED = "#9AA3AF"     # приглушённый текст


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Единая тёмная тема для всех графиков."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=13),
        margin=dict(l=8, r=8, t=40, b=8),
        colorway=SERIES_PALETTE,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=12),
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
    )
    return fig


def tier_for_score(score: float):
    """Тиер срочности по скору 0–100. Дублируется в dashboard для доступности."""
    if score >= 60:
        return "Срочно"
    if score >= 30:
        return "Средне"
    return "Не срочно"


# --- Таб 1: Обзор ----------------------------------------------------------

def score_distribution(df: pd.DataFrame) -> go.Figure:
    """Гистограмма распределения priority-скоров с разбиением по тиерам."""
    fig = go.Figure()
    for tier in TIER_ORDER:
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub["priority_score"],
                name=tier,
                marker_color=TIER_COLORS[tier],
                opacity=0.9,
                xbins=dict(start=0, end=100, size=5),
                hovertemplate="Приоритет: %{x}<br>Кол-во: %{y}<extra>" + tier + "</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Приоритет-скор (0–100)",
        yaxis_title="Кол-во обращений",
        bargap=0.05,
    )
    return _apply_theme(fig)


def appeals_by_month(df: pd.DataFrame) -> go.Figure:
    """Линейный график количества обращений по месяцам, стек по тиерам."""
    df = df.copy()
    df["month_dt"] = df["date"].dt.to_period("M").dt.to_timestamp()
    pivot = df.groupby([df["month_dt"], "tier"]).size().unstack(fill_value=0)

    fig = go.Figure()
    for tier in TIER_ORDER:
        if tier in pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[tier],
                    name=tier,
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5, color=TIER_COLORS[tier]),
                    fillcolor=TIER_COLORS[tier],
                    hovertemplate="%{x|%b %Y}<br>" + tier + ": %{y}<extra></extra>",
                )
            )
    fig.update_layout(
        xaxis_title="Месяц",
        yaxis_title="Кол-во обращений",
    )
    return _apply_theme(fig)


# --- Таб 3: Аналитика ------------------------------------------------------

def by_category_stacked(df: pd.DataFrame) -> go.Figure:
    """Количество обращений по категориям, стек по тиерам срочности."""
    pivot = df.groupby(["category", "tier"]).size().unstack(fill_value=0)
    # упорядочить категории по убыванию общего числа
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig = go.Figure()
    for tier in TIER_ORDER:
        if tier in pivot.columns:
            fig.add_trace(
                go.Bar(
                    y=pivot.index,
                    x=pivot[tier],
                    name=tier,
                    orientation="h",
                    marker_color=TIER_COLORS[tier],
                    hovertemplate="%{y}<br>" + tier + ": %{x}<extra></extra>",
                )
            )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Кол-во обращений",
        yaxis_title="",
        bargap=0.25,
    )
    return _apply_theme(fig)


def score_box_by_category(df: pd.DataFrame) -> go.Figure:
    """Box-plot распределения скоров по категориям."""
    cats = sorted(df["category"].unique())
    fig = go.Figure()
    for i, cat in enumerate(cats):
        sub = df[df["category"] == cat]
        fig.add_trace(
            go.Box(
                y=sub["priority_score"],
                name=cat,
                marker_color=SERIES_PALETTE[i % len(SERIES_PALETTE)],
                boxpoints="outliers",
                line_width=1.5,
                hovertemplate="%{y:.0f}<extra>" + cat + "</extra>",
            )
        )
    fig.update_layout(
        yaxis_title="Приоритет-скор",
        xaxis_title="",
        showlegend=False,
    )
    return _apply_theme(fig)


def by_district(df: pd.DataFrame) -> go.Figure:
    """Количество обращений по районам."""
    counts = df["district"].value_counts().sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            y=counts.index,
            x=counts.values,
            orientation="h",
            marker_color=SERIES_PALETTE[0],
            marker_line_width=0,
            hovertemplate="%{y}<br>Кол-во: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Кол-во обращений",
        yaxis_title="",
        bargap=0.25,
    )
    return _apply_theme(fig)


def heatmap_district_category(df: pd.DataFrame) -> go.Figure:
    """Тепловая карта: район × категория (количество обращений)."""
    pivot = df.groupby(["district", "category"]).size().unstack(fill_value=0)
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[[0, CARD_BG], [1, "#E5534B"]],
            hovertemplate="Район: %{y}<br>Категория: %{x}<br>Кол-во: %{z}<extra></extra>",
            colorbar=dict(title="Кол-во", tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        )
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
    )
    return _apply_theme(fig)


def repeat_vs_priority(df: pd.DataFrame) -> go.Figure:
    """Средний приоритет по количеству повторных обращений."""
    grp = df.groupby("repeat_count")["priority_score"].agg(["mean", "count"]).reset_index()
    grp = grp[grp["count"] >= 3]  # отбрасываем малочисленные группы
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grp["repeat_count"],
            y=grp["mean"],
            name="Средний приоритет",
            marker_color=SERIES_PALETTE[0],
            yaxis="y",
            hovertemplate="Повторов: %{x}<br>Ср. приоритет: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grp["repeat_count"],
            y=grp["count"],
            name="Кол-во обращений",
            mode="lines+markers",
            line=dict(color=MUTED, width=1.5, dash="dot"),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="Повторов: %{x}<br>Кол-во обращений: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Кол-во повторных обращений",
        yaxis=dict(title="Средний приоритет", side="left"),
        yaxis2=dict(title="Кол-во обращений", side="right", overlaying="y", showgrid=False),
        legend=dict(y=1.0, yanchor="bottom"),
    )
    return _apply_theme(fig)


# --- Таб 5: О модели -------------------------------------------------------

def roc_curve_fig(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """ROC-кривая."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.3f})",
            line=dict(color=SERIES_PALETTE[0], width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Случайный классификатор",
            line=dict(color=MUTED, width=1, dash="dash"),
            hovertemplate="",
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return _apply_theme(fig)


def pr_curve_fig(precision: np.ndarray, recall: np.ndarray, ap: float) -> go.Figure:
    """Precision-Recall кривая."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR (AP = {ap:.3f})",
            line=dict(color=TIER_COLORS["Средне"], width=2.5),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return _apply_theme(fig)


def confusion_matrix_fig(cm: np.ndarray) -> go.Figure:
    """Тепловая карта матрицы ошибок."""
    labels = ["Не срочно", "Срочно"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, CARD_BG], [1, "#5B8DEF"]],
            text=cm,
            texttemplate="%{text}",
            textfont=dict(color=TEXT, size=16),
            hovertemplate="Факт: %{y}<br>Предсказано: %{x}<br>Кол-во: %{z}<extra></extra>",
            colorbar=dict(title="Кол-во", tickfont=dict(color=MUTED), title_font=dict(color=MUTED)),
        )
    )
    fig.update_layout(
        xaxis_title="Предсказано",
        yaxis_title="Факт",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_theme(fig)


def top_features_bar(feature_names, coefs: np.ndarray, k: int = 20) -> go.Figure:
    """Топ-k признаков по модулю коэффициента: повышающие vs понижающие."""
    order = np.argsort(np.abs(coefs))[::-1][:k]
    names = [feature_names[i] for i in order]
    vals = [coefs[i] for i in order]

    colors = [TIER_COLORS["Срочно"] if v > 0 else TIER_COLORS["Не срочно"] for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Коэффициент: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Коэффициент модели (вклад в срочность)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        bargap=0.3,
    )
    return _apply_theme(fig)
