"""
Дашборд: очередь обращений граждан, отсортированная по приоритету.

Прототип отборочного этапа GovTech Camp.

Запуск:
    streamlit run app/dashboard.py
"""

import html
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from features import add_date_features, build_feature_matrix, load_raw
import plots as P

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# зона вокруг 50/100, где модель считается неуверенной
UNCERTAINTY_BAND = (35, 65)
TIER_THRESHOLDS = (60, 30)
PAGE_SIZE = 30

st.set_page_config(
    page_title="ELNIT — Приоритизация обращений",
    layout="wide",
    page_icon=str(Path(__file__).resolve().parent / "assets" / "favicon.png"),
)

# ---------------------------------------------------------------------------
# Оформление: сдержанная «казначейская» палитра. Моноширинный шрифт — для
# чисел и id (данные), гротеск — для текста. Карточки обращений с цветной
# рейлкой слева имитируют папку дела.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #12151C;
    }
    h1, h2, h3, h4 {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #E6E8EB !important;
        font-weight: 600;
    }
    /* подписи виджетов и caption фиксируем под тёмную палитру */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"] p {
        color: #E6E8EB !important;
    }
    .gt-subtitle {
        color: #9AA3AF;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 18px;
    }
    .gt-kpi {
        background: #1B1F28;
        border: 1px solid #2A303B;
        border-radius: 8px;
        padding: 16px 20px;
        height: 100%;
    }
    .gt-kpi-label {
        font-size: 12px;
        color: #9AA3AF;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .gt-kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #E6E8EB;
        margin-top: 6px;
    }
    .gt-kpi-delta {
        font-size: 12px;
        color: #9AA3AF;
        margin-top: 4px;
    }
    .gt-case {
        background: #1B1F28;
        border: 1px solid #2A303B;
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .gt-case-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        color: #9AA3AF;
    }
    .gt-case-score {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 16px;
        color: var(--accent);
        float: right;
    }
    .gt-case-tier {
        font-size: 13px;
        font-weight: 600;
        color: var(--accent);
        letter-spacing: .02em;
        margin-top: 6px;
    }
    .gt-uncertain-tag {
        display: inline-block;
        margin-left: 8px;
        font-size: 11px;
        font-weight: 500;
        color: #9AA3AF;
        background: #262C36;
        border: 1px solid #2A303B;
        border-radius: 3px;
        padding: 1px 6px;
        vertical-align: middle;
    }
    .gt-case summary {
        cursor: pointer;
        font-size: 14px;
        color: #E6E8EB;
        margin-top: 8px;
    }
    .gt-case-body {
        margin-top: 10px;
        font-size: 14px;
        color: #E6E8EB;
        line-height: 1.5;
    }
    /* explainability-бары: горизонтальные полоски вклада факторов */
    .gt-factors {
        margin-top: 12px;
    }
    .gt-factor-row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .gt-factor-name {
        font-family: 'IBM Plex Mono', monospace;
        color: #C3C9D1;
        width: 140px;
        flex-shrink: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .gt-factor-bar-wrap {
        flex: 1;
        background: #262C36;
        border-radius: 3px;
        height: 14px;
        margin: 0 10px;
        overflow: hidden;
    }
    .gt-factor-bar {
        height: 100%;
        background: var(--accent);
        border-radius: 3px;
    }
    .gt-factor-val {
        font-family: 'IBM Plex Mono', monospace;
        color: #9AA3AF;
        width: 55px;
        text-align: right;
        flex-shrink: 0;
    }
    .gt-footnote {
        color: #9AA3AF;
        font-size: 13px;
        line-height: 1.5;
    }
    .gt-callout {
        background: #1B1F28;
        border: 1px solid #2A303B;
        border-left: 3px solid #5B7FA6;
        border-radius: 4px;
        padding: 12px 16px;
        color: #C3C9D1;
        font-size: 13.5px;
        line-height: 1.5;
    }
    /* сайдбар */
    section[data-testid="stSidebar"] {
        background-color: #161A22;
    }
    .gt-sidebar-title {
        font-weight: 700;
        font-size: 16px;
        color: #E6E8EB;
    }
    .gt-sidebar-note {
        font-size: 12.5px;
        color: #9AA3AF;
        line-height: 1.5;
    }
    /* убрать лишние отступы у tab-ов */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Загрузка артефактов
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Вспомогательные функции (определены до загрузчиков, т.к. используются в них)
# ---------------------------------------------------------------------------
def tier_for_score(score: float) -> str:
    if score >= TIER_THRESHOLDS[0]:
        return "Срочно"
    if score >= TIER_THRESHOLDS[1]:
        return "Средне"
    return "Не срочно"


def tier_color(tier: str) -> str:
    return P.TIER_COLORS.get(tier, "#9AA3AF")


def get_top_factors(row_pos: int, k: int = 5):
    """Топ-k факторов с положительным вкладом в скор обращения."""
    row = contrib_matrix.getrow(row_pos).toarray().flatten()
    top_idx = np.argsort(row)[::-1][:k]
    return [(feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]


def render_explainability_bars(factors, accent: str):
    """Горизонтальные CSS-полоски вклада факторов."""
    if not factors:
        return "<div class='gt-footnote'>Явных повышающих факторов не найдено.</div>"
    max_val = max((v for _, v in factors), default=1)
    rows = []
    for feat, val in factors:
        width_pct = max(2, (val / max_val) * 100) if max_val > 0 else 0
        rows.append(
            f"<div class='gt-factor-row'>"
            f"<div class='gt-factor-name' title='{html.escape(feat)}'>{html.escape(feat)}</div>"
            f"<div class='gt-factor-bar-wrap'>"
            f"<div class='gt-factor-bar' style='width:{width_pct:.0f}%'></div>"
            f"</div>"
            f"<div class='gt-factor-val'>+{val:.2f}</div>"
            f"</div>"
        )
    return f"<div class='gt-factors' style='--accent:{accent}'>{''.join(rows)}</div>"


# ---------------------------------------------------------------------------
# Загрузка артефактов и данных (после определения вспомогательных функций)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(DATA_DIR / "processed" / "model.pkl")
    vectorizer = joblib.load(DATA_DIR / "processed" / "vectorizer.pkl")
    encoder = joblib.load(DATA_DIR / "processed" / "encoder.pkl")
    feature_names = joblib.load(DATA_DIR / "processed" / "feature_names.pkl")
    return model, vectorizer, encoder, feature_names


model, vectorizer, encoder, feature_names = load_artifacts()


@st.cache_data
def load_dataframe():
    """Загружаем сырые данные и считаем приоритеты для всех обращений."""
    df = load_raw()
    df = add_date_features(df)
    X_all, _, _, _ = build_feature_matrix(df, vectorizer=vectorizer, encoder=encoder, fit=False)
    df = df.reset_index(drop=True)
    df["priority_score"] = model.predict_proba(X_all)[:, 1] * 100
    df["_row_pos"] = np.arange(len(df))
    df["is_uncertain"] = df["priority_score"].between(*UNCERTAINTY_BAND)
    df["tier"] = df["priority_score"].apply(tier_for_score)
    # матрица вкладов факторов (векторно, один раз)
    coefs = model.coef_[0]
    contrib = X_all.multiply(coefs).tocsr()
    return df, contrib


df, contrib_matrix = load_dataframe()
coefs = model.coef_[0]


@st.cache_data
def load_test_predictions():
    """Метрики модели на тестовой выборке (тот же split, что в train_model.py)."""
    raw = load_raw()
    raw = add_date_features(raw)
    idx_train, idx_test = train_test_split(
        raw.index, test_size=0.25, random_state=42, stratify=raw["is_urgent"]
    )
    df_test = raw.loc[idx_test]
    X_test, _, _, _ = build_feature_matrix(
        df_test, vectorizer=vectorizer, encoder=encoder, fit=False
    )
    y_test = df_test["is_urgent"].values
    proba = model.predict_proba(X_test)[:, 1]
    return y_test, proba


def render_case_card(row, accent: str, compact: bool = False):
    """HTML-карточка обращения с explainability-барами."""
    score = row["priority_score"]
    tier_label = row["tier"]
    uncertain_html = (
        '<span class="gt-uncertain-tag">низкая уверенность модели</span>'
        if row["is_uncertain"]
        else ""
    )

    top_factors = get_top_factors(row["_row_pos"])
    factors_html = render_explainability_bars(top_factors, accent)

    text_escaped = html.escape(row["text"])
    preview = row["text"] if len(row["text"]) <= 70 else row["text"][:67] + "…"
    preview_escaped = html.escape(preview)

    body_extra = ""
    if not compact:
        body_extra = (
            f"<div class='gt-case-body'>{text_escaped}</div>"
            f"<div class='gt-case-factors' style='margin-top:10px;font-size:13px;color:#C3C9D1'>"
            f"Повторных обращений по похожей проблеме: <b>{int(row['repeat_count'])}</b><br>"
            f"<b style='color:#E6E8EB'>Что повысило приоритет:</b>"
            f"</div>"
            f"{factors_html}"
        )

    return f"""
    <div class="gt-case" style="--accent:{accent}">
        <div class="gt-case-score">{score:.0f}/100</div>
        <div class="gt-case-meta">№{int(row['appeal_id'])} · {html.escape(row['category'])} · {html.escape(row['district'])} · {row['date'].strftime('%d.%m.%Y')}</div>
        <div class="gt-case-tier">{tier_label}{uncertain_html}</div>
        <details>
            <summary>{preview_escaped}</summary>
            {body_extra}
        </details>
    </div>
    """


# ===========================================================================
# Сайдбар
# ===========================================================================
with st.sidebar:
    logo_col, title_col = st.columns([1, 3])
    with logo_col:
        st.image(str(Path(__file__).resolve().parent / "assets" / "logo.png"), width=56)
    with title_col:
        st.markdown(
            "<div class='gt-sidebar-title' style='padding-top:10px'>ELNIT</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='gt-sidebar-note'><b>Enhanced Local Network for Intelligent Triage</b><br><br>"
        "Прототип GovTech Camp.<br>"
        "Обращения ранжируются по вероятной срочности, а не по порядку поступления.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Источник данных**")
    st.markdown(
        "<div class='gt-sidebar-note'>Синтетические данные (1500 обращений), "
        "сгенерированные в <code>src/generate_synthetic_data.py</code>. "
        "Логика: 5 категорий, 3 уровня срочности, случайные детали. "
        "Подготовлено для замены на реальные данные eOtinish.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Роль человека**")
    st.markdown(
        "<div class='gt-sidebar-note'>Модель <b>не закрывает и не отклоняет</b> обращения. "
        "Она только переупорядочивает очередь и объясняет срочность. "
        "Финальное решение — за оператором ЦОН / акимата.</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"<div class='gt-sidebar-note'>Всего обращений: <b>{len(df)}</b><br>"
        f"Период: {df['date'].min().strftime('%m.%Y')}–{df['date'].max().strftime('%m.%Y')}</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Заголовок
# ===========================================================================
st.title("ELNIT — Приоритизация обращений граждан")
st.markdown(
    '<div class="gt-subtitle"><b>Enhanced Local Network for Intelligent Triage</b> · '
    "Прототип GovTech Camp — обращения ранжируются "
    "по вероятной срочности, а не по порядку поступления.</div>",
    unsafe_allow_html=True,
)

# ===========================================================================
# Табы
# ===========================================================================
tab_overview, tab_queue, tab_analytics, tab_new, tab_model = st.tabs(
    ["📊 Обзор", "📋 Очередь обращений", "📈 Аналитика", "➕ Новое обращение", "🧠 О модели"]
)

# ---------------------------------------------------------------------------
# Таб 1: Обзор
# ---------------------------------------------------------------------------
with tab_overview:
    urgent = int((df["priority_score"] >= TIER_THRESHOLDS[0]).sum())
    medium = int(((df["priority_score"] >= TIER_THRESHOLDS[1]) & (df["priority_score"] < TIER_THRESHOLDS[0])).sum())
    uncertain = int(df["is_uncertain"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f'<div class="gt-kpi"><div class="gt-kpi-label">Всего в очереди</div>'
        f'<div class="gt-kpi-value">{len(df)}</div>'
        f'<div class="gt-kpi-delta">за {df["date"].min().strftime("%b")}–{df["date"].max().strftime("%b %Y")}</div></div>',
        unsafe_allow_html=True,
    )
    k2.markdown(
        f'<div class="gt-kpi"><div class="gt-kpi-label">Срочных</div>'
        f'<div class="gt-kpi-value" style="color:#E5534B">{urgent}</div>'
        f'<div class="gt-kpi-delta">{urgent / len(df) * 100:.0f}% от очереди</div></div>',
        unsafe_allow_html=True,
    )
    k3.markdown(
        f'<div class="gt-kpi"><div class="gt-kpi-label">Средней срочности</div>'
        f'<div class="gt-kpi-value" style="color:#D9A73B">{medium}</div>'
        f'<div class="gt-kpi-delta">{medium / len(df) * 100:.0f}% от очереди</div></div>',
        unsafe_allow_html=True,
    )
    k4.markdown(
        f'<div class="gt-kpi"><div class="gt-kpi-label">Нужна ручная проверка</div>'
        f'<div class="gt-kpi-value" style="color:#9AA3AF">{uncertain}</div>'
        f'<div class="gt-kpi-delta">скор в зоне {UNCERTAINTY_BAND[0]}–{UNCERTAINTY_BAND[1]}</div></div>',
        unsafe_allow_html=True,
    )

    st.write("")
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.subheader("Распределение приоритетов")
        st.plotly_chart(P.score_distribution(df), use_container_width=True)
    with c_right:
        st.subheader("Топ-5 самых срочных")
        top5 = df.sort_values("priority_score", ascending=False).head(5)
        for _, row in top5.iterrows():
            st.markdown(
                render_case_card(row, tier_color(row["tier"]), compact=True),
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Нагрузка по месяцам")
    st.plotly_chart(P.appeals_by_month(df), use_container_width=True)

# ---------------------------------------------------------------------------
# Таб 2: Очередь обращений
# ---------------------------------------------------------------------------
with tab_queue:
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            categories = st.multiselect("Категория", options=sorted(df["category"].unique()))
        with f2:
            districts = st.multiselect("Район", options=sorted(df["district"].unique()))
        with f3:
            min_score = st.slider("Минимальный приоритет", 0, 100, 0)
        with f4:
            tiers = st.multiselect(
                "Тиер срочности", options=["Срочно", "Средне", "Не срочно"]
            )
        search = st.text_input(
            "Поиск по тексту", placeholder="например: прорвало, кипяток, лампочка…"
        )

    filtered = df.copy()
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if districts:
        filtered = filtered[filtered["district"].isin(districts)]
    if tiers:
        filtered = filtered[filtered["tier"].isin(tiers)]
    if search.strip():
        filtered = filtered[filtered["text"].str.contains(search.strip(), case=False, na=False)]
    filtered = filtered[filtered["priority_score"] >= min_score]
    filtered = filtered.sort_values("priority_score", ascending=False)

    st.subheader(f"Очередь обращений ({len(filtered)})")

    if filtered.empty:
        st.info("Нет обращений, удовлетворяющих фильтрам.")
    else:
        # пагинация «показать ещё»
        if "queue_page" not in st.session_state:
            st.session_state["queue_page"] = 1
        page = st.session_state["queue_page"]
        shown = filtered.head(page * PAGE_SIZE)

        for _, row in shown.iterrows():
            st.markdown(
                render_case_card(row, tier_color(row["tier"])),
                unsafe_allow_html=True,
            )

        if len(filtered) > page * PAGE_SIZE:
            col_a, col_b = st.columns([3, 1])
            remaining = len(filtered) - page * PAGE_SIZE
            col_a.markdown(
                f"<div class='gt-footnote'>Показано {page * PAGE_SIZE} из {len(filtered)} "
                f"(осталось {remaining})</div>",
                unsafe_allow_html=True,
            )
            if col_b.button("Показать ещё", use_container_width=True):
                st.session_state["queue_page"] += 1
                st.rerun()
        else:
            st.markdown(
                f"<div class='gt-footnote'>Показаны все {len(filtered)} обращений.</div>",
                unsafe_allow_html=True,
            )

        # сброс пагинации при смене фильтров — кнопка
        if page > 1:
            if st.button("↩ Сбросить пагинацию"):
                st.session_state["queue_page"] = 1
                st.rerun()

    st.divider()
    st.markdown(
        '<div class="gt-footnote">Модель не закрывает и не отклоняет обращения — она '
        "только помогает оператору ЦОН / акимата увидеть срочные случаи раньше. "
        "Обращения с пометкой «низкая уверенность» требуют ручной проверки. "
        "Финальное решение принимает человек.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Таб 3: Аналитика
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Обращения по категориям")
    st.plotly_chart(P.by_category_stacked(df), use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Распределение скоров по категориям")
        st.plotly_chart(P.score_box_by_category(df), use_container_width=True)
    with a2:
        st.subheader("Повторные обращения и приоритет")
        st.plotly_chart(P.repeat_vs_priority(df), use_container_width=True)

    st.subheader("Обращения по районам")
    st.plotly_chart(P.by_district(df), use_container_width=True)

    st.subheader("Тепловая карта: район × категория")
    st.plotly_chart(P.heatmap_district_category(df), use_container_width=True)

# ---------------------------------------------------------------------------
# Таб 4: Новое обращение
# ---------------------------------------------------------------------------
with tab_new:
    st.subheader("Оценка нового обращения")
    st.markdown(
        '<div class="gt-callout">Введите текст нового обращения — модель оценит '
        "его приоритет на лету и покажет, какие факторы повлияли на скор. "
        "Это демонстрирует, что модель работает «вживую», а не на заранее заготовленных данных.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    n1, n2 = st.columns(2)
    with n1:
        new_category = st.selectbox("Категория", options=sorted(df["category"].unique()))
        new_repeat = st.slider("Кол-во повторных обращений", 0, 8, 0)
    with n2:
        new_district = st.selectbox("Район", options=sorted(df["district"].unique()))
        new_date = st.date_input("Дата обращения", value=df["date"].max())

    new_text = st.text_area(
        "Текст обращения",
        height=120,
        placeholder="Например: Прорвало трубу с горячей водой в подвале, затапливает подземный паркинг, кипяток идёт на электрический щиток.",
    )

    if st.button("Оценить приоритет", type="primary"):
        if not new_text.strip():
            st.warning("Введите текст обращения.")
        else:
            # формируем однострочный DataFrame и считаем скор
            new_row = pd.DataFrame(
                [
                    {
                        "appeal_id": 0,
                        "date": pd.Timestamp(new_date),
                        "district": new_district,
                        "category": new_category,
                        "text": new_text.strip(),
                        "repeat_count": new_repeat,
                        "true_tier": None,
                        "is_urgent": None,
                    }
                ]
            )
            new_row = add_date_features(new_row)
            X_new, _, _, _ = build_feature_matrix(
                new_row, vectorizer=vectorizer, encoder=encoder, fit=False
            )
            score = float(model.predict_proba(X_new)[0, 1] * 100)
            tier = tier_for_score(score)
            accent = tier_color(tier)

            # топ-факторы для нового обращения
            row_contrib = X_new.multiply(coefs).toarray().flatten()
            top_idx = np.argsort(row_contrib)[::-1][:5]
            factors = [
                (feature_names[i], float(row_contrib[i]))
                for i in top_idx
                if row_contrib[i] > 0
            ]

            st.write("")
            res1, res2 = st.columns([1, 2])
            with res1:
                color = (
                    "#E5534B" if tier == "Срочно"
                    else "#D9A73B" if tier == "Средне"
                    else "#4FAF7D"
                )
                st.metric("Приоритет", f"{score:.0f}/100")
                st.markdown(
                    f"<div style='font-size:18px;font-weight:600;color:{color};"
                    f"margin-top:8px'>● {tier}</div>",
                    unsafe_allow_html=True,
                )
                if UNCERTAINTY_BAND[0] <= score <= UNCERTAINTY_BAND[1]:
                    st.markdown(
                        "<div class='gt-footnote' style='margin-top:8px'>"
                        "⚠ Скор в зоне неуверенности — рекомендуется ручная проверка.</div>",
                        unsafe_allow_html=True,
                    )
            with res2:
                st.markdown("**Что повлияло на приоритет:**")
                if factors:
                    st.markdown(
                        render_explainability_bars(factors, accent),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Явных повышающих факторов не найдено.")

            st.divider()
            st.markdown(
                '<div class="gt-callout"><b>Важно:</b> модель не принимает финальное '
                "решение. Она лишь помогает оператору быстрее оценить срочность. "
                "Назначение исполнителя и сроки реакции остаются за человеком.</div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Таб 5: О модели
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("AI/ML-подход")
    st.markdown(
        "<div class='gt-callout'>"
        "<b>Задача:</b> бинарная классификация / скоринг — обращение срочное или нет.<br><br>"
        "<b>Модель:</b> логистическая регрессия на TF-IDF (уни- и биграммы, до 2000 признаков) "
        "+ структурные признаки (категория one-hot, repeat_count, месяц, день недели).<br><br>"
        "<b>Почему логрег, а не бустинг:</b> коэффициенты линейно интерпретируемы — "
        "для каждого обращения можно точно показать, какие слова и признаки повысили "
        "приоритет (вклад = коэффициент × значение признака), без SHAP и дополнительных инструментов."
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    y_test, proba_test = load_test_predictions()
    auc = roc_auc_score(y_test, proba_test)
    ap = average_precision_score(y_test, proba_test)
    y_pred = (proba_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    m1, m2 = st.columns(2)
    m1.metric("ROC-AUC (test)", f"{auc:.3f}")
    m2.metric("PR-AUC (test)", f"{ap:.3f}")
    st.caption(
        "Метрики пересчитываются прямо в дашборде через тот же split (test_size=0.25, "
        "random_state=42), что и при обучении — без переобучения модели."
    )

    st.write("")
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    precision, recall, _ = precision_recall_curve(y_test, proba_test)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**ROC-кривая**")
        st.plotly_chart(P.roc_curve_fig(fpr, tpr, auc), use_container_width=True)
    with r2:
        st.markdown("**Precision-Recall кривая**")
        st.plotly_chart(P.pr_curve_fig(precision, recall, ap), use_container_width=True)

    st.write("")
    cm1, cm2 = st.columns([1, 2])
    with cm1:
        st.markdown("**Матрица ошибок**")
        st.plotly_chart(P.confusion_matrix_fig(cm), use_container_width=True)
    with cm2:
        st.markdown("**Топ-20 важных признаков**")
        st.plotly_chart(
            P.top_features_bar(feature_names, coefs, k=20), use_container_width=True
        )
        st.caption(
            "Красные — повышают срочность, зелёные — понижают. "
            "Это «глобальная» важность признаков; для каждого отдельного обращения "
            "вклад считается индивидуально (вклад = коэффициент × значение признака)."
        )

    st.write("")
    st.subheader("Ограничения")
    st.markdown(
        "<div class='gt-footnote'>"
        "• Данные синтетические — на реальных обращениях модель нужно переобучить и, "
        "скорее всего, скорректировать целевую переменную (реальной разметки «срочно/не срочно» "
        "обычно нет, нужен прокси или ручная разметка выборки).<br>"
        "• Модель не учитывает вложения (фото/видео к обращению) — только текст.<br>"
        "• TF-IDF не понимает синонимы и опечатки так же хорошо, как современные языковые "
        "модели — при необходимости можно заменить на embeddings."
        "</div>",
        unsafe_allow_html=True,
    )
