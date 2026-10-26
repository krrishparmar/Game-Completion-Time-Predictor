"""
🎮 Game Completion Time Predictor
Streamlit frontend for multi-output regression on HowLongToBeat data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎮 Game Completion Time Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #c4c4e0 !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
}
div[data-testid="stMetric"] label {
    color: #a5b4fc !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e0e7ff !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 60%, #6366f1 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
}
.hero-banner h1 {
    color: #fff;
    font-size: 2.4rem;
    font-weight: 900;
    margin: 0 0 8px 0;
    letter-spacing: -0.03em;
}
.hero-banner p {
    color: #c7d2fe;
    font-size: 1.1rem;
    margin: 0;
    font-weight: 400;
}

/* ── Prediction result cards ── */
.pred-card {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.6), rgba(67, 56, 202, 0.25));
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.pred-card:hover {
    border-color: rgba(129, 140, 248, 0.6);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    transform: translateY(-4px);
}
.pred-card .label {
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.pred-card .value {
    color: #fff;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.pred-card .unit {
    color: #818cf8;
    font-size: 0.9rem;
    font-weight: 500;
    margin-top: 4px;
}

/* ── Section dividers ── */
.section-header {
    border-left: 4px solid #6366f1;
    padding-left: 16px;
    margin: 40px 0 20px 0;
}
.section-header h2 {
    color: #e0e7ff;
    font-weight: 800;
    font-size: 1.5rem;
    margin: 0;
    letter-spacing: -0.02em;
}
.section-header p {
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}

/* ── Info boxes ── */
.info-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    color: #c7d2fe;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3730a3, #4f46e5) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
    transform: translateY(-1px);
}

/* ── Expander ── */
details {
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    background: rgba(99, 102, 241, 0.05) !important;
}

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme helper ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c4c4e0"),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[
        "#818cf8", "#a78bfa", "#c084fc", "#f472b6",
        "#fb923c", "#facc15", "#34d399", "#22d3ee",
    ],
)


# ── Data loading & model training (cached) ──────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning dataset …")
def load_and_clean(csv_path: str, min_polled: int = 5):
    TARGETS = ["main_story", "main_plus_sides", "completionist"]
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "game"]
    polled = [
        "main_story_polled",
        "main_plus_sides_polled",
        "completionist_polled",
    ]
    df = df.dropna(subset=TARGETS + polled + ["genres", "release_year"])
    df = df[df[polled].min(axis=1) >= min_polled]
    df = df[(df["release_year"] >= 1980) & (df["release_year"] <= 2026)]
    df = df[
        (df.main_story <= df.main_plus_sides)
        & (df.main_plus_sides <= df.completionist)
    ]
    df = df.drop_duplicates(subset=["name", "release_year"]).reset_index(drop=True)
    for t in TARGETS:
        df[t] = df[t].clip(upper=df[t].quantile(0.99))
    return df


def split_list(s):
    return [x.strip() for x in s.split(",")] if isinstance(s, str) else []


@st.cache_resource(show_spinner="Training models … this may take a minute ⏳")
def build_models(csv_path: str, min_polled: int = 5):
    TARGETS = ["main_story", "main_plus_sides", "completionist"]
    df = load_and_clean(csv_path, min_polled)

    # Feature engineering
    gen_counts = df.genres.apply(split_list).explode().value_counts()
    keep_genres = gen_counts[gen_counts >= 50].index.tolist()
    mlb_g = MultiLabelBinarizer(classes=keep_genres)
    G = pd.DataFrame(
        mlb_g.fit_transform(df.genres.apply(split_list)),
        columns=["genre_" + c for c in keep_genres],
    )

    plats = df.platform.fillna("").apply(split_list)
    plat_counts = plats.explode().value_counts()
    keep_plats = plat_counts[plat_counts >= 100].index.tolist()
    mlb_p = MultiLabelBinarizer(classes=keep_plats)
    P = pd.DataFrame(
        mlb_p.fit_transform(plats), columns=["plat_" + c for c in keep_plats]
    )

    dev_count = df.developer.map(df.developer.value_counts()).fillna(1)
    X = pd.concat([G, P], axis=1)
    X["n_genres"] = df.genres.apply(split_list).apply(len)
    X["n_platforms"] = plats.apply(len)
    X["release_year"] = df.release_year.values
    X["developer_games"] = dev_count.values
    X["has_developer"] = df.developer.notna().astype(int).values

    Y = np.log1p(df[TARGETS])

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=42
        ),
        "KNN": MultiOutputRegressor(
            Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=10, weights="distance")),
            ])
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
        "SVR": MultiOutputRegressor(
            Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(C=1.0, epsilon=0.1, max_iter=2500)),
            ]),
            n_jobs=-1,
        ),
    }

    rows, preds = [], {}
    for name, m in models.items():
        m.fit(Xtr, Ytr)
        p = m.predict(Xte)
        preds[name] = p
        for i, t in enumerate(TARGETS):
            hours_true = np.expm1(Yte.iloc[:, i])
            hours_pred = np.expm1(p[:, i])
            rows.append(
                {
                    "Model": name,
                    "Target": t,
                    "MAE (hours)": mean_absolute_error(hours_true, hours_pred),
                    "R² (log scale)": r2_score(Yte.iloc[:, i], p[:, i]),
                }
            )
    results = pd.DataFrame(rows)
    best = results.groupby("Model")["R² (log scale)"].mean().idxmax()

    # Feature importance
    imp = {}
    for i, t in enumerate(TARGETS):
        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42
        ).fit(Xtr, Ytr.iloc[:, i])
        imp[t] = pd.Series(rf.feature_importances_, index=X.columns)
    imp_df = pd.DataFrame(imp)

    return {
        "df": df,
        "X": X,
        "Y": Y,
        "Xtr": Xtr,
        "Xte": Xte,
        "Ytr": Ytr,
        "Yte": Yte,
        "models": models,
        "results": results,
        "preds": preds,
        "best": best,
        "imp": imp_df,
        "keep_genres": keep_genres,
        "keep_plats": keep_plats,
        "TARGETS": TARGETS,
    }


# ── Helper: predict a new game ──────────────────────────────────────────────
def predict_game(data, genres, platforms, year, developer_games, model_name=None):
    TARGETS = data["TARGETS"]
    model_name = model_name or data["best"]
    model = data["models"][model_name]
    X_cols = data["X"].columns
    row = pd.DataFrame(0, index=[0], columns=X_cols)
    for gname in genres:
        col = "genre_" + gname
        if col in row.columns:
            row[col] = 1
    for pl in platforms:
        col = "plat_" + pl
        if col in row.columns:
            row[col] = 1
    row["n_genres"] = len(genres)
    row["n_platforms"] = len(platforms)
    row["release_year"] = year
    row["developer_games"] = developer_games
    row["has_developer"] = 1
    out = np.expm1(model.predict(row[X_cols]))[0]
    return dict(zip(TARGETS, np.round(out, 1)))


def lookup_game_features(data, game_name):
    """Look up a game by name and return its features for prediction."""
    df = data["df"]
    match = df[df.name.str.lower() == game_name.strip().lower()]
    if match.empty:
        return None
    row = match.iloc[0]
    genres = split_list(row.genres) if isinstance(row.genres, str) else []
    platforms = split_list(row.platform) if isinstance(row.get("platform", ""), str) else []
    year = int(row.release_year) if pd.notna(row.release_year) else 2020
    dev = row.get("developer", None)
    dev_games = int(df[df.developer == dev].shape[0]) if pd.notna(dev) else 1
    actual = {t: round(float(row[t]), 1) for t in data["TARGETS"]}
    return {
        "name": row["name"],
        "genres": genres,
        "platforms": platforms,
        "year": year,
        "developer": dev if pd.notna(dev) else "Unknown",
        "developer_games": dev_games,
        "actual": actual,
    }


# ── Helper: format hours nicely ─────────────────────────────────────────────
def format_hours(h):
    if h < 1:
        return f"{int(h * 60)}m"
    elif h < 100:
        return f"{h:.1f}h"
    else:
        return f"{h:.0f}h"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = "/Users/krrishparmar/Downloads/archive/hltb_dataset.csv"

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🎮 Game Completion Time Predictor</h1>
    <p>Multi-output regression on HowLongToBeat data — explore, analyze, and predict how long it takes to beat any game.</p>
</div>
""", unsafe_allow_html=True)

# Load data
data = build_models(CSV_PATH)
df = data["df"]
TARGETS = data["TARGETS"]
TARGET_LABELS = {
    "main_story": "🏁 Main Story",
    "main_plus_sides": "📋 Main + Sides",
    "completionist": "🏆 Completionist",
}
TARGET_COLORS = {
    "main_story": "#818cf8",
    "main_plus_sides": "#a78bfa",
    "completionist": "#c084fc",
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Configuration")
    st.markdown("---")

    st.markdown("**📊 Dataset Overview**")
    st.metric("Total Games", f"{len(df):,}")
    st.metric("Features", f"{data['X'].shape[1]}")
    st.metric("Best Model", data["best"])
    st.markdown("---")

    st.markdown("**ℹ️ About**")
    st.markdown("""
    <div class="info-box">
        Built on the <strong>HowLongToBeat</strong> dataset.
        Models predict three completion styles in hours:
        <br>• <strong>Main Story</strong> — story only
        <br>• <strong>Main + Sides</strong> — story + side quests
        <br>• <strong>Completionist</strong> — 100% everything
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_predict, tab_eda, tab_models, tab_explore = st.tabs([
    "🔮 Predict",
    "📊 Exploratory Analysis",
    "🤖 Model Performance",
    "🔍 Explore Games",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1: PREDICT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_predict:
    st.markdown("""
    <div class="section-header">
        <h2>Predict Completion Time</h2>
        <p>Search for a game by name or configure a custom game to see predicted completion times.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Game name search ─────────────────────────────────────────────────────
    st.markdown("#### 🔎 Search for a Game")

    game_names_sorted = sorted(df.name.unique().tolist())
    search_input = st.selectbox(
        "Type a game name",
        options=[""] + game_names_sorted,
        index=0,
        placeholder="Start typing a game name … e.g. The Witcher 3, Elden Ring, Celeste",
        help="Search from the dataset of {:,} games.".format(len(game_names_sorted)),
    )

    model_choice = st.selectbox(
        "Model",
        list(data["models"].keys()),
        index=list(data["models"].keys()).index(data["best"]),
        help="Choose which trained model to use for prediction.",
        key="predict_model",
    )

    if search_input:
        game_info = lookup_game_features(data, search_input)
        if game_info:
            # ── Game info header ──
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(30, 27, 75, 0.5), rgba(67, 56, 202, 0.2));
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 16px;
                padding: 24px 28px;
                margin: 16px 0 24px 0;
            ">
                <div style="font-size: 1.6rem; font-weight: 800; color: #e0e7ff; margin-bottom: 8px;">
                    🎮 {game_info['name']}
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.8;">
                    <span style="color: #a5b4fc; font-weight: 600;">Genres:</span> {', '.join(game_info['genres'])} &nbsp;·&nbsp;
                    <span style="color: #a5b4fc; font-weight: 600;">Year:</span> {game_info['year']} &nbsp;·&nbsp;
                    <span style="color: #a5b4fc; font-weight: 600;">Developer:</span> {game_info['developer']}
                    ({game_info['developer_games']} game{'s' if game_info['developer_games'] != 1 else ''} in dataset)
                    <br>
                    <span style="color: #a5b4fc; font-weight: 600;">Platforms:</span> {', '.join(game_info['platforms']) if game_info['platforms'] else 'N/A'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Run prediction
            prediction = predict_game(
                data,
                game_info["genres"],
                game_info["platforms"],
                game_info["year"],
                game_info["developer_games"],
                model_choice,
            )

            # ── Predicted completion times ──
            st.markdown("##### 📈 Predicted Completion Times")

            cols = st.columns(3)
            for col, target_key in zip(cols, TARGETS):
                with col:
                    pred_h = max(prediction[target_key], 0)
                    st.markdown(f"""
                    <div class="pred-card">
                        <div class="label">{TARGET_LABELS[target_key]}</div>
                        <div class="value">{format_hours(pred_h)}</div>
                        <div class="unit">hours</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bar chart ──
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[TARGET_LABELS[t] for t in TARGETS],
                y=[max(prediction[t], 0) for t in TARGETS],
                marker=dict(
                    color=[TARGET_COLORS[t] for t in TARGETS],
                    line=dict(width=0),
                    cornerradius=8,
                ),
                text=[format_hours(max(prediction[t], 0)) for t in TARGETS],
                textposition="outside",
                textfont=dict(size=14, color="#e0e7ff", family="Inter"),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Predicted Hours — {game_info['name']}",
                yaxis_title="Hours",
                showlegend=False,
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Game **\"{search_input}\"** not found in the dataset. Try the search suggestions or use custom input below.")

    else:
        st.markdown("""
        <div class="info-box" style="margin-top: 12px; text-align: center; padding: 40px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🔎</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #a5b4fc;">Search for Any Game</div>
            <div style="margin-top: 8px; color: #94a3b8;">
                Start typing a game name above to see predicted completion times.<br>
                Or use the custom input section below to predict for a hypothetical game.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Custom game input (secondary) ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛠️ Custom Game — predict for a hypothetical game", expanded=False):
        custom_col1, custom_col2 = st.columns([1, 1], gap="large")

        with custom_col1:
            custom_genres = st.multiselect(
                "Genres",
                options=sorted(data["keep_genres"]),
                default=["Action", "Adventure"],
                help="Select one or more genres.",
                key="custom_genres",
            )
            custom_platforms = st.multiselect(
                "Platforms",
                options=sorted(data["keep_plats"]),
                default=["PC"],
                help="Select platform(s).",
                key="custom_platforms",
            )
            c1, c2 = st.columns(2)
            with c1:
                custom_year = st.slider("Release Year", 1980, 2026, 2024, key="custom_year")
            with c2:
                custom_dev = st.slider("Developer Experience (# games)", 1, 50, 5, key="custom_dev")
            custom_predict = st.button("🚀  Predict", use_container_width=True, key="custom_predict_btn")

        with custom_col2:
            if custom_predict and custom_genres and custom_platforms:
                custom_prediction = predict_game(
                    data, custom_genres, custom_platforms, custom_year, custom_dev, model_choice
                )
                for target_key in TARGETS:
                    hours = max(custom_prediction[target_key], 0)
                    st.markdown(f"""
                    <div class="pred-card">
                        <div class="label">{TARGET_LABELS[target_key]}</div>
                        <div class="value">{format_hours(hours)}</div>
                        <div class="unit">hours</div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)

                fig_custom = go.Figure()
                fig_custom.add_trace(go.Bar(
                    x=[TARGET_LABELS[t] for t in TARGETS],
                    y=[max(custom_prediction[t], 0) for t in TARGETS],
                    marker=dict(
                        color=[TARGET_COLORS[t] for t in TARGETS],
                        line=dict(width=0),
                        cornerradius=8,
                    ),
                    text=[format_hours(max(custom_prediction[t], 0)) for t in TARGETS],
                    textposition="outside",
                    textfont=dict(size=14, color="#e0e7ff", family="Inter"),
                ))
                fig_custom.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Predicted Hours by Play Style",
                    yaxis_title="Hours",
                    showlegend=False,
                    height=320,
                )
                st.plotly_chart(fig_custom, use_container_width=True)
            elif custom_predict:
                st.warning("Please select at least one genre and one platform.")
            else:
                st.markdown("""
                <div class="info-box" style="text-align: center; padding: 30px 20px;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">🛠️</div>
                    <div style="color: #94a3b8;">
                        Configure genres, platforms, year &amp; developer experience,<br>
                        then click <strong>Predict</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2: EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_eda:
    st.markdown("""
    <div class="section-header">
        <h2>Exploratory Data Analysis</h2>
        <p>Visualize distributions, trends, and genre comparisons across the dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick stats ──
    qs1, qs2, qs3, qs4 = st.columns(4)
    qs1.metric("Median Main Story", f"{df.main_story.median():.1f}h")
    qs2.metric("Median Main + Sides", f"{df.main_plus_sides.median():.1f}h")
    qs3.metric("Median Completionist", f"{df.completionist.median():.1f}h")
    qs4.metric("Year Range", f"{int(df.release_year.min())}–{int(df.release_year.max())}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution ──
    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        fig_dist = make_subplots(rows=1, cols=3, subplot_titles=[
            TARGET_LABELS[t] for t in TARGETS
        ])
        for i, t in enumerate(TARGETS):
            fig_dist.add_trace(
                go.Histogram(
                    x=df[t],
                    nbinsx=60,
                    marker_color=TARGET_COLORS[t],
                    opacity=0.85,
                    name=TARGET_LABELS[t],
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        fig_dist.update_layout(
            **PLOTLY_LAYOUT,
            title="Distribution of Completion Times (hours)",
            height=350,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with eda_col2:
        fig_log = make_subplots(rows=1, cols=3, subplot_titles=[
            f"log₁ₚ {TARGET_LABELS[t]}" for t in TARGETS
        ])
        for i, t in enumerate(TARGETS):
            fig_log.add_trace(
                go.Histogram(
                    x=np.log1p(df[t]),
                    nbinsx=60,
                    marker_color=TARGET_COLORS[t],
                    opacity=0.85,
                    name=TARGET_LABELS[t],
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        fig_log.update_layout(
            **PLOTLY_LAYOUT,
            title="Distribution (log-transformed)",
            height=350,
        )
        st.plotly_chart(fig_log, use_container_width=True)

    # ── Average hours by genre ──
    g = df.assign(genre=df.genres.str.split(", ")).explode("genre")
    top_genres = g.genre.value_counts().head(14).index
    avg_genre = g[g.genre.isin(top_genres)].groupby("genre")[TARGETS].mean().sort_values("completionist")

    fig_genre = go.Figure()
    for t in TARGETS:
        fig_genre.add_trace(go.Bar(
            y=avg_genre.index,
            x=avg_genre[t],
            name=TARGET_LABELS[t],
            orientation="h",
            marker_color=TARGET_COLORS[t],
            marker_line_width=0,
        ))
    fig_genre.update_layout(
        **PLOTLY_LAYOUT,
        title="Average Hours by Genre (Top 14)",
        barmode="group",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_genre, use_container_width=True)

    # ── Decade trend + Correlation ──
    dec_col1, dec_col2 = st.columns(2)

    with dec_col1:
        df_dec = df.copy()
        df_dec["decade"] = (df_dec.release_year // 10 * 10).astype(int)
        dec_avg = df_dec.groupby("decade")[TARGETS].mean().reset_index()

        fig_dec = go.Figure()
        for t in TARGETS:
            fig_dec.add_trace(go.Scatter(
                x=dec_avg.decade,
                y=dec_avg[t],
                mode="lines+markers",
                name=TARGET_LABELS[t],
                line=dict(width=3),
                marker=dict(size=8),
            ))
        fig_dec.update_layout(
            **PLOTLY_LAYOUT,
            title="Average Hours by Decade",
            xaxis_title="Decade",
            yaxis_title="Hours",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dec, use_container_width=True)

    with dec_col2:
        corr_cols = TARGETS + ["release_year"]
        corr = df[corr_cols].corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=[TARGET_LABELS.get(c, c) for c in corr.columns],
            y=[TARGET_LABELS.get(c, c) for c in corr.columns],
            colorscale=[[0, "#1e1b4b"], [0.5, "#4338ca"], [1, "#c084fc"]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=14, color="#e0e7ff"),
            showscale=False,
        ))
        fig_corr.update_layout(
            **PLOTLY_LAYOUT,
            title="Correlation Matrix (Pearson)",
            height=420,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Statistical Analysis: Covariance, Confidence Intervals & Hypothesis Testing ──
    st.markdown("""
    <div class="section-header">
        <h2>Statistical Inference & Analysis</h2>
        <p>Descriptive statistics, Covariance Matrix, and Hypothesis Testing derived from curriculum labs.</p>
    </div>
    """, unsafe_allow_html=True)

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        # Covariance Matrix (from Lab 8 & Practical 6)
        cov = df[corr_cols].cov()
        fig_cov = go.Figure(data=go.Heatmap(
            z=cov.values,
            x=[TARGET_LABELS.get(c, c) for c in cov.columns],
            y=[TARGET_LABELS.get(c, c) for c in cov.columns],
            colorscale="Viridis",
            text=np.round(cov.values, 1),
            texttemplate="%{text}",
            textfont=dict(size=12, color="#ffffff"),
            showscale=False,
        ))
        fig_cov.update_layout(
            **PLOTLY_LAYOUT,
            title="Covariance Matrix",
            height=380,
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with stat_col2:
        # Correlation & Significance Tests (Pearson & Spearman from Practical 6)
        st.markdown("##### 📐 Correlation & Significance (vs Release Year, α = 0.05)")
        stat_rows = []
        for t in TARGETS:
            pr, pp = stats.pearsonr(df["release_year"], df[t])
            sr, sp = stats.spearmanr(df["release_year"], df[t])
            stat_rows.append({
                "Target": TARGET_LABELS[t],
                "Pearson r": round(pr, 3),
                "Pearson p": f"{pp:.2e}",
                "Pearson Sig?": "Yes (Reject H0)" if pp < 0.05 else "No",
                "Spearman ρ": round(sr, 3),
                "Spearman p": f"{sp:.2e}",
                "Spearman Sig?": "Yes (Reject H0)" if sp < 0.05 else "No",
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        # Hypothesis Testing (t-test from Lab 5 & Practical 6)
        st.markdown("##### 🧪 Two-Sample Hypothesis Test (Modern vs Classic Games)")
        modern = df[df.release_year >= 2015]["main_story"]
        classic = df[df.release_year < 2015]["main_story"]
        t_stat, p_val = stats.ttest_ind(modern, classic, equal_var=False)
        decision = "Reject H0 (Statistically Significant Difference)" if p_val < 0.05 else "Fail to reject H0"
        st.markdown(f"""
        <div style="background: rgba(30, 27, 75, 0.4); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 14px 18px; font-size: 0.9rem; line-height: 1.6;">
            <strong>H0:</strong> Mean completion time for Modern games (≥2015) = Classic games (&lt;2015)<br>
            <strong>H1:</strong> Mean completion time differs between eras<br>
            • <strong>Modern Mean:</strong> {modern.mean():.1f}h &nbsp;|&nbsp; <strong>Classic Mean:</strong> {classic.mean():.1f}h<br>
            • <strong>t-statistic:</strong> {t_stat:.3f} &nbsp;|&nbsp; <strong>p-value:</strong> {p_val:.2e}<br>
            • <strong>Decision at α=0.05:</strong> <span style="color: #34d399; font-weight: 700;">{decision}</span>
        </div>
        """, unsafe_allow_html=True)

    # Descriptive Statistics Table (Mean, Median, Mode, Variance, Std, IQR, 95% CI from Lab 5)
    st.markdown("##### 📊 Descriptive Statistics & 95% Confidence Intervals")
    desc_rows = []
    for t in TARGETS:
        s = df[t]
        m = float(s.mean())
        sem = float(stats.sem(s))
        ci_low, ci_high = stats.t.interval(0.95, len(s) - 1, loc=m, scale=sem)
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        desc_rows.append({
            "Target": TARGET_LABELS[t],
            "Mean": round(m, 2),
            "Median": round(float(s.median()), 2),
            "Mode": round(float(s.mode()[0]), 2),
            "Std Dev": round(float(s.std()), 2),
            "Variance": round(float(s.var()), 2),
            "Range": round(float(s.max() - s.min()), 2),
            "IQR": round(q3 - q1, 2),
            "95% CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
        })
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

    # ── Central Limit Theorem & Normal Distribution (from Lab 4) ──
    st.markdown("""
    <div class="section-header">
        <h2>Central Limit Theorem (CLT) & Normal Distribution</h2>
        <p>Demonstrating how sampling distributions of sample means converge to a Gaussian Bell Curve (from Lab 4).</p>
    </div>
    """, unsafe_allow_html=True)

    pop_data = df["main_story"].values
    p_mean = float(np.mean(pop_data))
    p_std = float(np.std(pop_data))

    fig_clt = make_subplots(
        rows=1, cols=4,
        subplot_titles=[
            f"Population (N={len(pop_data)}, Skewed)",
            "Sample Size n = 5",
            "Sample Size n = 30",
            "Sample Size n = 100 (Normal Fit)",
        ],
    )

    # 1. Population
    fig_clt.add_trace(go.Histogram(
        x=pop_data, nbinsx=35, marker_color="#38bdf8", opacity=0.75, name="Population", showlegend=False
    ), row=1, col=1)

    # 2, 3, 4. Sampling distributions
    np.random.seed(42)
    clt_colors = ["#f59e0b", "#a855f7", "#10b981"]
    for i, (n_size, c) in enumerate(zip([5, 30, 100], clt_colors), 2):
        s_means = [float(np.mean(np.random.choice(pop_data, size=n_size))) for _ in range(1000)]
        fig_clt.add_trace(go.Histogram(
            x=s_means, nbinsx=25, marker_color=c, opacity=0.75, name=f"n={n_size}", showlegend=False
        ), row=1, col=i)

    fig_clt.update_layout(
        **PLOTLY_LAYOUT,
        title="Central Limit Theorem: Progression Towards Normal Distribution (1,000 Repeated Samples)",
        height=380,
    )
    st.plotly_chart(fig_clt, use_container_width=True)

    # CLT Verification Table & Key Insight
    clt_col1, clt_col2 = st.columns([1.2, 1])
    with clt_col1:
        st.markdown("##### 📐 CLT Parameters: Theoretical vs Actual Standard Error")
        clt_tbl = []
        for n_val in [5, 30, 100]:
            th_se = p_std / np.sqrt(n_val)
            np.random.seed(42)
            s_means = [np.mean(np.random.choice(pop_data, size=n_val)) for _ in range(1000)]
            act_se = float(np.std(s_means))
            clt_tbl.append({
                "Sample Size (n)": n_val,
                "Population Mean (μ)": f"{p_mean:.2f}h",
                "Sample Mean (X̄)": f"{np.mean(s_means):.2f}h",
                "Theoretical SE (σ/√n)": f"{th_se:.2f}",
                "Actual SE": f"{act_se:.2f}",
            })
        st.dataframe(pd.DataFrame(clt_tbl), use_container_width=True, hide_index=True)

    with clt_col2:
        st.markdown("##### 💡 Central Limit Theorem Takeaway")
        st.markdown("""
        <div style="background: rgba(30, 27, 75, 0.4); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 14px 18px; font-size: 0.88rem; line-height: 1.6;">
            • <strong>Non-Normal Population:</strong> The raw completion time is heavily right-skewed (skewness = 2.52).<br>
            • <strong>Convergence to Normality:</strong> As sample size increases to <strong>n = 30 and n = 100</strong>, the distribution of sample means becomes symmetric and bell-shaped.<br>
            • <strong>SE Reduction:</strong> The standard error decreases proportionally to <strong>1/√n</strong>, concentrating tightly around the true population mean (11.36h).
        </div>
        """, unsafe_allow_html=True)

    # ── Side-content gap ──
    st.markdown("""
    <div class="section-header">
        <h2>Side-Content Gap</h2>
        <p>Which genres have the biggest difference between main story and 100% completion?</p>
    </div>
    """, unsafe_allow_html=True)

    df_gap = df.copy()
    df_gap["gap"] = df_gap.completionist - df_gap.main_story
    df_gap["ratio"] = df_gap.completionist / df_gap.main_story
    gg = df_gap.assign(genre=df_gap.genres.str.split(", ")).explode("genre")
    gg = gg[gg.genre.isin(data["keep_genres"])]
    gap_stats = gg.groupby("genre").agg(
        games=("gap", "size"),
        mean_gap=("gap", "mean"),
        median_ratio=("ratio", "median"),
    )
    gap_stats = gap_stats[gap_stats.games >= 100].sort_values("mean_gap", ascending=False)

    gap_col1, gap_col2 = st.columns([2, 1])
    with gap_col1:
        top_gap = gap_stats.head(10).sort_values("mean_gap")
        fig_gap = go.Figure(go.Bar(
            y=top_gap.index,
            x=top_gap.mean_gap,
            orientation="h",
            marker=dict(
                color=top_gap.mean_gap,
                colorscale=[[0, "#4338ca"], [1, "#f472b6"]],
                line_width=0,
                cornerradius=6,
            ),
            text=[f"{v:.0f}h" for v in top_gap.mean_gap],
            textposition="outside",
            textfont=dict(color="#e0e7ff", size=12, family="Inter"),
        ))
        fig_gap.update_layout(
            **PLOTLY_LAYOUT,
            title="Top 10 Genres by Side-Content Gap (hours)",
            xaxis_title="Avg Extra Hours (Completionist − Main Story)",
            height=420,
            showlegend=False,
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    with gap_col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        display_gap = gap_stats.head(10).reset_index().rename(
            columns={"genre": "Genre", "games": "Games", "mean_gap": "Gap (hours)", "median_ratio": "Ratio"}
        ).round(2)
        st.dataframe(display_gap, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3: MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_models:
    st.markdown("""
    <div class="section-header">
        <h2>Model Comparison</h2>
        <p>Evaluating 5 models from the curriculum (Ridge, Decision Tree, KNN, Random Forest, SVR) on a 20% held-out test set.</p>
    </div>
    """, unsafe_allow_html=True)

    results = data["results"]

    # ── MAE and R² tables ──
    tbl1, tbl2 = st.columns(2)
    with tbl1:
        st.markdown("##### MAE (hours)")
        mae_pivot = results.pivot(index="Model", columns="Target", values="MAE (hours)")[TARGETS].round(2)
        mae_pivot.columns = [TARGET_LABELS[c] for c in mae_pivot.columns]
        st.dataframe(mae_pivot, use_container_width=True)
    with tbl2:
        st.markdown("##### R² (log scale)")
        r2_pivot = results.pivot(index="Model", columns="Target", values="R² (log scale)")[TARGETS].round(3)
        r2_pivot.columns = [TARGET_LABELS[c] for c in r2_pivot.columns]
        st.dataframe(r2_pivot, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual comparisons ──
    chart1, chart2 = st.columns(2)

    with chart1:
        fig_mae = go.Figure()
        model_names = results.Model.unique()
        model_colors = {
            "Ridge": "#818cf8",
            "Decision Tree": "#f472b6",
            "KNN": "#fb923c",
            "Random Forest": "#a78bfa",
            "SVR": "#34d399",
        }
        for model in model_names:
            sub = results[results.Model == model]
            fig_mae.add_trace(go.Bar(
                x=[TARGET_LABELS[t] for t in sub.Target],
                y=sub["MAE (hours)"],
                name=model,
                marker_color=model_colors.get(model, "#818cf8"),
                marker_line_width=0,
            ))
        fig_mae.update_layout(
            **PLOTLY_LAYOUT,
            title="MAE by Model & Target",
            yaxis_title="MAE (hours)",
            barmode="group",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with chart2:
        fig_r2 = go.Figure()
        for model in model_names:
            sub = results[results.Model == model]
            fig_r2.add_trace(go.Bar(
                x=[TARGET_LABELS[t] for t in sub.Target],
                y=sub["R² (log scale)"],
                name=model,
                marker_color=model_colors.get(model, "#818cf8"),
                marker_line_width=0,
            ))
        fig_r2.update_layout(
            **PLOTLY_LAYOUT,
            title="R² by Model & Target",
            yaxis_title="R² (log scale)",
            barmode="group",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # ── Actual vs Predicted scatter ──
    st.markdown("""
    <div class="section-header">
        <h2>Actual vs Predicted</h2>
        <p>Scatter plots for the best model ({best}) on the test set.</p>
    </div>
    """.format(best=data["best"]), unsafe_allow_html=True)

    best_preds = data["preds"][data["best"]]
    Yte = data["Yte"]

    fig_scatter = make_subplots(
        rows=1, cols=3,
        subplot_titles=[TARGET_LABELS[t] for t in TARGETS],
    )
    for i, t in enumerate(TARGETS):
        actual = Yte.iloc[:, i]
        predicted = best_preds[:, i]
        fig_scatter.add_trace(
            go.Scattergl(
                x=actual,
                y=predicted,
                mode="markers",
                marker=dict(
                    color=TARGET_COLORS[t],
                    size=3,
                    opacity=0.4,
                ),
                name=TARGET_LABELS[t],
                showlegend=False,
            ),
            row=1, col=i + 1,
        )
        lim_min, lim_max = float(actual.min()), float(actual.max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[lim_min, lim_max],
                y=[lim_min, lim_max],
                mode="lines",
                line=dict(color="#f87171", dash="dash", width=2),
                showlegend=False,
            ),
            row=1, col=i + 1,
        )
        fig_scatter.update_xaxes(title_text="Actual (log₁ₚ h)", row=1, col=i + 1)
        fig_scatter.update_yaxes(title_text="Predicted (log₁ₚ h)", row=1, col=i + 1)

    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Actual vs Predicted — {data['best']}",
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Feature importance ──
    st.markdown("""
    <div class="section-header">
        <h2>Feature Importance</h2>
        <p>Random Forest importance scores — what drives each target?</p>
    </div>
    """, unsafe_allow_html=True)

    imp = data["imp"]
    top_feats = imp.mean(axis=1).sort_values(ascending=False).head(14).index
    imp_top = imp.loc[top_feats].sort_values("completionist")

    fig_imp = go.Figure()
    for t in TARGETS:
        fig_imp.add_trace(go.Bar(
            y=imp_top.index.str.replace("genre_", "🎭 ").str.replace("plat_", "💻 "),
            x=imp_top[t],
            name=TARGET_LABELS[t],
            orientation="h",
            marker_color=TARGET_COLORS[t],
            marker_line_width=0,
        ))
    fig_imp.update_layout(
        **PLOTLY_LAYOUT,
        title="Top 14 Feature Importances by Target",
        xaxis_title="Relative Importance Score (0.0 to 1.0 — sums to 100% across all features)",
        barmode="group",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Feature Importance explanation & insights ──
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.45), rgba(67, 56, 202, 0.15)); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 14px; padding: 22px 26px; margin-top: 16px;">
        <h4 style="color: #e0e7ff; margin: 0 0 10px 0; font-size: 1.15rem;">📌 Understanding Feature Importance & the X-Axis</h4>
        <p style="color: #c7d2fe; font-size: 0.92rem; line-height: 1.6; margin-bottom: 16px;">
            <strong>What does the X-Axis show?</strong> The x-axis measures the <strong>Relative Feature Importance Score</strong> (Mean Decrease in Impurity / Gini Importance from Random Forest). All feature importances across the entire model sum up to <strong>1.0 (or 100%)</strong>. For example, a value of <strong>0.28</strong> means that <strong>28%</strong> of the model's total predictive weight comes from that single feature alone.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;">
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 14px 16px;">
                <div style="font-weight: 700; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 4px;">🎭 Role-Playing (#1 Driver)</div>
                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                    Accounts for <strong>~20%</strong> of story length, but spikes to <strong>~28%</strong> for completionist time. RPGs feature massive optional side quests, leveling systems, and collectibles.
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 14px 16px;">
                <div style="font-weight: 700; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 4px;">📅 Release Year (#2 Driver)</div>
                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                    Contributes <strong>~15%–17%</strong>. Modern games have systematically grown in scope, map size, and optional content volume compared to retro eras.
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 14px 16px;">
                <div style="font-weight: 700; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 4px;">🏢 Developer Scale (#3 Driver)</div>
                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                    Contributes <strong>~9%</strong>. Veteran studios with extensive catalogs (e.g. Ubisoft, Square Enix) consistently produce larger, more sprawling games than solo indies.
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 14px 16px;">
                <div style="font-weight: 700; color: #a5b4fc; font-size: 0.95rem; margin-bottom: 4px;">💻 Genres vs Platforms</div>
                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                    Platforms (PC, Switch) only contribute <strong>1%–2%</strong>. Game design and genre define playtime, not the hardware console you play it on.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4: EXPLORE GAMES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_explore:
    st.markdown("""
    <div class="section-header">
        <h2>Explore the Dataset</h2>
        <p>Search and filter games, and see their actual completion times.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ──
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        search_query = st.text_input("🔎 Search by game name", placeholder="e.g. Zelda, Witcher, Elden Ring …")
    with fc2:
        all_genres_list = sorted(
            df.genres.str.split(", ").explode().dropna().unique()
        )
        filter_genre = st.selectbox("Filter by genre", ["All"] + all_genres_list)
    with fc3:
        yr_range = st.slider(
            "Release year range",
            int(df.release_year.min()),
            int(df.release_year.max()),
            (2000, 2026),
        )

    # Apply filters
    filtered = df.copy()
    if search_query:
        filtered = filtered[filtered.name.str.contains(search_query, case=False, na=False)]
    if filter_genre != "All":
        filtered = filtered[filtered.genres.str.contains(filter_genre, na=False)]
    filtered = filtered[
        (filtered.release_year >= yr_range[0]) & (filtered.release_year <= yr_range[1])
    ]

    st.markdown(f"**Showing {len(filtered):,} games**")

    display_cols = ["name", "genres", "platform", "release_year"] + TARGETS
    display_df = filtered[display_cols].sort_values("name").head(500).reset_index(drop=True)
    display_df.columns = ["Game", "Genres", "Platforms", "Year", "🏁 Story (h)", "📋 +Sides (h)", "🏆 100% (h)"]
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)

    # ── Top longest / shortest ──
    long_col, short_col = st.columns(2)
    with long_col:
        st.markdown("##### 🐢 Longest Games (Completionist)")
        longest = df.nlargest(10, "completionist")[["name", "genres"] + TARGETS].reset_index(drop=True)
        longest.columns = ["Game", "Genres", "🏁 Story", "📋 +Sides", "🏆 100%"]
        st.dataframe(longest, use_container_width=True, hide_index=True)
    with short_col:
        st.markdown("##### ⚡ Shortest Games (Main Story)")
        shortest = df.nsmallest(10, "main_story")[["name", "genres"] + TARGETS].reset_index(drop=True)
        shortest.columns = ["Game", "Genres", "🏁 Story", "📋 +Sides", "🏆 100%"]
        st.dataframe(shortest, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "Built with Streamlit · Data from HowLongToBeat · Models from Lab Curriculum: Ridge, Decision Tree, Random Forest, KNN, SVR"
    "</div>",
    unsafe_allow_html=True,
)
