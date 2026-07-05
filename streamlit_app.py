import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Cost Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* All text default */
    .stMarkdown, .stText, p, h1, h2, h3, h4, label {
        color: #e0e0e0 !important;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 91, 255, 0.3);
    }
    .metric-card .value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }
    .metric-card .label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #9ca3af !important;
        margin-top: 6px;
    }
    .metric-card .sublabel {
        font-size: 0.72rem;
        color: #6b7280 !important;
        margin-top: 2px;
    }

    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(99,91,255,0.2), rgba(0,212,255,0.15));
        border: 1px solid rgba(99,91,255,0.5);
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin-top: 20px;
        animation: fadeSlideIn 0.5s ease-out;
        backdrop-filter: blur(10px);
    }
    .prediction-box .cost-value {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .prediction-box .cost-label {
        font-size: 1rem;
        color: #9ca3af !important;
        font-weight: 500;
        margin-bottom: 8px;
    }

    /* Error box */
    .error-box {
        background: rgba(239,68,68,0.15);
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #fca5a5 !important;
        font-size: 0.9rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0e0e0 !important;
        margin-bottom: 6px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99,91,255,0.4);
    }

    /* Hero banner */
    .hero {
        text-align: center;
        padding: 12px 0 28px 0;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .hero p {
        font-size: 1.05rem;
        color: #9ca3af !important;
        margin-top: 8px;
    }

    /* Info chip */
    .chip {
        display: inline-block;
        background: rgba(99,91,255,0.2);
        border: 1px solid rgba(99,91,255,0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #a78bfa !important;
        margin: 3px;
    }

    /* Insight card */
    .insight-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .insight-card .insight-title {
        font-weight: 700;
        font-size: 0.9rem;
        color: #a78bfa !important;
    }
    .insight-card .insight-text {
        font-size: 0.85rem;
        color: #9ca3af !important;
        margin-top: 4px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,91,255,0.3) !important;
        color: #a78bfa !important;
    }

    /* Input fields */
    .stNumberInput input, .stSelectbox select, .stTextInput input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #635bff, #38bdf8);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        padding: 12px 32px;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(99,91,255,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,91,255,0.5);
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Resources (cached) ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_metrics():
    with open("model_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    df = pd.read_csv("electricity_cost_dataset.csv")
    return df

@st.cache_data
def get_test_predictions():
    """Re-split data identically to training and return test actuals + predictions."""
    df = load_dataset()
    target = "electricity cost"
    X = df.drop(columns=[target])
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = load_model()
    preds = model.predict(X_test)
    return y_test.values, preds

@st.cache_data
def get_feature_correlations():
    df = load_dataset()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [c for c in num_cols if c != "electricity cost"]
    corr = df[num_cols + ["electricity cost"]].corr()["electricity cost"].drop("electricity cost")
    corr = corr.abs().sort_values(ascending=True)
    # Clean up column names for display
    name_map = {
        "site area": "Site Area",
        "water consumption": "Water Consumption",
        "recycling rate": "Recycling Rate",
        "utilisation rate": "Utilisation Rate",
        "air qality index": "Air Quality Index",
        "issue reolution time": "Issue Resolution Time",
        "resident count": "Resident Count",
    }
    corr.index = [name_map.get(c, c) for c in corr.index]
    return corr


model    = load_model()
metrics  = load_metrics()
df       = load_dataset()

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9ca3af", family="Inter"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Electricity Cost\n### Predictor")
    st.markdown("---")
    st.markdown(
        f"""
        <div class='metric-card' style='margin-bottom:12px'>
            <div class='value'>{metrics['r2']}</div>
            <div class='label'>Model R²</div>
        </div>
        <div class='metric-card' style='margin-bottom:12px'>
            <div class='value'>${metrics['rmse']}</div>
            <div class='label'>RMSE</div>
        </div>
        <div class='metric-card'>
            <div class='value'>${metrics['mae']}</div>
            <div class='label'>MAE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        **Algorithm** · Gradient Boosting\n
        **Tuning** · RandomizedSearchCV (5-fold)\n
        **Preprocessing** · StandardScaler + OHE + PCA\n
        **Dataset** · 10,000 buildings
        """
    )
    st.markdown("---")
    st.markdown(
        "<a href='https://github.com/Garvita-Joshi/electricity-cost-predictor' "
        "target='_blank' style='color:#a78bfa;font-weight:600;'>🔗 View on GitHub</a>",
        unsafe_allow_html=True,
    )


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class='hero'>
        <h1>⚡ Electricity Cost Predictor</h1>
        <p>ML-powered dashboard for real-time building electricity cost estimation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Predict",
    "📊  Model Performance",
    "🔍  Data Explorer",
    "ℹ️   About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Enter Building Details</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            site_area = st.number_input(
                "🏗️ Site Area (sq. meters)",
                min_value=1.0, max_value=100000.0, value=1500.0, step=10.0,
                help="Total site area of the building in square meters"
            )
            water_consumption = st.number_input(
                "💧 Water Consumption (liters)",
                min_value=0.0, value=2500.0, step=50.0,
                help="Monthly water consumption in liters"
            )
            utilization_rate = st.slider(
                "⚙️ Utilization Rate (0 – 1)",
                min_value=0.0, max_value=1.0, value=0.75, step=0.01,
                help="What fraction of the building's capacity is being used"
            )
            issue_resolution_time = st.number_input(
                "🔧 Issue Resolution Time (hours)",
                min_value=0.0, value=5.0, step=0.5,
                help="Average time to resolve maintenance issues"
            )

        with c2:
            structure_type = st.selectbox(
                "🏢 Structure Type",
                options=["Residential", "Commercial", "Mixed-use"],
                help="Primary use category of the building"
            )
            recycling_rate = st.slider(
                "♻️ Recycling Rate (%)",
                min_value=0, max_value=100, value=60,
                help="Percentage of waste that is recycled"
            )
            air_quality_index = st.number_input(
                "🌬️ Air Quality Index (AQI)",
                min_value=0.0, value=80.0, step=1.0,
                help="Local Air Quality Index reading"
            )
            resident_count = st.number_input(
                "👥 Resident Count",
                min_value=1, value=100, step=1,
                help="Total number of residents or occupants"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Calculate Predicted Cost", use_container_width=True)

    if submitted:
        try:
            if not model:
                st.error("Model not loaded. Please check the server.")
            else:
                input_df = pd.DataFrame([{
                    "site area":           site_area,
                    "structure type":      structure_type,
                    "water consumption":   water_consumption,
                    "recycling rate":      recycling_rate,
                    "utilisation rate":    utilization_rate,
                    "air qality index":    air_quality_index,
                    "issue reolution time": issue_resolution_time,
                    "resident count":      resident_count,
                }])

                prediction = model.predict(input_df)[0]

                # Percentile rank in dataset
                pct = (df["electricity cost"] < prediction).mean() * 100

                st.markdown(
                    f"""
                    <div class='prediction-box'>
                        <div class='cost-label'>PREDICTED MONTHLY ELECTRICITY COST</div>
                        <div class='cost-value'>${prediction:,.2f}</div>
                        <div style='color:#6b7280;font-size:0.85rem;margin-top:12px'>
                            Higher than <strong style='color:#a78bfa'>{pct:.0f}%</strong>
                            of buildings in the dataset
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    number={"prefix": "$", "font": {"size": 28, "color": "#a78bfa"}},
                    gauge={
                        "axis": {
                            "range": [
                                float(df["electricity cost"].min()),
                                float(df["electricity cost"].max()),
                            ],
                            "tickcolor": "#9ca3af",
                        },
                        "bar":       {"color": "#635bff"},
                        "bgcolor":   "rgba(0,0,0,0)",
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [df["electricity cost"].min(), df["electricity cost"].quantile(0.33)], "color": "rgba(52,211,153,0.2)"},
                            {"range": [df["electricity cost"].quantile(0.33), df["electricity cost"].quantile(0.66)], "color": "rgba(251,191,36,0.2)"},
                            {"range": [df["electricity cost"].quantile(0.66), df["electricity cost"].max()], "color": "rgba(239,68,68,0.2)"},
                        ],
                    },
                    title={"text": "Cost vs Dataset Range", "font": {"color": "#9ca3af", "size": 14}},
                ))
                fig_gauge.update_layout(**PLOTLY_THEME, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.markdown(
                f"<div class='error-box'>⚠️ Prediction failed: {str(e)}</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='metric-card'><div class='value'>{metrics['r2']}</div>"
            f"<div class='label'>R² Score</div><div class='sublabel'>Coefficient of Determination</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='metric-card'><div class='value'>${metrics['rmse']}</div>"
            f"<div class='label'>RMSE</div><div class='sublabel'>Root Mean Squared Error</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='metric-card'><div class='value'>${metrics['mae']}</div>"
            f"<div class='label'>MAE</div><div class='sublabel'>Mean Absolute Error</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"<div class='metric-card'><div class='value'>10K</div>"
            f"<div class='label'>Training Rows</div><div class='sublabel'>80/20 train-test split</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    # ── Feature Correlation Chart ──
    with col_left:
        st.markdown("#### 📌 Feature Correlation with Electricity Cost")
        corr = get_feature_correlations()
        fig_corr = px.bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            color=corr.values,
            color_continuous_scale=["#38bdf8", "#635bff", "#a78bfa"],
            labels={"x": "Absolute Correlation", "y": "Feature"},
        )
        fig_corr.update_coloraxes(showscale=False)
        fig_corr.update_traces(marker_line_width=0)
        fig_corr.update_layout(**PLOTLY_THEME, height=340,
                               xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Actual vs Predicted ──
    with col_right:
        st.markdown("#### 🎯 Actual vs Predicted (Test Set)")
        with st.spinner("Running test set evaluation…"):
            y_actual, y_pred = get_test_predictions()

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=y_actual, y=y_pred,
            mode="markers",
            marker=dict(
                color=np.abs(y_actual - y_pred),
                colorscale=[[0, "#34d399"], [0.5, "#635bff"], [1, "#f87171"]],
                size=4, opacity=0.7,
                colorbar=dict(title="Error ($)", thickness=10, tickfont=dict(color="#9ca3af")),
            ),
            name="Predictions",
        ))
        # Perfect prediction line
        lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        fig_scatter.add_trace(go.Scatter(
            x=lims, y=lims,
            mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dash"),
            name="Perfect Fit",
        ))
        fig_scatter.update_layout(
            **PLOTLY_THEME,
            height=340,
            xaxis=dict(title="Actual Cost ($)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Predicted Cost ($)", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(font=dict(color="#9ca3af")),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Error Distribution ──
    st.markdown("#### 📉 Prediction Error Distribution")
    errors = y_actual - y_pred
    fig_err = px.histogram(
        x=errors, nbins=60,
        color_discrete_sequence=["#635bff"],
        labels={"x": "Residual Error ($)", "y": "Count"},
    )
    fig_err.add_vline(x=0, line_dash="dash", line_color="#f59e0b", line_width=2)
    fig_err.update_layout(**PLOTLY_THEME, height=260,
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
    st.plotly_chart(fig_err, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset stats
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            "<div class='metric-card'><div class='value'>10,000</div>"
            "<div class='label'>Total Rows</div></div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='metric-card'><div class='value'>${df['electricity cost'].mean():,.0f}</div>"
            "<div class='label'>Avg Cost</div></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='metric-card'><div class='value'>${df['electricity cost'].min():,.0f}</div>"
            "<div class='label'>Min Cost</div></div>",
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"<div class='metric-card'><div class='value'>${df['electricity cost'].max():,.0f}</div>"
            "<div class='label'>Max Cost</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # ── Cost Distribution ──
    with col_a:
        st.markdown("#### 📊 Electricity Cost Distribution")
        fig_hist = px.histogram(
            df, x="electricity cost", nbins=50,
            color_discrete_sequence=["#635bff"],
            labels={"electricity cost": "Electricity Cost ($)"},
        )
        fig_hist.update_layout(**PLOTLY_THEME, height=320,
                               xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Cost by Structure Type ──
    with col_b:
        st.markdown("#### 🏢 Cost by Structure Type")
        fig_box = px.box(
            df, x="structure type", y="electricity cost",
            color="structure type",
            color_discrete_map={
                "Residential": "#a78bfa",
                "Commercial":  "#38bdf8",
                "Mixed-use":   "#34d399",
            },
            labels={"electricity cost": "Electricity Cost ($)", "structure type": ""},
        )
        fig_box.update_layout(**PLOTLY_THEME, height=320,
                              showlegend=False,
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown("#### 🌡️ Correlation Heatmap")
    num_df = df.select_dtypes(include=["int64", "float64"])
    corr_matrix = num_df.corr().round(2)
    display_names = {
        "site area": "Site Area",
        "water consumption": "Water Consumption",
        "recycling rate": "Recycling Rate",
        "utilisation rate": "Utilisation Rate",
        "air qality index": "Air Quality Index",
        "issue reolution time": "Issue Resolution Time",
        "resident count": "Resident Count",
        "electricity cost": "Electricity Cost",
    }
    corr_matrix.index = [display_names.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [display_names.get(c, c) for c in corr_matrix.columns]

    fig_heatmap = px.imshow(
        corr_matrix,
        color_continuous_scale=[[0, "#1e3a5f"], [0.5, "#302b63"], [1, "#a78bfa"]],
        zmin=-1, zmax=1,
        text_auto=True,
        aspect="auto",
    )
    fig_heatmap.update_traces(textfont=dict(size=11, color="white"))
    fig_heatmap.update_layout(**PLOTLY_THEME, height=420)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Feature vs Cost Scatter ──
    st.markdown("#### 🔎 Explore Feature vs Electricity Cost")
    feature_opts = {
        "Site Area":            "site area",
        "Water Consumption":    "water consumption",
        "Utilisation Rate":     "utilisation rate",
        "Recycling Rate":       "recycling rate",
        "Air Quality Index":    "air qality index",
        "Issue Resolution Time":"issue reolution time",
        "Resident Count":       "resident count",
    }
    selected_feature = st.selectbox("Select a feature", list(feature_opts.keys()))
    col_key = feature_opts[selected_feature]
    sample = df.sample(min(2000, len(df)), random_state=1)
    fig_feat = px.scatter(
        sample, x=col_key, y="electricity cost",
        color="structure type",
        color_discrete_map={
            "Residential": "#a78bfa",
            "Commercial":  "#38bdf8",
            "Mixed-use":   "#34d399",
        },
        opacity=0.6,
        trendline="ols",
        labels={col_key: selected_feature, "electricity cost": "Electricity Cost ($)"},
    )
    fig_feat.update_layout(**PLOTLY_THEME, height=360,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           legend=dict(font=dict(color="#9ca3af")))
    st.plotly_chart(fig_feat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>About This Project</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_about, col_stack = st.columns([3, 2])

    with col_about:
        st.markdown(
            """
            <div class='insight-card'>
                <div class='insight-title'>🎯 Project Goal</div>
                <div class='insight-text'>
                    Built an end-to-end ML pipeline to predict monthly electricity costs for buildings
                    based on structural and operational metrics. The model helps facility managers
                    and urban planners estimate energy expenditure before construction or during audits.
                </div>
            </div>
            <div class='insight-card'>
                <div class='insight-title'>🔬 ML Pipeline</div>
                <div class='insight-text'>
                    Conducted full EDA on 10,000 building records → applied IQR outlier clipping,
                    log-transformation for skewed features, StandardScaler + OneHotEncoder preprocessing,
                    PCA dimensionality reduction (10 components), and hyperparameter tuning via
                    RandomizedSearchCV with 5-fold cross-validation. Final model: Gradient Boosting Regressor.
                </div>
            </div>
            <div class='insight-card'>
                <div class='insight-title'>📈 Resume Bullets</div>
                <div class='insight-text'>
                    • Built a full ML pipeline to predict electricity costs for 10,000+ buildings.<br>
                    • Achieved <strong style='color:#a78bfa'>R²=0.96</strong> using Gradient Boosting with RandomizedSearchCV tuning.<br>
                    • Designed a Streamlit dashboard for real-time prediction and data exploration.<br>
                    • Deployed on Render with live API and interactive visualisations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_stack:
        st.markdown("#### 🛠️ Tech Stack")
        chips = [
            "Python 3.11", "Streamlit", "scikit-learn", "XGBoost",
            "Gradient Boosting", "Pandas", "NumPy", "Plotly",
            "Joblib", "Render", "PCA", "RandomizedSearchCV",
        ]
        chips_html = "".join(f"<span class='chip'>{c}</span>" for c in chips)
        st.markdown(chips_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📦 Dataset")
        st.markdown(
            """
            <div class='insight-card'>
                <div class='insight-text'>
                    <strong style='color:#e0e0e0'>10,000 rows</strong> · 8 features · 1 target<br><br>
                    Features: Site Area, Structure Type, Water Consumption,
                    Recycling Rate, Utilisation Rate, AQI,
                    Issue Resolution Time, Resident Count<br><br>
                    Target: <strong style='color:#a78bfa'>Monthly Electricity Cost ($)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🔗 Links")
        st.markdown(
            """
            <a href='https://github.com/Garvita-Joshi/electricity-cost-predictor'
               target='_blank'
               style='display:block;background:rgba(99,91,255,0.2);border:1px solid rgba(99,91,255,0.4);
                      border-radius:10px;padding:10px 16px;color:#a78bfa;font-weight:600;
                      text-decoration:none;margin-bottom:8px;text-align:center'>
               📂 GitHub Repository
            </a>
            """,
            unsafe_allow_html=True,
        )
