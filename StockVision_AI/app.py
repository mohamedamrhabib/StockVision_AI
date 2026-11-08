import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --------------------------- Page setup ---------------------------
st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b1220; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #101a2b; border-right: 1px solid #263449; }
    [data-testid="stHeader"] { background: rgba(11,18,32,0.9); }
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(115deg,#14243b 0%,#102b36 55%,#153b35 100%);
        border: 1px solid #29445a; margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; color: #f8fafc; font-size: 2rem; font-weight: 800; }
    .hero p { margin: .45rem 0 0; color: #a9bacd; }
    .eyebrow { color: #53d6a1; letter-spacing: .13em; font-size: .72rem; font-weight: 700; }
    div[data-testid="stMetric"] {
        background: #121f32; border: 1px solid #263a51;
        padding: 15px 16px; border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { color: #9aacc2; }
    div[data-testid="stMetricValue"] { color: #f8fafc; }
    .panel {
        background: #111e30; border: 1px solid #263a51;
        border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
    }
    .muted { color: #9aacc2; font-size: .88rem; }
    .up { color: #42d6a0; font-weight: 800; }
    .down { color: #ff727f; font-weight: 800; }
    .small-note { color: #9aacc2; font-size: .78rem; }
    h2, h3 { color: #edf4fc !important; }
    .stTabs [data-baseweb="tab"] { color: #b8c7d9; }
    .stTabs [aria-selected="true"] { color: #4adea6 !important; }
    .stButton>button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
DATA_PATH = ART / "Final_Stock_Data_Macro.csv"

@st.cache_resource(show_spinner="Loading trained neural network...")
def load_assets():
    model = tf.keras.models.load_model(ART / "stock_model.keras", compile=False)
    scaler = joblib.load(ART / "scaler.pkl")
    feature_columns = joblib.load(ART / "feature_columns.pkl")
    history = joblib.load(ART / "history.pkl")
    ev = np.load(ART / "evaluation.npz", allow_pickle=True)
    evaluation = {k: ev[k] for k in ev.files}
    return model, scaler, feature_columns, history, evaluation

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0)

    # Reproduce the original notebook feature engineering exactly.
    df["MA_14"] = df["Close"].rolling(window=14).mean()
    df["Smooth_Trend"] = df["MA_14"].rolling(window=5).mean()
    df["Trend_Velocity"] = df["Smooth_Trend"].pct_change(1)
    df["Trend_Accel"] = df["Trend_Velocity"].diff(1)
    df["Trend_Jerk"] = df["Trend_Accel"].diff(1)
    df["Dist_Smooth"] = (df["Close"] - df["Smooth_Trend"]) / (df["Smooth_Trend"] + 1e-8)
    df["Momentum_13D"] = df["Close"].pct_change(13)

    if "EPU" in df.columns and "GPR" in df.columns:
        df["EPU_Shock"] = df["EPU"] / (df["EPU"].rolling(window=90, min_periods=1).mean() + 1e-8)
        df["GPR_Shock"] = df["GPR"] / (df["GPR"].rolling(window=90, min_periods=1).mean() + 1e-8)
        df["Macro_Crisis_Index"] = (df["EPU_Shock"] + df["GPR_Shock"]) / 2.0

    df["Target_Trend"] = np.where(
        df["Smooth_Trend"].shift(-1) > df["Smooth_Trend"], 1, 0
    )
    df.dropna(inplace=True)

    target_cols = ["Target_Next_Close", "Target_Direction", "Target_Trend"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    ratio_cols = [
        "Trend_Velocity", "Trend_Accel", "Trend_Jerk", "Dist_Smooth",
        "Momentum_13D", "EPU_Shock", "GPR_Shock", "Macro_Crisis_Index"
    ]
    other_cols = [c for c in feature_cols if c not in ratio_cols]

    stationary = pd.DataFrame(index=df.index)
    for c in other_cols:
        r_mean = df[c].rolling(window=30, min_periods=1).mean()
        r_std = df[c].rolling(window=30, min_periods=1).std().replace(0, 1).fillna(1)
        stationary[c] = np.clip((df[c] - r_mean) / (r_std + 1e-8), -4.0, 4.0)

    for c in ratio_cols:
        if c in df.columns:
            r_std = df[c].rolling(window=30, min_periods=1).std().replace(0, 1).fillna(1)
            stationary[c] = np.clip(df[c] / (r_std + 1e-8), -4.0, 4.0)

    stationary.replace([np.inf, -np.inf], 0, inplace=True)
    stationary.fillna(0, inplace=True)
    return df, stationary

def make_latest_prediction(model, scaler, feature_columns, stationary):
    missing = [c for c in feature_columns if c not in stationary.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    x = stationary[feature_columns].values
    x_scaled = np.clip(scaler.transform(x), -4.0, 4.0)
    if len(x_scaled) < 14:
        raise ValueError("At least 14 rows are required for a prediction.")
    seq = x_scaled[-14:][np.newaxis, :, :]
    probability = float(model.predict(seq, verbose=0).ravel()[0])
    return probability

def plot_layout(fig, height=360):
    fig.update_layout(
        height=height, paper_bgcolor="#111e30", plot_bgcolor="#111e30",
        font=dict(color="#cbd5e1", family="Inter"),
        margin=dict(l=18, r=18, t=45, b=18),
        xaxis=dict(gridcolor="#26384d", zerolinecolor="#26384d"),
        yaxis=dict(gridcolor="#26384d", zerolinecolor="#26384d"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# --------------------------- Load ---------------------------
try:
    model, scaler, feature_columns, history, ev = load_assets()
    df, stationary = load_data()
    latest_prob = make_latest_prediction(model, scaler, feature_columns, stationary)
except Exception as exc:
    st.error(f"Could not initialize StockVision AI: {exc}")
    st.info("Check that the artifacts folder contains the trained model, scaler, feature columns, history, evaluation file, and CSV dataset.")
    st.stop()

y_test = np.asarray(ev["y_test"]).astype(int).ravel()
preds = np.asarray(ev["preds"]).astype(int).ravel()
probs = np.asarray(ev["preds_prob"]).astype(float).ravel()
dates = pd.to_datetime(np.asarray(ev["test_dates"]).astype(str), errors="coerce")
test_acc = accuracy_score(y_test, preds)
test_precision = precision_score(y_test, preds, zero_division=0)
test_recall = recall_score(y_test, preds, zero_division=0)
test_f1 = f1_score(y_test, preds, zero_division=0)

# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## 📈 StockVision AI")
    st.caption("AI-powered market trend analysis")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Trend Prediction", "Market Analysis", "Model Performance"],
        label_visibility="visible"
    )
    st.divider()
    st.markdown("**MODEL STATUS**")
    st.success("Trained model loaded")
    st.markdown(f"<span class='small-note'>Input window: 14 time steps<br>Features: {len(feature_columns)}<br>Dataset rows: {len(df):,}</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Academic project demo • Historical dataset")

# --------------------------- Overview ---------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI • DEEP LEARNING • FINANCIAL ANALYTICS</div>
      <h1>StockVision AI</h1>
      <p>Stock market trend classification powered by Bidirectional LSTM and Multi-Head Attention.</p>
    </div>
    """, unsafe_allow_html=True)

    direction = "UPTREND" if latest_prob >= 0.5 else "DOWNTREND"
    direction_color = "#42d6a0" if latest_prob >= 0.5 else "#ff727f"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest closing price", f"{df['Close'].iloc[-1]:,.2f}")
    c2.metric("Latest model signal", direction)
    c3.metric("Uptrend probability", f"{latest_prob*100:.1f}%")
    c4.metric("Test accuracy", f"{test_acc*100:.2f}%")
    st.markdown(
        f"<div class='panel'><span class='muted'>Latest available observation</span><br>"
        f"<span style='font-size:1.25rem;font-weight:700;color:{direction_color}'>{direction}</span>"
        f"<span class='muted'> • Data date: {df.index[-1].strftime('%d %b %Y')} • "
        f"Probability threshold: 50%</span><p class='small-note'>This is a model classification for the next trend step, not a forecasted stock price or investment recommendation.</p></div>",
        unsafe_allow_html=True
    )

    st.subheader("Closing price history")
    price_fig = px.line(
        df.reset_index(), x="Date", y="Close",
        labels={"Close": "Closing Price", "Date": "Date"},
        title="Historical closing price"
    )
    price_fig.update_traces(line=dict(color="#42d6a0", width=2))
    st.plotly_chart(plot_layout(price_fig, 390), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Market trend features")
        feature_fig = go.Figure()
        feature_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#42d6a0")))
        feature_fig.add_trace(go.Scatter(x=df.index, y=df["Smooth_Trend"], name="Smoothed trend", line=dict(color="#60a5fa")))
        st.plotly_chart(plot_layout(feature_fig, 320), use_container_width=True)
    with b:
        st.subheader("Prediction confidence")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=latest_prob*100,
            number={"suffix": "%", "font": {"color": "#f8fafc"}},
            title={"text": "Uptrend probability", "font": {"color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar": {"color": "#42d6a0" if latest_prob >= .5 else "#ff727f"},
                "bgcolor": "#1b2b40",
                "steps": [{"range": [0, 50], "color": "#293449"}, {"range": [50, 100], "color": "#173d38"}],
                "threshold": {"line": {"color": "#f8fafc", "width": 2}, "value": 50}
            }
        ))
        st.plotly_chart(plot_layout(gauge, 320), use_container_width=True)

# --------------------------- Prediction ---------------------------
elif page == "Trend Prediction":
    st.markdown("<div class='hero'><div class='eyebrow'>MODEL INFERENCE</div><h1>Trend Prediction</h1><p>Run the trained model on the latest 14-observation sequence in the supplied dataset.</p></div>", unsafe_allow_html=True)
    latest_date = df.index[-1]
    latest_close = float(df["Close"].iloc[-1])
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("### Latest available input")
        st.metric("Observation date", latest_date.strftime("%d %B %Y"))
        st.metric("Latest close", f"{latest_close:,.2f}")
        st.caption("The model uses the latest 14 time steps and the same saved scaler and feature order used during training.")
        if st.button("Run prediction", type="primary", use_container_width=True):
            st.session_state["show_prediction"] = True
    with col2:
        st.markdown("### Prediction result")
        if st.session_state.get("show_prediction", True):
            is_up = latest_prob >= 0.5
            label = "UPTREND" if is_up else "DOWNTREND"
            color = "#42d6a0" if is_up else "#ff727f"
            st.markdown(
                f"<div class='panel' style='text-align:center;padding:28px 12px'>"
                f"<div class='muted'>NEXT-STEP TREND CLASS</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{color};margin:8px 0'>{label}</div>"
                f"<div style='font-size:1.5rem;color:#f8fafc'>{latest_prob*100:.2f}%</div>"
                f"<div class='muted'>Probability of Uptrend</div></div>",
                unsafe_allow_html=True
            )
            st.progress(float(np.clip(latest_prob, 0, 1)))
            st.caption(f"Downtrend probability: {(1-latest_prob)*100:.2f}%")
        else:
            st.info("Click Run prediction to display the model output.")
    st.warning("Historical-data limitation: the supplied dataset ends on 06 September 2022. This output is based on that last available observation, not live market data. It is not financial advice.")
    st.subheader("Recent closing prices")
    recent = df.tail(90).reset_index()
    f = px.line(recent, x="Date", y="Close", title="Last 90 available observations")
    f.update_traces(line=dict(color="#60a5fa", width=2))
    st.plotly_chart(plot_layout(f, 340), use_container_width=True)

# --------------------------- Market Analysis ---------------------------
elif page == "Market Analysis":
    st.markdown("<div class='hero'><div class='eyebrow'>HISTORICAL EXPLORATION</div><h1>Market Analysis</h1><p>Explore historical prices and technical / macroeconomic indicators in the dataset.</p></div>", unsafe_allow_html=True)
    min_date, max_date = df.index.min().date(), df.index.max().date()
    selected = st.date_input("Select analysis period", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        start, end = pd.Timestamp(selected[0]), pd.Timestamp(selected[1])
        view = df.loc[(df.index >= start) & (df.index <= end)]
    else:
        view = df.copy()
    if view.empty:
        st.warning("No observations in this date range.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Observations", f"{len(view):,}")
    m2.metric("First close", f"{view['Close'].iloc[0]:,.2f}")
    m3.metric("Last close", f"{view['Close'].iloc[-1]:,.2f}")

    st.subheader("Price and smoothed trend")
    f = go.Figure()
    f.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Close", line=dict(color="#42d6a0", width=1.8)))
    f.add_trace(go.Scatter(x=view.index, y=view["Smooth_Trend"], name="Smooth Trend", line=dict(color="#60a5fa", width=2)))
    st.plotly_chart(plot_layout(f, 390), use_container_width=True)

    left, right = st.columns(2)
    with left:
        indicator = st.selectbox("Technical indicator", ["RSI", "MACD", "EMA 12", "EMA 26", "Volume", "Momentum_13D", "Trend_Velocity"])
        if indicator in view.columns:
            g = px.line(view.reset_index(), x="Date", y=indicator, title=f"{indicator} over time")
            g.update_traces(line=dict(color="#a78bfa", width=1.8))
            st.plotly_chart(plot_layout(g, 330), use_container_width=True)
        else:
            st.info(f"{indicator} is not available.")
    with right:
        macro_options = [c for c in ["Interest_Rate", "Inflation_CPI", "Insider_Sentiment", "EPU", "GPR", "Macro_Crisis_Index"] if c in view.columns]
        macro_indicator = st.selectbox("Economic indicator", macro_options)
        g2 = px.line(view.reset_index(), x="Date", y=macro_indicator, title=f"{macro_indicator} over time")
        g2.update_traces(line=dict(color="#fbbf24", width=1.8))
        st.plotly_chart(plot_layout(g2, 330), use_container_width=True)
    st.caption("Note: EPU and GPR are missing in the supplied CSV and are filled with zero by the notebook preprocessing; interpret those series accordingly.")

# --------------------------- Model Performance ---------------------------
elif page == "Model Performance":
    st.markdown("<div class='hero'><div class='eyebrow'>OUT-OF-SAMPLE EVALUATION</div><h1>Model Performance</h1><p>Evaluation metrics and diagnostic visualizations from the saved test-set predictions.</p></div>", unsafe_allow_html=True)
    st.caption(f"Test samples: {len(y_test):,} • Evaluation period: {dates.min().strftime('%d %b %Y')} – {dates.max().strftime('%d %b %Y')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{test_acc*100:.2f}%")
    c2.metric("Precision", f"{test_precision*100:.2f}%")
    c3.metric("Recall", f"{test_recall*100:.2f}%")
    c4.metric("F1-score", f"{test_f1*100:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"], text=cm, texttemplate="%{text}",
            colorscale=[[0, "#14243b"], [1, "#247a70"]], showscale=False
        ))
        cm_fig.update_layout(xaxis_title="Predicted label", yaxis_title="True label")
        st.plotly_chart(plot_layout(cm_fig, 370), use_container_width=True)
    with right:
        st.subheader("Prediction probability distribution")
        hist_fig = px.histogram(
            x=probs, nbins=35, labels={"x": "Predicted probability", "count": "Samples"},
            title="Probability scores"
        )
        hist_fig.update_traces(marker_color="#60a5fa", opacity=.85)
        hist_fig.add_vline(x=.5, line_dash="dash", line_color="#ff727f", annotation_text="Threshold 0.5")
        st.plotly_chart(plot_layout(hist_fig, 370), use_container_width=True)

    st.subheader("Predicted vs actual trend")
    n = min(200, len(y_test))
    timeline = pd.DataFrame({
        "Date": dates[:n], "Actual": y_test[:n], "Predicted": preds[:n]
    })
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=timeline["Date"], y=timeline["Actual"], name="Actual", mode="lines", line=dict(color="#60a5fa", width=2)))
    line_fig.add_trace(go.Scatter(x=timeline["Date"], y=timeline["Predicted"], name="Predicted", mode="lines", line=dict(color="#f59e0b", width=1.8, dash="dash")))
    line_fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Down", "Up"])
    st.plotly_chart(plot_layout(line_fig, 340), use_container_width=True)

    st.subheader("Training history")
    hdf = pd.DataFrame(history)
    hdf["Epoch"] = np.arange(1, len(hdf) + 1)
    a, b = st.columns(2)
    with a:
        hf = go.Figure()
        hf.add_trace(go.Scatter(x=hdf["Epoch"], y=hdf["accuracy"], name="Training accuracy", line=dict(color="#42d6a0")))
        hf.add_trace(go.Scatter(x=hdf["Epoch"], y=hdf["val_accuracy"], name="Validation accuracy", line=dict(color="#60a5fa")))
        hf.update_layout(title="Accuracy by epoch", xaxis_title="Epoch", yaxis_title="Accuracy")
        st.plotly_chart(plot_layout(hf, 320), use_container_width=True)
    with b:
        lf = go.Figure()
        lf.add_trace(go.Scatter(x=hdf["Epoch"], y=hdf["loss"], name="Training loss", line=dict(color="#f59e0b")))
        lf.add_trace(go.Scatter(x=hdf["Epoch"], y=hdf["val_loss"], name="Validation loss", line=dict(color="#a78bfa")))
        lf.update_layout(title="Loss by epoch", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(plot_layout(lf, 320), use_container_width=True)

    report = classification_report(y_test, preds, target_names=["Down (0)", "Up (1)"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.round(3)
    st.subheader("Classification report")
    st.dataframe(report_df, use_container_width=True)

st.markdown("<hr><div class='small-note' style='text-align:center'>StockVision AI • Academic Machine Learning Project • BiLSTM + Multi-Head Attention<br>For educational and research demonstration only — not investment advice.</div>", unsafe_allow_html=True)
