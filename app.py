from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_generator import generate_iot_dataset
from src.llm_assistant import OllamaError, generate_maintenance_advice, ollama_settings
from src.model_building import evaluate_model, train_model
from src.prediction import build_alert, predict_failure
from src.preprocessing import load_and_prepare_data


DATA_PATH = Path("data/industrial_iot_failure_data.csv")
FEATURE_LABELS = {
    "temperature_pressure_ratio": "Temperature / pressure",
    "vibration_load_index": "Vibration load",
    "rpm_current_ratio": "RPM / current",
    "stress_index": "Combined stress",
    "temperature": "Temperature",
    "vibration": "Vibration",
    "pressure": "Pressure",
    "rpm": "RPM",
    "current": "Current",
    "humidity": "Humidity",
    "coolant_temp": "Coolant temperature",
    "power_draw": "Power draw",
    "device_age_hours": "Device age",
    "maintenance_gap_days": "Maintenance gap",
}


st.set_page_config(
    page_title="Predictive maintenance | Industrial AI",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def train_and_evaluate_project():
    if not DATA_PATH.exists():
        generate_iot_dataset(output_path=DATA_PATH, n_samples=3000, seed=42)

    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(DATA_PATH)
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, model_dir="models")
    return X_test, y_test, feature_names, model, metrics


@st.cache_data(ttl=600)
def load_dataset():
    return pd.read_csv(DATA_PATH)


st.markdown(
    """
    <style>
    :root {
        --ink: #e7f0f4;
        --muted: #8ea4b3;
        --line: rgba(157, 185, 201, 0.16);
        --panel: rgba(16, 33, 45, 0.88);
        --lime: #b7f34a;
        --cyan: #55d6d0;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(26, 96, 100, 0.25), transparent 36%),
            radial-gradient(ellipse at 4% 26%, rgba(37, 78, 119, 0.16), transparent 32%),
            #07131c;
    }
    [data-testid="stHeader"] {
        background: rgba(7, 19, 28, 0.76);
    }
    [data-testid="stMain"] > div {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1d28 0%, #091720 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        letter-spacing: 0.02em;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19, 42, 54, 0.96), rgba(13, 29, 40, 0.96));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.14);
    }
    [data-testid="stMetricLabel"] {
        color: #9eb3c0;
        font-size: 0.76rem;
        font-weight: 650;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 720;
    }
    [data-testid="stMetricValue"] > div {
        color: var(--ink);
    }
    [data-testid="stMetric"] [data-testid="stHelpIcon"] {
        color: var(--cyan);
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.55rem;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stTabs"] button[role="tab"] {
        background: rgba(15, 32, 43, 0.7);
        border: 1px solid transparent;
        border-radius: 10px 10px 0 0;
        color: #94a9b7;
        padding: 0.65rem 1rem;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: rgba(183, 243, 74, 0.09);
        border-color: rgba(183, 243, 74, 0.22);
        color: var(--lime);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(150deg, rgba(17, 37, 49, 0.92), rgba(11, 26, 36, 0.96));
        border-color: var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stForm"] {
        background: rgba(13, 29, 40, 0.76);
        border-color: var(--line) !important;
        border-radius: 18px !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #0b1b25;
        border-color: rgba(157, 185, 201, 0.22);
        border-radius: 10px;
    }
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 650;
        transition: transform 120ms ease, border-color 120ms ease;
    }
    [data-testid="stButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--lime);
        transform: translateY(-1px);
    }
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--lime));
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    .pulse-hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        overflow: hidden;
        min-height: 220px;
        margin: 0.2rem 0 1.9rem;
        padding: 2rem 2.1rem;
        border: 1px solid rgba(108, 189, 190, 0.28);
        border-radius: 24px;
        background:
            radial-gradient(ellipse at 78% 42%, rgba(42, 172, 159, 0.23), transparent 34%),
            radial-gradient(ellipse at 99% 2%, rgba(183, 243, 74, 0.12), transparent 28%),
            linear-gradient(112deg, #102632 0%, #0b1b25 53%, #102a32 100%);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.3), inset 0 1px rgba(255, 255, 255, 0.035);
    }
    .pulse-hero::after {
        position: absolute;
        right: 26%;
        bottom: -110px;
        width: 320px;
        height: 220px;
        border: 1px solid rgba(85, 214, 208, 0.1);
        border-radius: 50%;
        content: "";
        pointer-events: none;
        box-shadow: 0 0 0 28px rgba(85, 214, 208, 0.025), 0 0 0 58px rgba(85, 214, 208, 0.02);
    }
    .pulse-brand {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        flex: 0 1 620px;
        min-width: 0;
    }
    .pulse-brand > div:last-child { min-width: 0; }
    .pulse-mark {
        display: grid;
        width: 68px;
        height: 68px;
        flex: 0 0 68px;
        place-items: center;
        border: 1px solid rgba(183, 243, 74, 0.48);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(183, 243, 74, 0.18), rgba(85, 214, 208, 0.08));
        box-shadow: inset 0 0 28px rgba(183, 243, 74, 0.08), 0 0 32px rgba(183, 243, 74, 0.07);
    }
    .pulse-mark svg { width: 40px; height: 40px; }
    .pulse-kicker {
        margin-bottom: 0.35rem;
        color: var(--lime);
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.21em;
    }
    .pulse-title {
        margin: 0;
        color: #f1f7f8;
        font-size: clamp(2.2rem, 4vw, 3.55rem);
        font-weight: 800;
        letter-spacing: -0.065em;
        line-height: 0.98;
    }
    .pulse-title span { color: var(--cyan); }
    .pulse-description {
        margin: 0.7rem 0 0;
        color: #9bb0bb;
        font-size: 0.94rem;
    }
    .pulse-status {
        z-index: 1;
        flex: 0 0 218px;
        min-width: 218px;
        padding: 1.1rem 1.15rem;
        border: 1px solid rgba(157, 185, 201, 0.19);
        border-radius: 16px;
        background: linear-gradient(150deg, rgba(5, 17, 25, 0.68), rgba(13, 37, 44, 0.68));
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.035), 0 12px 30px rgba(0, 0, 0, 0.13);
    }
    .pulse-live {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.55rem;
        color: #dff9bc;
        font-size: 0.77rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 99px;
        background: var(--lime);
        box-shadow: 0 0 12px rgba(183, 243, 74, 0.8);
    }
    .pulse-status small {
        color: #8198a5;
        font-size: 0.69rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .pulse-status strong {
        display: block;
        margin-top: 0.15rem;
        color: #d9e8ec;
        font-size: 0.94rem;
    }
    .pulse-meta {
        display: flex;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }
    .pulse-meta span {
        padding: 0.28rem 0.5rem;
        border: 1px solid rgba(157, 185, 201, 0.14);
        border-radius: 6px;
        color: #9bb0bb;
        font-family: monospace;
        font-size: 0.66rem;
        letter-spacing: 0.03em;
    }
    .pulse-visual {
        z-index: 1;
        display: grid;
        flex: 0 1 230px;
        min-width: 130px;
        align-content: center;
        justify-items: center;
        gap: 0.45rem;
        opacity: 0.88;
    }
    .pulse-visual svg { width: 100%; max-width: 220px; overflow: visible; }
    .pulse-visual-label {
        color: #8299a5;
        font-family: monospace;
        font-size: 0.63rem;
        letter-spacing: 0.15em;
    }
    @media (max-width: 1200px) {
        .pulse-hero { align-items: flex-start; flex-direction: column; gap: 1.1rem; }
        .pulse-status { width: 100%; }
        .pulse-title { font-size: clamp(2rem, 6vw, 3rem); }
        .pulse-visual { display: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="pulse-hero">
      <div class="pulse-brand">
        <div class="pulse-mark" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none">
            <path d="M5 26h9l5-13 8 23 6-15 4 5h6" stroke="#B7F34A" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="19" cy="13" r="2.2" fill="#55D6D0"/>
          </svg>
        </div>
        <div>
          <div class="pulse-kicker">PULSE / FAILURE INTELLIGENCE</div>
          <h1 class="pulse-title">See risk.<br><span>Before downtime.</span></h1>
          <p class="pulse-description">Equipment signals, translated into earlier decisions.</p>
        </div>
      </div>
      <div class="pulse-visual" aria-hidden="true">
        <svg viewBox="0 0 260 92" fill="none">
          <path d="M4 63H52L70 58L83 67L104 22L126 78L146 45L163 57H254" stroke="#55D6D0" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 80H254" stroke="#56717C" stroke-width="1" stroke-dasharray="3 6"/>
          <circle cx="104" cy="22" r="4" fill="#B7F34A"/>
          <circle cx="104" cy="22" r="10" stroke="#B7F34A" stroke-opacity=".38"/>
        </svg>
        <span class="pulse-visual-label">TELEMETRY / SIGNAL TRACE</span>
      </div>
      <div class="pulse-status">
        <div class="pulse-live"><span class="pulse-dot"></span> ANALYTICS ONLINE</div>
        <small>DEMO ENVIRONMENT</small>
        <strong>Synthetic sensor data</strong>
        <div class="pulse-meta"><span>10 INPUTS</span><span>RF MODEL</span></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### :material/precision_manufacturing: PULSE")
    st.caption("INDUSTRIAL HEALTH CONSOLE")
    st.markdown("#### Control panel")
    st.caption("Tune the risk view and manage the local model workspace.")

    risk_threshold = st.slider(
        "Chart reference threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help=(
            "Moves the reference line on the probability chart. "
            "Alert severity is determined separately by the prediction policy."
        ),
    )

    with st.expander("Local AI assistant", expanded=False):
        _, default_model = ollama_settings()
        ollama_model = st.text_input("Ollama model", value=default_model)
        st.caption(
            "Explanations are sent only to your configured Ollama endpoint. "
            "The assistant does not change the model prediction."
        )

    st.markdown("#### Dataset")
    st.caption(f"`{DATA_PATH.as_posix()}`")
    regenerate = st.button(
        "Regenerate data and retrain",
        type="secondary",
        icon=":material/refresh:",
        width="stretch",
    )
    if regenerate:
        generate_iot_dataset(output_path=DATA_PATH, n_samples=3000, seed=42)
        st.cache_resource.clear()
        st.cache_data.clear()
        st.toast("Synthetic data regenerated. The model will retrain now.", icon=":material/check:")

try:
    X_test, y_test, feature_names, model, metrics = train_and_evaluate_project()
    dataset = load_dataset()
except Exception as exc:  # pragma: no cover - user-facing startup failure
    st.error(f"Unable to prepare the dashboard: {exc}", icon=":material/error:")
    st.stop()

st.subheader("Model performance")
st.caption("HELD-OUT VALIDATION · SYNTHETIC SENSOR DATA")

kpi_columns = st.columns(4)
kpi_details = [
    ("Accuracy", metrics["accuracy"], "Overall correct classifications"),
    ("Precision", metrics["precision"], "Share of failure alerts that were correct"),
    ("Recall", metrics["recall"], "Share of failures detected by the model"),
    ("F1 score", metrics["f1_score"], "Balance between precision and recall"),
]
for column, (label, value, help_text) in zip(kpi_columns, kpi_details):
    column.metric(label, f"{value:.1%}", border=True, help=help_text)

overview_tab, assessment_tab, data_tab = st.tabs(
    [
        "Overview",
        "Live assessment",
        "Dataset",
    ]
)

with overview_tab:
    st.subheader("Risk signals")
    st.caption("Compare the model's failure scores and inspect the strongest contributing signals.")

    chart_column, importance_column = st.columns(2, gap="large")
    risk_scores = model.predict_proba(X_test)[:, 1]

    with chart_column:
        with st.container(border=True):
            st.markdown("**Failure probability distribution**")
            st.caption("Predicted risk across healthy and failure test examples")
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#10212d")
            ax.set_facecolor("#10212d")
            ax.hist(
                risk_scores[y_test == 0],
                bins=20,
                alpha=0.82,
                label="Healthy",
                color="#55d6d0",
            )
            ax.hist(
                risk_scores[y_test == 1],
                bins=20,
                alpha=0.78,
                label="Failure",
                color="#ff7b72",
            )
            ax.axvline(
                risk_threshold,
                color="#b7f34a",
                linestyle="--",
                linewidth=1.5,
                label=f"Reference · {risk_threshold:.0%}",
            )
            ax.set_xlabel("Predicted probability of failure")
            ax.set_ylabel("Test examples")
            ax.tick_params(colors="#9eb3c0")
            ax.xaxis.label.set_color("#9eb3c0")
            ax.yaxis.label.set_color("#9eb3c0")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#314653")
            ax.grid(axis="y", color="#314653", alpha=0.52)
            ax.set_axisbelow(True)
            ax.legend(
                frameon=True,
                facecolor="#142a36",
                edgecolor="#314653",
                labelcolor="#d5e2e8",
            )
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    with importance_column:
        with st.container(border=True):
            st.markdown("**Top factors influencing risk**")
            st.caption("Random Forest feature importance · relative contribution")
            importances = model.named_steps["model"].feature_importances_
            sorted_idx = importances.argsort()[-10:]
            labels = [
                FEATURE_LABELS.get(feature_names[i], feature_names[i].replace("_", " ").title())
                for i in sorted_idx
            ]
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#10212d")
            ax.set_facecolor("#10212d")
            ax.barh(labels, importances[sorted_idx], color="#55d6d0", height=0.68)
            ax.set_xlabel("Relative importance")
            ax.tick_params(axis="y", length=0, colors="#c5d4da")
            ax.tick_params(axis="x", colors="#9eb3c0")
            ax.xaxis.label.set_color("#9eb3c0")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#314653")
            ax.grid(axis="x", color="#314653", alpha=0.52)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    with st.expander("Preview sensor records"):
        st.dataframe(dataset.head(10), hide_index=True, width="stretch")

with assessment_tab:
    st.subheader("Assess equipment health")
    st.caption(
        "Enter current operating measurements to estimate failure probability. "
        "Use consistent units and verify sensor readings before acting."
    )

    with st.form("risk_form", border=True):
        st.markdown("**Operating conditions**")
        operating_left, operating_right = st.columns(2)
        with operating_left:
            temperature = st.number_input(
                "Temperature (°C)", value=98.0, min_value=20.0, max_value=150.0
            )
            vibration = st.number_input(
                "Vibration (mm/s)", value=3.8, min_value=0.1, max_value=10.0, step=0.1
            )
            pressure = st.number_input(
                "Pressure (psi)", value=155.0, min_value=30.0, max_value=200.0
            )
            rpm = st.number_input("Speed (RPM)", value=1765.0, min_value=500.0, max_value=2200.0)
            current = st.number_input(
                "Current (A)", value=62.0, min_value=5.0, max_value=90.0
            )
        with operating_right:
            humidity = st.number_input(
                "Humidity (%)", value=54.0, min_value=10.0, max_value=95.0
            )
            coolant_temp = st.number_input(
                "Coolant temperature (°C)", value=96.0, min_value=20.0, max_value=130.0
            )
            power_draw = st.number_input(
                "Power draw (kW)", value=73.0, min_value=10.0, max_value=100.0
            )

        st.markdown("**Equipment history**")
        history_left, history_right = st.columns(2)
        with history_left:
            device_age_hours = st.number_input(
                "Device age (hours)", value=4800.0, min_value=100.0, max_value=10000.0
            )
        with history_right:
            maintenance_gap_days = st.number_input(
                "Days since maintenance", value=42.0, min_value=1.0, max_value=180.0
            )

        submitted = st.form_submit_button(
            "Run risk assessment",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

    if submitted:
        sample = {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "current": current,
            "humidity": humidity,
            "coolant_temp": coolant_temp,
            "power_draw": power_draw,
            "device_age_hours": device_age_hours,
            "maintenance_gap_days": maintenance_gap_days,
        }
        prediction, probability = predict_failure(model, sample)
        st.session_state["last_prediction"] = {
            "sample": sample,
            "prediction": prediction,
            "probability": probability,
        }

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        prediction = result["prediction"]
        probability = result["probability"]
        alert = build_alert(prediction, probability)
        is_critical = alert.startswith("CRITICAL")
        is_warning = alert.startswith("WARNING")

        with st.container(border=True):
            st.markdown("**Assessment result**")
            result_column, probability_column = st.columns([1, 2], vertical_alignment="center")
            with result_column:
                if is_critical:
                    st.error("High failure risk", icon=":material/error:")
                elif is_warning:
                    st.warning("Elevated failure risk", icon=":material/warning:")
                else:
                    st.success("Normal operating range", icon=":material/check_circle:")
                st.caption(alert)
            with probability_column:
                st.metric("Estimated failure probability", f"{probability:.1%}")
                st.progress(min(1.0, max(0.0, probability)))
                st.caption(
                    "Model score for this sample; not a calibrated guarantee or safety decision."
                )

        with st.expander("Local AI maintenance explanation"):
            st.caption(
                "Optional advisory guidance from your configured local Ollama model. "
                "It does not alter the risk estimate."
            )
            if st.button(
                "Generate maintenance explanation",
                type="secondary",
                icon=":material/psychology:",
            ):
                try:
                    with st.spinner(f"Requesting a local explanation from {ollama_model}..."):
                        advice = generate_maintenance_advice(
                            result["sample"],
                            prediction,
                            probability,
                            alert,
                            model=ollama_model,
                        )
                    st.markdown(advice)
                except OllamaError as exc:
                    st.error(str(exc), icon=":material/error:")
    else:
        st.info(
            "Submit the assessment form to see the estimated risk and alert.",
            icon=":material/info:",
        )

with data_tab:
    st.subheader("Sensor dataset")
    st.caption(f"Source: `{DATA_PATH.as_posix()}` · Synthetic industrial telemetry")

    healthy_count = int((dataset["failure"] == 0).sum())
    failure_count = int((dataset["failure"] == 1).sum())
    summary_columns = st.columns(3)
    summary_columns[0].metric("Records", f"{len(dataset):,}", border=True)
    summary_columns[1].metric("Healthy examples", f"{healthy_count:,}", border=True)
    summary_columns[2].metric("Failure examples", f"{failure_count:,}", border=True)

    label_filter = st.selectbox(
        "Filter records by outcome",
        ["All outcomes", "Healthy", "Failure"],
    )
    if label_filter == "Healthy":
        visible_dataset = dataset.loc[dataset["failure"] == 0]
    elif label_filter == "Failure":
        visible_dataset = dataset.loc[dataset["failure"] == 1]
    else:
        visible_dataset = dataset

    st.dataframe(
        visible_dataset,
        hide_index=True,
        width="stretch",
        height=440,
    )
    st.caption(
        f"Showing {len(visible_dataset):,} of {len(dataset):,} records. "
        "Failure label: 0 = healthy, 1 = failure."
    )

st.caption(
    "Prototype for education and portfolio demonstration. Validate with representative "
    "equipment data and qualified engineering review before operational use."
)
