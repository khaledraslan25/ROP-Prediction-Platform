import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ROP Prediction Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(90deg, #08111f 0%, #0b1730 45%, #07212b 100%);
        color: white;
    }

    header, [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="collapsedControl"] {
        color: white;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08111f 0%, #0b1730 100%);
        border-right: 1px solid rgba(110, 175, 255, 0.12);
    }

    .hero-box {
        background: linear-gradient(90deg, rgba(16,31,58,0.95) 0%, rgba(11,26,59,0.95) 45%, rgba(7,48,57,0.92) 100%);
        border: 1px solid rgba(110, 175, 255, 0.18);
        border-radius: 26px;
        padding: 2.2rem 2.2rem 1.8rem 2.2rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.22);
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.35rem;
        letter-spacing: 0.2px;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #eef4ff;
        margin-bottom: 0.5rem;
    }

    .card-box {
        background: linear-gradient(180deg, rgba(20,31,58,0.92) 0%, rgba(15,23,45,0.92) 100%);
        border: 1px solid rgba(124, 154, 255, 0.16);
        border-radius: 22px;
        padding: 1.7rem 1.5rem 1.4rem 1.5rem;
        min-height: 180px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .card-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 0.8rem;
    }

    .card-text {
        font-size: 1rem;
        color: #f1f5ff;
        text-align: center;
        line-height: 1.45;
        margin-bottom: 1.25rem;
    }

    .section-box {
        background: rgba(10, 18, 35, 0.70);
        border: 1px solid rgba(125, 156, 255, 0.14);
        border-radius: 22px;
        padding: 1.4rem 1.4rem 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.25rem;
    }

    .section-sub {
        color: #d8e6ff;
        margin-bottom: 1rem;
        font-size: 0.98rem;
    }

    .small-note {
        color: #d2def5;
        font-size: 0.92rem;
    }

    .mapping-box {
        background: rgba(14, 28, 54, 0.75);
        border: 1px solid rgba(124, 154, 255, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.7rem 1rem;
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(7,36,89,0.95) 0%, rgba(10,71,125,0.92) 100%);
        border: 1px solid rgba(122, 181, 255, 0.35);
        border-radius: 24px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        text-align: center;
        margin-top: 0.7rem;
        margin-bottom: 1rem;
    }

    .result-label {
        font-size: 1.0rem;
        color: #d9ebff;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .result-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.5px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(124, 154, 255, 0.22);
        background: linear-gradient(90deg, #0e2c63 0%, #113b77 100%);
        color: white;
        font-weight: 700;
        padding: 0.65rem 0.9rem;
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        font-weight: 700;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.3rem;
    }

    .sidebar-sub {
        font-size: 0.92rem;
        color: #cfe0ff;
        margin-bottom: 1rem;
    }

    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PATHS
# ============================================================
APP_DIR = Path(".")
MODEL_DIR = APP_DIR / "models"

MODEL_FILE = MODEL_DIR / "ann_rop_model.pkl"
SCALER_X_FILE = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_FILE = MODEL_DIR / "scaler_y.pkl"
CONFIG_FILE = MODEL_DIR / "model_config.json"

# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_FEATURES = ["WOB", "RPM", "Torque", "SPP", "Flow in", "M.Temp in", "M.Wt in"]
DEFAULT_DEPTH_COL = "Measured Depth (m)"
DEFAULT_TARGET_COL = "ROP"

DISPLAY_LABELS = {
    "WOB": "WOB",
    "RPM": "RPM",
    "Torque": "Torque",
    "SPP": "SPP",
    "Flow in": "Flow in",
    "M.Temp in": "M.Temp in",
    "M.Wt in": "M.Wt in",
    "Measured Depth (m)": "Measured Depth (m)",
    "ROP": "ROP",
}

SYNONYMS = {
    "WOB": [
        "wob", "weight on bit", "weight_on_bit", "weight-on-bit",
        "bit weight", "bit load", "hook load on bit"
    ],
    "RPM": [
        "rpm", "rotary rpm", "rotation rpm", "rotary speed", "bit rpm", "revolutions per minute"
    ],
    "Torque": [
        "torque", "rotary torque", "surface torque", "bit torque"
    ],
    "SPP": [
        "spp", "stand pipe pressure", "standpipe pressure", "stand_pipe_pressure",
        "standpipe_press", "surface pump pressure"
    ],
    "Flow in": [
        "flow in", "flow_in", "flow-in", "inlet flow", "input flow", "pump flow", "flow rate in", "flowrate in"
    ],
    "M.Temp in": [
        "m temp in", "m.temp in", "m_temp_in", "motor temp in", "motor temperature in",
        "motor inlet temp", "motor input temperature", "temperature in", "temp in"
    ],
    "M.Wt in": [
        "m wt in", "m.wt in", "m_wt_in", "mud weight in",
        "mud wt in", "mudweight in", "inlet mud weight"
    ],
    "Measured Depth (m)": [
        "measured depth", "measured_depth", "measured-depth", "depth", "depth m",
        "depth_m", "md", "md m", "measured depth m", "measureddepthm"
    ],
    "ROP": [
        "rop", "actual rop", "rop actual", "rop_actual", "actual_rop", "actual-rop",
        "rate of penetration", "rate_of_penetration", "actual rate of penetration"
    ],
    "Well": [
        "well", "well name", "well_name", "well-name", "wellid", "well id", "well_id"
    ]
}

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "single_history" not in st.session_state:
    st.session_state.single_history = []

# ============================================================
# HELPERS
# ============================================================
def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


@st.cache_resource
def load_artifacts():
    missing = []
    for p in [MODEL_FILE, SCALER_X_FILE, SCALER_Y_FILE]:
        if not p.exists():
            missing.append(p.name)

    if missing:
        return None, None, None, None, missing

    model = joblib.load(MODEL_FILE)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}

    return model, scaler_x, scaler_y, config, []


def get_feature_columns(config: dict):
    features = config.get("features", DEFAULT_FEATURES)
    if not isinstance(features, list) or len(features) == 0:
        features = DEFAULT_FEATURES
    return features


def predict_rop(df_features: pd.DataFrame, model, scaler_x, scaler_y):
    X_scaled = scaler_x.transform(df_features)
    y_scaled = model.predict(X_scaled)
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    y_pred = scaler_y.inverse_transform(y_scaled).ravel()

    # Ensure no negative ROP predictions
    y_pred = np.abs(y_pred)

    return y_pred


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Predictions"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


def make_sample_bulk_template(features):
    return pd.DataFrame(columns=features)


def make_sample_depth_template(features):
    cols = [DEFAULT_DEPTH_COL, "Well"] + features + [DEFAULT_TARGET_COL]
    return pd.DataFrame(columns=cols)


def add_back_button():
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.button("← Home", key=f"home_btn_{st.session_state.page}"):
            go_to("home")


def hero():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">ROP Prediction Platform</div>
            <div class="hero-sub">A smart drilling workflow for rapid rate-of-penetration estimation across individual records, uploaded datasets, and depth-profile analysis.</div>
            <div class="hero-sub">Built to support faster evaluation, clearer visualization, and easier interpretation of drilling performance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_depth_profile(df_plot, depth_col, pred_col, actual_col=None, well_col=None):
    fig = go.Figure()

    if well_col and well_col in df_plot.columns:
        wells = df_plot[well_col].astype(str).fillna("Unknown").unique()

        for well in wells:
            tmp = df_plot[df_plot[well_col].astype(str) == str(well)].copy()

            if actual_col and actual_col in tmp.columns:
                fig.add_trace(
                    go.Scatter(
                        x=tmp[actual_col],
                        y=tmp[depth_col],
                        mode="lines+markers",
                        name=f"{well} - Actual ROP",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=tmp[pred_col],
                    y=tmp[depth_col],
                    mode="lines+markers",
                    name=f"{well} - Predicted ROP",
                )
            )
    else:
        if actual_col and actual_col in df_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot[actual_col],
                    y=df_plot[depth_col],
                    mode="lines+markers",
                    name="Actual ROP",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=df_plot[pred_col],
                y=df_plot[depth_col],
                mode="lines+markers",
                name="Predicted ROP",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=750,
        title="ROP vs Measured Depth",
        xaxis_title="ROP (m/hr)",
        yaxis_title=depth_col,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=30, r=20, t=60, b=30),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig


def normalize_name(name):
    s = str(name).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_synonym_lookup():
    lookup = {}
    for canonical, variants in SYNONYMS.items():
        all_variants = set(variants + [canonical])
        for v in all_variants:
            lookup[normalize_name(v)] = canonical
    return lookup


SYNONYM_LOOKUP = build_synonym_lookup()


def auto_map_columns(df_columns):
    mapping = {}
    used_targets = set()

    for original_col in df_columns:
        norm = normalize_name(original_col)

        matched = None
        if norm in SYNONYM_LOOKUP:
            matched = SYNONYM_LOOKUP[norm]
        else:
            for alias_norm, canonical in SYNONYM_LOOKUP.items():
                if norm == alias_norm:
                    matched = canonical
                    break
                if norm in alias_norm or alias_norm in norm:
                    matched = canonical
                    break

        if matched is not None and matched not in used_targets:
            mapping[original_col] = matched
            used_targets.add(matched)
        else:
            mapping[original_col] = original_col

    return mapping


def show_column_mapping(df_before, df_after):
    records = []
    for old, new in zip(df_before.columns, df_after.columns):
        if old != new:
            records.append({
                "Uploaded Column": old,
                "System Column": new
            })

    if len(records) == 0:
        return

    map_df = pd.DataFrame(records)

    st.markdown("<div class='mapping-box'>", unsafe_allow_html=True)
    st.markdown("### Column interpretation window")
    st.dataframe(map_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def validate_columns(df: pd.DataFrame, required_cols):
    return [c for c in required_cols if c not in df.columns]


def build_missing_values_report(df: pd.DataFrame, check_cols):
    null_mask = df[check_cols].isna()
    rows_with_missing = null_mask.any(axis=1)

    if not rows_with_missing.any():
        return None, None, None

    missing_counts = null_mask.sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]
    missing_counts_df = missing_counts.reset_index()
    missing_counts_df.columns = ["Column", "Missing Count"]

    missing_locations = []
    for idx in df.index[rows_with_missing]:
        row_missing_cols = [col for col in check_cols if pd.isna(df.loc[idx, col])]
        missing_locations.append({
            "Row Index": idx,
            "Missing Columns": ", ".join(row_missing_cols)
        })

    missing_locations_df = pd.DataFrame(missing_locations)

    highlighted_df = df.copy()
    highlighted_df.insert(0, "Row Index", highlighted_df.index)

    def highlight_missing(v):
        if pd.isna(v):
            return "background-color: #8b0000; color: white; font-weight: bold;"
        return ""

    styled = highlighted_df.style.map(highlight_missing)

    return missing_counts_df, missing_locations_df, styled


def show_missing_values_message(df: pd.DataFrame, check_cols, section_name="file"):
    missing_counts_df, missing_locations_df, styled = build_missing_values_report(df, check_cols)

    if missing_counts_df is None:
        return False

    st.error(f"Prediction cannot proceed because the uploaded {section_name} contains missing values in required input columns.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Missing values by column")
        st.dataframe(missing_counts_df, use_container_width=True)

    with c2:
        st.markdown("### Missing value locations")
        st.dataframe(missing_locations_df, use_container_width=True)

    st.markdown("### Highlighted preview of missing values")
    st.dataframe(styled, use_container_width=True)

    st.info("Please fill or remove the highlighted missing values, then upload the corrected file again.")
    return True


def read_uploaded_table(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def standardize_uploaded_columns(df):
    original_df = df.copy()
    mapping = auto_map_columns(df.columns)
    df = df.rename(columns=mapping)
    return original_df, df


def find_well_column(df):
    for candidate in ["Well", "Well Name", "well", "well name"]:
        if candidate in df.columns:
            return candidate
    return None


# ============================================================
# LOAD MODEL
# ============================================================
model, scaler_x, scaler_y, config, missing_files = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Open any module directly from here.</div>', unsafe_allow_html=True)

    if st.button("Home", key="sb_home"):
        go_to("home")
    if st.button("Single Entry", key="sb_single"):
        go_to("single")
    if st.button("Bulk Upload", key="sb_bulk"):
        go_to("bulk")
    if st.button("Depth Profile", key="sb_depth"):
        go_to("depth")

    st.markdown("---")
    st.markdown(
        f"<div class='small-note'><b>Current Page:</b> {st.session_state.page.title()}</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# HOME PAGE
# ============================================================
def render_home():
    hero()

    if missing_files:
        st.error(
            "Missing model artifacts in ./models/: "
            + ", ".join(missing_files)
            + "\n\nPlace ann_rop_model.pkl, scaler_X.pkl, scaler_y.pkl, and optional model_config.json inside the models folder."
        )
        st.stop()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <div class="card-box">
                <div>
                    <div class="card-title">Single Entry</div>
                    <div class="card-text">Quick one-point prediction from manual inputs.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Single Entry", key="open_single"):
            go_to("single")

    with col2:
        st.markdown(
            """
            <div class="card-box">
                <div>
                    <div class="card-title">Bulk Upload</div>
                    <div class="card-text">Predict many rows and export enriched results.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Bulk Upload", key="open_bulk"):
            go_to("bulk")

    with col3:
        st.markdown(
            """
            <div class="card-box">
                <div>
                    <div class="card-title">Depth Profile</div>
                    <div class="card-text">Plot predicted vs actual ROP across measured depth.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Depth Profile", key="open_depth"):
            go_to("depth")


# ============================================================
# SINGLE ENTRY PAGE
# ============================================================
def render_single_entry():
    add_back_button()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Single Entry</div>
            <div class="section-sub">Enter one set of drilling parameters and predict ROP instantly.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = get_feature_columns(config)

    with st.form("single_prediction_form"):
        cols = st.columns(3)
        inputs = {}

        for i, feat in enumerate(features):
            with cols[i % 3]:
                inputs[feat] = st.number_input(
                    DISPLAY_LABELS.get(feat, feat),
                    value=0.0,
                    step=0.1,
                    format="%g",
                    key=f"single_{feat}"
                )

        submitted = st.form_submit_button("Predict ROP")

    if submitted:
        try:
            df_input = pd.DataFrame([inputs], columns=features)
            pred = predict_rop(df_input, model, scaler_x, scaler_y)[0]

            record = df_input.copy()
            record["Predicted ROP (m/hr)"] = pred
            st.session_state.single_history.append(record.iloc[0].to_dict())

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted ROP (m/hr)</div>
                    <div class="result-value">{pred:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("### Prediction Table")
    if len(st.session_state.single_history) > 0:
        hist_df = pd.DataFrame(st.session_state.single_history)
        hist_df.index = np.arange(1, len(hist_df) + 1)
        hist_df.index.name = "Trial No."
        st.dataframe(hist_df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download Session Results as Excel",
                data=to_excel_bytes(hist_df.reset_index(), sheet_name="Single_History"),
                file_name="single_entry_session_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            if st.button("Clear Session Results", key="clear_single_history"):
                st.session_state.single_history = []
                st.rerun()
    else:
        st.info("No predictions recorded yet in this session.")


# ============================================================
# BULK UPLOAD PAGE
# ============================================================
def render_bulk_upload():
    add_back_button()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Bulk Upload</div>
            <div class="section-sub">Upload CSV or Excel, predict all rows, and download enriched results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = get_feature_columns(config)

    c1, c2 = st.columns([1.1, 1.5], gap="large")

    with c1:
        st.markdown("#### Input template")
        sample_df = make_sample_bulk_template(features)

        st.download_button(
            "Download sample CSV template",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_bulk_template.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download sample Excel template",
            data=to_excel_bytes(sample_df, sheet_name="Bulk_Template"),
            file_name="sample_bulk_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown(
            "<div class='small-note'>Flexible names supported. Example: weight on bit → WOB, motor temperature in → M.Temp in.</div>",
            unsafe_allow_html=True,
        )

    with c2:
        uploaded_file = st.file_uploader(
            "Upload bulk file",
            type=["csv", "xlsx", "xls"],
            key="bulk_upload",
        )

    if uploaded_file is not None:
        try:
            df_raw = read_uploaded_table(uploaded_file)
            df_before, df = standardize_uploaded_columns(df_raw)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(df_raw.head(20), use_container_width=True)

            show_column_mapping(df_before, df)

            missing_cols = validate_columns(df, features)
            if missing_cols:
                st.error("The system could not identify these required columns after intelligent mapping: " + ", ".join(missing_cols))
                st.stop()

            has_missing = show_missing_values_message(df, features, section_name="bulk file")
            if has_missing:
                st.stop()

            pred = predict_rop(df[features], model, scaler_x, scaler_y)
            result_df = df.copy()
            result_df["Predicted ROP"] = pred

            st.markdown("### Prediction Results")
            st.dataframe(result_df.head(50), use_container_width=True)

            st.download_button(
                "Download Predicted Excel",
                data=to_excel_bytes(result_df, sheet_name="Bulk_Predictions"),
                file_name="bulk_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                "Download Predicted CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")


# ============================================================
# DEPTH PROFILE PAGE
# ============================================================
def render_depth_profile():
    add_back_button()

    st.markdown(
        """
        <div class="section-box">
            <div class="section-title">Depth Profile</div>
            <div class="section-sub">Upload a depth-wise file to generate predicted ROP along measured depth and compare against actual ROP when available.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features = get_feature_columns(config)

    c1, c2 = st.columns([1.1, 1.5], gap="large")

    with c1:
        st.markdown("#### Input template")
        sample_depth_df = make_sample_depth_template(features)

        st.download_button(
            "Download depth-profile CSV template",
            data=sample_depth_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_depth_profile_template.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download depth-profile Excel template",
            data=to_excel_bytes(sample_depth_df, sheet_name="Depth_Profile_Template"),
            file_name="sample_depth_profile_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown(
            "<div class='small-note'>Flexible names supported for depth, well, and actual ROP. Example: depth_m → Measured Depth (m), actual rop → ROP, well_name → Well.</div>",
            unsafe_allow_html=True,
        )

    with c2:
        uploaded_file = st.file_uploader(
            "Upload depth-profile file",
            type=["csv", "xlsx", "xls"],
            key="depth_upload",
        )

    if uploaded_file is not None:
        try:
            df_raw = read_uploaded_table(uploaded_file)
            df_before, df = standardize_uploaded_columns(df_raw)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(df_raw.head(20), use_container_width=True)

            show_column_mapping(df_before, df)

            required = [DEFAULT_DEPTH_COL] + features
            missing_cols = validate_columns(df, required)
            if missing_cols:
                st.error("The system could not identify these required columns after intelligent mapping: " + ", ".join(missing_cols))
                st.stop()

            has_missing = show_missing_values_message(df, required, section_name="depth profile file")
            if has_missing:
                st.stop()

            df_plot = df.copy()
            df_plot["Predicted ROP"] = predict_rop(df_plot[features], model, scaler_x, scaler_y)

            well_col = "Well" if "Well" in df_plot.columns else find_well_column(df_plot)

            # Filters
            st.markdown("### Filters")
            f1, f2, f3 = st.columns(3)

            with f1:
                if well_col:
                    well_options = ["All Wells"] + sorted(df_plot[well_col].dropna().astype(str).unique().tolist())
                    selected_well = st.selectbox("Select Well", well_options, index=0)
                else:
                    selected_well = "All Wells"
                    st.info("No well column detected. Plot will use all data.")

            depth_min_data = float(df_plot[DEFAULT_DEPTH_COL].min())
            depth_max_data = float(df_plot[DEFAULT_DEPTH_COL].max())

            with f2:
                selected_depth_min = st.number_input(
                    "Depth From",
                    value=depth_min_data,
                    min_value=depth_min_data,
                    max_value=depth_max_data,
                    step=1.0,
                    format="%g"
                )

            with f3:
                selected_depth_max = st.number_input(
                    "Depth To",
                    value=depth_max_data,
                    min_value=depth_min_data,
                    max_value=depth_max_data,
                    step=1.0,
                    format="%g"
                )

            if selected_depth_min > selected_depth_max:
                st.warning("Depth From is greater than Depth To. Please correct the interval.")
                st.stop()

            filtered_df = df_plot.copy()

            if well_col and selected_well != "All Wells":
                filtered_df = filtered_df[filtered_df[well_col].astype(str) == str(selected_well)].copy()

            filtered_df = filtered_df[
                (filtered_df[DEFAULT_DEPTH_COL] >= selected_depth_min) &
                (filtered_df[DEFAULT_DEPTH_COL] <= selected_depth_max)
            ].copy()

            sort_choice = st.checkbox("Sort by Measured Depth before plotting", value=True)
            if sort_choice:
                sort_cols = [DEFAULT_DEPTH_COL]
                if well_col and well_col in filtered_df.columns:
                    sort_cols = [well_col, DEFAULT_DEPTH_COL]
                filtered_df = filtered_df.sort_values(sort_cols).reset_index(drop=True)

            if filtered_df.empty:
                st.warning("No rows found for the selected well/depth interval.")
                st.stop()

            st.markdown("### Depth Profile Data Preview")
            st.dataframe(filtered_df.head(100), use_container_width=True)

            actual_col = DEFAULT_TARGET_COL if DEFAULT_TARGET_COL in filtered_df.columns else None

            fig = plot_depth_profile(
                df_plot=filtered_df,
                depth_col=DEFAULT_DEPTH_COL,
                pred_col="Predicted ROP",
                actual_col=actual_col,
                well_col=well_col if selected_well == "All Wells" else None,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Depth Profile Results",
                data=to_excel_bytes(filtered_df, sheet_name="Depth_Profile"),
                file_name="depth_profile_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                "Download Plot Data as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="depth_profile_predictions.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Depth-profile processing failed: {e}")


# ============================================================
# ROUTER
# ============================================================
if missing_files and st.session_state.page != "home":
    st.session_state.page = "home"

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "single":
    render_single_entry()
elif st.session_state.page == "bulk":
    render_bulk_upload()
elif st.session_state.page == "depth":
    render_depth_profile()
else:
    render_home()