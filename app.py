import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_autorefresh import st_autorefresh
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Virtual CNC + AI Tool Monitoring",
    page_icon="⚙️",
    layout="wide"
)

st.title("Virtual CNC Machine & AI Tool Condition Monitoring")
st.caption(
    "Software simulation | Vibration + spindle current | "
    "No physical CNC machine is connected"
)

FEATURES = [
    "Vibration_RMS",
    "Vibration_STD",
    "Vibration_Peak",
    "Vibration_Kurtosis",
    "Current_RMS",
    "Current_Mean",
    "Current_STD",
    "Current_Peak"
]

MODEL_PATH = "CNC_TCM_Project/models/tool_condition_model.pkl"


# --------------------------------------------------
# SIMULATED SENSOR SIGNALS
# --------------------------------------------------
def generate_signals(condition, seed=1, n=2000, fs=2000):
    """Generate illustrative signals, not real machine measurements."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs

    settings = {
        "Healthy":  {"vib": 0.25, "current": 4.0, "noise": 0.05},
        "Worn":     {"vib": 0.65, "current": 5.5, "noise": 0.12},
        "Critical": {"vib": 1.15, "current": 7.5, "noise": 0.22}
    }
    s = settings[condition]

    vibration = (
        s["vib"] * np.sin(2 * np.pi * 50 * t)
        + 0.30 * s["vib"] * np.sin(2 * np.pi * 120 * t)
        + rng.normal(0, s["noise"], n)
    )

    current = (
        s["current"]
        + 0.25 * s["current"] * np.sin(2 * np.pi * 10 * t)
        + rng.normal(0, s["noise"] * 1.5, n)
    )

    return t, vibration, current


def extract_features(vibration, current):
    return {
        "Vibration_RMS": float(np.sqrt(np.mean(vibration ** 2))),
        "Vibration_STD": float(np.std(vibration)),
        "Vibration_Peak": float(np.max(np.abs(vibration))),
        "Vibration_Kurtosis": float(
            kurtosis(vibration, fisher=False, bias=False)
        ),
        "Current_RMS": float(np.sqrt(np.mean(current ** 2))),
        "Current_Mean": float(np.mean(current)),
        "Current_STD": float(np.std(current)),
        "Current_Peak": float(np.max(np.abs(current)))
    }


# --------------------------------------------------
# DEMONSTRATION AI MODEL
# --------------------------------------------------
@st.cache_resource
def get_model():
    """
    Use the saved project model only if its feature names match.
    Otherwise create a separate demonstration model from simulated data.
    """
    if os.path.exists(MODEL_PATH):
        try:
            saved_model = joblib.load(MODEL_PATH)
            saved_features = list(
                getattr(saved_model, "feature_names_in_", [])
            )
            if saved_features == FEATURES:
                return saved_model, "Saved project model"
        except Exception:
            pass

    rng = np.random.default_rng(42)
    rows = []
    labels = []

    for condition in ["Healthy", "Worn", "Critical"]:
        for i in range(100):
            _, vibration, current = generate_signals(
                condition,
                seed=int(rng.integers(0, 1_000_000))
            )
            rows.append(extract_features(vibration, current))
            labels.append(condition)

    X = pd.DataFrame(rows, columns=FEATURES)
    y = np.array(labels)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)

    return model, "Demonstration model trained on simulated data"


model, model_source = get_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "progress": 0.0,
    "running": False,
    "machine_status": "READY",
    "cycle_count": 0
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Virtual CNC Controls")

spindle_rpm = st.sidebar.slider(
    "Spindle speed (RPM)", 500, 6000, 2500, step=100
)
feed_rate = st.sidebar.slider(
    "Feed rate (mm/min)", 50, 1500, 400, step=25
)
cycle_duration = st.sidebar.slider(
    "Simulated cycle duration (seconds)", 10, 120, 40, step=5
)

condition_mode = st.sidebar.selectbox(
    "Tool condition for simulation",
    ["Automatic progression", "Healthy", "Worn", "Critical"]
)

st.sidebar.caption(
    "These controls change the virtual simulation only."
)

c1, c2 = st.sidebar.columns(2)
if c1.button("▶ Start", use_container_width=True):
    st.session_state.progress = 0.0
    st.session_state.running = True
    st.session_state.machine_status = "RUNNING"
    st.session_state.cycle_count += 1

if c2.button("⏸ Pause", use_container_width=True):
    st.session_state.running = False
    st.session_state.machine_status = "PAUSED"

c3, c4 = st.sidebar.columns(2)
if c3.button("▶ Resume", use_container_width=True):
    if st.session_state.progress < 1.0:
        st.session_state.running = True
        st.session_state.machine_status = "RUNNING"

if c4.button("■ Stop", use_container_width=True):
    st.session_state.running = False
    st.session_state.progress = 0.0
    st.session_state.machine_status = "STOPPED"


# Refresh the page every second so the simulation can advance.
st_autorefresh(interval=1000, key="virtual_cnc_refresh")

if st.session_state.running:
    st.session_state.progress += 1.0 / cycle_duration
    if st.session_state.progress >= 1.0:
        st.session_state.progress = 1.0
        st.session_state.running = False
        st.session_state.machine_status = "COMPLETED"


# --------------------------------------------------
# MACHINE STATUS
# --------------------------------------------------
progress = float(st.session_state.progress)

if condition_mode == "Automatic progression":
    if progress < 0.40:
        condition = "Healthy"
    elif progress < 0.75:
        condition = "Worn"
    else:
        condition = "Critical"
else:
    condition = condition_mode

status_col, rpm_col, feed_col, condition_col = st.columns(4)

status_col.metric("Machine status", st.session_state.machine_status)
rpm_col.metric("Spindle speed", f"{spindle_rpm} RPM")
feed_col.metric("Feed rate", f"{feed_rate} mm/min")
condition_col.metric("Simulation condition", condition)

st.progress(progress, text=f"Machining cycle: {progress * 100:.0f}%")


# --------------------------------------------------
# VIRTUAL WORKPIECE AND MOVING TOOLPATH
# --------------------------------------------------
st.subheader("1. Virtual CNC Machine")

# Create a rectangular workpiece surface.
x_grid = np.linspace(-40, 40, 24)
y_grid = np.linspace(-30, 30, 20)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X)

# Generate a zigzag toolpath over the workpiece.
path_x, path_y = [], []
rows = 8
points_per_row = 50

for row in range(rows):
    y = -27 + row * 54 / (rows - 1)
    xs = np.linspace(-37, 37, points_per_row)
    if row % 2 == 1:
        xs = xs[::-1]
    for x in xs:
        path_x.append(x)
        path_y.append(y)

path_x = np.array(path_x)
path_y = np.array(path_y)
tool_index = min(
    int(progress * (len(path_x) - 1)),
    len(path_x) - 1
)

tool_x = float(path_x[tool_index])
tool_y = float(path_y[tool_index])
tool_z = 5.0

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale=[
        [0, "#64748b"],
        [1, "#cbd5e1"]
    ],
    opacity=0.85,
    showscale=False,
    name="Workpiece"
))

fig.add_trace(go.Scatter3d(
    x=path_x,
    y=path_y,
    z=np.full_like(path_x, 0.7),
    mode="lines",
    line=dict(color="gray", width=3),
    name="Planned toolpath"
))

fig.add_trace(go.Scatter3d(
    x=path_x[:tool_index + 1],
    y=path_y[:tool_index + 1],
    z=np.full(tool_index + 1, 1.0),
    mode="lines",
    line=dict(color="blue", width=5),
    name="Completed path"
))

fig.add_trace(go.Scatter3d(
    x=[tool_x],
    y=[tool_y],
    z=[tool_z],
    mode="markers+text",
    marker=dict(size=9, color="red", symbol="diamond"),
    text=["Virtual tool"],
    textposition="top center",
    name="Cutting tool"
))

fig.update_layout(
    height=560,
    margin=dict(l=0, r=0, t=20, b=0),
    scene=dict(
        xaxis_title="X axis (mm)",
        yaxis_title="Y axis (mm)",
        zaxis_title="Z axis (mm)",
        xaxis=dict(range=[-45, 45]),
        yaxis=dict(range=[-35, 35]),
        zaxis=dict(range=[0, 12]),
        aspectmode="manual",
        aspectratio=dict(x=1.5, y=1.1, z=0.5)
    ),
    legend=dict(orientation="h", y=-0.05)
)

st.plotly_chart(fig, use_container_width=True)

axis1, axis2, axis3 = st.columns(3)
axis1.metric("Virtual X position", f"{tool_x:.1f} mm")
axis2.metric("Virtual Y position", f"{tool_y:.1f} mm")
axis3.metric("Virtual Z position", f"{tool_z:.1f} mm")

st.caption(
    "The red marker moves along the virtual toolpath. "
    "The displayed axes are simulated coordinates, not machine feedback."
)


# --------------------------------------------------
# SIMULATED SENSOR DATA AND AI PREDICTION
# --------------------------------------------------
st.subheader("2. AI Tool Condition Monitoring")

seed = int(progress * 10000) + int(spindle_rpm) + int(feed_rate)
t, vibration, current = generate_signals(condition, seed=seed)

features = extract_features(vibration, current)
feature_frame = pd.DataFrame([features], columns=FEATURES)

try:
    prediction = str(model.predict(feature_frame)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
except Exception as e:
    prediction = "Prediction unavailable"
    probabilities = None
    st.warning(f"Model prediction error: {e}")

left, right = st.columns(2)

with left:
    st.markdown("**Simulated vibration signal**")
    vib_fig = go.Figure()
    vib_fig.add_trace(go.Scatter(
        x=t[:400],
        y=vibration[:400],
        mode="lines",
        name="Vibration"
    ))
    vib_fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (simulated)",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(vib_fig, use_container_width=True)

with right:
    st.markdown("**Simulated spindle current**")
    cur_fig = go.Figure()
    cur_fig.add_trace(go.Scatter(
        x=t[:400],
        y=current[:400],
        mode="lines",
        name="Current"
    ))
    cur_fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Current (simulated units)",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(cur_fig, use_container_width=True)

pred_col, source_col = st.columns(2)
pred_col.metric("AI-predicted condition", prediction)
source_col.info(f"Model source: {model_source}")

if probabilities is not None:
    st.markdown("**Model class scores**")
    classes = list(model.classes_)
    prob_df = pd.DataFrame({
        "Condition": classes,
        "Model score": probabilities
    })
    st.bar_chart(prob_df.set_index("Condition"))

st.markdown("**Extracted signal features**")
st.dataframe(
    pd.DataFrame(
        {"Feature": list(features.keys()),
         "Value": list(features.values())}
    ),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# RESULTS EXPORT
# --------------------------------------------------
st.subheader("3. Export Simulation Results")

result = {
    "Machine_Status": st.session_state.machine_status,
    "Cycle_Progress": progress,
    "Spindle_RPM": spindle_rpm,
    "Feed_Rate_mm_min": feed_rate,
    "Selected_Simulation_Condition": condition,
    "AI_Predicted_Condition": prediction,
    "Virtual_X_mm": tool_x,
    "Virtual_Y_mm": tool_y,
    "Virtual_Z_mm": tool_z,
    **features
}

result_df = pd.DataFrame([result])
st.dataframe(result_df, use_container_width=True)

st.download_button(
    "Download current results as CSV",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="virtual_cnc_monitoring_results.csv",
    mime="text/csv"
)

st.warning(
    "Research limitation: sensor signals and machining are simulated. "
    "The AI result is a demonstration, not validated real-machine accuracy."
)
