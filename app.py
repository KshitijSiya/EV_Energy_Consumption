import base64
import io
import json
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ZERO-SCROLL BALANCED HUD STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EV Powertrain Telemetry & Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700;800&display=swap');

    /* Global Dark Automotive Cockpit Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 12%, #0e172a 0%, #060913 100%) !important;
        color: #e2ecf8 !important;
        font-family: 'Chakra Petch', sans-serif !important;
    }

    /* Balanced padding so the page fills the full viewport without scrolling */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    footer {
        display: none !important;
    }

    /* Disable Streamlit full-screen icon on images */
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    /* Top Navigation / Status Bar */
    .hud-title-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, rgba(0, 242, 254, 0.1) 0%, rgba(13, 22, 41, 0.8) 50%, rgba(0, 255, 135, 0.08) 100%);
        border: 1px solid rgba(0, 242, 254, 0.28);
        border-radius: 8px;
        padding: 6px 18px;
        margin-bottom: 8px;
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.08);
    }

    .hud-title-text {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #ffffff;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hud-status-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #00ff87;
        background: rgba(0, 255, 135, 0.12);
        padding: 3px 10px;
        border-radius: 4px;
        border: 1px solid rgba(0, 255, 135, 0.35);
        letter-spacing: 0.06em;
    }

    /* Master Column Panel Headers */
    .hud-panel-header {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        color: #00f2fe;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(0, 242, 254, 0.2);
        padding-bottom: 4px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Comfortable Widget Spacing to fill vertical space nicely */
    .stSlider {
        margin-bottom: 6px !important;
    }

    .stSlider > label {
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        color: #a4bede !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0px !important;
    }

    .stSelectbox {
        margin-bottom: 4px !important;
    }

    .stSelectbox > label {
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        color: #a4bede !important;
        margin-bottom: 0px !important;
    }

    .stRadio > label {
        font-size: 0.77rem !important;
        font-weight: 600 !important;
        color: #a4bede !important;
        margin-bottom: 2px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #070d1a !important;
        border: 1px solid rgba(0, 242, 254, 0.35) !important;
        color: #ffffff !important;
        font-size: 0.83rem !important;
        min-height: 35px !important;
        border-radius: 6px !important;
    }

    /* Prominent Centered Car Display */
    .car-display-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: radial-gradient(ellipse at center, rgba(0, 242, 254, 0.09) 0%, rgba(6, 12, 24, 0.95) 75%);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 8px;
        padding: 6px 10px;
        margin: 4px 0 8px 0;
        height: 215px;
        max-height: 215px;
        box-sizing: border-box;
        overflow: hidden;
    }

    .car-display-img {
        display: block;
        margin: 0 auto;
        max-height: 200px;
        max-width: 96%;
        width: auto;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 0 14px rgba(0, 242, 254, 0.45));
        pointer-events: none;
        user-select: none;
    }

    /* Single-Line Telemetry Chips */
    .hud-chips-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
        white-space: nowrap;
        flex-wrap: nowrap;
    }

    .hud-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 3px 10px;
        background: rgba(0, 242, 254, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 4px;
        color: #b0d2f8;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Telemetry Info Cards for Lower Columns */
    .telemetry-footer-card {
        background: rgba(0, 242, 254, 0.04);
        border: 1px solid rgba(0, 242, 254, 0.18);
        border-radius: 6px;
        padding: 8px 12px;
        margin-top: 10px;
        font-size: 0.73rem;
        color: #8da4c4;
    }

    .telemetry-footer-title {
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: #00f2fe;
        text-transform: uppercase;
        margin-bottom: 3px;
        display: flex;
        justify-content: space-between;
    }

    /* Result Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(9, 18, 36, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3), inset 0 0 12px rgba(0, 242, 254, 0.06);
    }

    .metric-card-accent {
        background: linear-gradient(135deg, rgba(0, 255, 135, 0.1) 0%, rgba(9, 28, 24, 0.95) 100%);
        border: 1px solid rgba(0, 255, 135, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3), inset 0 0 12px rgba(0, 255, 135, 0.06);
    }

    .metric-title {
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        font-weight: 700;
        color: #88a7ce;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 800;
        color: #00f2fe;
        line-height: 1.15;
    }

    .metric-value-accent {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 800;
        color: #00ff87;
        line-height: 1.15;
    }

    .metric-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #9cb5d3;
        margin-top: 2px;
    }

    /* Simulation Control Inputs and Sync Button */
    input[type="number"] {
        background-color: #080f1e !important;
        border: 1px solid rgba(0, 242, 254, 0.35) !important;
        color: #00f2fe !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        border-radius: 5px !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00f2fe, #00c3ff) !important;
        color: #040914 !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 5px 10px !important;
        width: 100% !important;
        box-shadow: 0 2px 10px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 16px rgba(0, 242, 254, 0.7) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. CAR SCALABILITY DATABASE (Configurable Presets with Custom Image Support)
# -----------------------------------------------------------------------------
car_database = {
    "Tata Nexon EV": {
        "base_weight_kg": 1400,
        "max_payload_kg": 400,
        "battery_capacity_kwh": 30.2,
        "max_speed_kmh": 120,
        "image": "nexon_ev.png",
    },
    "MG ZS EV": {
        "base_weight_kg": 1570,
        "max_payload_kg": 450,
        "battery_capacity_kwh": 50.3,
        "max_speed_kmh": 140,
        "image": "mg_zs_ev.png",
    },
    "Tesla Model 3 (RWD)": {
        "base_weight_kg": 1610,
        "max_payload_kg": 430,
        "battery_capacity_kwh": 60.0,
        "max_speed_kmh": 200,
        "image": "tesla_model3.png",
    },
    "Hyundai Ioniq 5": {
        "base_weight_kg": 1910,
        "max_payload_kg": 520,
        "battery_capacity_kwh": 72.6,
        "max_speed_kmh": 185,
        "image": "hyundai-ioniq_ev.png",
    },
}

# -----------------------------------------------------------------------------
# 3. ROBUST MODEL LOADER (Cross-Version Fallback for XGBoost 3.x/2.x)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ev_model(model_path="xgboost_ev_model.pkl"):
    """Loads the trained XGBoost model seamlessly across different library versions."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        class UBJSONReader:
            def __init__(self, data):
                self.stream = io.BytesIO(data)

            def read_val(self, type_byte=None):
                if not type_byte:
                    type_byte = self.stream.read(1)
                    if not type_byte:
                        return None
                if type_byte in (b"Z", b"N"):
                    return None
                if type_byte == b"T":
                    return True
                if type_byte == b"F":
                    return False
                if type_byte == b"i":
                    return int.from_bytes(self.stream.read(1), "big", signed=True)
                if type_byte == b"U":
                    return int.from_bytes(self.stream.read(1), "big", signed=False)
                if type_byte == b"I":
                    return int.from_bytes(self.stream.read(2), "big", signed=True)
                if type_byte == b"l":
                    return int.from_bytes(self.stream.read(4), "big", signed=True)
                if type_byte == b"L":
                    return int.from_bytes(self.stream.read(8), "big", signed=True)
                if type_byte == b"d":
                    import struct

                    return struct.unpack(">f", self.stream.read(4))[0]
                if type_byte == b"D":
                    import struct

                    return struct.unpack(">d", self.stream.read(8))[0]
                if type_byte in (b"S", b"s"):
                    length = self.read_val()
                    return self.stream.read(length).decode("utf-8", "ignore")
                if type_byte == b"[":
                    tag = self.stream.read(1)
                    item_type, count = None, None
                    if tag == b"$":
                        item_type = self.stream.read(1)
                        tag = self.stream.read(1)
                    if tag == b"#":
                        count = self.read_val()
                    else:
                        self.stream.seek(-1, 1)
                    items = []
                    if count is not None:
                        for _ in range(count):
                            items.append(self.read_val(item_type))
                    else:
                        while True:
                            b = self.stream.read(1)
                            if b == b"]":
                                break
                            items.append(self.read_val(b))
                    return items
                if type_byte == b"{":
                    tag = self.stream.read(1)
                    item_type, count = None, None
                    if tag == b"$":
                        item_type = self.stream.read(1)
                        tag = self.stream.read(1)
                    if tag == b"#":
                        count = self.read_val()
                    else:
                        self.stream.seek(-1, 1)
                    obj = {}
                    if count is not None:
                        for _ in range(count):
                            k_len = self.read_val()
                            k = self.stream.read(k_len).decode("utf-8", "ignore")
                            obj[k] = self.read_val(item_type)
                    else:
                        while True:
                            k_len_type = self.stream.read(1)
                            if k_len_type == b"}":
                                break
                            k_len = self.read_val(k_len_type)
                            k = self.stream.read(k_len).decode("utf-8", "ignore")
                            obj[k] = self.read_val(item_type)
                    return obj
                raise ValueError(f"Unknown type: {type_byte}")

        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if "xgboost" in module:
                    class Dummy:
                        def __setstate__(self, state):
                            self.state = state

                        def __call__(self, *a, **k):
                            return self

                    return Dummy
                return super().find_class(module, name)

        with open(model_path, "rb") as f:
            unpickled = SafeUnpickler(f).load()

        handle_bytes = bytes(unpickled.state["_Booster"].state["handle"])
        parsed_ubjson = UBJSONReader(handle_bytes).read_val()
        model_json_bytes = json.dumps(parsed_ubjson["Model"]).encode("utf-8")

        booster = xgb.Booster()
        booster.load_model(bytearray(model_json_bytes))
        regressor = xgb.XGBRegressor()
        regressor._Booster = booster
        return regressor


model = load_ev_model("xgboost_ev_model.pkl")

# Helper to load and encode image to Base64
def get_image_base64(filepath):
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

# -----------------------------------------------------------------------------
# 4. TOP HUD TELEMETRY HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hud-title-bar">
        <div class="hud-title-text">
            <span>⚡ EV POWERTRAIN ENERGY & RANGE PREDICTOR</span>
            <span style="font-size: 0.74rem; color: #00f2fe; font-weight: 500;">| TELEMETRY INFERENCE CONSOLE</span>
        </div>
        <div class="hud-status-badge">● INFERENCE ENGINE: ONLINE (XGBOOST)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 5. STRICT 3-COLUMN BALANCED GRID
# -----------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1.15, 1.1, 1.15], gap="medium")

# =============================================================================
# CENTER COLUMN (THE VEHICLE HUD & ARCHITECTURE)
# =============================================================================
with col_center:
    st.markdown(
        """
        <div class="hud-panel-header">
            <span>VEHICLE ARCHITECTURE</span>
            <span style="color: #7998c2;">PLATFORM SPECS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Car Selection Dropdown
    selected_car_name = st.selectbox(
        "ACTIVE VEHICLE MODEL",
        options=list(car_database.keys()),
        index=0,
        label_visibility="collapsed",
    )
    car_specs = car_database[selected_car_name]

    # Resolve Car Image: Car-specific image if present, else fallback
    car_img_target = car_specs.get("image")
    if car_img_target and os.path.exists(car_img_target):
        active_car_img_path = car_img_target
    elif os.path.exists("ev_car.png"):
        active_car_img_path = "ev_car.png"
    elif os.path.exists("edited-image.png"):
        active_car_img_path = "edited-image.png"
    elif os.path.exists("car_image.png"):
        active_car_img_path = "car_image.png"
    else:
        active_car_img_path = None

    car_b64 = get_image_base64(active_car_img_path)

    # 2. Prominent Centered Car Display (Height 215px)
    if car_b64:
        st.markdown(
            f"""
            <div class="car-display-container">
                <img src="data:image/png;base64,{car_b64}" class="car-display-img" alt="{selected_car_name}" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Single-Line Telemetry Specs Chips
    st.markdown(
        f"""
        <div class="hud-chips-row">
            <span class="hud-chip">BATT: {car_specs['battery_capacity_kwh']} kWh</span>
            <span class="hud-chip">BASE: {car_specs['base_weight_kg']} kg</span>
            <span class="hud-chip">V-MAX: {car_specs['max_speed_kmh']} km/h</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Inline Payload & Total Weight Header
    current_payload_preview = st.session_state.get("slider_payload", 75)
    current_total_weight = car_specs["base_weight_kg"] + current_payload_preview

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px; margin-bottom: 2px;">
            <span style="font-size: 0.76rem; font-weight: 600; color: #a4bede;">PAYLOAD (KG)</span>
            <span style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #8ba2c4;">TOTAL: <strong style="color: #00f2fe; font-size: 0.82rem;">{current_total_weight} kg</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    payload_kg = st.slider(
        "Payload Weight Slider",
        min_value=0,
        max_value=int(car_specs["max_payload_kg"]),
        value=min(current_payload_preview, int(car_specs["max_payload_kg"])),
        step=5,
        key="slider_payload",
        label_visibility="collapsed",
        help="Passenger + Luggage payload added to base vehicle curb weight.",
    )
    total_vehicle_weight_kg = car_specs["base_weight_kg"] + payload_kg

    # 5. Driving Mode (1=Eco, 2=Normal, 3=Sport)
    driving_mode_str = st.radio(
        "Powertrain Mode",
        options=["Eco", "Normal", "Sport"],
        index=1,
        horizontal=True,
        help="Adjusts throttle map, regenerative braking, and auxiliary draw.",
    )
    driving_mode_map = {"Eco": 1, "Normal": 2, "Sport": 3}
    driving_mode_val = driving_mode_map[driving_mode_str]

    payload_utilization_pct = (payload_kg / float(car_specs["max_payload_kg"])) * 100.0
    mode_descriptions = {
        "Eco": "High Regen | Capped Throttle Profile",
        "Normal": "Standard Energy Profile | Adaptive Regen",
        "Sport": "Instant Peak Torque | Aggressive Draw",
    }

    # Center Column Bottom Telemetry Status Card (Balances height)
    st.markdown(
        f"""
        <div class="telemetry-footer-card">
            <div class="telemetry-footer-title">
                <span>CHASSIS & POWERTRAIN PROFILE</span>
                <span style="color: #00ff87;">LOAD: {payload_utilization_pct:.0f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                <span>CURB MASS: <strong style="color: #00f2fe;">{total_vehicle_weight_kg} kg</strong></span>
                <span style="color: #a0c2e6;">MAP: <strong style="color: #00ff87;">{driving_mode_str.upper()}</strong></span>
            </div>
            <div style="font-size: 0.68rem; color: #738ca8; margin-top: 2px;">
                {mode_descriptions[driving_mode_str]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# LEFT COLUMN (TELEMETRY & ENVIRONMENT — ALL 10 INPUTS)
# =============================================================================
with col_left:
    st.markdown(
        """
        <div class="hud-panel-header">
            <span>DYNAMICS & ENVIRONMENT</span>
            <span style="color: #7998c2;">SENSORS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: Speed & Acceleration
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        speed_kmh = st.slider(
            "Speed (km/h)",
            min_value=0,
            max_value=int(car_specs["max_speed_kmh"]),
            value=min(65, int(car_specs["max_speed_kmh"])),
            step=1,
        )
    with r1_c2:
        accel_ms2 = st.slider(
            "Accel/Brake (m/s²)",
            min_value=-3.0,
            max_value=3.0,
            value=0.2,
            step=0.1,
            help="Positive = acceleration, Negative = regenerative braking.",
        )

    # Row 2: Temperature & Slope
    r2_c1, r2_c2 = st.columns(2)
    with r2_c1:
        temp_c = st.slider(
            "Ambient Temp (°C)",
            min_value=-10,
            max_value=50,
            value=25,
            step=1,
        )
    with r2_c2:
        slope_pct = st.slider(
            "Road Incline (%)",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            help="Grade percentage: positive for uphill, negative for downhill.",
        )

    # Row 3: Tire Pressure & Road Type
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        tire_psi = st.slider(
            "Tire Pressure (PSI)",
            min_value=28,
            max_value=35,
            value=32,
            step=1,
        )
    with r3_c2:
        road_type_str = st.selectbox(
            "Road Classification",
            options=["Highway", "City"],
            index=0,
        )
        road_type_val = 1 if road_type_str == "Highway" else 2

    # Row 4: Weather & Traffic
    r4_c1, r4_c2 = st.columns(2)
    with r4_c1:
        weather_str = st.selectbox(
            "Atmosphere / Weather",
            options=["Sunny", "Cloudy", "Rain", "Snow"],
            index=0,
        )
        weather_map = {"Sunny": 1, "Cloudy": 2, "Rain": 3, "Snow": 4}
        weather_val = weather_map[weather_str]
    with r4_c2:
        traffic_str = st.selectbox(
            "Traffic Density",
            options=["Light", "Moderate", "Heavy"],
            index=0,
        )
        traffic_map = {"Light": 1, "Moderate": 2, "Heavy": 3}
        traffic_val = traffic_map[traffic_str]

    # Row 5: Humidity & Wind Speed (Direct ML Input Features)
    r5_c1, r5_c2 = st.columns(2)
    with r5_c1:
        humidity_val = st.slider(
            "Humidity (%)",
            min_value=20,
            max_value=100,
            value=50,
            step=1,
            help="Relative atmospheric humidity percentage.",
        )
    with r5_c2:
        wind_speed_val = st.slider(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
            help="Ambient aerodynamic wind velocity.",
        )

    # Physical Estimation of Aero Drag Power: P = 0.5 * rho * Cd * A * v_rel^2 * v
    v_mps = speed_kmh / 3.6
    relative_v = v_mps + (wind_speed_val * 0.5)
    est_aero_kw = round(0.5 * 1.225 * 0.32 * 2.2 * (relative_v ** 2) * v_mps / 1000.0, 1)

    # Left Column Bottom Telemetry Status Card (Balances height)
    st.markdown(
        f"""
        <div class="telemetry-footer-card">
            <div class="telemetry-footer-title">
                <span>AERODYNAMIC & ENVIRONMENTAL FORCES</span>
                <span style="color: #00f2fe;">ACTIVE SENSORS</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;">
                <span>AERO DRAG POWER: <strong style="color: #00f2fe;">{est_aero_kw} kW</strong></span>
                <span>INCLINE: <strong style="color: {'#ff8c00' if slope_pct > 0 else '#00ff87' if slope_pct < 0 else '#a0c2e6'};">{slope_pct:+d}%</strong></span>
            </div>
            <div style="font-size: 0.68rem; color: #738ca8; margin-top: 2px;">
                Atmosphere: {weather_str} | Traffic Density: {traffic_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# RIGHT COLUMN (BATTERY STATE & ML RESULTS / TWO-WAY ESTIMATOR)
# =============================================================================
with col_right:
    st.markdown(
        """
        <div class="hud-panel-header">
            <span>POWERTRAIN & PREDICTIONS</span>
            <span style="color: #00ff87;">ML ANALYTICS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    r_top1, r_top2 = st.columns(2)
    with r_top1:
        distance_km = st.slider(
            "Trip Distance (km)",
            min_value=1,
            max_value=300,
            value=45,
            step=1,
        )
    with r_top2:
        battery_soc = st.slider(
            "State of Charge (%)",
            min_value=10,
            max_value=100,
            value=85,
            step=1,
        )

    r_mid1, r_mid2 = st.columns(2)
    with r_mid1:
        battery_temp_c = st.slider(
            "Battery Temp (°C)",
            min_value=10,
            max_value=50,
            value=28,
            step=1,
        )
    with r_mid2:
        # Realistic Battery Voltage Curve based on SOC: 320V @ 10% to 400V @ 100%
        battery_voltage_v = round(320.0 + ((battery_soc - 10) / 90.0) * 80.0, 1)
        st.markdown(
            f"""
            <div style="margin-top: 14px; padding: 5px 8px; background: rgba(0, 242, 254, 0.07); border: 1px solid rgba(0, 242, 254, 0.25); border-radius: 6px; text-align: center;">
                <div style="font-size: 0.66rem; color: #8ba2c4; letter-spacing: 0.08em;">PACK VOLTAGE</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #00f2fe;">{battery_voltage_v} V</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # ML INFERENCE (Exact 16 Features in Strict Order)
    # -------------------------------------------------------------------------
    features_ordered = [
        "Speed_kmh",
        "Acceleration_ms2",
        "Battery_State_%",
        "Battery_Voltage_V",
        "Battery_Temperature_C",
        "Driving_Mode",
        "Road_Type",
        "Traffic_Condition",
        "Slope_%",
        "Weather_Condition",
        "Temperature_C",
        "Humidity_%",
        "Wind_Speed_ms",
        "Tire_Pressure_psi",
        "Vehicle_Weight_kg",
        "Distance_Travelled_km",
    ]

    input_data = [
        [
            float(speed_kmh),
            float(accel_ms2),
            float(battery_soc),
            float(battery_voltage_v),
            float(battery_temp_c),
            int(driving_mode_val),
            int(road_type_val),
            int(traffic_val),
            float(slope_pct),
            int(weather_val),
            float(temp_c),
            float(humidity_val),
            float(wind_speed_val),
            float(tire_psi),
            float(total_vehicle_weight_kg),
            float(distance_km),
        ]
    ]

    input_df = pd.DataFrame(input_data, columns=features_ordered)

    # Execute Prediction
    ml_predicted_kwh = float(model.predict(input_df)[0])
    ml_predicted_kwh = max(ml_predicted_kwh, 0.1)

    # Dynamic Math
    efficiency_kwh_per_km = ml_predicted_kwh / max(float(distance_km), 0.1)
    available_energy_kwh = (float(battery_soc) / 100.0) * float(car_specs["battery_capacity_kwh"])
    ml_max_remaining_range_km = available_energy_kwh / max(efficiency_kwh_per_km, 0.001)

    # -------------------------------------------------------------------------
    # BIDIRECTIONAL EDITING SYSTEM (Session State Synchronization)
    # -------------------------------------------------------------------------
    if "override_consumption_kwh" not in st.session_state:
        st.session_state.override_consumption_kwh = ml_predicted_kwh
    if "override_range_km" not in st.session_state:
        st.session_state.override_range_km = ml_max_remaining_range_km
    if "last_ml_hash" not in st.session_state:
        st.session_state.last_ml_hash = None

    # Track inputs changed so default tracks ML until manually overridden
    current_inputs_hash = hash((
        speed_kmh,
        accel_ms2,
        battery_soc,
        battery_temp_c,
        distance_km,
        total_vehicle_weight_kg,
        driving_mode_val,
        road_type_val,
        traffic_val,
        slope_pct,
        weather_val,
        temp_c,
        tire_psi,
        humidity_val,
        wind_speed_val,
    ))
    if st.session_state.last_ml_hash != current_inputs_hash:
        st.session_state.override_consumption_kwh = round(ml_predicted_kwh, 2)
        st.session_state.override_range_km = round(ml_max_remaining_range_km, 1)
        st.session_state.last_ml_hash = current_inputs_hash

    # Callbacks for Two-Way Editing
    def on_consumption_change():
        c_kwh = st.session_state.input_consumption
        if c_kwh > 0 and distance_km > 0:
            eff = c_kwh / distance_km
            st.session_state.override_range_km = round(available_energy_kwh / max(eff, 0.001), 1)
            st.session_state.override_consumption_kwh = round(c_kwh, 2)

    def on_range_change():
        r_km = st.session_state.input_range
        if r_km > 0:
            eff = available_energy_kwh / r_km
            st.session_state.override_consumption_kwh = round(eff * distance_km, 2)
            st.session_state.override_range_km = round(r_km, 1)

    def reset_to_ml():
        st.session_state.override_consumption_kwh = round(ml_predicted_kwh, 2)
        st.session_state.override_range_km = round(ml_max_remaining_range_km, 1)

    # Primary Display Metrics
    display_kwh = st.session_state.override_consumption_kwh
    display_range = st.session_state.override_range_km
    display_efficiency = (display_kwh / distance_km) * 100.0  # kWh / 100 km

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">PREDICTED TRIP CONSUMPTION</div>
            <div class="metric-value">{display_kwh:.2f} <span style="font-size: 1.05rem; font-weight: 500; color: #a2c6ed;">kWh</span></div>
            <div class="metric-sub">EFFICIENCY: <strong>{display_efficiency:.1f} kWh/100km</strong> ({display_kwh/distance_km:.3f} kWh/km)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card-accent">
            <div class="metric-title">ESTIMATED MAXIMUM RANGE</div>
            <div class="metric-value-accent">{display_range:.1f} <span style="font-size: 1.05rem; font-weight: 500; color: #a2edca;">km</span></div>
            <div class="metric-sub">AVAILABLE ENERGY: <strong>{available_energy_kwh:.2f} kWh</strong> @ {battery_soc}% SOC</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # TWO-WAY INTERACTIVE SIMULATION ADJUSTER
    # -------------------------------------------------------------------------
    sim_c1, sim_c2, sim_c3 = st.columns([1.1, 1.1, 0.9])
    with sim_c1:
        st.number_input(
            "Target kWh",
            value=float(st.session_state.override_consumption_kwh),
            step=0.5,
            key="input_consumption",
            on_change=on_consumption_change,
            help="Edit consumption to see resulting range.",
        )
    with sim_c2:
        st.number_input(
            "Target Range (km)",
            value=float(st.session_state.override_range_km),
            step=5.0,
            key="input_range",
            on_change=on_range_change,
            help="Edit target range to see required trip consumption.",
        )
    with sim_c3:
        st.write("")
        st.write("")
        st.button("🔄 Sync ML", on_click=reset_to_ml, help="Re-sync values to exact XGBoost ML prediction.")
