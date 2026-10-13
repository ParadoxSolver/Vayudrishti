"""
NAIP v2.0 — Streamlit Policy Command Center
Real-time AQI Dashboard + Policy Impact Simulator
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import pickle
import os
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NAIP — National Air Intelligence Platform",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")
APP_DIR = Path(__file__).resolve().parent

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:        #060d17;
    --surface:   #0d1f35;
    --card:      #0a1929;
    --border:    #1a3a5c;
    --accent:    #00d4ff;
    --accent2:   #ff6b35;
    --accent3:   #39ff14;
    --text:      #e2f0ff;
    --muted:     #5a7a9a;
    --good:      #39ff14;
    --moderate:  #ffd700;
    --poor:      #ff6b35;
    --hazardous: #ff2d55;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.good::before    { background: var(--good); }
.metric-card.moderate::before { background: var(--moderate); }
.metric-card.poor::before    { background: var(--poor); }
.metric-card.hazardous::before { background: var(--hazardous); }

.metric-value {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-top: 0.4rem;
}
.metric-city {
    font-size: 0.85rem;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.3rem;
}

/* AQI badge */
.aqi-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Policy panel */
.policy-card {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1929 100%);
    border: 1px solid var(--accent);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    box-shadow: 0 0 30px rgba(0,212,255,0.06);
}
.policy-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.rec-action {
    background: rgba(0,212,255,0.06);
    border-left: 3px solid var(--accent2);
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0.6rem 0;
}
.authority-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 4px;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Live indicator */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--good);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.4; transform:scale(1.3); }
}

/* Impact delta */
.delta-positive { color: var(--good); font-weight: 700; }
.delta-negative { color: var(--hazardous); font-weight: 700; }
.delta-neutral  { color: var(--muted); }

/* Streamlit overrides */
.stSelectbox > div > div { background: var(--card); border-color: var(--border); }
.stSlider > div         { color: var(--text); }
div[data-testid="metric-container"] { background: var(--card); border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

ZONE_CONFIG = {
    "North_Delhi": {
        "region_id": 0, "zone_label": "North India", "color": "#ff6b35",
        "spike_bands": [6,7,8,9],
        "source_weights": {"Agricultural":0.60,"Traffic":0.30,"Industrial":0.10},
        "wind_mean": 3.5, "inv_layer_mean": 1200, "base_aqi": 180, "aqi_std": 60,
        "cities": [
            {"name":"Delhi - Anand Vihar", "station_id":"site_233","state":"Delhi", "lat":28.6468, "lon":77.3160},
            {"name":"Delhi - ITO",          "station_id":"site_234","state":"Delhi", "lat":28.6286, "lon":77.2497},
            {"name":"Delhi - RK Puram",     "station_id":"site_235","state":"Delhi", "lat":28.5633, "lon":77.1869},
            {"name":"Noida - Sector 62",    "station_id":"site_369","state":"Uttar Pradesh", "lat":28.6289, "lon":77.3649},
            {"name":"Gurugram - Vikas Sadan","station_id":"site_152","state":"Haryana", "lat":28.4501, "lon":77.0263},
        ]
    },
    "West_Vadodara": {
        "region_id": 1, "zone_label": "West India", "color": "#00d4ff",
        "spike_bands": [2,3,4,5],
        "source_weights": {"Industrial":0.65,"Traffic":0.25,"Agricultural":0.10},
        "wind_mean": 5.0, "inv_layer_mean": 950, "base_aqi": 155, "aqi_std": 50,
        "cities": [
            {"name":"Vadodara - GIDC",    "station_id":"site_424","state":"Gujarat", "lat":22.2760, "lon":73.2010},
            {"name":"Surat - Udhna",      "station_id":"site_411","state":"Gujarat", "lat":21.1583, "lon":72.8417},
            {"name":"Mumbai - Mazagaon",  "station_id":"site_274","state":"Maharashtra", "lat":18.9690, "lon":72.8460},
            {"name":"Pune - Karve Road",  "station_id":"site_336","state":"Maharashtra", "lat":18.5074, "lon":73.8290},
        ]
    },
    "South_Bangalore": {
        "region_id": 2, "zone_label": "South India", "color": "#39ff14",
        "spike_bands": [1,2,8,9],
        "source_weights": {"Traffic":0.55,"Industrial":0.30,"Agricultural":0.15},
        "wind_mean": 4.2, "inv_layer_mean": 1100, "base_aqi": 120, "aqi_std": 40,
        "cities": [
            {"name":"Bangalore - BTM Layout","station_id":"site_070","state":"Karnataka", "lat":12.9166, "lon":77.6101},
            {"name":"Hyderabad - ICRISAT",  "station_id":"site_168","state":"Telangana", "lat":17.5170, "lon":78.2750},
            {"name":"Chennai - Alandur",    "station_id":"site_105","state":"Tamil Nadu", "lat":12.9970, "lon":80.2040},
        ]
    },
    "East_Kolkata": {
        "region_id": 3, "zone_label": "East India", "color": "#ff2d55",
        "spike_bands": [5,6,7,8],
        "source_weights": {"Industrial":0.50,"Traffic":0.35,"Agricultural":0.15},
        "wind_mean": 3.8, "inv_layer_mean": 1050, "base_aqi": 165, "aqi_std": 55,
        "cities": [
            {"name":"Kolkata - Rabindra Bharati","station_id":"site_209","state":"West Bengal", "lat":22.6270, "lon":88.3810},
            {"name":"Patna - IGSC Planetarium",  "station_id":"site_315","state":"Bihar", "lat":25.6050, "lon":85.1160},
        ]
    },
    "Central_Bhopal": {
        "region_id": 4, "zone_label": "Central India", "color": "#b44fff",
        "spike_bands": [3,4,6,7],
        "source_weights": {"Agricultural":0.45,"Industrial":0.35,"Traffic":0.20},
        "wind_mean": 4.0, "inv_layer_mean": 1000, "base_aqi": 150, "aqi_std": 50,
        "cities": [
            {"name":"Bhopal - T.T. Nagar","station_id":"site_076","state":"Madhya Pradesh", "lat":23.2330, "lon":77.3960},
            {"name":"Indore - Polo Ground","station_id":"site_173","state":"Madhya Pradesh", "lat":22.7240, "lon":75.8790},
            {"name":"Nagpur - Civil Lines","station_id":"site_280","state":"Maharashtra", "lat":21.1490, "lon":79.0730},
        ]
    },
}

WAVELENGTHS_NM = [412,443,490,510,560,620,665,705,740,865]
N_BANDS        = 10
SOURCES        = ["Agricultural","Industrial","Traffic"]

# Policy templates for the simulator
POLICY_TEMPLATES = {
    "🚗 Odd-Even Vehicle Restriction": {
        "description": "Restricts private vehicles by number plate on alternate days.",
        "wind_boost": +0.3, "inv_reduction": -50,
        "source_shift": {"Traffic": -0.25, "Agricultural": +0.05, "Industrial": +0.05},
        "aqi_base_reduction": -18, "effectiveness_hours": 12,
    },
    "🏭 Industrial Shutdown Order": {
        "description": "Shuts down heavy industries and brick kilns for 48 hours.",
        "wind_boost": 0.0, "inv_reduction": -30,
        "source_shift": {"Industrial": -0.40, "Agricultural": +0.10, "Traffic": +0.10},
        "aqi_base_reduction": -30, "effectiveness_hours": 6,
    },
    "🌾 Stubble Burning Ban": {
        "description": "Emergency ban on crop residue burning with penalty enforcement.",
        "wind_boost": 0.0, "inv_reduction": 0,
        "source_shift": {"Agricultural": -0.50, "Industrial": +0.10, "Traffic": +0.10},
        "aqi_base_reduction": -22, "effectiveness_hours": 18,
    },
    "🌳 Green Corridor + Water Sprinklers": {
        "description": "Deploy water sprinkler trucks and activate green corridors.",
        "wind_boost": +0.1, "inv_reduction": -20,
        "source_shift": {"Traffic": -0.10, "Agricultural": -0.05, "Industrial": -0.05},
        "aqi_base_reduction": -8, "effectiveness_hours": 2,
    },
    "🚌 Free Public Transport": {
        "description": "Free metro/bus pass to reduce private vehicle usage.",
        "wind_boost": +0.2, "inv_reduction": -15,
        "source_shift": {"Traffic": -0.20, "Agricultural": +0.05, "Industrial": +0.05},
        "aqi_base_reduction": -12, "effectiveness_hours": 24,
    },
    "⚗️ CEMS Mandatory Reporting": {
        "description": "24-hr mandatory continuous emission monitoring at all factories.",
        "wind_boost": 0.0, "inv_reduction": -10,
        "source_shift": {"Industrial": -0.20, "Agricultural": +0.05, "Traffic": +0.05},
        "aqi_base_reduction": -15, "effectiveness_hours": 8,
    },
    "🔥 Custom Policy": {
        "description": "Define your own policy parameters below.",
        "wind_boost": 0.0, "inv_reduction": 0,
        "source_shift": {"Traffic": 0.0, "Agricultural": 0.0, "Industrial": 0.0},
        "aqi_base_reduction": 0, "effectiveness_hours": 6,
    },
}

PLAYBOOK = {
    ("North_Delhi","Agricultural"):  ("🌾 Activate crop-residue burning ban. Alert Punjab & Haryana PPCB. Deploy drone surveillance over stubble fields.", "Central CPCB + State PPCB", "#ff6b35"),
    ("North_Delhi","Traffic"):       ("🚗 Enforce odd-even vehicular restriction in NCR. Expand CNG mandate; suspend high-polluting diesel fleets.", "Delhi Transport Authority", "#ffd700"),
    ("North_Delhi","Industrial"):    ("🏭 Issue 48-hr stack emission compliance notice to NCR industries. Halt brick kilns.", "DPCC + MoEFCC", "#ff6b35"),
    ("West_Vadodara","Industrial"):  ("⚗️ Enforce SO₂/VOC emission caps in Vadodara GIDC. CEMS reporting within 24 hrs. GPCB inspection.", "GPCB + MoEFCC", "#ff2d55"),
    ("West_Vadodara","Traffic"):     ("🚗 Restrict HGV on NH48 (06:00–22:00). Activate green zone buffers near residential areas.", "Vadodara Municipal Corp", "#ffd700"),
    ("West_Vadodara","Agricultural"):("🌾 Advisory to restrict field burning in South Gujarat. Engage ATMA extension officers.", "Gujarat State PPCB", "#00d4ff"),
    ("South_Bangalore","Traffic"):   ("🚗 Expand metro/BRT capacity. Odd-even enforcement on Outer Ring Road and MG Road.", "BBMP + Karnataka SPCB", "#ffd700"),
    ("South_Bangalore","Industrial"):("🏭 Inspection of Peenya & Whitefield industrial zones. Enforce IT park DG norms.", "KSPCB + MoEFCC", "#ff6b35"),
    ("East_Kolkata","Industrial"):   ("🏭 Halt coal-fired units at Durgapur/Asansol. STO to Haldia petrochemical cluster.", "WBPCB + MoEFCC", "#ff2d55"),
    ("East_Kolkata","Traffic"):      ("🚗 Strengthen vehicle fitness inspection in Kolkata metro. Expand e-bus corridor.", "Kolkata Traffic Police", "#ffd700"),
    ("Central_Bhopal","Agricultural"):("🌾 Activate stubble-burning ban in MP. Alert district collectors in soybean belt.", "MP PPCB + Agriculture Dept", "#ff2d55"),
    ("Central_Bhopal","Industrial"): ("🏭 Inspect cement/steel plants in Jabalpur and Satna. Issue show-cause to non-compliant units.", "MP PPCB + MoEFCC", "#ff6b35"),
    ("Central_Bhopal","Traffic"):    ("🚗 Fleet modernisation advisory. Strengthen PUC enforcement in Bhopal and Indore.", "MP Transport Authority", "#ffd700"),
}


# ══════════════════════════════════════════════════════════════════
#  DATA & MODEL HELPERS
# ══════════════════════════════════════════════════════════════════

class NAIPPreprocessor:
    def __init__(self):
        self.feat_scaler = StandardScaler()
        self.tgt_scaler  = StandardScaler()
        self.label_enc   = LabelEncoder()
        self.n_classes   = None
        self.class_names = None

    def inverse_transform_reg(self, y_scaled):
        return self.tgt_scaler.inverse_transform(y_scaled)


@st.cache_data(ttl=3600)
def fetch_cpcb_station(api_key: str, base_url: str, station_id: str,
                        city_name: str, fetch_hours: int = 72):
    """Fetch hourly AQI from CPCB API. Returns list of (timestamp, aqi) or None."""
    end_dt   = datetime.now(IST)
    start_dt = end_dt - timedelta(hours=fetch_hours)

    # Try multiple auth styles automatically
    auth_attempts = [
        # 1. Bearer token header
        {"headers": {"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"}},
        # 2. API key as query param
        {"params": {"apikey": api_key, "key": api_key}},
        # 3. Basic auth
        {"auth": (api_key, "")},
    ]

    payload = {
        "stationId": station_id,
        "fromDate":  start_dt.strftime("%Y-%m-%d %H:00:00"),
        "toDate":    end_dt.strftime("%Y-%m-%d %H:00:00"),
        "criteria":  "hourly",
    }

    for attempt in auth_attempts:
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/getStationData",
                json=payload, timeout=12, **attempt
            )
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, dict):
                    continue
                records = data.get("data", data.get("records", []))
                if isinstance(records, dict):
                    records = records.get("data", records.get("records", []))
                if not isinstance(records, list):
                    continue
                rows = []
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    ts     = pd.to_datetime(rec.get("timestamp", rec.get("from_date", None)), errors="coerce")
                    params = rec.get("parameters", rec.get("pollutants", {}))
                    if not isinstance(params, dict):
                        continue
                    aqi    = None
                    if "AQI" in params:
                        try:
                            aqi = float(params["AQI"])
                        except (TypeError, ValueError):
                            pass
                    if aqi is None and "PM2.5" in params:
                        try:
                            pm25 = float(params["PM2.5"])
                            aqi  = _pm25_to_aqi(pm25)
                        except (TypeError, ValueError):
                            pass
                    if aqi is not None and np.isfinite(aqi) and pd.notna(ts):
                        rows.append({"Timestamp": ts, "Hourly_AQI": float(aqi)})
                if rows:
                    return (pd.DataFrame(rows).drop_duplicates("Timestamp", keep="last")
                            .sort_values("Timestamp").reset_index(drop=True))
        except (requests.RequestException, ValueError, TypeError):
            continue
    return None


def _pm25_to_aqi(pm25):
    if not np.isfinite(pm25) or pm25 < 0:
        return None
    bps = [(0,30,0,50),(30,60,51,100),(60,90,101,200),
           (90,120,201,300),(120,250,301,400),(250,500,401,500)]
    for lc,hc,la,ha in bps:
        if lc<=pm25<=hc:
            return la+(pm25-lc)*(ha-la)/(hc-lc)
    return 500.0


def rim_simulate(zone_key: str, fetch_hours: int = 72, seed: int = 42) -> pd.DataFrame:
    """RIM-based synthetic data fallback."""
    cfg  = ZONE_CONFIG[zone_key]
    rng  = np.random.default_rng(seed)
    n    = fetch_hours
    hrs  = np.arange(n) % 24
    base = rng.normal(cfg["base_aqi"], cfg["aqi_std"], n)
    diu  = 30 * np.sin(np.pi * (hrs - 7) / 12) * -1
    aqi  = np.clip(base + diu, 30, 500).astype(np.float32)
    end  = datetime.now(IST)
    timestamps = [end - timedelta(hours=(n-1-i)) for i in range(n)]
    return pd.DataFrame({"Timestamp": timestamps, "Hourly_AQI": aqi})


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_station_aqi(api_key: str, base_url: str, fetch_hours: int) -> pd.DataFrame:
    """Fetch the latest AQI for every configured station, with per-station RIM fallback."""
    rows = []
    for zone_key, zone_cfg in ZONE_CONFIG.items():
        for city in zone_cfg["cities"]:
            station_data = None
            if api_key:
                station_data = fetch_cpcb_station(
                    api_key, base_url, city["station_id"], city["name"], fetch_hours
                )

            # A stable, station-specific seed avoids every fallback in a zone showing
            # the same synthetic AQI while retaining the zone's RIM characteristics.
            station_seed = sum(ord(char) for char in city["station_id"])
            source = "CPCB live"
            if station_data is None or station_data.empty:
                station_data = rim_simulate(zone_key, fetch_hours, seed=station_seed)
                source = "RIM fallback"

            rows.append({
                "zone_key": zone_key,
                "Zone": zone_cfg["zone_label"],
                "Station": city["name"],
                "State": city["state"],
                "station_id": city["station_id"],
                "lat": city["lat"],
                "lon": city["lon"],
                "AQI": float(station_data["Hourly_AQI"].iloc[-1]),
                "Source": source,
            })
    return pd.DataFrame(rows)


def derive_features(aqi_arr, zone_cfg, seed=42):
    rng  = np.random.default_rng(seed)
    n    = len(aqi_arr)
    base = zone_cfg.get("base_aqi", 160)
    wind = np.clip(rng.normal(zone_cfg["wind_mean"],1.5,n)
                   - 0.004*(aqi_arr-base), 0.1, 20).astype(np.float32)
    inv  = np.clip(rng.normal(zone_cfg["inv_layer_mean"],200,n)
                   + 0.3*(aqi_arr-base), 200, 3000).astype(np.float32)
    spec = np.clip(rng.uniform(0.02,0.15,(n,N_BANDS)), 0, 1).astype(np.float32)
    aqi_n = (aqi_arr - aqi_arr.min()) / (aqi_arr.max() - aqi_arr.min() + 1e-6)
    for b in zone_cfg["spike_bands"]:
        spec[:,b] += 0.45*aqi_n + rng.normal(0,0.02,n)
    spec = np.clip(spec,0,1)
    return wind, inv, spec


def trend_forecast(aqi_values, horizons=(24, 48, 72)):
    """Return a deterministic linear forecast from a recent AQI series."""
    recent = np.asarray(aqi_values, dtype=float)
    recent = recent[np.isfinite(recent)]
    if len(recent) == 0:
        return np.full(len(horizons), 20.0)
    current = recent[-1]
    if len(recent) < 2:
        return np.full(len(horizons), np.clip(current, 20, 500))
    trend = (recent[-1] - recent[0]) / (len(recent) - 1)
    return np.clip([current + trend * horizon for horizon in horizons], 20, 500)


@st.cache_resource(show_spinner=False)
def load_model_and_preprocessor():
    """Load saved model/preprocessor or return None if not yet trained."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(APP_DIR / "naip_model_v2.keras")
        with (APP_DIR / "naip_preprocessor_v2.pkl").open("rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except (ImportError, OSError, pickle.UnpicklingError, ValueError, TypeError):
        return None, None


def aqi_category(v):
    if v <= 50:   return ("Good",        "#39ff14")
    if v <= 100:  return ("Satisfactory","#a8ff3e")
    if v <= 200:  return ("Moderate",    "#ffd700")
    if v <= 300:  return ("Poor",        "#ff6b35")
    if v <= 400:  return ("Very Poor",   "#ff2d55")
    return ("Hazardous", "#c800ff")


def detect_pollution_hotspots(stations: pd.DataFrame, threshold: float,
                              radius_km: float = 35.0):
    """Cluster high-AQI stations with DBSCAN using great-circle distance."""
    high = stations.loc[stations["AQI"] > threshold].copy()
    high["hotspot_cluster"] = -1
    if len(high) < 2:
        return high, []

    coordinates = np.radians(high[["lat", "lon"]].to_numpy())
    labels = DBSCAN(
        eps=radius_km / 6371.0088, min_samples=2, metric="haversine"
    ).fit_predict(coordinates)
    high["hotspot_cluster"] = labels

    hotspots = []
    for cluster_id in sorted(label for label in np.unique(labels) if label != -1):
        members = high.loc[high["hotspot_cluster"] == cluster_id]
        zone_counts = members["Zone"].value_counts()
        hotspots.append({
            "Cluster": f"Hotspot {cluster_id + 1}",
            "lat": members["lat"].mean(),
            "lon": members["lon"].mean(),
            "Average AQI": members["AQI"].mean(),
            "Stations": len(members),
            "Zones": ", ".join(
                f"{zone} ({count})" for zone, count in zone_counts.items()
            ),
            "Station names": members["Station"].tolist(),
        })
    return high, hotspots


def apply_policy_to_aqi(base_aqi: float, policy: dict,
                          activation_delay_hours: int = 0) -> dict:
    """
    Simulates policy impact on AQI at T+24, T+48, T+72.
    Uses a physics-inspired decay model:
    - Policy effect ramps up after `effectiveness_hours`
    - Effect decays exponentially from peak
    - Wind boost amplifies dispersion
    - Inversion layer reduction improves mixing
    """
    eff_h   = max(1, policy["effectiveness_hours"])
    base_r  = policy["aqi_base_reduction"]
    wind_b  = policy["wind_boost"]
    inv_r   = policy["inv_reduction"]

    results = {}
    for horizon in [24, 48, 72]:
        hours_active = max(0, horizon - activation_delay_hours)

        # Ramp factor: 0 → 1 after activation, over `effectiveness_hours`.
        ramp = min(1.0, hours_active / eff_h)

        # Decay: effect diminishes after 36 hours without follow-up
        decay = np.exp(-max(0, hours_active - eff_h - 36) / 48.0)

        # Wind dispersion bonus
        wind_factor = 1.0 + wind_b * 0.15

        # Inversion improvement (better mixing = lower AQI)
        inv_factor  = 1.0 + (abs(inv_r) / 500.0) * 0.10

        total_reduction = base_r * ramp * decay * wind_factor * inv_factor

        # AQI can't go below 20 (natural background)
        projected = max(20.0, base_aqi + total_reduction)
        results[f"T+{horizon}h"] = round(projected, 1)

    return results


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
        <div style='font-size:2rem'>🇮🇳</div>
        <div style='font-family:Space Mono;font-size:0.65rem;letter-spacing:3px;
                    color:#00d4ff;text-transform:uppercase'>National Air Intelligence</div>
        <div style='font-size:1.2rem;font-weight:800;letter-spacing:-0.5px'>NAIP v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**🔑 CPCB API Configuration**")
    api_key  = st.text_input("API Key", type="password",
                               placeholder="Paste your CPCB key here",
                               help="Your CPCB CAAQMS API key")
    base_url = st.text_input("Base URL",
                              value="https://api.cpcbccr.com/caaqms",
                              help="CPCB API base URL")

    st.divider()

    st.markdown("**🗺️ Zone & City**")
    selected_zone = st.selectbox(
        "Zone", list(ZONE_CONFIG.keys()),
        format_func=lambda k: ZONE_CONFIG[k]["zone_label"]
    )
    cities_in_zone = [c["name"] for c in ZONE_CONFIG[selected_zone]["cities"]]
    selected_city  = st.selectbox("City / Station", cities_in_zone)
    selected_city_obj = next(
        c for c in ZONE_CONFIG[selected_zone]["cities"] if c["name"] == selected_city
    )

    st.divider()

    fetch_hours = st.slider("History window (hours)", 24, 168, 72, step=24)

    if st.button("🔄 Refresh Live Data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.7rem;color:#5a7a9a'>
        <span class='live-dot'></span>
        Last updated: {datetime.now(IST).strftime('%H:%M IST')}<br>
        Data: CPCB CAAQMS API<br>
        Fallback: RIM Simulation
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────
col_title, col_time = st.columns([3,1])
with col_title:
    st.markdown("""
    <h1 style='font-family:Syne;font-size:2rem;font-weight:800;
               letter-spacing:-1px;margin-bottom:0'>
        🇮🇳 National Air Intelligence Platform
    </h1>
    <div style='font-family:Space Mono;font-size:0.7rem;color:#5a7a9a;
                letter-spacing:2px;text-transform:uppercase'>
        Policy Command Center — Real-Time AQI & Impact Simulator
    </div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style='text-align:right;padding-top:0.5rem'>
        <div style='font-family:Space Mono;font-size:1.1rem;font-weight:700;
                    color:#00d4ff'>{datetime.now(IST).strftime('%H:%M')}</div>
        <div style='font-family:Space Mono;font-size:0.65rem;color:#5a7a9a'>
            {datetime.now(IST).strftime('%d %b %Y  IST')}
        </div>
        <div><span class='live-dot' style='display:inline-block;
            width:7px;height:7px;background:#39ff14;border-radius:50%;
            animation:pulse 1.5s infinite'></span>
            <span style='font-size:0.65rem;color:#39ff14'>LIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Dashboard",
    "🔮 72-hr Forecast",
    "⚡ Policy Simulator",
    "🗺️ All Zones Overview",
])


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:

    # Fetch data
    live_data = None
    data_source_label = "RIM Simulation"

    if api_key:
        with st.spinner(f"Fetching live data for {selected_city}..."):
            live_data = fetch_cpcb_station(
                api_key, base_url,
                selected_city_obj["station_id"], selected_city,
                fetch_hours
            )
        if live_data is not None and not live_data.empty:
            data_source_label = "🟢 CPCB Live API"
        else:
            st.warning(f"⚠️ CPCB API unreachable for {selected_city} — showing RIM simulation.")
    else:
        st.info("ℹ️ Enter your CPCB API key in the sidebar to fetch live data. Showing simulation.")

    if live_data is None or live_data.empty:
        live_data = rim_simulate(selected_zone, fetch_hours)
        data_source_label = "🔄 RIM Simulation"

    current_aqi  = float(live_data["Hourly_AQI"].iloc[-1])
    prev_aqi     = float(live_data["Hourly_AQI"].iloc[-2]) if len(live_data) > 1 else current_aqi
    avg_aqi      = float(live_data["Hourly_AQI"].mean())
    max_aqi      = float(live_data["Hourly_AQI"].max())
    category, cat_color = aqi_category(current_aqi)
    delta_val    = current_aqi - prev_aqi

    # ── KPI Row ───────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)

    def card_class(v):
        if v<=100: return "good"
        if v<=200: return "moderate"
        if v<=300: return "poor"
        return "hazardous"

    with k1:
        st.markdown(f"""
        <div class="metric-card {card_class(current_aqi)}">
            <div class="metric-city">{selected_city}</div>
            <div class="metric-value" style="color:{cat_color}">{current_aqi:.0f}</div>
            <div class="metric-label">Current AQI</div>
            <span class="aqi-badge" style="background:{cat_color}22;color:{cat_color};
                  border:1px solid {cat_color}44">{category}</span>
        </div>""", unsafe_allow_html=True)

    with k2:
        delta_color = "#ff2d55" if delta_val > 0 else "#39ff14"
        delta_arrow = "▲" if delta_val > 0 else "▼"
        st.markdown(f"""
        <div class="metric-card moderate">
            <div class="metric-city">vs previous hour</div>
            <div class="metric-value" style="color:{delta_color}">
                {delta_arrow}{abs(delta_val):.0f}
            </div>
            <div class="metric-label">AQI Change</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        cat2, col2 = aqi_category(avg_aqi)
        st.markdown(f"""
        <div class="metric-card {card_class(avg_aqi)}">
            <div class="metric-city">Last {fetch_hours} hours</div>
            <div class="metric-value" style="color:{col2}">{avg_aqi:.0f}</div>
            <div class="metric-label">Mean AQI</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        cat3, col3 = aqi_category(max_aqi)
        st.markdown(f"""
        <div class="metric-card {card_class(max_aqi)}">
            <div class="metric-city">Peak episode</div>
            <div class="metric-value" style="color:{col3}">{max_aqi:.0f}</div>
            <div class="metric-label">Max AQI</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.72rem;color:#5a7a9a;margin-top:0.5rem'>"
                f"Data source: {data_source_label} &nbsp;|&nbsp; "
                f"Zone: {ZONE_CONFIG[selected_zone]['zone_label']} &nbsp;|&nbsp; "
                f"State: {selected_city_obj['state']}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AQI Time-Series Chart ─────────────────────────────────────
    st.markdown('<div class="section-header">📡 AQI Time Series</div>', unsafe_allow_html=True)

    zone_color = ZONE_CONFIG[selected_zone]["color"]
    fig_ts = go.Figure()

    # Threshold bands
    fig_ts.add_hrect(y0=0,   y1=50,  fillcolor="#39ff14", opacity=0.04, line_width=0)
    fig_ts.add_hrect(y0=50,  y1=100, fillcolor="#a8ff3e", opacity=0.04, line_width=0)
    fig_ts.add_hrect(y0=100, y1=200, fillcolor="#ffd700", opacity=0.04, line_width=0)
    fig_ts.add_hrect(y0=200, y1=300, fillcolor="#ff6b35", opacity=0.05, line_width=0)
    fig_ts.add_hrect(y0=300, y1=500, fillcolor="#ff2d55", opacity=0.06, line_width=0)

    # Threshold lines
    for val, label, col in [(300,"Hazardous","#ff2d55"),(200,"Very Poor","#ff6b35"),
                             (100,"Moderate","#ffd700"),(50,"Good","#39ff14")]:
        fig_ts.add_hline(y=val, line_dash="dot", line_color=col,
                          line_width=1, opacity=0.5,
                          annotation_text=label, annotation_position="right",
                          annotation_font_color=col, annotation_font_size=10)

    # Main AQI line with gradient color
    fig_ts.add_trace(go.Scatter(
        x=live_data["Timestamp"], y=live_data["Hourly_AQI"],
        mode="lines", name="AQI",
        line=dict(color=zone_color, width=2.5),
        fill="tozeroy", fillcolor=f"rgba({int(zone_color[1:3],16)},"
                                  f"{int(zone_color[3:5],16)},"
                                  f"{int(zone_color[5:7],16)},0.08)",
        hovertemplate="<b>%{x|%d %b %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))

    # Current point
    fig_ts.add_trace(go.Scatter(
        x=[live_data["Timestamp"].iloc[-1]], y=[current_aqi],
        mode="markers", name="Current",
        marker=dict(size=12, color=cat_color,
                    line=dict(width=2, color="#060d17")),
        hovertemplate=f"<b>Now</b><br>AQI: {current_aqi:.0f}<extra></extra>",
    ))

    fig_ts.update_layout(
        plot_bgcolor="#0a1929", paper_bgcolor="#0a1929",
        font=dict(color="#e2f0ff", family="Syne"),
        xaxis=dict(gridcolor="#1a3a5c", showgrid=True, tickfont_size=11),
        yaxis=dict(gridcolor="#1a3a5c", showgrid=True, title="AQI", range=[0, max(max_aqi*1.2, 350)]),
        margin=dict(l=10, r=80, t=20, b=10),
        height=320, showlegend=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig_ts, width="stretch")

    # ── Source Attribution + Recommendation ──────────────────────
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown('<div class="section-header">🔬 Source Attribution</div>', unsafe_allow_html=True)

        model, preprocessor = load_model_and_preprocessor()
        if model is not None and preprocessor is not None:
            zcfg = ZONE_CONFIG[selected_zone]
            aqi_arr = np.array([current_aqi], dtype=np.float32)
            wind, inv, spec = derive_features(aqi_arr, zcfg)
            region_id = zcfg["region_id"]
            feat = np.array([[region_id, current_aqi, float(wind[0]),
                              float(inv[0])] + list(spec[0])], dtype=np.float32)
            X_in = preprocessor.feat_scaler.transform(feat).astype(np.float32)
            _, p_cls = model.predict(X_in, verbose=0)
            
            probs = {name: float(p_cls[0][i]) for i, name in enumerate(preprocessor.class_names)}
        else:
            # Simple RIM-based probability for current AQI
            zcfg    = ZONE_CONFIG[selected_zone]
            weights = zcfg["source_weights"]
            intensity = min((current_aqi - 100) / 200, 1.0)
            p = np.array([weights[s] for s in SOURCES])
            p[np.argmax(p)] += intensity * (1 - p.max()) * 0.5
            p /= p.sum()
            probs = {s: float(p[i]) for i, s in enumerate(SOURCES)}
        dominant = max(probs, key=probs.get)

        src_colors = {"Agricultural": "#ff6b35", "Industrial": "#00d4ff", "Traffic": "#b44fff"}
        fig_src = go.Figure(go.Bar(
            x=list(probs.values()), y=list(probs.keys()),
            orientation="h",
            marker_color=[src_colors[s] for s in probs.keys()],
            text=[f"{v:.1%}" for v in probs.values()],
            textposition="outside", textfont=dict(color="#e2f0ff", size=12),
        ))
        fig_src.update_layout(
            plot_bgcolor="#0a1929", paper_bgcolor="#0a1929",
            font=dict(color="#e2f0ff", family="Syne"),
            xaxis=dict(range=[0,1.2], showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=13)),
            margin=dict(l=10, r=60, t=10, b=10), height=180,
            showlegend=False,
        )
        st.plotly_chart(fig_src, width="stretch")

    with c_right:
        st.markdown('<div class="section-header">⚡ Surgical Recommendation</div>', unsafe_allow_html=True)
        key     = (selected_zone, dominant)
        entry   = PLAYBOOK.get(key)
        if entry:
            action, authority, rec_color = entry
        else:
            action    = "⚠️ Mixed episode. Activate GRAP Stage II. Increase monitoring to 30-min intervals."
            authority = "Central CPCB"
            rec_color = "#5a7a9a"

        confidence = probs[dominant]
        st.markdown(f"""
        <div class="policy-card" style="border-color:{rec_color}44">
            <div style="font-size:0.8rem;color:{rec_color};font-weight:700;margin-bottom:0.5rem">
                {dominant} — {confidence:.1%} confidence
            </div>
            <div class="rec-action">{action}</div>
            <div class="authority-tag">Issuing Authority: {authority}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — 72-HR FORECAST
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔮 72-Hour AQI Forecast</div>', unsafe_allow_html=True)

    # Check for trained model
    model, preprocessor = load_model_and_preprocessor()

    if live_data is None or live_data.empty:
        live_data = rim_simulate(selected_zone, fetch_hours)

    current_aqi_fc = float(live_data["Hourly_AQI"].iloc[-1])

    if model is not None and preprocessor is not None:
        # Use trained neural network
        zcfg = ZONE_CONFIG[selected_zone]
        aqi_arr = np.array([current_aqi_fc], dtype=np.float32)
        wind, inv, spec = derive_features(aqi_arr, zcfg)
        region_id = zcfg["region_id"]
        feat = np.array([[region_id, current_aqi_fc, float(wind[0]),
                          float(inv[0])] + list(spec[0])], dtype=np.float32)
        X_in = preprocessor.feat_scaler.transform(feat).astype(np.float32)
        p_reg, p_cls = model.predict(X_in, verbose=0)
        forecast_vals = preprocessor.inverse_transform_reg(p_reg)[0]
        forecast_vals = np.clip(forecast_vals, 20, 500)
        forecast_source = "Neural Network"
    else:
        # Fallback: simple trend extrapolation
        forecast_vals = trend_forecast(live_data["Hourly_AQI"].tail(24).values)
        forecast_source = "Trend Extrapolation (train model for NN forecast)"

    st.info(f"📊 Forecast method: **{forecast_source}**")

    horizons    = ["Now", "+24h", "+48h", "+72h"]
    x_pos       = [0, 24, 48, 72]
    aqi_series  = np.array([current_aqi_fc] + list(forecast_vals))
    seg_colors  = [aqi_category(v)[1] for v in aqi_series]

    fig_fc = go.Figure()

    # Confidence band
    upper = aqi_series * 1.12
    lower = aqi_series * 0.88
    fig_fc.add_trace(go.Scatter(
        x=x_pos+x_pos[::-1], y=list(upper)+list(lower)[::-1],
        fill="toself", fillcolor=f"rgba({int(ZONE_CONFIG[selected_zone]['color'][1:3],16)},"
                                  f"{int(ZONE_CONFIG[selected_zone]['color'][3:5],16)},"
                                  f"{int(ZONE_CONFIG[selected_zone]['color'][5:7],16)},0.10)",
        line=dict(width=0), name="±12% confidence",
        hoverinfo="skip",
    ))

    # Threshold bands
    for val,col in [(300,"#ff2d55"),(200,"#ff6b35"),(100,"#ffd700"),(50,"#39ff14")]:
        fig_fc.add_hline(y=val, line_dash="dot", line_color=col,
                          line_width=1, opacity=0.5)

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=x_pos, y=aqi_series, mode="lines+markers+text",
        line=dict(color=ZONE_CONFIG[selected_zone]["color"], width=3),
        marker=dict(size=[10,14,14,14], color=seg_colors,
                    line=dict(width=2, color="#060d17")),
        text=[f"<b>{int(v)}</b>" for v in aqi_series],
        textposition="top center",
        textfont=dict(size=13, color=seg_colors),
        name="AQI Forecast",
    ))

    fig_fc.update_layout(
        plot_bgcolor="#0a1929", paper_bgcolor="#0a1929",
        font=dict(color="#e2f0ff", family="Syne"),
        xaxis=dict(tickvals=x_pos, ticktext=horizons,
                   gridcolor="#1a3a5c", showgrid=True, tickfont_size=13),
        yaxis=dict(gridcolor="#1a3a5c", showgrid=True, title="AQI",
                   range=[0, max(aqi_series)*1.35]),
        margin=dict(l=10, r=30, t=30, b=10), height=380,
        legend=dict(bgcolor="#0a1929", bordercolor="#1a3a5c"),
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Forecast summary cards
    fc1, fc2, fc3 = st.columns(3)
    for col, h, val in zip([fc1,fc2,fc3], ["+24h","+48h","+72h"], forecast_vals):
        cat, c = aqi_category(val)
        delta  = val - current_aqi_fc
        with col:
            st.markdown(f"""
            <div class="metric-card {card_class(val)}">
                <div class="metric-city">{h} forecast</div>
                <div class="metric-value" style="color:{c}">{val:.0f}</div>
                <div class="metric-label">{cat}</div>
                <div style='font-size:0.8rem;color:{"#ff2d55" if delta>0 else "#39ff14"};margin-top:4px'>
                    {"▲" if delta>0 else "▼"} {abs(delta):.0f} from now
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">⚡ Policy Impact Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        "Select a policy intervention and see its predicted effect on AQI at +24h, +48h, +72h "
        "based on source attribution, dispersion physics, and historical effectiveness.",
        unsafe_allow_html=False
    )

    if live_data is None or live_data.empty:
        live_data = rim_simulate(selected_zone, fetch_hours)
    current_aqi_ps = float(live_data["Hourly_AQI"].iloc[-1])

    p_col, r_col = st.columns([1, 1])

    with p_col:
        st.markdown("**Select Policy**")
        selected_policy_name = st.selectbox(
            "Policy", list(POLICY_TEMPLATES.keys()),
            label_visibility="collapsed"
        )
        policy = deepcopy(POLICY_TEMPLATES[selected_policy_name])
        st.markdown(f"""
        <div class="policy-card">
            <div class="policy-title">Policy Description</div>
            <div style='font-size:0.9rem;color:#e2f0ff'>{policy['description']}</div>
            <div style='margin-top:0.8rem;font-size:0.75rem;color:#5a7a9a'>
                <b>Effectiveness delay:</b> {policy['effectiveness_hours']} hours after activation<br>
                <b>Base AQI reduction:</b> {policy['aqi_base_reduction']} AQI units at peak
            </div>
        </div>""", unsafe_allow_html=True)

        if selected_policy_name == "🔥 Custom Policy":
            st.markdown("**Custom Parameters**")
            policy["aqi_base_reduction"] = st.slider("Base AQI reduction", -80, 0, -15)
            policy["effectiveness_hours"]= st.slider("Hours to take effect", 1, 48, 12)
            policy["wind_boost"]         = st.slider("Wind speed boost (m/s)", 0.0, 2.0, 0.0, 0.1)
            policy["inv_reduction"]      = st.slider("Inversion layer reduction (m)", -200, 0, -30)
            st.markdown("**Source contribution shift:**")
            policy["source_shift"]["Traffic"]     = st.slider("Traffic shift %",     -0.5, 0.3, 0.0, 0.05)
            policy["source_shift"]["Agricultural"]= st.slider("Agricultural shift %",-0.5, 0.3, 0.0, 0.05)
            policy["source_shift"]["Industrial"]  = st.slider("Industrial shift %",  -0.5, 0.3, 0.0, 0.05)

        st.markdown("**Activation Timing**")
        activation_delay = st.radio(
            "Activate policy",
            ["Immediately (now)", "In 6 hours", "In 12 hours", "In 24 hours"],
            horizontal=True
        )
        delay_map = {"Immediately (now)": 0, "In 6 hours": 6,
                     "In 12 hours": 12, "In 24 hours": 24}
        extra_delay = delay_map[activation_delay]

    with r_col:
        st.markdown("**Predicted Impact**")

        # Compute impact
        baseline_fc = {}
        for h, value in zip([24, 48, 72], trend_forecast(live_data["Hourly_AQI"].tail(24).values)):
            baseline_fc[f"T+{h}h"] = float(value)

        # Apply each policy reduction to the same no-policy baseline, so the
        # displayed difference is attributable to the intervention alone.
        raw_policy_fc = apply_policy_to_aqi(
            current_aqi_ps, policy, activation_delay_hours=extra_delay
        )
        policy_fc = {
            key: max(20.0, baseline_fc[key] + (raw_policy_fc[key] - current_aqi_ps))
            for key in baseline_fc
        }

        # Impact chart
        horizons_p = ["Now", "+24h", "+48h", "+72h"]
        x_p        = [0, 24, 48, 72]
        baseline_s = [current_aqi_ps] + [baseline_fc[f"T+{h}h"] for h in [24,48,72]]
        policy_s   = [current_aqi_ps] + [policy_fc[f"T+{h}h"]   for h in [24,48,72]]

        fig_pol = go.Figure()

        # Baseline (no policy)
        fig_pol.add_trace(go.Scatter(
            x=x_p, y=baseline_s, mode="lines+markers",
            name="Without Policy (Trend)",
            line=dict(color="#5a7a9a", width=2, dash="dot"),
            marker=dict(size=8, color="#5a7a9a"),
        ))

        # With policy
        fig_pol.add_trace(go.Scatter(
            x=x_p, y=policy_s, mode="lines+markers+text",
            name="With Policy",
            line=dict(color="#00d4ff", width=3),
            marker=dict(size=10, color=[aqi_category(v)[1] for v in policy_s],
                        line=dict(width=2,color="#060d17")),
            text=[f"<b>{int(v)}</b>" for v in policy_s],
            textposition="top center",
            textfont=dict(size=12, color="#00d4ff"),
        ))

        # Shaded improvement area
        fig_pol.add_trace(go.Scatter(
            x=x_p+x_p[::-1], y=policy_s+baseline_s[::-1],
            fill="toself", fillcolor="rgba(57,255,20,0.07)",
            line=dict(width=0), name="AQI improvement zone",
            hoverinfo="skip",
        ))

        fig_pol.add_hline(y=200, line_dash="dot", line_color="#ff6b35",
                           line_width=1, opacity=0.5, annotation_text="Very Poor")
        fig_pol.add_hline(y=100, line_dash="dot", line_color="#ffd700",
                           line_width=1, opacity=0.5, annotation_text="Moderate")

        fig_pol.update_layout(
            plot_bgcolor="#0a1929", paper_bgcolor="#0a1929",
            font=dict(color="#e2f0ff", family="Syne"),
            xaxis=dict(tickvals=x_p, ticktext=horizons_p,
                       gridcolor="#1a3a5c", showgrid=True),
            yaxis=dict(gridcolor="#1a3a5c", showgrid=True, title="AQI"),
            margin=dict(l=10, r=30, t=30, b=10), height=320,
            legend=dict(bgcolor="#0a1929", bordercolor="#1a3a5c",
                        orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_pol, width="stretch")

        # Delta summary
        d1, d2, d3 = st.columns(3)
        for dcol, h in zip([d1,d2,d3], [24,48,72]):
            base_v   = baseline_fc[f"T+{h}h"]
            policy_v = policy_fc[f"T+{h}h"]
            delta    = policy_v - base_v
            pct      = (abs(delta) / base_v * 100) if base_v > 0 else 0
            dcol.metric(
                label=f"+{h}h AQI",
                value=f"{policy_v:.0f}",
                delta=f"{delta:+.0f} ({pct:.1f}%)",
                delta_color="inverse",
            )

        # Effectiveness summary
        total_reduction = current_aqi_ps - policy_fc["T+24h"]
        effectiveness_pct = abs(total_reduction) / current_aqi_ps * 100 if current_aqi_ps > 0 else 0

        st.markdown(f"""
        <div class="policy-card" style="border-color:#39ff1444;margin-top:0.5rem">
            <div class="policy-title">Policy Effectiveness Summary</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem'>
                <div>
                    <div style='font-size:1.6rem;font-weight:800;color:#39ff14'>
                        {effectiveness_pct:.1f}%
                    </div>
                    <div style='font-size:0.7rem;color:#5a7a9a;text-transform:uppercase;
                                letter-spacing:1px'>AQI Reduction at 24h</div>
                </div>
                <div>
                    <div style='font-size:1.6rem;font-weight:800;color:#00d4ff'>
                        {policy["effectiveness_hours"] + extra_delay}h
                    </div>
                    <div style='font-size:0.7rem;color:#5a7a9a;text-transform:uppercase;
                                letter-spacing:1px'>Time to Full Effect</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 4 — ALL ZONES OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">🗺️ All Zones — National Overview</div>',
                unsafe_allow_html=True)

    zone_summary = []
    for zk, zcfg in ZONE_CONFIG.items():
        # Use cached data if available, else simulate
        zdata = rim_simulate(zk, 24, seed=sum(map(ord, zk)))
        latest_aqi = float(zdata["Hourly_AQI"].iloc[-1])
        cat, col   = aqi_category(latest_aqi)
        zone_summary.append({
            "zone_key":  zk,
            "Zone":      zcfg["zone_label"],
            "Cities":    len(zcfg["cities"]),
            "AQI":       latest_aqi,
            "Category":  cat,
            "Color":     col,
            "Dominant":  max(zcfg["source_weights"], key=zcfg["source_weights"].get),
        })

    # National map (approximate lat/lon per zone)
    zone_coords = {
        "North_Delhi":    (28.65, 77.23),
        "West_Vadodara":  (22.31, 73.18),
        "South_Bangalore":(12.97, 77.59),
        "East_Kolkata":   (22.57, 88.36),
        "Central_Bhopal": (23.26, 77.41),
    }

    # Zone cards
    cols = st.columns(len(ZONE_CONFIG))
    for col, zs in zip(cols, zone_summary):
        with col:
            st.markdown(f"""
            <div class="metric-card {card_class(zs['AQI'])}">
                <div class="metric-city" style="font-size:0.75rem">{zs['Zone']}</div>
                <div class="metric-value" style="color:{zs['Color']};font-size:2rem">
                    {zs['AQI']:.0f}
                </div>
                <div class="metric-label">AQI</div>
                <span class="aqi-badge" style="background:{zs['Color']}22;
                      color:{zs['Color']};border:1px solid {zs['Color']}44">
                    {zs['Category']}
                </span>
                <div style='font-size:0.7rem;color:#5a7a9a;margin-top:0.5rem'>
                    Dominant: {zs['Dominant']}<br>
                    {zs['Cities']} stations
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # India map scatter
    st.markdown('<div class="section-header">🗺️ Spatial AQI Map</div>', unsafe_allow_html=True)

    lats   = [zone_coords[z["zone_key"]][0] for z in zone_summary]
    lons   = [zone_coords[z["zone_key"]][1] for z in zone_summary]
    aqis   = [z["AQI"] for z in zone_summary]
    labels = [f"{z['Zone']}<br>AQI: {z['AQI']:.0f}<br>{z['Category']}" for z in zone_summary]
    colors = [z["Color"] for z in zone_summary]

    fig_map = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, text=labels,
        mode="markers+text",
        marker=dict(
            size=[max(15, a/8) for a in aqis],
            color=aqis,
            colorscale=[[0,"#39ff14"],[0.2,"#ffd700"],[0.4,"#ff6b35"],
                         [0.6,"#ff2d55"],[1,"#c800ff"]],
            cmin=0, cmax=400,
            showscale=True,
            colorbar=dict(title=dict(text="AQI", font=dict(color="#e2f0ff")),
                          tickfont=dict(color="#e2f0ff")),
            line=dict(width=2, color="#060d17"),
        ),
        hovertext=labels, hoverinfo="text",
        textfont=dict(color="#e2f0ff", size=10),
        textposition="top center",
    ))

    fig_map.update_geos(
        scope="asia",
        center=dict(lat=22, lon=80),
        projection_scale=4,
        showland=True, landcolor="#0d1f35",
        showocean=True, oceancolor="#060d17",
        showcountries=True, countrycolor="#1a3a5c",
        showsubunits=True, subunitcolor="#1a3a5c",
        bgcolor="#060d17",
    )
    fig_map.update_layout(
        geo=dict(bgcolor="#060d17"),
        paper_bgcolor="#060d17",
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        font=dict(color="#e2f0ff"),
    )
    st.plotly_chart(fig_map, width="stretch")

    # ── Station-level hotspot detection ───────────────────────────
    st.markdown('<div class="section-header">🔥 Station-Level Pollution Hotspots</div>',
                unsafe_allow_html=True)
    control_col, radius_col = st.columns(2)
    with control_col:
        hotspot_threshold = st.number_input(
            "Hotspot AQI threshold", min_value=0, max_value=500, value=200,
            step=10, help="Only stations whose AQI exceeds this value are clustered."
        )
    with radius_col:
        hotspot_radius = st.slider(
            "Cluster radius (km)", min_value=5, max_value=100, value=35, step=5,
            help="Stations within this great-circle distance can form one DBSCAN cluster."
        )

    with st.spinner("Loading current AQI for all configured stations..."):
        station_aqi = fetch_all_station_aqi(api_key, base_url, fetch_hours)
    station_aqi[["Category", "Category color"]] = station_aqi["AQI"].apply(
        lambda value: pd.Series(aqi_category(value))
    )
    high_aqi_stations, hotspots = detect_pollution_hotspots(
        station_aqi, hotspot_threshold, hotspot_radius
    )

    station_hover = [
        f"<b>{row.Station}</b><br>{row.Zone} · {row.State}"
        f"<br>AQI: {row.AQI:.0f} ({row.Category})"
        f"<br>Source: {row.Source}"
        for row in station_aqi.itertuples()
    ]
    fig_hotspots = go.Figure()
    fig_hotspots.add_trace(go.Scattergeo(
        lat=station_aqi["lat"], lon=station_aqi["lon"], mode="markers",
        name="Monitoring stations",
        marker=dict(size=8, color=station_aqi["Category color"],
                    line=dict(width=1, color="#060d17"), opacity=0.9),
        text=station_hover, hoverinfo="text",
    ))

    if hotspots:
        hotspot_hover = [
            f"<b>{hotspot['Cluster']}</b><br>Average AQI: {hotspot['Average AQI']:.0f}"
            f"<br>Stations ({hotspot['Stations']}): "
            f"{', '.join(hotspot['Station names'])}"
            f"<br>Zone(s): {hotspot['Zones']}"
            for hotspot in hotspots
        ]
        fig_hotspots.add_trace(go.Scattergeo(
            lat=[hotspot["lat"] for hotspot in hotspots],
            lon=[hotspot["lon"] for hotspot in hotspots], mode="markers",
            name="Detected hotspots",
            marker=dict(symbol="star", size=20, color="#ff2d55",
                        line=dict(width=2, color="#ffd700")),
            text=hotspot_hover, hoverinfo="text",
        ))

    fig_hotspots.update_geos(
        scope="asia", center=dict(lat=22, lon=80), projection_scale=4,
        showland=True, landcolor="#0d1f35", showocean=True, oceancolor="#060d17",
        showcountries=True, countrycolor="#1a3a5c",
        showsubunits=True, subunitcolor="#1a3a5c", bgcolor="#060d17",
    )
    fig_hotspots.update_layout(
        geo=dict(bgcolor="#060d17"), paper_bgcolor="#060d17",
        margin=dict(l=0, r=0, t=10, b=0), height=480,
        font=dict(color="#e2f0ff", family="Syne"),
        legend=dict(bgcolor="#0a1929", bordercolor="#1a3a5c", borderwidth=1),
    )
    st.plotly_chart(fig_hotspots, width="stretch")

    st.caption(
        f"{len(station_aqi)} stations loaded; {len(high_aqi_stations)} exceed AQI "
        f"{hotspot_threshold:.0f}. Live CPCB readings fall back to station-specific "
        "RIM simulation when unavailable."
    )
    if hotspots:
        hotspot_table = pd.DataFrame([{
            "Cluster": hotspot["Cluster"],
            "Centroid": f"{hotspot['lat']:.4f}, {hotspot['lon']:.4f}",
            "Average AQI": round(hotspot["Average AQI"], 1),
            "Stations": hotspot["Stations"],
            "Zone(s) involved": hotspot["Zones"],
        } for hotspot in hotspots])
        st.dataframe(hotspot_table, width="stretch", hide_index=True)
    else:
        st.info("No multi-station hotspots detected at the selected AQI threshold and radius.")

    # Zone comparison bar chart
    st.markdown('<div class="section-header">📊 Zone Comparison</div>', unsafe_allow_html=True)
    fig_bar = go.Figure(go.Bar(
        x=[z["Zone"] for z in zone_summary],
        y=[z["AQI"] for z in zone_summary],
        marker_color=[z["Color"] for z in zone_summary],
        marker_line_color="#060d17", marker_line_width=2,
        text=[f"{z['AQI']:.0f}" for z in zone_summary],
        textposition="outside", textfont=dict(color="#e2f0ff", size=13),
    ))
    for val, col, label in [(300,"#ff2d55","Hazardous"),(200,"#ff6b35","Very Poor"),
                              (100,"#ffd700","Moderate"),(50,"#39ff14","Good")]:
        fig_bar.add_hline(y=val, line_dash="dot", line_color=col,
                           line_width=1, opacity=0.5, annotation_text=label,
                           annotation_font_color=col)
    fig_bar.update_layout(
        plot_bgcolor="#0a1929", paper_bgcolor="#0a1929",
        font=dict(color="#e2f0ff", family="Syne"),
        xaxis=dict(showgrid=False, tickfont_size=12),
        yaxis=dict(gridcolor="#1a3a5c", title="AQI", range=[0, max(aqis)*1.3]),
        margin=dict(l=10, r=30, t=20, b=10), height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, width="stretch")
