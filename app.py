import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from physarum_model import run_simulation, ROAD_OMEGA
from physarum_network_model import run_network_simulation
from cost_model import add_cost_benefit, verify_top_candidates, DEFAULT_COST_PER_KM
from routing import find_alternate_route
from baseline_model import compute_static_baseline, compare_rankings
from road_aggregation import aggregate_by_road_name, bulk_set_congestion, bulk_set_congestion_multi

# Custom warm colormap replacing matplotlib's default RdYlGn, so the
# color-graded tables match the brand palette: rust (weak) -> gold (mid)
# -> champagne (strong), instead of red-yellow-green.
WARM_CMAP = LinearSegmentedColormap.from_list(
    "malyx_warm", ["#9C4A2E", "#D8C9A8", "#C9BFAE"]
)

AUTHOR_NAME = "Muhammad Hamza Malik"
AUTHOR_SUBTITLE = "BS Biotechnology, University of Karachi -- YIC-26 Sindh HEC Fellow"

FAVICON_PATH = Path(__file__).parent / "assets" / "favicon_32.png"
try:
    from PIL import Image
    page_icon = Image.open(FAVICON_PATH)
except Exception:
    page_icon = "🧬"  # falls back if assets/ wasn't deployed alongside app.py

st.set_page_config(
    page_title="Physarum Urban Network Intelligence",
    page_icon=page_icon,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design system: organic/bio-inspired palette (moss, soil, ochre) instead of
# default Streamlit styling. Fraunces for display type (soft organic serif),
# IBM Plex Sans/Mono for body and data, matching the scientific-fieldnotes
# feel of the project.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  color-scheme: dark;
  --canvas: #0A1110;
  --surface: #101B19;
  --surface-raised: #14221F;
  --surface-soft: #192925;
  --mint: #72E0B1;
  --mint-bright: #A5F2D1;
  --sand: #DFC99A;
  --coral: #D98268;
  --ink: #F1F6F3;
  --muted: #9AAEA6;
  --line: rgba(183, 224, 207, .14);
}

html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 86% 0%, rgba(64, 143, 112, .12), transparent 34rem),
    radial-gradient(ellipse at 4% 28%, rgba(48, 89, 78, .12), transparent 30rem),
    var(--canvas);
  color: var(--ink);
}
[data-testid="stHeader"] { background: rgba(10, 17, 16, .72); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101B19, #0D1715 70%, #0A1110);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
.main .block-container {
  max-width: 1480px;
  padding: 1.6rem clamp(1rem, 3vw, 2.6rem) 3rem;
}

.hero-banner {
  background:
    radial-gradient(circle at 88% 20%, rgba(114, 224, 177, .13), transparent 15rem),
    linear-gradient(125deg, rgba(20, 34, 31, .98), rgba(12, 22, 20, .98));
  border: 1px solid rgba(114, 224, 177, .2);
  border-radius: 22px;
  padding: clamp(1.6rem, 3.2vw, 3rem);
  margin: .3rem 0 1.35rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 22px 70px rgba(0, 0, 0, .22), inset 0 1px rgba(255,255,255,.025);
}
.hero-banner::after {
  content: "";
  position: absolute;
  inset: auto -8% -110% 45%;
  height: 18rem;
  border: 1px solid rgba(114,224,177,.12);
  border-radius: 50%;
  box-shadow: 0 0 0 28px rgba(114,224,177,.025), 0 0 0 58px rgba(114,224,177,.02);
  pointer-events: none;
}
.hero-eyebrow {
  font: 600 .68rem "IBM Plex Mono", monospace;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--mint);
  margin-bottom: .65rem;
}
.hero-title {
  font: 600 clamp(2rem, 4vw, 3.15rem)/1.08 "Space Grotesk", sans-serif;
  letter-spacing: -.045em;
  color: var(--ink);
  margin: 0 0 .55rem;
}
.hero-location {
  font: 500 .72rem "IBM Plex Mono", monospace;
  letter-spacing: .07em;
  color: var(--muted);
  margin-bottom: 1.1rem;
}
.hero-byline {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  background: rgba(114, 224, 177, .07);
  border: 1px solid rgba(114, 224, 177, .2);
  border-radius: 999px;
  padding: .42rem .78rem;
  font-size: .8rem;
  color: var(--ink);
}
.hero-byline b { color: var(--mint-bright); font-weight: 600; }
.hero-logo { position: absolute; top: 1.7rem; right: 2rem; width: 76px; height: 76px; opacity: .72; }

.kpi-card {
  background: linear-gradient(150deg, rgba(23, 39, 35, .97), rgba(15, 26, 24, .98));
  border: 1px solid var(--line);
  border-radius: 17px;
  padding: 1rem 1.15rem;
  box-shadow: 0 10px 28px rgba(0,0,0,.14), inset 0 1px rgba(255,255,255,.025);
  height: 100%;
  transition: border-color .18s ease, transform .18s ease, box-shadow .18s ease;
}
.kpi-card:hover { transform: translateY(-2px); border-color: rgba(114,224,177,.36); box-shadow: 0 14px 34px rgba(0,0,0,.2); }
.kpi-label { font: 600 .65rem "IBM Plex Mono", monospace; letter-spacing: .105em; text-transform: uppercase; color: var(--muted); margin-bottom: .52rem; }
.kpi-value { font: 600 1.55rem/1.2 "Space Grotesk", sans-serif; letter-spacing: -.035em; color: var(--ink); }
.kpi-sub { font-size: .76rem; color: var(--muted); margin-top: .35rem; }
.section-tag {
  font: 600 .68rem "IBM Plex Mono", monospace;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--mint-bright);
  border-bottom: 1px solid var(--line);
  padding-bottom: .62rem;
  margin: .45rem 0 .8rem;
}
.footer-credit {
  margin-top: 2rem;
  padding: 1.1rem 1.4rem;
  background: rgba(16,27,25,.8);
  border: 1px solid var(--line);
  border-radius: 16px;
  text-align: center;
}
.footer-credit .name { font: 600 1rem "Space Grotesk", sans-serif; color: var(--ink); }
.footer-credit .sub { font: 500 .68rem "IBM Plex Mono", monospace; color: var(--muted); margin-top: .3rem; }

[data-testid="stMetricValue"] { font-family: "Space Grotesk", sans-serif; letter-spacing: -.03em; }
[data-testid="stTabs"] [role="tablist"] { gap: .32rem; border-bottom: 1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] {
  font: 500 .78rem "DM Sans", sans-serif;
  color: var(--muted);
  border-radius: 10px 10px 0 0;
  padding: .6rem .75rem;
  transition: color .16s ease, background .16s ease;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--ink); background: rgba(114,224,177,.05); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--mint-bright);
  background: rgba(114,224,177,.08);
}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
  border: 1px solid rgba(114,224,177,.25);
  border-radius: 11px;
  font-weight: 600;
  transition: border-color .16s ease, background .16s ease, transform .16s ease;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  border-color: rgba(114,224,177,.62);
  transform: translateY(-1px);
}
[data-testid="stAlert"] { border-radius: 13px; border: 1px solid var(--line); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stCustomComponentV1"] { border-radius: 16px; overflow: hidden; }
[data-testid="stFileUploader"] section {
  border: 1px dashed rgba(114,224,177,.34);
  border-radius: 14px;
  background: rgba(114,224,177,.025);
}
[data-testid="stSlider"] [role="slider"] { accent-color: var(--mint); }

.map-empty {
  display: grid;
  grid-template-columns: 3.2rem 1fr;
  gap: .25rem 1rem;
  align-items: start;
  padding: 1.3rem 1.4rem;
  margin: .6rem 0;
  border: 1px solid rgba(114,224,177,.2);
  border-radius: 17px;
  background:
    radial-gradient(circle at 95% 0, rgba(114,224,177,.08), transparent 16rem),
    linear-gradient(145deg, rgba(20,34,31,.96), rgba(13,23,21,.96));
}
.map-empty-icon {
  width: 3rem; height: 3rem;
  display: grid; place-items: center;
  border: 1px solid rgba(114,224,177,.26);
  border-radius: 14px;
  background: rgba(114,224,177,.08);
  color: var(--mint);
  font: 600 1.4rem "Space Grotesk", sans-serif;
  grid-row: span 2;
}
.map-empty-title { color: var(--ink); font: 600 1.05rem "Space Grotesk", sans-serif; }
.map-empty-copy { color: var(--muted); font-size: .88rem; line-height: 1.55; }
.map-empty-fields { grid-column: 2; margin-top: .35rem; color: var(--muted); font-size: .78rem; }
.map-empty-fields code { color: var(--mint-bright); background: rgba(114,224,177,.08); border-radius: 6px; padding: .2rem .4rem; }

@media (max-width: 720px) {
  .main .block-container { padding: 1rem .8rem 2rem; }
  .hero-banner { border-radius: 17px; padding: 1.45rem 1.2rem; }
  .hero-logo { width: 48px; height: 48px; right: 1rem; top: 1rem; opacity: .55; }
  .hero-location { font-size: .62rem; line-height: 1.6; max-width: 82%; }
  .kpi-card { border-radius: 14px; }
  [data-testid="stTabs"] button[role="tab"] { padding: .48rem .55rem; font-size: .72rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-banner">
    <svg class="hero-logo" viewBox="0 0 240 240" aria-hidden="true">
        <defs>
            <radialGradient id="gradMid" cx="50%" cy="50%" r="55%" fx="35%" fy="32%">
                <stop offset="0%" stop-color="#EFE9DD"/><stop offset="100%" stop-color="#8C8175"/>
            </radialGradient>
            <radialGradient id="gradDark" cx="50%" cy="50%" r="55%" fx="35%" fy="32%">
                <stop offset="0%" stop-color="#7A7166"/><stop offset="100%" stop-color="#3A342C"/>
            </radialGradient>
            <radialGradient id="gradHub" cx="50%" cy="50%" r="58%" fx="35%" fy="30%">
                <stop offset="0%" stop-color="#FFFDF6"/><stop offset="100%" stop-color="#D8C9A8"/>
            </radialGradient>
        </defs>
        <polygon points="120,25 202.27,72.5 202.27,167.5 120,215 37.73,167.5 37.73,72.5" fill="none" stroke="#C9BFAE" stroke-width="2"/>
        <polygon points="120,36 192.75,78 192.75,162 120,204 47.25,162 47.25,78" fill="none" stroke="#C9BFAE" stroke-width="1" opacity="0.4"/>
        <line x1="65" y1="165" x2="95" y2="75" stroke="#8C8175" stroke-width="4"/>
        <line x1="95" y1="75" x2="120" y2="120" stroke="#C9BFAE" stroke-width="6"/>
        <line x1="120" y1="120" x2="145" y2="75" stroke="#C9BFAE" stroke-width="6"/>
        <line x1="145" y1="75" x2="175" y2="165" stroke="#8C8175" stroke-width="4"/>
        <circle cx="65" cy="165" r="12" fill="url(#gradDark)"/>
        <circle cx="95" cy="75" r="14" fill="url(#gradMid)"/>
        <circle cx="145" cy="75" r="14" fill="url(#gradMid)"/>
        <circle cx="175" cy="165" r="12" fill="url(#gradDark)"/>
        <circle cx="120" cy="120" r="21" fill="url(#gradHub)"/>
    </svg>
    <div class="hero-eyebrow">Bio-Inspired Urban Network Optimization</div>
    <div class="hero-title">Physarum Network Intelligence</div>
    <div class="hero-location">GULSHAN-E-IQBAL &nbsp;·&nbsp; DISTRICT EAST, KARACHI &nbsp;·&nbsp; YIC-26 SINDH HEC</div>
    <div class="hero-byline">🧬 Developed by <b>{AUTHOR_NAME}</b></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### 🧬 {AUTHOR_NAME}")
    st.caption(AUTHOR_SUBTITLE)
    st.divider()

    st.header("Data")
    uploaded = st.file_uploader(
        "Upload edges CSV (from auto_discover_corridors.py, with congestion values filled in)",
        type="csv",
    )
    demand_uploaded = st.file_uploader(
        "Optional: upload node demand CSV (from compute_node_demand.py --full). "
        "Required if your edges use auto-discovered N<id> node names -- the built-in "
        "8-landmark demand values won't match those.",
        type="csv",
        key="demand_csv",
    )
    use_sample = st.checkbox("Use illustrative sample dataset instead", value=uploaded is None)

    st.header("Model parameters")
    sim_mode = st.radio(
        "Simulation mode",
        ["Network-Coupled (experimental)", "Independent-edge baseline"],
        help="Network-Coupled solves an undirected Kirchhoff flow problem for a single-sink "
             "scenario. It is Physarum-inspired, not calibrated traffic assignment; one-way "
             "restrictions are not enforced. Independent scores edges without network flow.",
    )
    decay_rate = st.slider("Decay rate", 0.05, 0.30, 0.15, 0.01)
    mu = st.slider(
        "Reinforcement sharpness (mu)", 0.5, 3.0, 1.0, 0.1,
        help="Exponent in the normalized saturating response f(|Q|)=|Q|^mu/(1+|Q|^mu). "
             "Higher values sharpen reinforcement. It is not calibrated to Karachi traffic.",
    )
    generations = st.slider("Maximum iterations", 10, 100, 50, 5)

    st.header("Cost assumptions (PKR millions/km)")
    st.caption("Grounded in published Pakistani road-cost references -- replace with real KDA/SULTA figures for the actual submission")
    cost_main = st.number_input("Main Artery", value=DEFAULT_COST_PER_KM["Main Artery"], step=10.0)
    cost_collector = st.number_input("Collector Road", value=DEFAULT_COST_PER_KM["Collector Road"], step=10.0)
    cost_internal = st.number_input("Internal Street", value=DEFAULT_COST_PER_KM["Internal Street"], step=10.0)
    cost_per_km = {
        "Main Artery": cost_main,
        "Collector Road": cost_collector,
        "Internal Street": cost_internal,
    }

DEFAULT_DEMANDS = {
    "AKU": 1.2, "Kashmir Park": 1.5, "Hassan Square": 1.2, "FUAAST": 0.7,
    "NIPA": 1.5, "Gulshan Chowrangi": 1.2, "Maskan": 0.9, "KU Gate": 0.7,
}

SAMPLE_EDGES = pd.DataFrame([
    ["E1", "AKU", "Kashmir Park", 1.4, "Main Artery", 8],
    ["E3", "Kashmir Park", "Hassan Square", 1.4, "Main Artery", 8],
    ["E4", "Hassan Square", "FUAAST", 2.7, "Main Artery", 8],
    ["E7", "FUAAST", "NIPA", 1.2, "Main Artery", 4],
    ["E10", "Gulshan Chowrangi", "Maskan", 2.3, "Main Artery", 8],
    ["E14", "Maskan", "KU Gate", 2.6, "Main Artery", 9],
], columns=["edge_id", "u", "v", "distance_km", "road_class", "congestion"])

COORDINATE_COLUMNS = ["u_lat", "u_lon", "v_lat", "v_lon"]

if uploaded is not None and not use_sample:
    edges_df = pd.read_csv(uploaded)
    missing_coordinate_columns = [c for c in COORDINATE_COLUMNS if c not in edges_df.columns]
    valid_coordinate_rows = pd.Series(False, index=edges_df.index)
    if not missing_coordinate_columns:
        for column in COORDINATE_COLUMNS:
            edges_df[column] = pd.to_numeric(edges_df[column], errors="coerce")
        valid_coordinate_rows = (
            edges_df["u_lat"].between(-90, 90)
            & edges_df["v_lat"].between(-90, 90)
            & edges_df["u_lon"].between(-180, 180)
            & edges_df["v_lon"].between(-180, 180)
        ).fillna(False)
    valid_coordinate_count = int(valid_coordinate_rows.sum())
    invalid_coordinate_count = len(edges_df) - valid_coordinate_count if not missing_coordinate_columns else 0
    has_coords = valid_coordinate_count > 0
    edges_df["congestion"] = pd.to_numeric(edges_df["congestion"], errors="coerce")
    if edges_df["congestion"].isnull().any():
        st.error(
            "Some rows have no valid congestion value. Fill in the congestion column "
            "(1-10) before running the model. Record its source, date, and collection method."
        )
        st.stop()
    if edges_df["congestion"].nunique(dropna=True) == 1:
        congestion_value = float(edges_df["congestion"].iloc[0])
        if congestion_value == 5:
            st.warning(
                "All uploaded corridors have congestion=5. The repository's network "
                "extraction workflow assigns 5 as a placeholder; this is not an observed "
                "traffic measurement. These results cannot distinguish congestion between "
                "corridors. Replace the values with sourced, dated observations before using "
                "the scores as congestion analysis. With this input, scores are driven by "
                "network structure, the demand proxy, distance, and road class."
            )
        else:
            st.info(
                f"All uploaded corridors have the same congestion value ({congestion_value:g}). "
                "The model therefore has no corridor-to-corridor congestion variation to use."
            )
else:
    edges_df = SAMPLE_EDGES
    missing_coordinate_columns = COORDINATE_COLUMNS.copy()
    valid_coordinate_rows = pd.Series(False, index=edges_df.index)
    valid_coordinate_count = 0
    invalid_coordinate_count = 0
    has_coords = False
    st.info(
        "Using the illustrative 6-corridor sample dataset. Upload an edges CSV "
        "from auto_discover_corridors.py to analyze a real network."
    )

node_demands = DEFAULT_DEMANDS
if demand_uploaded is not None:
    demand_df = pd.read_csv(demand_uploaded)
    node_demands = dict(zip(demand_df["node"], demand_df["demand"]))
elif uploaded is not None and not use_sample:
    unmatched = set(edges_df["u"]).union(set(edges_df["v"])) - set(DEFAULT_DEMANDS.keys())
    if unmatched:
        st.warning(
            f"{len(unmatched)} node(s) in your edges CSV aren't in the built-in demand "
            f"list (e.g. {list(unmatched)[:3]}...). They'll default to demand=1.0. "
            f"Upload a node demand CSV from compute_node_demand.py for accurate values."
        )

@st.cache_data(show_spinner="Running simulation...", max_entries=5)
def cached_simulation(edges_df, node_demands, decay_rate, mu, generations, sim_mode):
    if sim_mode == "Network-Coupled (experimental)":
        result_df, sink_used = run_network_simulation(edges_df, node_demands, decay_rate, mu, generations)
        return result_df
    return run_simulation(edges_df, node_demands, decay_rate, mu, generations)


@st.cache_data(show_spinner=False, max_entries=5)
def cached_sink_node(edges_df, node_demands):
    """Cheap to compute on its own -- avoids relying on DataFrame.attrs surviving
    st.cache_data's pickling, which isn't a guaranteed-stable mechanism."""
    nodes = sorted(set(edges_df["u"]).union(set(edges_df["v"])))
    return max(nodes, key=lambda n: node_demands.get(n, 1.0))


@st.cache_data(show_spinner=False, max_entries=5)
def cached_cost_benefit(result, node_demands, cost_per_km):
    return add_cost_benefit(result, node_demands, cost_per_km)


@st.cache_resource(show_spinner="Building map...", max_entries=3)
def build_main_map(result, edges_df, highlighted):
    """
    Cached so panning/zooming the map (which triggers a Streamlit rerun via
    st_folium) doesn't rebuild 1000+ polylines every time -- only rebuilds
    when the actual data or the highlighted corridor changes.

    Each segment is styled using its WHOLE ROAD's aggregated score, not its
    own individual segment score. OpenStreetMap splits every real road into
    many small pieces at each intersection, so styling segments
    individually made a single physical road flicker between different
    colors and widths along its length. Using the same road-level
    aggregation as the Conductance Ranking tab keeps a road looking like one
    consistent line, styled by its weakest segment (a road is only as
    strong as its weakest link).
    """
    coordinate_values = edges_df[COORDINATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid_edges = coordinate_values[
        coordinate_values["u_lat"].between(-90, 90)
        & coordinate_values["v_lat"].between(-90, 90)
        & coordinate_values["u_lon"].between(-180, 180)
        & coordinate_values["v_lon"].between(-180, 180)
    ].dropna()
    if valid_edges.empty:
        raise ValueError("No valid latitude/longitude pairs are available to draw.")
    center_lat = valid_edges[["u_lat", "v_lat"]].values.mean()
    center_lon = valid_edges[["u_lon", "v_lon"]].values.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="CartoDB dark_matter", name="Dark (fallback)", overlay=False, control=True,
    ).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    # Precompute road-level style (color + width) once, shared by every
    # segment of the same road -- avoids the old flickering effect and
    # avoids an O(n^2) rank lookup by computing each edge's rank position
    # once up front instead of re-searching the list per segment.
    road_style = {}
    if "road_name" in result.columns:
        aggregated = aggregate_by_road_name(result).reset_index(drop=True)
        max_road_eff = aggregated["worst_bio_efficiency"].max() or 1
        n_roads = len(aggregated)
        for i, arow in aggregated.iterrows():
            rank_frac = i / max(n_roads - 1, 1)
            color = "#9C4A2E" if rank_frac < 0.34 else "#D8C9A8" if rank_frac < 0.67 else "#C9BFAE"
            weight = 2 + (arow["worst_bio_efficiency"] / max_road_eff) * 8
            for eid in arow["segment_edge_ids"]:
                road_style[eid] = (color, weight)

    max_eff = result["bio_efficiency"].max() or 1
    n_segments = len(result)
    edge_rank = {eid: i for i, eid in enumerate(result["edge_id"])}

    highlight_coords = None
    for row in result.itertuples():
        src_match = edges_df[edges_df.edge_id == row.edge_id]
        if src_match.empty:
            continue  # skip rather than crash the whole map over one bad row
        src = src_match.iloc[0]
        edge_coords = pd.to_numeric(pd.Series([src.u_lat, src.u_lon, src.v_lat, src.v_lon]), errors="coerce")
        if edge_coords.isna().any() or not (-90 <= edge_coords.iloc[0] <= 90 and -180 <= edge_coords.iloc[1] <= 180
                                             and -90 <= edge_coords.iloc[2] <= 90 and -180 <= edge_coords.iloc[3] <= 180):
            continue
        if pd.isna(row.bio_efficiency):
            color, weight = "#77746D", 2
        elif row.edge_id in road_style:
            color, weight = road_style[row.edge_id]
        else:
            weight = 2 + (row.bio_efficiency / max_eff) * 8
            rank_frac = edge_rank[row.edge_id] / max(n_segments - 1, 1)
            color = "#9C4A2E" if rank_frac < 0.34 else "#D8C9A8" if rank_frac < 0.67 else "#C9BFAE"
        is_highlighted = highlighted and row.edge_id == highlighted
        coords = [(src.u_lat, src.u_lon), (src.v_lat, src.v_lon)]
        m_ = folium.PolyLine(coords, color="#1B1815", weight=weight + 2.5, opacity=0.6 if not is_highlighted else 0.3)
        m_.add_to(m)
        if is_highlighted:
            highlight_coords = coords
            folium.PolyLine(coords, color=color, weight=weight, opacity=0.35).add_to(m)
        else:
            folium.PolyLine(
                coords, color=color, weight=weight,
                opacity=0.9 if not highlighted else 0.3,
                tooltip=f"{getattr(row, 'road_name', row.edge_id)}: {row.u} to {row.v} ({row.bio_efficiency:.2f})",
            ).add_to(m)

    if highlight_coords:
        folium.PolyLine(highlight_coords, color="#1B1815", weight=10, opacity=0.7).add_to(m)
        folium.PolyLine(
            highlight_coords, color="#F2ECDD", weight=7, opacity=0.95,
            tooltip=f"SELECTED: {result[result.edge_id == highlighted].iloc[0].get('road_name', highlighted)}",
        ).add_to(m)
        mid = [(highlight_coords[0][0] + highlight_coords[1][0]) / 2,
               (highlight_coords[0][1] + highlight_coords[1][1]) / 2]
        folium.Marker(mid, icon=folium.Icon(color="beige", icon="star")).add_to(m)
        m.location = mid
        m.zoom_start = 16
    return m


result = cached_simulation(edges_df, node_demands, decay_rate, mu, generations, sim_mode)
if "converged" in result.columns:
    scored_rows = result[result["bio_efficiency"].notna()]
    if not scored_rows.empty and scored_rows["converged"].all():
        n_used = int(scored_rows["iterations_run"].max())
        st.success(
            f"Scored corridors met the numerical stopping threshold within {n_used} iteration(s). "
            "This indicates stability for the selected model and inputs; it does not validate traffic predictions."
        )
    elif not scored_rows.empty:
        n_used = int(scored_rows["iterations_run"].max())
        st.warning(
            f"At least part of the scored network did not meet the numerical stopping threshold within "
            f"{n_used} iteration(s). Treat these scores as iteration-limited."
        )
if result.empty:
    st.error(
        "The simulation produced no results -- this usually means the uploaded CSV has a formatting "
        "issue (e.g. edited in Excel and saved with a different structure than expected). Check that "
        "edge_id, u, v, distance_km, road_class, and congestion columns are all present and non-empty, "
        "then re-upload."
    )
    st.stop()
ranked_value = cached_cost_benefit(result, node_demands, cost_per_km)
has_road_name = "road_name" in edges_df.columns
if "note" in result.columns and result["note"].astype(str).str.contains("disconnected").any():
    st.warning(
        "Some corridors are disconnected from the selected sink and receive no network-coupled score. "
        "They remain visible with a blank score and are excluded from model-score comparisons."
    )

if sim_mode == "Network-Coupled (experimental)":
    sink_assumption = cached_sink_node(edges_df, node_demands)
    st.info(
        f"Scenario assumption: **{sink_assumption}** is the single sink; all other nodes inject "
        "flow in proportion to their POI-demand proxy. This is not an OD matrix or observed trip pattern."
    )
if "two_way" in edges_df.columns:
    one_way_flags = edges_df["two_way"].astype(str).str.strip().str.lower().isin(["false", "0", "no"])
    if one_way_flags.any():
        st.warning(
            "This dataset contains one-way corridors. The Physarum adaptation model treats the "
            "network as undirected; direction restrictions affect alternate-route search only, "
            "not adaptation scores."
        )

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
top_priority = result.iloc[0]
total_nodes = len(set(edges_df["u"]).union(set(edges_df["v"])))
top5_cost = ranked_value[ranked_value["bio_efficiency"].notna()].sort_values(
    "bio_efficiency"
).head(5)["estimated_cost_pkr_m"].sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Corridors Mapped</div>
    <div class="kpi-value">{len(edges_df)}</div>
    <div class="kpi-sub">{total_nodes} intersections</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label"><span class="pulse-dot"></span>Lowest Model Score</div>
    <div class="kpi-value" style="font-size:1.15rem;">{top_priority['u']} → {top_priority['v']}</div>
    <div class="kpi-sub">bio-efficiency {top_priority['bio_efficiency']:.2f}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg. Bio-Efficiency</div>
    <div class="kpi-value">{result['bio_efficiency'].mean():.2f}</div>
    <div class="kpi-sub">across all corridors</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Est. Cost: 5 Lowest-Score Edges</div>
    <div class="kpi-value">Rs {top5_cost:,.0f}M</div>
    <div class="kpi-sub">placeholder cost model</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Shared helpers (used by Cost Screening and Resilience Routing tabs)
# ---------------------------------------------------------------------------
def node_label(node_id):
    if not has_road_name:
        return node_id  # sample dataset already uses human names like "AKU"
    names = set()
    touching = edges_df[(edges_df.u == node_id) | (edges_df.v == node_id)]
    for rn in touching["road_name"]:
        if rn and not str(rn).startswith("Unnamed"):
            names.add(rn)
    if names:
        return f"{' / '.join(sorted(names))}  ({node_id})"
    return f"Unnamed junction ({node_id})"


def edge_label(edge_id):
    match = edges_df[edges_df.edge_id == edge_id]
    if match.empty:
        return f"(no longer in dataset: {edge_id})"
    row = match.iloc[0]
    name = row.get("road_name", edge_id) if has_road_name else edge_id
    return f"{name}  [{node_label(row.u)} \u2192 {node_label(row.v)}]"


with st.expander("🌧️ Climate risk: mark flood-prone corridors (optional, shared across tabs)"):
    st.caption(
        "Based on local knowledge or historical monsoon reports -- mark which corridors are known "
        "to flood. This feeds both the Resilience Routing tab's monsoon scenario simulation and "
        "the Cost Screening tab's climate resilience cost premium. Not derived from any dataset -- "
        "you're telling the tool what you already know about this area."
    )
    flood_prone_edges = st.multiselect(
        "Flood-prone corridors",
        list(edges_df["edge_id"]),
        format_func=edge_label,
        key="flood_prone_edges",
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_priority, tab_cost, tab_map, tab_resilience, tab_baseline, tab_notes = st.tabs(
    ["🧬 Conductance Ranking", "💰 Cost Screening", "🗺️ Map", "🚧 Resilience Routing", "🔬 Baseline Comparison", "📖 Model Notes"]
)

with tab_priority:
    st.markdown('<div class="section-tag">Lowest conductance scores</div>', unsafe_allow_html=True)
    st.caption("These edges received the lowest final conductance scores under the selected inputs. "
               "This is a structural model signal, not evidence of congestion or a widening recommendation. "
               "Click a row to highlight it on the Map tab.")
    display_cols = ["edge_id"]
    if "road_name" in result.columns:
        display_cols.append("road_name")
    display_cols += ["u", "v", "distance_km", "road_class", "congestion", "impedance", "bio_efficiency"]
    if "note" in result.columns:
        display_cols.append("note")
    display_df = result[display_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1
    styled = display_df.style.background_gradient(
        subset=["bio_efficiency"], cmap=WARM_CMAP, vmin=display_df["bio_efficiency"].min(),
        vmax=display_df["bio_efficiency"].max(),
    )
    selection = st.dataframe(
        styled, width="stretch",
        on_select="rerun", selection_mode="single-row", key="priority_table",
    )
    try:
        selected_rows = selection.selection.rows
    except AttributeError:
        selected_rows = selection.get("selection", {}).get("rows", []) if selection else []
    if selected_rows:
        st.session_state["highlighted_edge"] = display_df.iloc[selected_rows[0]]["edge_id"]
        st.info(f"Selected **{display_df.iloc[selected_rows[0]]['edge_id']}** -- view it highlighted on the 🗺️ Map tab.")

    st.download_button(
        "⬇ Download results as CSV",
        display_df.to_csv().encode("utf-8"),
        "bio_efficiency_results.csv",
        "text/csv",
    )

    if has_road_name:
        st.markdown('<div class="section-tag">Grouped by real road, not by OSM segment</div>', unsafe_allow_html=True)
        st.caption(
            "OpenStreetMap splits real roads into a separate segment at every intersection they "
            "cross -- necessary for accurate routing, but it means one physical road can appear as "
            "many rows above. This groups those segments back up by their real name, using the "
            "weakest segment's score to represent the whole road (a road is only as strong as its "
            "weakest link)."
        )
        aggregated = aggregate_by_road_name(result)
        agg_display = aggregated[["road_name", "segment_count", "total_distance_km", "worst_bio_efficiency", "avg_congestion"]].reset_index(drop=True)
        agg_display.index = agg_display.index + 1
        agg_display.columns = ["Road", "Segments", "Total Length (km)", "Weakest Segment Score", "Avg. Congestion"]
        st.dataframe(
            agg_display.style.background_gradient(subset=["Weakest Segment Score"], cmap=WARM_CMAP),
            width="stretch",
        )

        st.markdown('<div class="section-tag">Bulk-fill congestion for multiple roads at once</div>', unsafe_allow_html=True)
        st.caption(
            "Pick as many roads as you want, set each one's congestion value in the table below, "
            "then apply all of them in a single click -- instead of repeating select-and-apply "
            "once per road. Every segment of a given road gets that road's value."
        )
        roads_to_edit = st.multiselect(
            "Roads to update",
            sorted(edges_df["road_name"].unique()),
        )

        if roads_to_edit:
            agg_lookup = aggregate_by_road_name(cached_simulation(edges_df, node_demands, decay_rate, mu, generations, sim_mode))
            agg_lookup = agg_lookup.set_index("road_name")

            edit_rows = []
            for road in roads_to_edit:
                current_avg = agg_lookup.loc[road, "avg_congestion"] if road in agg_lookup.index else 5.0
                seg_count = agg_lookup.loc[road, "segment_count"] if road in agg_lookup.index else (edges_df["road_name"] == road).sum()
                edit_rows.append({
                    "Road": road,
                    "Segments": int(seg_count),
                    "Current Avg. Congestion": round(float(current_avg), 1),
                    "New Congestion (1-10)": int(round(current_avg)),
                })
            edit_df = pd.DataFrame(edit_rows)

            edited = st.data_editor(
                edit_df,
                column_config={
                    "Road": st.column_config.TextColumn(disabled=True),
                    "Segments": st.column_config.NumberColumn(disabled=True),
                    "Current Avg. Congestion": st.column_config.NumberColumn(disabled=True),
                    "New Congestion (1-10)": st.column_config.NumberColumn(min_value=1, max_value=10, step=1),
                },
                hide_index=True,
                width="stretch",
                key="congestion_batch_editor",
            )

            if st.button(f"Apply to all {len(roads_to_edit)} selected road(s)"):
                road_congestion_map = dict(zip(edited["Road"], edited["New Congestion (1-10)"]))
                updated_edges, counts = bulk_set_congestion_multi(edges_df, road_congestion_map)
                summary = ", ".join(f"{road} ({n} segments \u2192 {road_congestion_map[road]})" for road, n in counts.items())
                st.success(f"Updated: {summary}")
                st.download_button(
                    "⬇ Download updated edges CSV",
                    updated_edges.to_csv(index=False).encode("utf-8"),
                    "edges_with_updated_congestion.csv",
                    "text/csv",
                )

with tab_cost:
    st.markdown('<div class="section-tag">Heuristic cost screening</div>', unsafe_allow_html=True)

    top_value = ranked_value.iloc[0]
    top_label = top_value.get("road_name", f"{top_value['u']} to {top_value['v']}") if has_road_name else f"{top_value['u']} to {top_value['v']}"
    st.markdown(
        f"Under the current demand proxy, model scores, and editable cost assumptions, **{top_label}** "
        f"has the highest heuristic value ratio. Its estimated cost is **Rs {top_value['estimated_cost_pkr_m']:,.0f} million**. "
        "This ratio is not a forecast of traffic relief, travel-time savings, or return on investment."
    )

    simple_display = ranked_value[["edge_id", "u", "v", "estimated_cost_pkr_m", "value_ratio"]].reset_index(drop=True)
    simple_display.index = simple_display.index + 1
    simple_display.columns = ["Edge", "From", "To", "Est. Cost (PKR M)", "Value Score"]
    st.dataframe(
        simple_display.style.background_gradient(subset=["Value Score"], cmap=WARM_CMAP),
        width="stretch",
    )
    st.caption("Higher Value Score = higher demand-to-score proxy per estimated cost. It is a screening heuristic, not a benefit estimate.")

    with st.expander("Show full technical table (bio-efficiency, demand pressure, etc.)"):
        value_display = ranked_value[["edge_id", "u", "v", "bio_efficiency",
                                       "estimated_cost_pkr_m", "demand_pressure", "value_ratio"]].reset_index(drop=True)
        value_display.index = value_display.index + 1
        value_display.columns = ["Edge", "From", "To", "Bio-Efficiency",
                                  "Est. Cost (PKR M)", "Demand Pressure", "Value Ratio"]
        st.caption(
            "Demand Pressure = average endpoint POI-demand proxy divided by model conductance. "
            "Value Ratio = this proxy divided by estimated cost. Neither measures unmet trips or benefits."
        )
        styled_value = value_display.style.background_gradient(subset=["Value Ratio"], cmap=WARM_CMAP)
        st.dataframe(styled_value, width="stretch")

    st.markdown('<div class="section-tag">Climate resilience cost adjustment</div>', unsafe_allow_html=True)
    st.caption(
        "Flood-resistant construction (elevated roadbed, improved drainage) genuinely costs more "
        "than standard widening. This applies an adjustable premium to corridors marked flood-prone "
        "in the Climate risk expander above the tabs, and shows whether accounting for that cost "
        "changes which corridor has the highest heuristic value ratio."
    )
    if not flood_prone_edges:
        st.info("No corridors marked flood-prone yet -- open the \"🌧️ Climate risk\" expander above the tabs to mark some.")
    else:
        climate_premium_pct = st.slider(
            "Climate resilience premium for flood-prone corridors (%)", 0, 100, 20, 5,
            help="Extra construction cost for flood-resistant infrastructure on corridors you've "
                 "marked as flood-prone. This is an adjustable assumption, not a measured figure.",
        )
        rv_climate = ranked_value.copy()
        is_flood_prone = rv_climate["edge_id"].isin(flood_prone_edges)
        rv_climate["climate_cost"] = rv_climate["estimated_cost_pkr_m"] * is_flood_prone.map(
            {True: 1 + climate_premium_pct / 100, False: 1.0}
        )
        rv_climate["climate_value_ratio"] = rv_climate["demand_pressure"] / rv_climate["climate_cost"]
        rv_climate["Flood-Prone"] = is_flood_prone.map({True: "🌧️ Yes", False: ""})

        climate_display = rv_climate.sort_values("climate_value_ratio", ascending=False)[
            ["edge_id", "u", "v", "Flood-Prone", "estimated_cost_pkr_m", "climate_cost", "climate_value_ratio"]
        ].reset_index(drop=True)
        climate_display.index = climate_display.index + 1
        climate_display.columns = ["Edge", "From", "To", "Flood-Prone", "Original Cost (PKR M)",
                                    "Climate-Adjusted Cost (PKR M)", "Climate-Adjusted Value Score"]
        st.dataframe(
            climate_display.style.background_gradient(subset=["Climate-Adjusted Value Score"], cmap=WARM_CMAP),
            width="stretch",
        )

        original_top = ranked_value.sort_values("value_ratio", ascending=False).iloc[0]["edge_id"]
        climate_top = rv_climate.sort_values("climate_value_ratio", ascending=False).iloc[0]["edge_id"]
        if original_top != climate_top:
            st.warning(
                f"**The heuristic ranking changes under this cost assumption:** without the flood-resilience "
                f"premium, {original_top} has the highest value ratio; with it, {climate_top} does. "
                "This is not a measured benefit or investment recommendation."
            )
        else:
            st.success(
                f"{original_top} retains the highest heuristic value ratio after the assumed climate premium. "
                "This is not a measured benefit or investment recommendation."
            )

    st.markdown('<div class="section-tag">Model-score scenario sensitivity</div>', unsafe_allow_html=True)
    st.caption(
        "This applies hypothetical road-class and congestion changes to each candidate and reruns "
        "the model. It reports internal conductance-score changes only; it does not predict "
        "real traffic relief, travel times, or economic return."
    )
    verify_n = st.slider("Analyze top N heuristic candidates", 5, 30, 15, 5)
    if st.button("🔬 Run model-score scenarios"):
        if sim_mode == "Network-Coupled (experimental)":
            def _sim_fn(e, d, dr, m, g):
                r, _ = run_network_simulation(e, d, dr, m, g)
                return r
        else:
            _sim_fn = run_simulation

        with st.spinner(f"Running model-score scenarios for {verify_n} candidates..."):
            verified = verify_top_candidates(
                edges_df, node_demands, ranked_value, decay_rate, mu, generations,
                cost_per_km, top_n=verify_n, simulation_fn=_sim_fn,
            )
        st.dataframe(
            verified.style.background_gradient(subset=["simulated_value_ratio"], cmap=WARM_CMAP),
            width="stretch",
        )
        if sim_mode == "Network-Coupled (experimental)":
            st.caption(
                "Note: in Network-Coupled mode, a corridor with no alternate route in the network "
                "(a dead-end or a chain with no loops) will correctly show zero simulated gain -- "
                "its flow is fixed by conservation regardless of its own width. This is expected "
                "behavior, not a bug; it means coupling only has something to redistribute where "
                "real alternate paths exist."
            )
        if list(verified["edge_id"])[0] != list(ranked_value["edge_id"])[0]:
            st.warning(
                f"Note: the simulation-verified top pick ({verified.iloc[0]['edge_id']}) "
                f"differs from the static heuristic's top pick ({ranked_value.iloc[0]['edge_id']}). "
                f"Treat this as model-score sensitivity only; validate recommendations against observed traffic and travel-time data."
            )

    st.markdown('<div class="section-tag">Malyx vs. a conventional consultancy</div>', unsafe_allow_html=True)
    st.caption(
        "For pitching purposes. The conventional-study figures below are editable placeholder "
        "assumptions, not a verified quote -- before using this in a real pitch, get an actual "
        "quote from a firm that does this work in Pakistan (e.g. NESPAK or Think Transportation) "
        "and replace the numbers below with it."
    )
    cvc1, cvc2 = st.columns(2)
    with cvc1:
        conv_cost = st.number_input("Conventional study cost (PKR millions)", value=15.0, step=1.0)
    with cvc2:
        conv_weeks = st.number_input("Conventional study timeline (weeks)", value=16, step=1)

    malyx_cost_est = max(top5_cost * 0.0, 0.05)  # negligible marginal compute cost, not zero to stay honest
    comp1, comp2, comp3 = st.columns(3)
    with comp1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Time to result</div>
        <div class="kpi-value" style="font-size:1.3rem;">Minutes</div>
        <div class="kpi-sub">vs. ~{conv_weeks:.0f} weeks conventional</div></div>""", unsafe_allow_html=True)
    with comp2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Marginal cost per re-run</div>
        <div class="kpi-value" style="font-size:1.3rem;">~Rs {malyx_cost_est:.2f}M</div>
        <div class="kpi-sub">vs. Rs {conv_cost:.0f}M conventional</div></div>""", unsafe_allow_html=True)
    with comp3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Update when conditions change</div>
        <div class="kpi-value" style="font-size:1.3rem;">Re-run instantly</div>
        <div class="kpi-sub">vs. new engagement conventionally</div></div>""", unsafe_allow_html=True)
    st.caption(
        "\"Marginal cost per re-run\" means the cost of updating results after the first full data "
        "collection is done -- it does not include the one-time cost of the initial field congestion "
        "survey, which still takes real time and effort (see the Operator's Guide)."
    )

with tab_map:
    st.markdown('<div class="section-tag">Spatial network view</div>', unsafe_allow_html=True)
    if has_coords:
        st.caption(f"Showing {valid_coordinate_count:,} corridor(s) with valid geographic coordinates.")
        if invalid_coordinate_count:
            st.warning(f"{invalid_coordinate_count:,} corridor(s) have missing or out-of-range coordinates and are omitted from the map.")
        highlighted = st.session_state.get("highlighted_edge")
        if highlighted:
            st.caption(f"Highlighting **{highlighted}** -- click a different row on the Conductance Ranking tab to change this, or clear the selection there to unhighlight.")

        m = build_main_map(result, edges_df, highlighted)

        st_folium(m, width=None, height=520, key="main_network_map", returned_objects=[])
    else:
        if uploaded is None or use_sample:
            title = "Your sample network is schematic"
            copy = "The built-in example contains corridor relationships and model inputs, but no real intersection locations. Upload a geocoded edge file to place the network on a map."
        elif missing_coordinate_columns:
            title = "Location fields are missing"
            copy = "The edge file loaded, but it does not include all four endpoint coordinates needed to draw corridors."
        else:
            title = "No usable locations found"
            copy = "Coordinate columns were found, but every corridor has missing or out-of-range latitude/longitude values. Check the values and coordinate reference system."
        fields = ", ".join(missing_coordinate_columns) if missing_coordinate_columns else ", ".join(COORDINATE_COLUMNS)
        st.markdown(
            f'<div class="map-empty"><div class="map-empty-icon">⌖</div>'
            f'<div class="map-empty-title">{title}</div>'
            f'<div class="map-empty-copy">{copy} Upload an edge CSV generated by '
            f'<code>auto_discover_corridors.py</code> with coordinates in decimal degrees.</div>'
            f'<div class="map-empty-fields">Required fields: <code>{fields}</code></div></div>',
            unsafe_allow_html=True,
        )

with tab_resilience:
    st.markdown('<div class="section-tag">What if roads are blocked?</div>', unsafe_allow_html=True)
    st.caption(
        "Pick two real intersections from your uploaded network below, then optionally close any "
        "number of corridors and/or intersections between them (simulating flooding, a protest, an "
        "accident, or construction -- real closures often take out several roads at once, not just "
        "one). We'll show the best alternate route and how much worse it is than normal."
    )

    all_nodes = sorted(set(edges_df["u"]).union(set(edges_df["v"])))

    rc1, rc2 = st.columns(2)
    with rc1:
        origin = st.selectbox("From (starting point)", all_nodes, index=0, format_func=node_label)
    with rc2:
        destination = st.selectbox("To (destination)", all_nodes, index=min(1, len(all_nodes) - 1), format_func=node_label)

    rc3, rc4 = st.columns(2)
    with rc3:
        blocked_corridors = st.multiselect(
            "Corridors to close (pick any number)",
            list(edges_df["edge_id"]), format_func=edge_label,
        )
    with rc4:
        blocked_intersections = st.multiselect(
            "Intersections to close (pick any number)",
            all_nodes, format_func=node_label,
        )

    btn1, btn2 = st.columns([1, 1.4])
    with btn1:
        find_clicked = st.button("🔍 Find route")
    with btn2:
        monsoon_clicked = st.button(
            f"🌧️ Simulate monsoon scenario ({len(flood_prone_edges)} flood-prone corridor(s) marked)",
            disabled=len(flood_prone_edges) == 0,
            help="Closes every corridor marked flood-prone in the expander above, all at once -- "
                 "a more realistic worst-case than closing roads one at a time." if flood_prone_edges
                 else "Mark at least one flood-prone corridor above (in the Climate risk expander) to enable this.",
        )

    if find_clicked or monsoon_clicked:
        if monsoon_clicked:
            kwargs = {"blocked_edge_ids": list(flood_prone_edges)}
            st.info(f"🌧️ Monsoon scenario: simulating all {len(flood_prone_edges)} flood-prone corridors closing at once.")
        else:
            kwargs = {}
            if blocked_corridors:
                kwargs["blocked_edge_ids"] = blocked_corridors
            if blocked_intersections:
                kwargs["blocked_nodes"] = blocked_intersections

        route = find_alternate_route(result, origin, destination, **kwargs)

        if not route["found"]:
            n_blocked = len(route.get("blocked_summary", []))
            st.error(
                f"No route exists between {node_label(origin)} and {node_label(destination)} "
                f"once {'this closure is' if n_blocked <= 1 else f'these {n_blocked} closures are'} applied. "
                f"This is a real finding: your network has no "
                f"redundancy here, meaning {'this closure' if n_blocked <= 1 else 'this combination of closures'} "
                f"would completely cut off this connection. That itself is worth flagging to planners."
            )
        else:
            path_labels = " → ".join(node_label(n) for n in route["path"])
            st.success(f"**Route found:** {path_labels}")

            if kwargs:
                baseline = find_alternate_route(result, origin, destination)
                if baseline["found"]:
                    extra_km = route["total_distance_km"] - baseline["total_distance_km"]
                    extra_pct = (extra_km / baseline["total_distance_km"] * 100) if baseline["total_distance_km"] > 0 else 0
                    if extra_km > 0.01:
                        st.warning(
                            f"Compared to the normal route ({baseline['total_distance_km']} km), "
                            f"this closure adds **{extra_km:.2f} km ({extra_pct:.0f}% longer)** to the trip."
                        )
                    else:
                        st.info("This closure doesn't add meaningful extra distance -- a nearby alternate exists at essentially the same length.")

            mc1, mc2 = st.columns(2)
            mc1.metric("Route distance", f"{route['total_distance_km']} km")
            mc2.metric(
                "Route difficulty score", route["total_impedance"],
                help="Combines distance, congestion, and road type into one number. "
                     "Lower is easier to drive. Only meaningful when compared to another route, "
                     "not as a number on its own.",
            )

            if has_coords:
                node_coords = {}
                for r in edges_df.itertuples():
                    node_coords[r.u] = (r.u_lat, r.u_lon)
                    node_coords[r.v] = (r.v_lat, r.v_lon)

                route_coords = [node_coords[n] for n in route["path"] if n in node_coords]
                if route_coords:
                    rm = folium.Map(location=route_coords[len(route_coords) // 2], zoom_start=14, tiles=None)
                    folium.TileLayer(
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attr="Tiles &copy; Esri", name="Satellite", overlay=False, control=True,
                    ).add_to(rm)
                    folium.TileLayer(
                        tiles="CartoDB dark_matter", name="Dark (fallback)", overlay=False, control=True,
                    ).add_to(rm)
                    folium.LayerControl(position="topright").add_to(rm)

                    # Full network dimmed in the background, for context
                    for r in edges_df.itertuples():
                        folium.PolyLine(
                            [(r.u_lat, r.u_lon), (r.v_lat, r.v_lon)],
                            color="#C9BFAE", weight=1.5, opacity=0.25,
                        ).add_to(rm)

                    # Faded dashed baseline route, only shown when a closure actually changes the path
                    if kwargs and baseline["found"]:
                        baseline_coords = [node_coords[n] for n in baseline["path"] if n in node_coords]
                        if baseline_coords and baseline_coords != route_coords:
                            folium.PolyLine(
                                baseline_coords, color="#F2ECDD", weight=3, opacity=0.45, dash_array="6,6",
                                tooltip="Normal route (before closure)",
                            ).add_to(rm)

                    # The actual route, with dark casing so it stays legible over satellite imagery
                    folium.PolyLine(route_coords, color="#1B1815", weight=9, opacity=0.6).add_to(rm)
                    folium.PolyLine(route_coords, color="#D8C9A8", weight=5.5, opacity=0.95, tooltip="Route used").add_to(rm)

                    folium.Marker(node_coords[origin], icon=folium.Icon(color="green", icon="play"),
                                  tooltip=f"Start: {node_label(origin)}").add_to(rm)
                    folium.Marker(node_coords[destination], icon=folium.Icon(color="darkred", icon="flag"),
                                  tooltip=f"End: {node_label(destination)}").add_to(rm)

                    blocked_edge_ids_used = kwargs.get("blocked_edge_ids", [])
                    blocked_nodes_used = kwargs.get("blocked_nodes", [])

                    for beid in blocked_edge_ids_used:
                        brow_match = edges_df[edges_df.edge_id == beid]
                        if not brow_match.empty:
                            brow = brow_match.iloc[0]
                            bmid = [(brow.u_lat + brow.v_lat) / 2, (brow.u_lon + brow.v_lon) / 2]
                            folium.Marker(bmid, icon=folium.Icon(color="black", icon="remove-sign"),
                                          tooltip=f"Blocked: {edge_label(beid)}").add_to(rm)
                    for bnode in blocked_nodes_used:
                        if bnode in node_coords:
                            folium.Marker(node_coords[bnode], icon=folium.Icon(color="black", icon="remove-sign"),
                                          tooltip=f"Blocked: {node_label(bnode)}").add_to(rm)

                    st_folium(rm, width=None, height=420, key="resilience_route_map", returned_objects=[])
            else:
                st.caption(
                    "No coordinates in this dataset, so no map here -- upload a CSV with "
                    "lat/lon columns (from auto_discover_corridors.py) to see routes visually."
                )

with tab_baseline:
    st.markdown('<div class="section-tag">Does the simulation actually beat a simple formula?</div>', unsafe_allow_html=True)
    st.caption(
        "An honest check: this compares the full simulation against a single-step static score "
        "(the same demand/impedance ratio, computed once, with no iteration). If the two rankings "
        "mostly agree, the simulation isn't adding much for ranking purposes -- worth knowing "
        "before claiming the bio-inspired approach is better."
    )

    if sim_mode == "Network-Coupled (experimental)":
        sink_display = cached_sink_node(edges_df, node_demands)
        st.info(f"Network-Coupled mode active. Flow sink node for this run: **{sink_display}** "
                f"(highest-demand node -- see physarum_network_model.py docstring for why this "
                f"modeling choice was made).")

    baseline_result = compute_static_baseline(edges_df, node_demands)
    network_scored_result = result[result["bio_efficiency"].notna()]
    comparison = compare_rankings(
        network_scored_result, baseline_result,
        top_n=min(10, len(network_scored_result)),
    )

    corr = comparison["spearman_corr"]
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Rank Correlation (Spearman)</div>
        <div class="kpi-value">{corr:.3f}</div>
        <div class="kpi-sub">1.0 = identical ranking, 0 = no relationship, -1 = opposite</div></div>""", unsafe_allow_html=True)
    with bc2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Top-{comparison['top_n']} Overlap</div>
        <div class="kpi-value">{comparison['top_n_overlap']} / {comparison['top_n']}</div>
        <div class="kpi-sub">corridors both methods agree belong in the top tier</div></div>""", unsafe_allow_html=True)

    if sim_mode == "Independent-edge baseline" and corr is not None and corr > 0.9:
        st.warning(
            f"**Honest finding:** correlation is {corr:.3f} -- essentially identical rankings. "
            f"In Independent mode, each corridor evolves separately during simulation (no edge "
            f"ever influences another), so the adaptation is mathematically guaranteed "
            f"to preserve the same order as this one-line static formula. For *ranking* purposes, "
            f"Independent mode isn't adding information a much simpler calculation wouldn't give "
            f"you. Switch to Network-Coupled mode in the sidebar to see genuine flow-competition "
            f"effects between corridors."
        )
    elif sim_mode == "Network-Coupled (experimental)" and corr is not None and corr > 0.9:
        st.info(
            f"Correlation is {corr:.3f} on this network -- the two methods happen to agree closely "
            f"here. This can genuinely happen on networks with few loops (a chain-like topology has "
            f"only one path between most points, so there's little for coupling to redistribute). "
            f"Check the Map tab -- if this network has few loops, a high correlation is an honest "
            f"result, not a sign anything is wrong."
        )
    elif corr is not None:
        st.success(
            f"Correlation is {corr:.3f} -- the two methods meaningfully disagree on ranking. "
            f"This difference is caused by the selected network-coupled model assumptions. "
            f"Worth investigating the specific corridors below where they diverge most."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Full simulation ranking**")
        st.dataframe(
            network_scored_result[["edge_id", "u", "v", "bio_efficiency"]].reset_index(drop=True),
            width="stretch",
        )
    with col_b:
        st.markdown("**Static baseline ranking**")
        st.dataframe(
            baseline_result[["edge_id", "u", "v", "vc_score"]].reset_index(drop=True),
            width="stretch",
        )

    st.markdown("**Corridors ranked most differently between the two methods**")
    st.caption("If this table is all zeros, the two methods agree completely on this dataset (see finding above).")
    st.dataframe(
        comparison["most_divergent"][["edge_id", "u", "v", "bio_rank", "baseline_rank", "rank_diff"]].reset_index(drop=True),
        width="stretch",
    )

with tab_notes:
    st.markdown('<div class="section-tag">Model notes for the pitch</div>', unsafe_allow_html=True)
    st.markdown(
        """
- **Impedance** combines distance, congestion, and road class into a single
  friction score per corridor.
- **Bio-efficiency** is the edge conductance after the solver's numerical
  stopping check or iteration cap. This checks stability within the selected
  model; it does not measure road capacity or traffic flow.
- The network-coupled solver uses one assumed sink (the highest POI-demand
  node), injects proxy demand at other nodes, and treats links as undirected.
  It is not an origin-destination traffic assignment.
- The saturating response is related to forms studied in Physarum research
  (Tero et al., 2008). Its parameters are not calibrated to local traffic data.
- Cost estimates are grounded in published Pakistani road-cost references
  (NHA, Lahore carriageway projects), not a site-specific quote.
"""
    )

st.markdown(f"""
<div class="footer-credit">
    <div class="name">{AUTHOR_NAME}</div>
    <div class="sub">Physarum Network Intelligence &nbsp;·&nbsp; YIC-26 Sindh HEC Youth Innovation Fellowship</div>
</div>
""", unsafe_allow_html=True)
