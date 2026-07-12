import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from physarum_model import run_simulation, ROAD_OMEGA
from cost_model import add_cost_benefit, verify_top_candidates, DEFAULT_COST_PER_KM
from routing import find_alternate_route

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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-deep: #1B1815;
    --bg-panel: #221D18;
    --bg-panel-light: #2B241D;
    --accent-moss: #C9BFAE;
    --accent-moss-bright: #E8DFCE;
    --accent-ochre: #D8C9A8;
    --accent-rust: #9C4A2E;
    --text-cream: #F2ECDD;
    --text-muted: #A79C8C;
    --border-subtle: #3A342C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.hero-banner {
    background: linear-gradient(120deg, #1B1815 0%, #2B241D 55%, #3A3126 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--accent-moss-bright);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.3rem;
    color: var(--text-cream);
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}
.hero-location {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 1.1rem;
}
.hero-byline {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(201,191,174,0.12);
    border: 1px solid var(--accent-moss);
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    font-size: 0.85rem;
    color: var(--text-cream);
}
.hero-byline b { color: var(--accent-moss-bright); font-weight: 600; }

.hero-logo {
    position: absolute;
    top: 1.6rem;
    right: 2rem;
    width: 68px;
    height: 68px;
    opacity: 0.95;
}

.kpi-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: transform 0.18s ease, border-color 0.18s ease;
    height: 100%;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent-moss);
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-cream);
}
.kpi-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.2rem; }

.pulse-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--accent-ochre);
    margin-right: 6px;
    box-shadow: 0 0 0 0 rgba(216,201,168,0.7);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(216,201,168,0.55); }
    70%  { box-shadow: 0 0 0 9px rgba(216,201,168,0); }
    100% { box-shadow: 0 0 0 0 rgba(216,201,168,0); }
}

.section-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-moss-bright);
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 0.4rem;
    margin-bottom: 0.9rem;
}

.footer-credit {
    margin-top: 2.2rem;
    padding: 1.3rem 1.6rem;
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    text-align: center;
}
.footer-credit .name { font-family: 'Fraunces', serif; font-size: 1.1rem; color: var(--text-cream); }
.footer-credit .sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }

div[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }
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
        "Upload edges CSV (from extract_gulshan_network.py or auto_discover_corridors.py, congestion filled in)",
        type="csv",
    )
    demand_uploaded = st.file_uploader(
        "Optional: upload node demand CSV (from compute_node_demand.py --full). "
        "Required if your edges use auto-discovered N<id> node names -- the built-in "
        "8-landmark demand values won't match those.",
        type="csv",
        key="demand_csv",
    )
    use_sample = st.checkbox("Use validated sample dataset instead", value=uploaded is None)

    st.header("Model parameters")
    decay_rate = st.slider("Decay rate", 0.05, 0.30, 0.15, 0.01)
    mu = st.slider(
        "Reinforcement sharpness (mu)", 0.5, 3.0, 1.0, 0.1,
        help="Controls the saturating reinforcement function f(Q) = Q^mu / (1 + Q^mu), "
             "from Tero et al. 2010 (Science). Higher mu = more winner-take-all behavior.",
    )
    generations = st.slider("Generations", 10, 60, 30, 5)

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

if uploaded is not None and not use_sample:
    edges_df = pd.read_csv(uploaded)
    has_coords = {"u_lat", "u_lon", "v_lat", "v_lon"}.issubset(edges_df.columns)
    if edges_df["congestion"].isnull().any():
        st.error(
            "Some rows have no congestion value. Fill in the congestion column "
            "(1-10) before running the model -- see extract_gulshan_network.py "
            "docstring for how to estimate it from observed traffic patterns."
        )
        st.stop()
else:
    edges_df = SAMPLE_EDGES
    has_coords = False
    st.info(
        "Using the validated 6-corridor sample dataset. Upload a real edges CSV "
        "in the sidebar once you've run extract_gulshan_network.py in Colab."
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

result = run_simulation(edges_df, node_demands, decay_rate, mu, generations)
ranked_value = add_cost_benefit(result, node_demands, cost_per_km)
has_road_name = "road_name" in edges_df.columns

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
top_priority = result.iloc[0]
total_nodes = len(set(edges_df["u"]).union(set(edges_df["v"])))
top5_cost = ranked_value.sort_values("bio_efficiency").head(5)["estimated_cost_pkr_m"].sum()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Corridors Mapped</div>
    <div class="kpi-value">{len(edges_df)}</div>
    <div class="kpi-sub">{total_nodes} intersections</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label"><span class="pulse-dot"></span>Top Priority Corridor</div>
    <div class="kpi-value" style="font-size:1.15rem;">{top_priority['u']} → {top_priority['v']}</div>
    <div class="kpi-sub">bio-efficiency {top_priority['bio_efficiency']:.2f}</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg. Bio-Efficiency</div>
    <div class="kpi-value">{result['bio_efficiency'].mean():.2f}</div>
    <div class="kpi-sub">across all corridors</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Est. Top-5 Priority Investment</div>
    <div class="kpi-value">Rs {top5_cost:,.0f}M</div>
    <div class="kpi-sub">placeholder cost model</div></div>""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_priority, tab_cost, tab_map, tab_resilience, tab_notes = st.tabs(
    ["🧬 Priority Ranking", "💰 Cost-Benefit", "🗺️ Map", "🚧 Resilience Routing", "📖 Model Notes"]
)

with tab_priority:
    st.markdown('<div class="section-tag">Weakest corridors first</div>', unsafe_allow_html=True)
    st.caption("Lowest bio-efficiency = weakest link the network cannot reinforce = top priority for widening. "
               "Click a row to highlight that corridor on the Map tab.")
    display_cols = ["edge_id"]
    if "road_name" in result.columns:
        display_cols.append("road_name")
    display_cols += ["u", "v", "distance_km", "road_class", "congestion", "impedance", "bio_efficiency"]
    display_df = result[display_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1
    styled = display_df.style.background_gradient(
        subset=["bio_efficiency"], cmap=WARM_CMAP, vmin=display_df["bio_efficiency"].min(),
        vmax=display_df["bio_efficiency"].max(),
    )
    selection = st.dataframe(
        styled, use_container_width=True,
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

with tab_cost:
    st.markdown('<div class="section-tag">Return on investment</div>', unsafe_allow_html=True)

    top_value = ranked_value.iloc[0]
    top_label = top_value.get("road_name", f"{top_value['u']} to {top_value['v']}") if has_road_name else f"{top_value['u']} to {top_value['v']}"
    st.markdown(
        f"**In plain terms:** of all the corridors we analyzed, **{top_label}** gives you the most "
        f"traffic relief for the money -- an estimated **Rs {top_value['estimated_cost_pkr_m']:,.0f} million** "
        f"to fix, for the biggest improvement per rupee spent. That's not necessarily the single most "
        f"broken road (see the Priority Ranking tab for that) -- it's the smartest one to fix *first* "
        f"if your budget is limited."
    )

    simple_display = ranked_value[["edge_id", "u", "v", "estimated_cost_pkr_m", "value_ratio"]].reset_index(drop=True)
    simple_display.index = simple_display.index + 1
    simple_display.columns = ["Edge", "From", "To", "Est. Cost (PKR M)", "Value Score"]
    st.dataframe(
        simple_display.style.background_gradient(subset=["Value Score"], cmap=WARM_CMAP),
        use_container_width=True,
    )
    st.caption("Higher Value Score = more traffic relief per rupee spent. Ranked best investment first.")

    with st.expander("Show full technical table (bio-efficiency, demand pressure, etc.)"):
        value_display = ranked_value[["edge_id", "u", "v", "bio_efficiency",
                                       "estimated_cost_pkr_m", "demand_pressure", "value_ratio"]].reset_index(drop=True)
        value_display.index = value_display.index + 1
        value_display.columns = ["Edge", "From", "To", "Bio-Efficiency",
                                  "Est. Cost (PKR M)", "Demand Pressure", "Value Ratio"]
        st.caption(
            "Demand Pressure = how much unmet traffic demand is being throttled by this corridor's "
            "weakness. Value Ratio = Demand Pressure divided by cost -- the raw number behind the "
            "Value Score above."
        )
        styled_value = value_display.style.background_gradient(subset=["Value Ratio"], cmap=WARM_CMAP)
        st.dataframe(styled_value, use_container_width=True)

    st.markdown('<div class="section-tag">Simulation-verified ranking</div>', unsafe_allow_html=True)
    st.caption(
        "The table above uses a fast static formula, which can misrank corridors. "
        "This instead actually simulates widening each top candidate and re-runs the "
        "network to measure the REAL resulting efficiency gain -- slower, but the "
        "number you can defend if asked 'how do you know'."
    )
    verify_n = st.slider("Verify top N candidates", 5, 30, 15, 5)
    if st.button("🔬 Run simulation verification"):
        with st.spinner(f"Simulating widening for top {verify_n} candidates..."):
            verified = verify_top_candidates(
                edges_df, node_demands, ranked_value, decay_rate, mu, generations,
                cost_per_km, top_n=verify_n,
            )
        st.dataframe(
            verified.style.background_gradient(subset=["simulated_value_ratio"], cmap=WARM_CMAP),
            use_container_width=True,
        )
        if list(verified["edge_id"])[0] != list(ranked_value["edge_id"])[0]:
            st.warning(
                f"Note: the simulation-verified top pick ({verified.iloc[0]['edge_id']}) "
                f"differs from the static heuristic's top pick ({ranked_value.iloc[0]['edge_id']}). "
                f"Use the simulated result for your actual recommendation."
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
    st.markdown('<div class="section-tag">Corridor network</div>', unsafe_allow_html=True)
    if has_coords:
        highlighted = st.session_state.get("highlighted_edge")
        if highlighted:
            st.caption(f"Highlighting **{highlighted}** -- click a different row on the Priority Ranking tab to change this, or clear the selection there to unhighlight.")

        center_lat = edges_df[["u_lat", "v_lat"]].values.mean()
        center_lon = edges_df[["u_lon", "v_lon"]].values.mean()
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=13,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
        )
        max_eff = result["bio_efficiency"].max() or 1
        highlight_coords = None
        for row in result.itertuples():
            src = edges_df[edges_df.edge_id == row.edge_id].iloc[0]
            weight = 2 + (row.bio_efficiency / max_eff) * 8
            rank_frac = list(result["edge_id"]).index(row.edge_id) / max(len(result) - 1, 1)
            color = "#9C4A2E" if rank_frac < 0.34 else "#D8C9A8" if rank_frac < 0.67 else "#C9BFAE"
            is_highlighted = highlighted and row.edge_id == highlighted
            coords = [(src.u_lat, src.u_lon), (src.v_lat, src.v_lon)]
            # Dark casing drawn first so colored routes stay legible over
            # busy satellite imagery, same technique real map cartography uses
            folium.PolyLine(coords, color="#1B1815", weight=weight + 2.5, opacity=0.6 if not is_highlighted else 0.3).add_to(m)
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

        st_folium(m, width=None, height=480, key="main_network_map")
    else:
        st.warning(
            "No coordinates in the current dataset, so no map -- this appears "
            "once you upload a CSV from extract_gulshan_network.py or auto_discover_corridors.py."
        )

with tab_resilience:
    st.markdown('<div class="section-tag">What if a corridor is blocked?</div>', unsafe_allow_html=True)
    st.caption(
        "Pick two real intersections from your uploaded network below, then optionally block a "
        "corridor or intersection between them (simulating flooding, a protest, an accident, or "
        "construction). We'll show the best alternate route and how much worse it is than the "
        "normal route -- no external service or API needed."
    )

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
        row = edges_df[edges_df.edge_id == edge_id].iloc[0]
        name = row.get("road_name", edge_id) if has_road_name else edge_id
        return f"{name}  [{node_label(row.u)} \u2192 {node_label(row.v)}]"

    all_nodes = sorted(set(edges_df["u"]).union(set(edges_df["v"])))

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        origin = st.selectbox("From (starting point)", all_nodes, index=0, format_func=node_label)
    with rc2:
        destination = st.selectbox("To (destination)", all_nodes, index=min(1, len(all_nodes) - 1), format_func=node_label)
    with rc3:
        block_type = st.radio("What's closed?", ["Nothing yet", "A corridor (road segment)", "An intersection"])
    with rc4:
        if block_type == "A corridor (road segment)":
            block_choice = st.selectbox("Which corridor?", list(edges_df["edge_id"]), format_func=edge_label)
        elif block_type == "An intersection":
            block_choice = st.selectbox("Which intersection?", all_nodes, format_func=node_label)
        else:
            block_choice = None
            st.caption("Choose \"A corridor\" or \"An intersection\" to simulate a closure.")

    if st.button("🔍 Find route"):
        kwargs = {}
        if block_choice is not None:
            if block_type == "A corridor (road segment)":
                kwargs["blocked_edge_id"] = block_choice
            else:
                kwargs["blocked_node"] = block_choice

        route = find_alternate_route(result, origin, destination, **kwargs)

        if not route["found"]:
            st.error(
                f"No route exists between {node_label(origin)} and {node_label(destination)} "
                f"once this closure is applied. This is a real finding: your network has no "
                f"redundancy here, meaning this closure would completely cut off this connection. "
                f"That itself is worth flagging to planners."
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
                    rm = folium.Map(
                        location=route_coords[len(route_coords) // 2], zoom_start=14,
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri World Imagery",
                    )

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

                    if block_choice is not None:
                        if block_type == "A corridor (road segment)":
                            brow = edges_df[edges_df.edge_id == block_choice].iloc[0]
                            bmid = [(brow.u_lat + brow.v_lat) / 2, (brow.u_lon + brow.v_lon) / 2]
                            folium.Marker(bmid, icon=folium.Icon(color="black", icon="remove-sign"),
                                          tooltip=f"Blocked: {edge_label(block_choice)}").add_to(rm)
                        elif block_choice in node_coords:
                            folium.Marker(node_coords[block_choice], icon=folium.Icon(color="black", icon="remove-sign"),
                                          tooltip=f"Blocked: {node_label(block_choice)}").add_to(rm)

                    st_folium(rm, width=None, height=420, key="resilience_route_map")
            else:
                st.caption(
                    "No coordinates in this dataset, so no map here -- upload a CSV with "
                    "lat/lon columns (from auto_discover_corridors.py) to see routes visually."
                )

with tab_notes:
    st.markdown('<div class="section-tag">Model notes for the pitch</div>', unsafe_allow_html=True)
    st.markdown(
        """
- **Impedance** combines distance, congestion, and road class into a single
  friction score per corridor.
- **Bio-efficiency** is the converged conductance value after simulated
  Physarum-style adaptation: corridors the network can sustain flow through
  strengthen, corridors it can't (regardless of demand) weaken toward zero.
- The adaptation rule uses the saturating reinforcement function from
  Tero et al. 2010 (*Science*) -- the same mechanism used to replicate the
  Tokyo rail network with real slime mold -- rather than an invented
  damping term.
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
