# Physarum-inspired urban network optimization

YIC-26, Sindh HEC. Gulshan-e-Iqbal, District East, Karachi.

## What's in this folder

- `physarum_model.py` — the core algorithm (single source of truth, imported
  by everything else). Fixed to actually converge: the original formulation
  compounded exponentially rather than settling to a stable value; this
  version adds a self-limiting term (gamma) so bio-efficiency scores reach a
  genuine fixed point.
- `extract_gulshan_network.py` — pulls the real Gulshan-e-Iqbal road network
  from OpenStreetMap: real distances, real road classes. Run this in
  **Google Colab**, not locally in a restricted environment, since it needs
  live access to OSM's geocoding and routing services.
- `app.py` — the Streamlit dashboard: interactive sliders, ranked priority
  list, and a map of corridors colored/weighted by bio-efficiency.
- `sample_edges.csv` — the validated 6-corridor dataset, used as a fallback
  so the dashboard works out of the box before you've collected real data.
- `requirements.txt` — dependencies for both steps.

## Step 1 — get real road data (Colab)

**Option A — the 8 named landmarks** (what you've already done):
1. Open a new Colab notebook.
2. First cell: `!pip install osmnx -q`
3. Upload `extract_gulshan_network.py` and run it. Fix any geocoding
   failures with `NODE_COORDS_OVERRIDE` (see comments in the script).

**Option B — the full corridor network (recommended for a stronger
submission)**: instead of relying on 8 hand-named landmarks, auto-discover
every primary/secondary/tertiary road segment in the area:
1. Same Colab setup (`!pip install osmnx -q`).
2. Upload and run `auto_discover_corridors.py`. It downloads the entire
   drivable network for Gulshan-e-Iqbal, filters to structural roads
   (skips residential streets), and writes:
   - `full_network_edges.csv` — every discovered corridor, with
     `congestion` defaulted to 5 (a flat placeholder — see step 2)
   - `full_network_nodes.csv` — every intersection, with real coordinates
3. This gives you dozens of real corridors and a genuinely connected
   network (with actual redundancy/alternate routes), instead of the
   original 6 hand-picked ones that turned out to be two disconnected
   chains with zero redundancy.

## Step 1b — real node demand for the full network

Node demand (commuter pull) can't stay hand-picked once you have dozens
of generic intersections. Run, still in Colab:
```
python compute_node_demand.py --full full_network_nodes.csv
```
This pulls real point-of-interest density (shops, offices, schools, etc.)
around each intersection from OSM and writes `node_demand.csv`. It takes
a few minutes on a large network due to deliberate pacing between
requests (avoids Overpass API rate limits) — that's expected, not a bug.

## Step 2 — fill in congestion (manual, and that's OK)

OpenStreetMap has no live traffic layer, and free real-time traffic APIs
don't reliably cover Karachi yet.

- **8-landmark dataset**: sample all 6 corridors by hand from Google Maps
  traffic coloring at rush hour.
- **Full auto-discovered network** (dozens of corridors): manually
  surveying every single one isn't realistic. Two-pass approach: run the
  model once with the flat default (5) for everything to see the
  topology-driven ranking, then manually refine congestion via Google
  Maps only for your top 10-15 lowest-bio-efficiency candidates — the
  ones that actually matter for your widening recommendation. State this
  plainly in your report: congestion was refined for priority candidates,
  not exhaustively surveyed. That's honest and still rigorous.

## Step 3 — run the dashboard

```
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar: upload your edges CSV, and if you used the full
auto-discovered network, also upload `node_demand.csv` (the built-in
8-landmark demand values won't match generic `N<id>` node names).
Otherwise leave the sample dataset checked to demo immediately.

The dashboard now includes:
- **Priority ranking** — bio-efficiency per corridor (weakest = top
  priority for widening)
- **Cost-benefit ranking** — value ratio (relief per PKR million spent),
  with editable cost-per-km assumptions in the sidebar
- **Resilience routing** — pick an origin, destination, and a corridor or
  intersection to block; finds the best alternate route using real graph
  shortest-path search (free, no external API needed)

Deploy to Streamlit Community Cloud the same way you deployed the
microscope app: push this folder to a GitHub repo, connect it on
share.streamlit.io, point it at `app.py`.

## Honest caveats to state in your pitch

- The self-limiting coefficient (gamma) is a modeling choice for stability,
  not a measured biological constant — say so if asked.
- Congestion values are manually sampled from visible traffic data, not a
  live feed, until a traffic API partnership is in place (pitch this as
  Phase 2).
- Node demand multipliers (1.2, 1.5, etc.) are currently assumed based on
  land use; a stronger version would derive these from population density
  or footfall counts if that data becomes available.
