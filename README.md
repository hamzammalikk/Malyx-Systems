# Malyx Systems: Physarum-inspired road-network screening

Research prototype for exploring road-network structure and hypothetical
corridor scenarios in Gulshan-e-Iqbal, Karachi. It is a decision-support
experiment, not a validated traffic forecast or engineering recommendation.

## Current workflow

1. In Google Colab, install the OSM extraction dependency and run
   `auto_discover_corridors.py`:
   ```bash
   pip install osmnx pandas
   python auto_discover_corridors.py
   ```
   This writes `full_network_edges.csv` and `full_network_nodes.csv`.
2. Estimate a destination-activity proxy around nodes:
   ```bash
   python compute_node_demand.py --full full_network_nodes.csv
   ```
   This writes `node_demand.csv`. POI density is not observed trip demand.
3. Fill or replace the default congestion values with a sourced, dated
   dataset. The script sets them to 5 as placeholders.
4. Install app dependencies and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Upload the edge CSV and matching node-demand CSV in the dashboard.

The repository currently documents the auto-discovery workflow. The older
`extract_gulshan_network.py` landmark workflow referenced in earlier
versions is not present on the default branch.

## What the models calculate

- `physarum_model.py` is an independent-edge baseline. It scores each edge
  from endpoint-demand proxies and impedance without solving network flow.
- `physarum_network_model.py` solves a Kirchhoff-style flow problem within
  the selected sink's connected component. Each edge uses effective friction
  `W = distance × (1 + 1.5 × (congestion / 10)^2) × road_class_factor`;
  flow conductance is `D / W`. Conductance then adapts using a normalized
  saturating response.
- The sink defaults to the node with the highest POI-demand proxy. Other
  nodes inject flow in proportion to that proxy. This is an explicit
  single-sink scenario, not an origin-destination matrix.
- The adaptation solver is undirected. A `two_way` column is respected by
  the routing feature, but one-way restrictions are not enforced in the
  Physarum flow solve.
- The model stops early if the maximum relative conductance change stays
  below tolerance for three consecutive iterations after at least ten steps;
  otherwise it reaches the iteration limit. Numerical stability does not
  validate the result against observed traffic or imply a global optimum.
- Disconnected edges are scored with the independent-edge model and marked
  in the result notes. Do not treat those scores as directly comparable to
  coupled scores without separate analysis.

The saturated reinforcement response has precedent in Physarum modeling,
including the multi-food-source model of Tero et al. (2008). The current
normalization and parameter values are project choices, not values calibrated
from biological measurements or Karachi traffic data.

## Interpreting dashboard outputs

- A low conductance score means the modeled adaptation did not reinforce the
  edge under the chosen inputs. It is a structural screening signal; by itself
  it does not show that the road is congested, under-capacity, or a widening
  priority.
- The cost-benefit tab uses assumed construction costs and a demand-to-score
  ratio. It is a heuristic, not traffic relief per rupee or a return-on-
  investment estimate.
- The scenario tool changes road-class and congestion inputs, reruns the
  selected model, and reports changes in its internal conductance score.
  It does not model the engineering effect of widening or forecast vehicle
  delay, flow, safety, or economic benefits.
- Routing finds a shortest path using the supplied impedance. It does not
  model queueing, time-dependent traffic, or capacity constraints.

## Data and validation limitations

- POI density is a destination-activity proxy. It does not estimate where
  trips start, the number of trips, mode choice, or an origin-destination
  matrix.
- Congestion values are placeholders until replaced by observations. Keep
  source, timestamp, and collection method with any supplied values.
- OpenStreetMap completeness varies by place. Review extracted geometry,
  road classes, direction, names, and missing links before analysis.
- Construction costs are order-of-magnitude placeholders, not local tender
  estimates or site-specific widening costs.
- Validate rankings against independent traffic counts, travel times, and a
  transparent baseline before using them to support public investment
  decisions.

## Research references

- Tero, Kobayashi & Nakagaki (2007), “A mathematical model for adaptive
  transport network in path finding by true slime mold,” *Journal of
  Theoretical Biology* 244, 553–564.
  https://doi.org/10.1016/j.jtbi.2006.07.015
- Tero et al. (2008), “Flow-network adaptation in Physarum amoebae,”
  *Theory in Biosciences* 127, 89–94.
  https://doi.org/10.1007/s12064-008-0037-9
- Tero et al. (2010), “Rules for biologically inspired adaptive network
  design,” *Science* 327, 439–442.
  https://doi.org/10.1126/science.1177894

OpenStreetMap data requires attribution. Review the ODbL and data-provider
terms before distributing derived data or operating a public service:
https://www.openstreetmap.org/copyright
