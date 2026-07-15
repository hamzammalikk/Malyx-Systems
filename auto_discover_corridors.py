"""
Auto-discover the full corridor network for Gulshan-e-Iqbal, instead of
relying on 8 hand-picked landmark names.

RUN THIS IN COLAB -- needs live OSM access.

What this does differently from extract_gulshan_network.py:
  - That script needed YOU to name each landmark and hoped Nominatim
    could find it (which is why Kashmir Park and KU Gate failed).
  - This script instead pulls every primary/secondary/tertiary road
    segment in the area automatically, so nothing depends on guessing
    landmark names. You get every real structural corridor, not just
    the ones you thought to name.

Scope choice: filtered to primary/secondary/tertiary roads only (not
residential streets). That's a deliberate choice -- these are the roads
that actually carry cross-neighborhood commuter traffic and are
plausible widening/bypass candidates. Residential streets rarely are.
If you want them included later, add 'residential' to ROAD_FILTER.

Congestion at this scale: you will NOT be able to manually survey every
corridor's traffic by hand once you have 30-50+ edges instead of 6.
Recommended two-pass approach:
  1. Run this script, run the model once with a flat default congestion
     value (say 5) for everything, just to see the topology-driven
     ranking.
  2. Manually verify/refine congestion (via Google Maps, as before) only
     for your top 10-15 lowest-bio-efficiency candidates -- the ones
     that matter for your actual widening recommendation. Say plainly in
     your report that congestion was refined for priority candidates,
     not exhaustively surveyed -- that's honest and still rigorous.
"""

import osmnx as ox
import pandas as pd

# Centered on the average of your 7 already-geocoded landmarks. Adjust if
# your actual study area is centered elsewhere.
CENTER_POINT = (24.91536, 67.08332)
RADIUS_M = 2500  # 2.5km radius -- adjust up/down to control network size.
                  # A named-place query ("Gulshan-e-Iqbal, Karachi, Pakistan")
                  # can silently match a much larger OSM boundary than you
                  # intend (in testing this returned 4000+ corridors --
                  # several neighborhoods' worth, not one). A radius around
                  # a known center point gives you direct control over scope.
ROAD_FILTER = ["primary", "secondary", "tertiary"]

HIGHWAY_TO_CLASS = {
    "motorway": "Main Artery", "trunk": "Main Artery", "primary": "Main Artery",
    "secondary": "Collector Road", "tertiary": "Collector Road",
}

DEFAULT_CONGESTION = 5  # flat placeholder -- refine top candidates by hand (see docstring)


def discover_network(center_point=CENTER_POINT, radius_m=RADIUS_M):
    print(f"Downloading drivable network within {radius_m}m of {center_point}")
    G_full = ox.graph_from_point(center_point, dist=radius_m, network_type="drive")

    # Keep only edges on the road classes we care about
    edges_to_keep = []
    for u, v, k, data in G_full.edges(keys=True, data=True):
        hwy = data.get("highway", "residential")
        if isinstance(hwy, list):
            hwy = hwy[0]
        if hwy in ROAD_FILTER:
            edges_to_keep.append((u, v, k))

    G = G_full.edge_subgraph(edges_to_keep).copy()
    print(f"Full network: {len(G_full.edges)} segments -> "
          f"filtered to {len(G.edges)} major-road segments")
    if len(G.nodes) > 300:
        print(f"WARNING: {len(G.nodes)} nodes is a lot -- compute_node_demand.py "
              f"will take a long time and may hit rate limits. Consider reducing "
              f"radius_m (currently {radius_m}m).")
    return G


def build_edges_csv(G, out_path="full_network_edges.csv"):
    """
    IMPORTANT: osmnx already builds G as a directed graph that correctly
    respects OpenStreetMap's oneway tags -- a one-way street only has an
    edge in the legal direction of travel, a two-way street has edges both
    ways. Earlier versions of this script deduplicated edges by a
    direction-blind sorted (u, v) pair, which silently destroyed this
    information and made every corridor look bidirectional downstream.
    This version preserves real direction and records whether the reverse
    direction is also legal, so routing.py can build a correctly directed
    graph instead of assuming every road can be driven both ways.
    """
    rows = []
    seen_directed_pairs = set()
    edge_counter = 1

    for u, v, data in G.edges(data=True):
        if (u, v) in seen_directed_pairs:
            continue  # duplicate parallel segment in the same direction
        seen_directed_pairs.add((u, v))

        hwy = data.get("highway", "tertiary")
        if isinstance(hwy, list):
            hwy = hwy[0]
        road_class = HIGHWAY_TO_CLASS.get(hwy, "Collector Road")

        length_m = data.get("length", None)
        if length_m is None:
            continue

        u_lat, u_lon = G.nodes[u]["y"], G.nodes[u]["x"]
        v_lat, v_lon = G.nodes[v]["y"], G.nodes[v]["x"]

        # Real street name from OSM, when tagged -- this is what turns
        # "N4471982893" into "University Road" in your tables.
        road_name = data.get("name", None)
        if isinstance(road_name, list):
            road_name = road_name[0]
        if not road_name:
            road_name = f"Unnamed {road_class} (#{edge_counter})"

        # Whether the reverse direction (v -> u) is also legal to drive.
        # Read from the graph's actual directed structure (which osmnx
        # already built correctly from OSM's oneway tags) rather than
        # re-parsing the raw tag ourselves, which has inconsistent values
        # ("yes", "-1", "reversible", etc.) across different mappers.
        two_way = G.has_edge(v, u)

        rows.append({
            "edge_id": f"C{edge_counter}",
            "u": f"N{u}", "v": f"N{v}",
            "road_name": road_name,
            "u_lat": u_lat, "u_lon": u_lon,
            "v_lat": v_lat, "v_lon": v_lon,
            "distance_km": round(length_m / 1000, 3),
            "road_class": road_class,
            "congestion": DEFAULT_CONGESTION,
            "two_way": two_way,
        })
        edge_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    named = (~df["road_name"].str.startswith("Unnamed")).sum()
    one_way = (~df["two_way"]).sum()
    print(f"\nWrote {len(df)} corridors to {out_path} ({named}/{len(df)} have a real OSM street name)")
    print(f"{one_way}/{len(df)} corridors are one-way only -- routing.py now respects this correctly.")
    print("Use the 'road_name' column instead of raw node IDs when presenting results.")
    return df


def build_nodes_csv(G, out_path="full_network_nodes.csv"):
    """
    For feeding into compute_node_demand.py at full scale. Also derives a
    human-readable junction label from the names of streets meeting at
    that node (e.g. "University Road / NIPA Chowrangi"), instead of a
    raw OSM ID -- built entirely from data already downloaded, no extra
    API calls needed.
    """
    rows = []
    for n, data in G.nodes(data=True):
        street_names = set()
        for _, _, edge_data in G.edges(n, data=True):
            name = edge_data.get("name")
            if isinstance(name, list):
                street_names.update(name)
            elif name:
                street_names.add(name)
        label = " / ".join(sorted(street_names)) if street_names else f"Unnamed Junction (N{n})"

        rows.append({"node": f"N{n}", "label": label, "lat": data["y"], "lon": data["x"]})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} nodes to {out_path} (with derived junction labels where street names exist)")
    return df


if __name__ == "__main__":
    G = discover_network()
    build_edges_csv(G)
    build_nodes_csv(G)
