"""
Estimate a destination-activity proxy from OpenStreetMap point-of-interest
(POI) density around network nodes.

The supported repository workflow is the full auto-discovered road network:
    python compute_node_demand.py --full full_network_nodes.csv

This writes node_demand.csv for the dashboard. POI density is not observed
trip demand: it does not identify trip origins, destinations, volumes, or
mode choice. Results depend on OSM tagging completeness and the chosen
search radius.
"""

import time
import pandas as pd
import osmnx as ox

# Tags that count as a "destination" for demand purposes. Extend this
# list if you want demand to weight certain POI types differently later.
DEMAND_TAGS = {
    "amenity": True,   # schools, hospitals, restaurants, banks, etc.
    "shop": True,       # retail
    "office": True,     # workplaces
}

RADIUS_M = 400  # how far around each landmark to count POIs
PAUSE_SECONDS = 1.0  # politeness delay between Overpass calls -- avoids rate limiting on larger node sets


def compute_poi_demand(coords, radius_m=RADIUS_M, pause=PAUSE_SECONDS):
    """
    coords: dict of node name -> (lat, lon)
    Returns: dict of node name -> demand multiplier, normalized so the
    mean across all nodes is 1.0 (keeps the scale comparable to the old
    hand-picked 0.7-1.5 range so the rest of the model doesn't need
    retuning).
    """
    raw_counts = {}
    n = len(coords)
    for i, (name, (lat, lon)) in enumerate(coords.items(), 1):
        try:
            pois = ox.features_from_point((lat, lon), tags=DEMAND_TAGS, dist=radius_m)
            raw_counts[name] = len(pois)
            print(f"  [{i}/{n}] {name:20s} -> {len(pois):4d} POIs within {radius_m}m")
        except Exception as e:
            print(f"  [{i}/{n}] WARNING: POI lookup failed for {name}: {e}")
            raw_counts[name] = 0
        if i < n:
            time.sleep(pause)

    if not raw_counts:
        return {}

    mean_count = sum(raw_counts.values()) / len(raw_counts)
    if mean_count == 0:
        print("  WARNING: no POIs found anywhere -- check radius or coordinates")
        return {name: 1.0 for name in raw_counts}

    # Normalize around 1.0, floor at 0.3 so a node never drops to zero
    # demand (a real intersection always carries some baseline traffic
    # even with few POIs nearby -- residential-only areas aren't "no
    # demand", just lower demand).
    demands = {
        name: max(0.3, round(count / mean_count, 2))
        for name, count in raw_counts.items()
    }
    return demands


def compute_poi_demand_from_nodes_csv(nodes_csv_path, center_point, radius_m,
                                       poi_radius_m=RADIUS_M, out_path="node_demand.csv"):
    """
    Efficient version for large networks (hundreds of nodes): makes ONE
    bulk POI request covering the whole study area, then counts POIs near
    each node locally (no per-node API calls). This avoids both the long
    runtime and the OSM rate-limit risk of querying node-by-node.

    nodes_csv_path: CSV with columns node, lat, lon
    center_point: (lat, lon) of the study area center (same one used in
        auto_discover_corridors.py)
    radius_m: the same radius_m used to build the network in
        auto_discover_corridors.py
    poi_radius_m: how far around each node to count POIs (default 400m)
    """
    nodes_df = pd.read_csv(nodes_csv_path)
    print(f"Fetching all POIs within {radius_m + poi_radius_m}m of {center_point} "
          f"in a single request (covers the whole study area + buffer)...")
    pois = ox.features_from_point(center_point, tags=DEMAND_TAGS, dist=radius_m + poi_radius_m)

    # Get POI centroids as plain lat/lon points
    poi_points = []
    for geom in pois.geometry:
        c = geom.centroid
        poi_points.append((c.y, c.x))
    print(f"Found {len(poi_points)} total POIs in the study area.")

    def haversine_m(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    raw_counts = {}
    n = len(nodes_df)
    for i, row in enumerate(nodes_df.itertuples(), 1):
        count = sum(
            1 for (plat, plon) in poi_points
            if haversine_m(row.lat, row.lon, plat, plon) <= poi_radius_m
        )
        raw_counts[row.node] = count
        if i % 100 == 0 or i == n:
            print(f"  [{i}/{n}] nodes processed")

    mean_count = sum(raw_counts.values()) / len(raw_counts) if raw_counts else 0
    if mean_count == 0:
        print("  WARNING: no POIs found near any node -- check radius or coordinates")
        demands = {name: 1.0 for name in raw_counts}
    else:
        demands = {name: max(0.3, round(c / mean_count, 2)) for name, c in raw_counts.items()}

    out_df = pd.DataFrame([{"node": name, "demand": val} for name, val in demands.items()])
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} node demand values to {out_path}")
    return demands


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="required: compute POI demand for the full-network node CSV",
    )
    parser.add_argument(
        "nodes_csv",
        nargs="?",
        default="full_network_nodes.csv",
        help="node CSV written by auto_discover_corridors.py",
    )
    parser.add_argument("--lat", type=float, default=None, help="Study-area center latitude.")
    parser.add_argument("--lon", type=float, default=None, help="Study-area center longitude.")
    parser.add_argument("--radius-m", type=int, default=None, help="Network radius in meters.")
    parser.add_argument("--poi-radius-m", type=int, default=RADIUS_M, help="POI count radius around each node.")
    parser.add_argument("--out", default="node_demand.csv", help="Output demand CSV.")
    args = parser.parse_args()
    if not args.full:
        parser.error(
            "the landmark extraction workflow is not included in this repository; "
            "run with --full after auto_discover_corridors.py"
        )

    # Must match the CENTER_POINT and RADIUS_M used by auto_discover_corridors.py.
    from auto_discover_corridors import CENTER_POINT, RADIUS_M

    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be supplied together")
    center = (args.lat, args.lon) if args.lat is not None else CENTER_POINT
    radius = args.radius_m if args.radius_m is not None else RADIUS_M
    if radius <= 0:
        parser.error("--radius-m must be positive")
    compute_poi_demand_from_nodes_csv(
        args.nodes_csv, center, radius, poi_radius_m=args.poi_radius_m, out_path=args.out
    )
