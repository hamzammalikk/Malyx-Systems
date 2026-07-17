"""
Real roads in OpenStreetMap are split into many segments, one per
intersection crossed -- necessary for accurate routing, but it means a
single physical road (e.g. "University Road") can appear as ten or more
separate rows in the corridor tables, each needing its own congestion
value. This module aggregates segment-level data back up to the
road-name level for reporting and bulk data entry, without touching the
underlying fine-grained routing graph (which needs the full segment
detail to route correctly).
"""

import pandas as pd


def aggregate_by_road_name(result_df):
    """
    result_df: output of run_simulation() or run_network_simulation()
        (has road_name, distance_km, bio_efficiency, congestion columns)

    Returns one row per unique road_name with:
        segment_count       : how many OSM segments make up this road
        total_distance_km   : summed length across all segments
        worst_bio_efficiency: the minimum (weakest) segment's score --
                              a road is only as strong as its weakest link
        avg_congestion      : mean congestion across its segments
        segment_edge_ids    : list of the underlying edge_ids, for lookup
    Sorted weakest (worst_bio_efficiency) first, same convention as the
    per-segment ranking.
    """
    if "road_name" not in result_df.columns:
        raise ValueError(
            "This dataset has no road_name column -- run auto_discover_corridors.py "
            "(not extract_gulshan_network.py) to get real street names first."
        )

    rows = []
    for name, group in result_df.groupby("road_name"):
        if name.startswith("Unnamed"):
            # Each "unnamed" segment is likely a genuinely distinct minor
            # connector, not fragments of one named road -- aggregating
            # them together would be misleading, so keep them separate.
            for _, seg in group.iterrows():
                rows.append({
                    "road_name": seg.get("road_name", "Unnamed"),
                    "segment_count": 1,
                    "total_distance_km": seg["distance_km"],
                    "worst_bio_efficiency": seg["bio_efficiency"],
                    "avg_congestion": seg["congestion"],
                    "segment_edge_ids": [seg["edge_id"]],
                })
            continue
        rows.append({
            "road_name": name,
            "segment_count": len(group),
            "total_distance_km": round(group["distance_km"].sum(), 3),
            "worst_bio_efficiency": group["bio_efficiency"].min(),
            "avg_congestion": round(group["congestion"].mean(), 1),
            "segment_edge_ids": list(group["edge_id"]),
        })

    return pd.DataFrame(rows).sort_values("worst_bio_efficiency")


def bulk_set_congestion(edges_df, road_name, new_congestion):
    """
    Set the same congestion value for every segment sharing road_name --
    fill it in once per real road instead of once per OSM segment.
    Returns a modified copy of edges_df.
    """
    df = edges_df.copy()
    mask = df["road_name"] == road_name
    df.loc[mask, "congestion"] = new_congestion
    return df


def bulk_set_congestion_multi(edges_df, road_congestion_map):
    """
    Apply a DIFFERENT congestion value to each of several roads at once --
    for batch-editing many roads in a single pass instead of repeating the
    single-road update one road at a time.

    road_congestion_map: dict of road_name -> new_congestion value
    Returns a modified copy of edges_df, plus a dict of
    {road_name: segments_updated_count} for confirming what actually changed.
    """
    df = edges_df.copy()
    counts = {}
    for road_name, new_congestion in road_congestion_map.items():
        mask = df["road_name"] == road_name
        counts[road_name] = int(mask.sum())
        df.loc[mask, "congestion"] = new_congestion
    return df, counts


if __name__ == "__main__":
    # Quick self-test with a synthetic road split into segments, like a
    # real OSM-extracted road crossing several intersections
    from physarum_model import run_simulation

    node_demands = {f"N{i}": 1.0 for i in range(6)}
    edges = pd.DataFrame([
        ["C1", "N0", "N1", 0.3, "Main Artery", "University Road", 7],
        ["C2", "N1", "N2", 0.4, "Main Artery", "University Road", 8],
        ["C3", "N2", "N3", 0.2, "Main Artery", "University Road", 6],
        ["C4", "N3", "N4", 0.5, "Collector Road", "Rashid Minhas Road", 5],
        ["C5", "N4", "N5", 1.2, "Internal Street", "Unnamed Internal Street (#5)", 3],
    ], columns=["edge_id", "u", "v", "distance_km", "road_class", "road_name", "congestion"])

    result = run_simulation(edges, node_demands)
    aggregated = aggregate_by_road_name(result)
    print("Per-segment view (what you see today):")
    print(result[["edge_id", "road_name", "distance_km", "bio_efficiency"]].to_string(index=False))
    print("\nAggregated per-road view (what this module adds):")
    print(aggregated.to_string(index=False))

    print("\nBulk congestion update test: setting all 'University Road' segments to congestion=9")
    updated = bulk_set_congestion(edges, "University Road", 9)
    print(updated[updated.road_name == "University Road"][["edge_id", "congestion"]].to_string(index=False))
