"""
Cost and model-score heuristics for scenario screening.

The default PKR/km values are unverified placeholder assumptions. The
demand_pressure/value_ratio values are heuristic ratios, not estimates of
traveler delay, congestion relief, or economic return. The intervention
scenario function changes model inputs and measures the resulting internal
conductance-score change; it does not simulate engineering capacity,
vehicle flows, travel times, or benefits. Replace these assumptions and
outputs with locally sourced costs and a validated traffic assignment
model before using them for investment decisions.
"""

import pandas as pd

# PKR millions per km. These are order-of-magnitude estimates grounded in
# published Pakistani road-cost references, NOT a quote for this specific
# project -- real costs depend heavily on land acquisition, drainage, and
# utility relocation, which vary block by block and aren't in this model.
#
# Sources checked (July 2026):
#   - NHA-cited dual-lane road: ~Rs111M/km; four-lane: ~Rs170M/km
#     (Dawn, PM Imran Khan road-cost briefing)
#   - Lahore 3-lane dual carriageway incl. storm drain: ~Rs450M/km
#     (dense urban context with drainage -- closest analog to Karachi)
#   - KPK MRS-2020 standard, pavement layers only, no land acquisition:
#     ~Rs32.5M/km (this is a FLOOR, not a realistic full-widening cost)
#
# Main Artery here assumes urban widening with drainage (closer to the
# Lahore figure than the bare NHA highway figure, since Gulshan-e-Iqbal
# is dense urban, not open highway). Collector/Internal scaled down
# proportionally. Replace these with actual KDA/SULTA tender figures for
# the real submission if you can get them -- these are defensible
# estimates, not verified project costs.
DEFAULT_COST_PER_KM = {
    "Main Artery": 250.0,
    "Collector Road": 130.0,
    "Internal Street": 55.0,
}


def add_cost_benefit(result_df, node_demands, cost_per_km=None):
    """
    Fast heuristic ranking for the FULL table (hundreds/thousands of rows).
    result_df: output of physarum_model.run_simulation()
    node_demands: dict of node -> demand multiplier

    Adds:
        estimated_cost_pkr_m : rough widening cost estimate
        demand_pressure      : avg_demand / bio_efficiency -- a proxy ratio
                                with no direct interpretation as unmet trips
        value_ratio           : demand_pressure / cost -- fast to compute
                                for every row, but a heuristic; it is
                                not a traffic-relief or benefit estimate.
                                verify_top_candidates() measures only the
                                model-score response to assumed input changes.
    """
    cost_map = cost_per_km or DEFAULT_COST_PER_KM
    df = result_df.copy()

    df["estimated_cost_pkr_m"] = df.apply(
        lambda r: r["distance_km"] * cost_map.get(r["road_class"], 90.0), axis=1
    )

    def pressure(row):
        avg_demand = (node_demands.get(row["u"], 1.0) + node_demands.get(row["v"], 1.0)) / 2.0
        return avg_demand / max(row["bio_efficiency"], 0.01)

    df["demand_pressure"] = df.apply(pressure, axis=1)
    df["value_ratio"] = df["demand_pressure"] / df["estimated_cost_pkr_m"]

    return df.sort_values("value_ratio", ascending=False)


ROAD_CLASS_UPGRADE = {
    "Internal Street": "Collector Road",
    "Collector Road": "Main Artery",
    "Main Artery": "Main Artery",  # already top tier
}


def simulate_widening(edges_df, edge_id, congestion_reduction_frac=0.4, upgrade_class=True):
    """
    Models a concrete widening intervention on ONE corridor:
      - road class upgraded one tier (Internal -> Collector -> Main Artery)
      - input congestion score reduced by congestion_reduction_frac
        (default 40%), floored at 1

    These are hypothetical input perturbations, not engineering effects.
    They do not establish that widening causes this congestion reduction.
    Returns a modified copy of edges_df.
    """
    df = edges_df.copy()
    mask = df["edge_id"] == edge_id
    if not mask.any():
        return df
    if upgrade_class:
        df.loc[mask, "road_class"] = df.loc[mask, "road_class"].map(
            lambda rc: ROAD_CLASS_UPGRADE.get(rc, rc)
        )
    df.loc[mask, "congestion"] = df.loc[mask, "congestion"].apply(
        lambda c: max(1, round(c * (1 - congestion_reduction_frac)))
    )
    return df


def verify_top_candidates(edges_df, node_demands, ranked_df, decay_rate, mu, generations,
                           cost_per_km=None, top_n=15, simulation_fn=None):
    """
    For each of the top_n heuristic candidates, apply the assumed input
    changes and re-run the selected model. The returned simulated_gain is
    a change in internal bio_efficiency, not measured or forecast traffic
    relief, travel-time savings, or economic return. This is scenario
    sensitivity analysis, not empirical validation.

    simulation_fn: which model to re-run with. Defaults to the
        independent-edge model (physarum_model.run_simulation) for
        backward compatibility. Pass a wrapped version of
        physarum_network_model.run_network_simulation to keep this
        consistent with network-coupled mode -- otherwise the score changes would
        compare different model configurations. Must have the
        signature (edges_df, node_demands, decay_rate, mu, generations)
        -> DataFrame with a bio_efficiency column.

    Returns model-score changes plus heuristic costs and ratios, sorted
    by simulated_value_ratio. Column names are retained for compatibility;
    none of these fields is a measured benefit.
    """
    if simulation_fn is None:
        from physarum_model import run_simulation as simulation_fn

    cost_map = cost_per_km or DEFAULT_COST_PER_KM
    eligible = ranked_df[ranked_df["bio_efficiency"].notna()]
    candidates = eligible.sort_values("value_ratio", ascending=False).head(top_n)

    rows = []
    for row in candidates.itertuples():
        modified_edges = simulate_widening(edges_df, row.edge_id)
        after_result = simulation_fn(modified_edges, node_demands, decay_rate, mu, generations)
        after_row = after_result[after_result.edge_id == row.edge_id].iloc[0]

        cost = row.distance_km * cost_map.get(row.road_class, 90.0)
        gain = after_row["bio_efficiency"] - row.bio_efficiency

        rows.append({
            "edge_id": row.edge_id,
            "u": row.u, "v": row.v,
            "bio_efficiency_before": round(row.bio_efficiency, 3),
            "bio_efficiency_after": round(after_row["bio_efficiency"], 3),
            "simulated_gain": round(gain, 3),
            "estimated_cost_pkr_m": round(cost, 1),
            "simulated_value_ratio": round(gain / cost, 6) if cost > 0 else 0,
        })

    return pd.DataFrame(rows).sort_values("simulated_value_ratio", ascending=False)


if __name__ == "__main__":
    from physarum_model import run_simulation

    node_demands = {
        "AKU": 1.2, "Kashmir Park": 1.5, "Hassan Square": 1.2, "FUAAST": 0.7,
        "NIPA": 1.5, "Gulshan Chowrangi": 1.2, "Maskan": 0.9, "KU Gate": 0.7,
    }
    edges = pd.DataFrame([
        ["E1", "AKU", "Kashmir Park", 1.4, "Main Artery", 8],
        ["E3", "Kashmir Park", "Hassan Square", 1.4, "Main Artery", 8],
        ["E4", "Hassan Square", "FUAAST", 2.7, "Main Artery", 8],
        ["E7", "FUAAST", "NIPA", 1.2, "Main Artery", 4],
        ["E10", "Gulshan Chowrangi", "Maskan", 2.3, "Main Artery", 8],
        ["E14", "Maskan", "KU Gate", 2.6, "Main Artery", 9],
    ], columns=["edge_id", "u", "v", "distance_km", "road_class", "congestion"])

    result = run_simulation(edges, node_demands)
    ranked = add_cost_benefit(result, node_demands)
    print(ranked[["edge_id", "u", "v", "bio_efficiency", "estimated_cost_pkr_m",
                   "demand_pressure", "value_ratio"]].to_string(index=False))
