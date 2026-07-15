"""
A static baseline for honestly testing whether the iterative Physarum
adaptation simulation actually earns its complexity, versus a much
simpler method a traffic engineer could compute directly.

The baseline is a single-step score:
    vc_score = avg_demand / impedance
This is exactly the Physarum "flux" term computed ONCE, with no
iteration, no decay, no reinforcement dynamics. That makes it a fair,
apples-to-apples comparison -- same inputs, same impedance function,
the only difference is whether 30 generations of adaptation change the
answer. This is a deliberate choice: comparing against a strawman
baseline would prove nothing.

IMPORTANT direction note (caught and fixed during testing -- worth
documenting so it isn't reintroduced): in the iterative model, a HIGH
demand/impedance ratio relative to the decay rate drives conductance
UP toward a strong, reinforced steady state (high bio_efficiency = NOT
a priority corridor). A LOW ratio causes conductance to decay toward
zero (low bio_efficiency = priority corridor). So the fair single-step
analog uses the SAME direction as bio_efficiency: LOW vc_score = weak
= priority. Sorting the other way (as if this were a real-world V/C
congestion ratio, where high = bad) silently compares two DIFFERENT
questions and produces a misleading near-total inverse correlation
that looks like a dramatic finding but is actually a direction bug.
"""

import pandas as pd
from physarum_model import impedance


def compute_static_baseline(edges_df, node_demands):
    """
    edges_df: DataFrame with edge_id, u, v, distance_km, road_class, congestion
    node_demands: dict of node -> demand multiplier

    Returns edges_df with 'impedance' and 'vc_score' columns added,
    sorted LOWEST vc_score (highest priority) first -- same convention
    as bio_efficiency, see module docstring for why.
    """
    df = edges_df.copy()
    df["impedance"] = df.apply(
        lambda r: impedance(r["distance_km"], r["congestion"], r["road_class"]), axis=1
    )

    def vc(row):
        avg_demand = (node_demands.get(row["u"], 1.0) + node_demands.get(row["v"], 1.0)) / 2.0
        return avg_demand / row["impedance"]

    df["vc_score"] = df.apply(vc, axis=1)
    return df.sort_values("vc_score", ascending=True)


def compare_rankings(bio_result, baseline_result, top_n=10):
    """
    bio_result: output of physarum_model.run_simulation() (has bio_efficiency)
    baseline_result: output of compute_static_baseline() (has vc_score)
    top_n: how many top-priority corridors to check for overlap

    Returns a dict with:
        merged             : DataFrame with both methods' rank per corridor
        spearman_corr      : rank correlation, -1 to 1. Near 1 = the two
                              methods agree almost completely (the extra
                              simulation complexity isn't changing much).
                              Notably lower = they genuinely disagree on
                              priority order (the simulation IS adding
                              something a static formula misses).
        top_n_overlap       : how many corridors both methods place in
                              their own top N
        top_n               : the N used above
        most_divergent      : corridors ranked most differently between
                              the two methods, most-divergent first --
                              these are the specific, checkable cases to
                              look at by hand
    """
    bio_sorted = bio_result.sort_values("bio_efficiency", ascending=True).reset_index(drop=True)
    bio_sorted["bio_rank"] = bio_sorted.index + 1

    base_sorted = baseline_result.sort_values("vc_score", ascending=True).reset_index(drop=True)
    base_sorted["baseline_rank"] = base_sorted.index + 1

    merged = bio_sorted[["edge_id", "u", "v", "bio_efficiency", "bio_rank"]].merge(
        base_sorted[["edge_id", "vc_score", "baseline_rank"]], on="edge_id"
    )

    n = len(merged)
    spearman_corr = merged["bio_rank"].corr(merged["baseline_rank"], method="spearman") if n > 2 else None

    top_n = min(top_n, n)
    bio_top = set(bio_sorted.head(top_n)["edge_id"])
    base_top = set(base_sorted.head(top_n)["edge_id"])
    overlap = len(bio_top & base_top)

    merged["rank_diff"] = (merged["bio_rank"] - merged["baseline_rank"]).abs()
    most_divergent = merged.sort_values("rank_diff", ascending=False)

    return {
        "merged": merged,
        "spearman_corr": spearman_corr,
        "top_n_overlap": overlap,
        "top_n": top_n,
        "n_corridors": n,
        "most_divergent": most_divergent,
    }


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

    bio_result = run_simulation(edges, node_demands)
    baseline_result = compute_static_baseline(edges, node_demands)
    comparison = compare_rankings(bio_result, baseline_result, top_n=3)

    print("=== Full simulation ranking ===")
    print(bio_result[["edge_id", "u", "v", "bio_efficiency"]].to_string(index=False))
    print("\n=== Static baseline ranking ===")
    print(baseline_result[["edge_id", "u", "v", "vc_score"]].to_string(index=False))
    print(f"\nSpearman rank correlation: {comparison['spearman_corr']:.3f}")
    print(f"Top-{comparison['top_n']} overlap: {comparison['top_n_overlap']}/{comparison['top_n']}")
    print("\nMost divergent corridors:")
    print(comparison["most_divergent"][["edge_id", "u", "v", "bio_rank", "baseline_rank", "rank_diff"]].to_string(index=False))
