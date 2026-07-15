"""
Physarum-inspired urban network optimization -- core model, v2.

v1 fix (previous version) added an ad-hoc quadratic damping term (gamma)
to stop the original formula from growing exponentially forever. It
worked, but "gamma" wasn't grounded in anything -- it was invented purely
for numerical stability.

v2 replaces that patch with the reinforcement rule actually used in the
Physarum literature (Tero, Kobayashi & Nakagaki, 2007, "A mathematical
model for adaptive transport network in path finding by true slime
mold", Journal of Theoretical Biology; and the follow-up Tero et al.
2010 Science paper that replicated the Tokyo rail network with this
exact mechanism):

    dD_ij/dt = f(Q_ij) - decay * D_ij
    f(Q) = Q^mu / (1 + Q^mu)          [saturating/Hill-type response]

f(Q) naturally bounds between 0 and 1 as flux grows, so conductance
converges to a fixed point at D* <= 1/decay without needing any invented
damping term. mu controls how sharply the tube "commits" once flux
crosses a threshold -- mu=1 is the standard mild-saturation case; higher
mu makes the network behave more like a winner-take-all switch.

This is the single source of truth imported by extraction/analysis
scripts and the Streamlit app.
"""

import pandas as pd


ROAD_OMEGA = {
    "Main Artery": 1.0,
    "Collector Road": 1.2,
    "Internal Street": 1.5,
}


def impedance(distance_km, congestion, road_class):
    """Multi-criteria impedance W_e for one edge."""
    omega = ROAD_OMEGA.get(road_class, 1.0)
    return distance_km * (1 + 1.5 * (congestion / 10.0) ** 2) * omega


def reinforcement(Q, mu=1.0):
    """Saturating reinforcement function f(Q) = Q^mu / (1 + Q^mu)."""
    if Q <= 0:
        return 0.0
    return (Q ** mu) / (1 + Q ** mu)


def run_simulation(edges_df, node_demands, decay_rate=0.15, mu=1.0, generations=30):
    """
    edges_df columns required: edge_id, u, v, distance_km, road_class, congestion
    node_demands: dict of node name -> demand multiplier
    decay_rate: baseline conductance decay per generation
    mu: saturation sharpness of the reinforcement function (see docstring)
    generations: number of adaptation steps to run

    Returns a DataFrame with the original columns plus 'impedance' and
    'bio_efficiency' (final converged conductance), sorted weakest first.
    """
    df = edges_df.copy()
    df["impedance"] = df.apply(
        lambda r: impedance(r["distance_km"], r["congestion"], r["road_class"]), axis=1
    )
    conductance = {row.edge_id: 1.0 for row in df.itertuples()}

    for _ in range(generations):
        flux = {}
        for row in df.itertuples():
            u_demand = node_demands.get(row.u, 1.0)
            v_demand = node_demands.get(row.v, 1.0)
            avg_demand = (u_demand + v_demand) / 2.0
            c = conductance[row.edge_id]
            flux[row.edge_id] = (c / row.impedance) * avg_demand

        for row in df.itertuples():
            c = conductance[row.edge_id]
            delta = reinforcement(flux[row.edge_id], mu) - (decay_rate * c)
            conductance[row.edge_id] = max(0.0, c + delta)

    df["bio_efficiency"] = df["edge_id"].map(conductance)
    return df.sort_values("bio_efficiency")


if __name__ == "__main__":
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
    print(result[["edge_id", "u", "v", "impedance", "bio_efficiency"]].to_string(index=False))
