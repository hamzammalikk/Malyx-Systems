"""
Physarum-inspired conductance adaptation and an independent-edge baseline.

The reinforcement response is a normalized saturating form,
f(Q) = Q^mu / (1 + Q^mu), related to the saturating response studied for
multi-food-source Physarum networks by Tero et al. (2008). The normalization
and its parameters are modeling choices here; they have not been calibrated
to biological or urban traffic observations.

The adaptation step is:
    D_e <- max(0, D_e + f(|Q_e|) - decay_rate * D_e)

This module's run_simulation() is an independent-edge approximation: each
edge uses its own mean endpoint-demand proxy and impedance. It does not solve
network flow conservation. Use physarum_network_model.run_network_simulation
for the network-coupled model. That model also has explicit single-sink and
undirected-network assumptions.

The number of iterations is fixed by the caller. No convergence criterion is
currently evaluated, so results are final-iteration conductance scores, not
guaranteed equilibria.

References:
- Tero, Kobayashi & Nakagaki (2007), J. Theor. Biol. 244, 553-564.
  https://doi.org/10.1016/j.jtbi.2006.07.015
- Tero et al. (2008), Theory in Biosciences 127, 89-94.
  https://doi.org/10.1007/s12064-008-0037-9
- Tero et al. (2010), Science 327, 439-442.
  https://doi.org/10.1126/science.1177894
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
    """Normalized saturating response f(|Q|) = |Q|^mu / (1 + |Q|^mu)."""
    if Q <= 0:
        return 0.0
    return (Q ** mu) / (1 + Q ** mu)


def run_simulation(edges_df, node_demands, decay_rate=0.15, mu=1.0,
                   generations=30, tolerance=1e-5, min_generations=10,
                   stable_iterations=3):
    """
    edges_df columns required: edge_id, u, v, distance_km, road_class, congestion
    node_demands: dict of node name -> demand multiplier
    decay_rate: baseline conductance decay per generation
    mu: saturation sharpness of the reinforcement function (see docstring)
    generations: maximum adaptation steps
    tolerance: relative maximum-conductance change required for stability
    min_generations: minimum steps before convergence can be declared
    stable_iterations: consecutive stable steps required to stop

    Returns the input columns plus 'impedance', 'bio_efficiency',
    'iterations_run', and 'converged'. Convergence means only numerical
    stability under this model's update rule, not validation against traffic.
    This independent-edge baseline is not a network flow solve.
    """
    df = edges_df.copy()
    df["impedance"] = df.apply(
        lambda r: impedance(r["distance_km"], r["congestion"], r["road_class"]), axis=1
    )
    conductance = {row.edge_id: 1.0 for row in df.itertuples()}

    stable_count = 0
    converged = False
    iterations_run = 0

    for generation in range(generations):
        previous = conductance.copy()
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

        iterations_run = generation + 1
        scale = max(1.0, max(conductance.values(), default=0.0))
        max_change = max(
            (abs(conductance[eid] - previous[eid]) for eid in conductance),
            default=0.0,
        )
        if generation + 1 >= min_generations and max_change <= tolerance * scale:
            stable_count += 1
            if stable_count >= stable_iterations:
                converged = True
                break
        else:
            stable_count = 0

    df["bio_efficiency"] = df["edge_id"].map(conductance)
    df["iterations_run"] = iterations_run
    df["converged"] = converged
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
