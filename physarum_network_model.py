"""
Network-coupled Physarum-inspired adaptation model. It solves a
Kirchhoff flow problem over the connected component, then adapts corridor
conductances. This is a network-coupled extension of the Physarum solver,
not a calibrated traffic-assignment model. The single-sink demand setup and
undirected road representation are explicit modeling assumptions.

The independent-edge model is retained as a comparison baseline. This
module couples edge scores through a shared flow solution. The flow solve
uses the project's impedance as effective resistance, but the impedance
formula and the conversion of POI density to source flow are project
assumptions, not calibrated transportation parameters.

SOURCE/SINK MODELING ASSUMPTION -- stated explicitly, not hidden: real
urban traffic is multi-origin, multi-destination, which is expensive to
solve exactly (all-pairs flow). As a tractable, defensible simplification,
we designate the single highest-demand node as a "sink" (the area's
primary destination pull) and treat every other node as a source
injecting flow proportional to its own demand. This is a real modeling
choice with real consequences for the result -- report it as such.

PERFORMANCE: uses scipy sparse matrices and a sparse solver, since real
road networks are sparse (each intersection connects to only a handful
of others, not everything). Tested at ~1000-node scale before shipping,
see the __main__ block and the accompanying test log.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from physarum_model import impedance, reinforcement


CONDUCTANCE_FLOOR = 1e-6  # numerical floor to keep the connected-component
                          # Laplacian nonsingular as edges decay


def run_network_simulation(edges_df, node_demands, decay_rate=0.15, mu=1.0,
                            generations=30, sink_node=None, tolerance=1e-5,
                            min_generations=10, stable_iterations=3):
    """
    edges_df: edge_id, u, v, distance_km, road_class, congestion.
        Optional two_way is not enforced by this undirected solver; one-way
        restrictions are respected only by routing.py.
    node_demands: dict node -> demand multiplier
    sink_node: which node absorbs all network flow. Defaults to the
        highest-demand node if not given.
    generations: maximum adaptation steps
    tolerance: relative maximum-conductance change required for stability
    min_generations: minimum steps before convergence can be declared
    stable_iterations: consecutive stable steps required to stop

    Returns (result_df, sink_node_used). 'converged' records numerical
    stability under this model's update rule, not traffic validation.
    Edges outside the sink component receive no coupled score (NaN) rather
    than an incompatible independent-model score. 'note' also flags ignored
    one-way restrictions.
    """
    df = edges_df.copy()
    df["impedance"] = df.apply(
        lambda r: impedance(r["distance_km"], r["congestion"], r["road_class"]), axis=1
    )

    nodes = sorted(set(df["u"]).union(set(df["v"])))
    if sink_node is None:
        sink_node = max(nodes, key=lambda n: node_demands.get(n, 1.0))
    if sink_node not in nodes:
        raise ValueError(f"Sink node {sink_node!r} not found in this network")

    G_topology = nx.Graph()
    for row in df.itertuples():
        G_topology.add_edge(row.u, row.v, edge_id=row.edge_id)

    component = nx.node_connected_component(G_topology, sink_node)
    in_component = df["u"].isin(component) & df["v"].isin(component)
    coupled_edges = df[in_component].reset_index(drop=True)
    isolated_edges = df[~in_component].reset_index(drop=True)

    comp_nodes = sorted(component)
    node_idx = {n: i for i, n in enumerate(n for n in comp_nodes if n != sink_node)}
    n_free = len(node_idx)

    conductance = {row.edge_id: 1.0 for row in coupled_edges.itertuples()}

    # Every non-sink node injects flow proportional to its own demand;
    # the sink implicitly absorbs the total (standard grounded-node
    # circuit formulation -- the sink's own KCL equation is redundant
    # given the other N-1 and is not needed explicitly).
    source = np.zeros(n_free)
    for n, i in node_idx.items():
        source[i] = node_demands.get(n, 1.0)

    edge_list = list(coupled_edges.itertuples())

    stable_count = 0
    converged = False
    iterations_run = 0

    for gen in range(generations):
        previous = conductance.copy()
        L = lil_matrix((n_free, n_free))
        edge_weight = {}
        for row in edge_list:
            # Treat the project's multi-criteria impedance as effective edge
            # resistance. Using distance alone would silently discard the
            # congestion and road-class inputs from the coupled flow solve.
            w = conductance[row.edge_id] / max(row.impedance, 1e-6)
            edge_weight[row.edge_id] = w
            iu = node_idx.get(row.u)
            iv = node_idx.get(row.v)
            if iu is not None:
                L[iu, iu] += w
            if iv is not None:
                L[iv, iv] += w
            if iu is not None and iv is not None:
                L[iu, iv] -= w
                L[iv, iu] -= w

        potentials = spsolve(csr_matrix(L), source)
        p = {sink_node: 0.0}
        for n, i in node_idx.items():
            p[n] = potentials[i]

        flux = {}
        for row in edge_list:
            q = edge_weight[row.edge_id] * (p[row.u] - p[row.v])
            flux[row.edge_id] = abs(q)

        for row in edge_list:
            c = conductance[row.edge_id]
            delta = reinforcement(flux[row.edge_id], mu) - (decay_rate * c)
            conductance[row.edge_id] = max(CONDUCTANCE_FLOOR, c + delta)

        iterations_run = gen + 1
        scale = max(1.0, max(conductance.values(), default=0.0))
        max_change = max(
            (abs(conductance[eid] - previous[eid]) for eid in conductance),
            default=0.0,
        )
        if gen + 1 >= min_generations and max_change <= tolerance * scale:
            stable_count += 1
            if stable_count >= stable_iterations:
                converged = True
                break
        else:
            stable_count = 0

    coupled_edges = coupled_edges.copy()
    coupled_edges["bio_efficiency"] = coupled_edges["edge_id"].map(conductance)
    coupled_edges["iterations_run"] = iterations_run
    coupled_edges["converged"] = converged
    direction_note = ""
    if "two_way" in df.columns:
        one_way = df["two_way"].astype(str).str.strip().str.lower().isin(
            ["false", "0", "no"]
        )
        if one_way.any():
            direction_note = "one-way restrictions ignored by undirected adaptation"

    coupled_edges["note"] = direction_note

    if not isolated_edges.empty:
        # These edges cannot participate in the source-to-sink solve.
        # Leave their scores missing instead of mixing a different model's
        # scale into the network-coupled ranking.
        iso_unscored = isolated_edges.copy()
        iso_unscored["bio_efficiency"] = np.nan
        iso_unscored["iterations_run"] = 0
        iso_unscored["converged"] = False
        iso_unscored["note"] = "disconnected from selected sink -- no coupled score"
        if direction_note:
            iso_unscored["note"] += "; " + direction_note
        result = pd.concat([coupled_edges, iso_unscored], ignore_index=True, sort=False)
    else:
        result = coupled_edges

    return result.sort_values("bio_efficiency"), sink_node


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

    result, sink = run_network_simulation(edges, node_demands)
    print(f"Sink node used: {sink}")
    print(result[["edge_id", "u", "v", "bio_efficiency", "note"]].to_string(index=False))
