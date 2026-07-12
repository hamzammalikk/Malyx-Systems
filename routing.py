"""
Resilience routing: given the corridor network, find the best alternate
route between two points if a specific edge or node is blocked (flooding,
protest, accident, construction closure -- all common causes of road
closure in Karachi).

Pure graph algorithm (networkx Dijkstra), no API or external service
needed.

Key limitation to be upfront about: this can only find an alternate
route if one topologically exists in your dataset. With only 6
hand-picked corridors, the network is two disconnected chains, so most
blockages will correctly report "no alternate route in this dataset" --
that's not a bug, it's telling you the truth about a sparse graph. This
feature becomes genuinely useful once you run auto_discover_corridors.py
and have the real, densely connected road network.
"""

import networkx as nx


def build_graph(edges_df):
    """
    Build an undirected weighted graph from the edges DataFrame.
    Weight = impedance (already computed by physarum_model.impedance),
    so shortest path here means "least friction", not just least distance.
    """
    G = nx.Graph()
    for row in edges_df.itertuples():
        G.add_edge(
            row.u, row.v,
            weight=row.impedance,
            distance_km=row.distance_km,
            edge_id=row.edge_id,
        )
    return G


def find_alternate_route(edges_df, origin, destination, blocked_edge_id=None, blocked_node=None):
    """
    Returns a dict describing the best route from origin to destination
    given an optional blocked edge (by edge_id) or blocked node (by name).

    Keys: found (bool), path (list of node names), total_impedance,
    total_distance_km, message (human-readable summary).
    """
    G = build_graph(edges_df)

    if blocked_node:
        if blocked_node in (origin, destination):
            return {"found": False, "message":
                    f"{blocked_node} is the origin or destination itself -- can't route through a blocked endpoint."}
        if G.has_node(blocked_node):
            G.remove_node(blocked_node)

    if blocked_edge_id:
        row = edges_df[edges_df.edge_id == blocked_edge_id]
        if not row.empty:
            u, v = row.iloc[0]["u"], row.iloc[0]["v"]
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    if origin not in G or destination not in G:
        return {"found": False, "message":
                f"{origin} or {destination} not in the network after removing the block."}

    try:
        path = nx.shortest_path(G, origin, destination, weight="weight")
        total_impedance = nx.shortest_path_length(G, origin, destination, weight="weight")
        total_distance = sum(
            G[path[i]][path[i + 1]]["distance_km"] for i in range(len(path) - 1)
        )
        return {
            "found": True,
            "path": path,
            "total_impedance": round(total_impedance, 2),
            "total_distance_km": round(total_distance, 2),
            "message": f"Alternate route found: {' -> '.join(path)} "
                       f"({total_distance:.2f} km, impedance {total_impedance:.2f})",
        }
    except nx.NetworkXNoPath:
        return {"found": False, "message":
                f"No alternate route exists between {origin} and {destination} in this dataset "
                f"once the block is applied. With only {len(edges_df)} corridors, this often "
                f"means the network genuinely has no redundancy here -- expand the dataset "
                f"(auto_discover_corridors.py) to test this properly."}


if __name__ == "__main__":
    import pandas as pd
    from physarum_model import run_simulation

    node_demands = {
        "AKU": 1.2, "Kashmir Park": 1.5, "Hassan Square": 1.2, "FUAAST": 0.7,
        "NIPA": 1.5, "Gulshan Chowrangi": 1.2, "Maskan": 0.9, "KU Gate": 0.7,
    }
    edges = pd.read_csv("sample_edges.csv")
    result = run_simulation(edges, node_demands)  # adds 'impedance' column

    print("Test 1: block E3 (Kashmir Park to Hassan Square), route AKU -> FUAAST")
    print(find_alternate_route(result, "AKU", "FUAAST", blocked_edge_id="E3")["message"])

    print("\nTest 2: block Maskan node, route Gulshan Chowrangi -> KU Gate")
    print(find_alternate_route(result, "Gulshan Chowrangi", "KU Gate", blocked_node="Maskan")["message"])

    print("\nTest 3: no block, route AKU -> NIPA (should succeed, full chain)")
    print(find_alternate_route(result, "AKU", "NIPA")["message"])
