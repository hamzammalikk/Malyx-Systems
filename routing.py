"""
Resilience routing: given the corridor network, find the best alternate
route between two points if a specific edge or node is blocked (flooding,
protest, accident, construction closure -- all common causes of road
closure in Karachi).

Pure graph algorithm (networkx Dijkstra), no API or external service
needed.

Respects one-way streets: the graph is directed, and a corridor is only
traversable in reverse if its 'two_way' column says so (from
auto_discover_corridors.py, which reads this from OSM's actual directed
graph structure). CSVs without a 'two_way' column (e.g. older exports,
or extract_gulshan_network.py) are treated as two-way for backward
compatibility -- less accurate, but matches this module's original
behavior rather than silently changing results for existing datasets.

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
    Build a DIRECTED weighted graph from the edges DataFrame, respecting
    one-way streets. A corridor with two_way=True gets edges in both
    directions; two_way=False (or missing, for backward compatibility
    with older CSVs that predate this column -- treated as two-way since
    that was this function's original, if inaccurate, assumption) gets
    only the recorded u -> v direction.

    Weight = impedance (already computed by physarum_model.impedance),
    so shortest path here means "least friction", not just least distance.
    """
    G = nx.DiGraph()
    has_direction_data = "two_way" in edges_df.columns
    for row in edges_df.itertuples():
        G.add_edge(
            row.u, row.v,
            weight=row.impedance,
            distance_km=row.distance_km,
            edge_id=row.edge_id,
        )
        is_two_way = getattr(row, "two_way", True) if has_direction_data else True
        if is_two_way:
            G.add_edge(
                row.v, row.u,
                weight=row.impedance,
                distance_km=row.distance_km,
                edge_id=row.edge_id,
            )
    return G


def find_alternate_route(edges_df, origin, destination, blocked_edge_id=None, blocked_node=None,
                          blocked_edge_ids=None, blocked_nodes=None):
    """
    Returns a dict describing the best route from origin to destination
    given any combination of blocked edges and blocked nodes -- simulating
    a real closure event, which often takes out several roads and/or
    intersections at once (e.g. area flooding), not just one.

    Accepts both the original singular kwargs (blocked_edge_id,
    blocked_node -- kept for backward compatibility) and new plural
    kwargs (blocked_edge_ids, blocked_nodes) that take a list. All four
    can be combined; every blocked item is applied together in one route
    computation.

    Keys: found (bool), path (list of node names), total_impedance,
    total_distance_km, message (human-readable summary), blocked_summary
    (list of strings describing everything that was blocked, for display).
    """
    G = build_graph(edges_df)

    all_blocked_nodes = list(blocked_nodes or [])
    if blocked_node:
        all_blocked_nodes.append(blocked_node)

    all_blocked_edge_ids = list(blocked_edge_ids or [])
    if blocked_edge_id:
        all_blocked_edge_ids.append(blocked_edge_id)

    blocked_summary = []

    for node in all_blocked_nodes:
        if node in (origin, destination):
            return {"found": False, "message":
                    f"{node} is the origin or destination itself -- can't route through a blocked endpoint.",
                    "blocked_summary": blocked_summary}
        if G.has_node(node):
            G.remove_node(node)
            blocked_summary.append(f"intersection {node}")

    for edge_id in all_blocked_edge_ids:
        row = edges_df[edges_df.edge_id == edge_id]
        if not row.empty:
            u, v = row.iloc[0]["u"], row.iloc[0]["v"]
            # A physical closure (flood, accident, construction) blocks the
            # road in both directions, regardless of which way is recorded
            # as the "forward" direction -- remove both if both exist.
            removed = False
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                removed = True
            if G.has_edge(v, u):
                G.remove_edge(v, u)
                removed = True
            if removed:
                blocked_summary.append(f"corridor {edge_id}")

    if origin not in G or destination not in G:
        return {"found": False, "message":
                f"{origin} or {destination} not in the network after removing the block(s).",
                "blocked_summary": blocked_summary}

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
            "blocked_summary": blocked_summary,
            "message": f"Alternate route found: {' -> '.join(path)} "
                       f"({total_distance:.2f} km, impedance {total_impedance:.2f})",
        }
    except nx.NetworkXNoPath:
        n_blocks = len(blocked_summary)
        return {"found": False, "blocked_summary": blocked_summary, "message":
                f"No alternate route exists between {origin} and {destination} in this dataset "
                f"once {'this block is' if n_blocks == 1 else f'these {n_blocks} blocks are'} applied. "
                f"With only {len(edges_df)} corridors, this often "
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
