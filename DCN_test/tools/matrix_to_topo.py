import os
import json
import argparse
import ast
from typing import List

import numpy as np
import networkx as nx


def parse_matrix(matrix_arg: str) -> np.ndarray:
    """Parse matrix from a file path or a Python-style literal string.

    Accepts:
    - Path to a file containing CSV or whitespace-separated integers
    - A Python list literal string like "[[0,1],[1,0]]" or with spaces
    """
    if os.path.exists(matrix_arg):
        try:
            return np.loadtxt(matrix_arg, delimiter=",")
        except Exception:
            return np.loadtxt(matrix_arg)
    # Try literal eval
    try:
        data = ast.literal_eval(matrix_arg)
        return np.array(data)
    except Exception as exc:
        raise ValueError(f"Unable to parse matrix from input: {exc}")


def build_graph_from_matrix(adj: np.ndarray, base_capacity: float, directed: bool = True) -> nx.Graph:
    n = adj.shape[0]
    G = nx.DiGraph() if directed else nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            weight = adj[i, j]
            if weight and int(weight) != 0:
                cap = float(base_capacity) * float(int(weight))
                G.add_edge(int(i), int(j), capacity=cap)
    return G


def write_topology_json(G: nx.Graph, out_json_path: str) -> None:
    data = nx.node_link_data(G)
    if not os.path.exists(os.path.dirname(out_json_path)):
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def k_shortest_simple_paths(G: nx.Graph, src: int, dst: int, k: int) -> List[List[int]]:
    try:
        gen = nx.shortest_simple_paths(G, source=src, target=dst, weight=None)
        paths = []
        for _ in range(k):
            try:
                paths.append(next(gen))
            except StopIteration:
                break
        return paths
    except nx.NetworkXNoPath:
        return []


def write_tunnels(G: nx.Graph, k: int, out_tunnels_path: str) -> None:
    n = G.number_of_nodes()
    lines = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            paths = k_shortest_simple_paths(G, i, j, k)
            if not paths:
                # still write an empty line for consistency
                lines.append(f"{i} {j}:\n")
                continue
            path_strs = ["-".join(map(str, p)) for p in paths]
            lines.append(f"{i} {j}:{','.join(path_strs)}\n")

    if not os.path.exists(os.path.dirname(out_tunnels_path)):
        os.makedirs(os.path.dirname(out_tunnels_path), exist_ok=True)
    with open(out_tunnels_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Generate topology JSON and tunnels.txt from adjacency matrix")
    parser.add_argument("--name", required=True, help="Topology name (output folder and json name)")
    parser.add_argument("--matrix", required=True, help="Path or literal of adjacency matrix")
    parser.add_argument("--capacity", type=float, default=10000.0, help="Base capacity per single link")
    parser.add_argument("--k_paths", type=int, default=3, help="Number of candidate paths per (src,dst)")
    parser.add_argument("--directed", action="store_true", help="Treat matrix as directed (default true)")
    parser.add_argument(
        "--output_dir",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Data")),
        help="Output directory to place <name>.json and tunnels.txt inside <output_dir>/<name>/",
    )

    args = parser.parse_args()

    adj = parse_matrix(args.matrix)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    # Ensure integer weights
    adj = np.array(adj, dtype=int)

    G = build_graph_from_matrix(adj, base_capacity=args.capacity, directed=True)

    topo_dir = os.path.join(args.output_dir, args.name)
    json_path = os.path.join(topo_dir, f"{args.name}.json")
    tunnels_path = os.path.join(topo_dir, "tunnels.txt")

    write_topology_json(G, json_path)
    write_tunnels(G, k=args.k_paths, out_tunnels_path=tunnels_path)

    print(f"Wrote topology JSON: {json_path}")
    print(f"Wrote tunnels file:   {tunnels_path}")


if __name__ == "__main__":
    main()


