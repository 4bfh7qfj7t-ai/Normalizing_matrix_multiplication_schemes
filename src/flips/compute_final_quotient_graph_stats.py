import csv
from pathlib import Path
from collections import Counter, defaultdict

import networkx as nx

base = Path("rank8_recursive_graphs/quotient_normal_form_graph")

nodes_path = base / "quotient_nodes.csv"
edges_path = base / "quotient_edges.csv"
out_path = base / "final_quotient_graph_metrics.txt"

nodes = list(csv.DictReader(open(nodes_path)))
edges = list(csv.DictReader(open(edges_path)))

G = nx.Graph()

rank_by_node = {}

for row in nodes:
    node = row["node_id"]
    rank = row["rank"]
    G.add_node(node, rank=rank)
    rank_by_node[node] = rank

self_loop_count = Counter()
edge_type_count = Counter()
reduction_edges = []
flip_edges = []

for row in edges:
    s = row["source"]
    t = row["target"]
    edge_type = row.get("edge_type", "")

    edge_type_count[edge_type] += 1

    if s == t or row.get("is_self_loop", "").lower() in {"yes", "true", "1"}:
        self_loop_count[s] += 1
        continue

    G.add_edge(s, t, edge_type=edge_type)

    if "reduction" in edge_type:
        reduction_edges.append((s, t))
    elif "flip" in edge_type:
        flip_edges.append((s, t))

components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)

main_component = components[0]
main = G.subgraph(main_component).copy()

diameter = nx.diameter(main)
radius = nx.radius(main)
center_nodes = nx.center(main)
periphery_nodes = nx.periphery(main)

degree_counter = Counter(dict(main.degree()).values())

rank_counts_main = Counter(rank_by_node[n] for n in main.nodes)
rank_counts_all = Counter(rank_by_node[n] for n in G.nodes)

reduction_endpoint_counts = Counter()
for s, t in reduction_edges:
    reduction_endpoint_counts[s] += 1
    reduction_endpoint_counts[t] += 1

rank7_nodes = [n for n, r in rank_by_node.items() if r == "7"]
isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]

lines = []

def add(x=""):
    lines.append(str(x))

add("Final quotient graph metrics")
add()
add(f"Nodes total: {G.number_of_nodes()}")
add(f"Edges total ignoring self-loops: {G.number_of_edges()}")
add(f"Self-loop rows: {sum(self_loop_count.values())}")
add()
add(f"Rank counts, all nodes: {dict(rank_counts_all)}")
add(f"Connected components: {len(components)}")
add(f"Component sizes: {[len(c) for c in components]}")
add(f"Isolated nodes: {isolated_nodes}")
add()
add("Main component")
add(f"  size: {main.number_of_nodes()}")
add(f"  edges: {main.number_of_edges()}")
add(f"  rank counts: {dict(rank_counts_main)}")
add(f"  diameter: {diameter}")
add(f"  radius: {radius}")
add(f"  center node count: {len(center_nodes)}")
add(f"  periphery node count: {len(periphery_nodes)}")
add()
add("Edge type counts from CSV")
for edge_type, count in edge_type_count.items():
    add(f"  {edge_type}: {count}")
add()
add(f"Flip edges ignoring self-loops: {len(flip_edges)}")
add(f"Reduction edges ignoring self-loops: {len(reduction_edges)}")
add()
add("Degree distribution in main component")
for degree in sorted(degree_counter):
    add(f"  degree {degree}: {degree_counter[degree]} nodes")
add()
add("Rank-7 nodes")
for n in rank7_nodes:
    add(f"  {n} degree={G.degree(n)}")
add()
add("Reduction edges")
for i, (s, t) in enumerate(reduction_edges, start=1):
    add(f"  {i}. {s} rank {rank_by_node[s]}  --  {t} rank {rank_by_node[t]}")
add()
add("Self-loop nodes")
for n, count in self_loop_count.most_common():
    add(f"  {n} rank {rank_by_node.get(n)} self_loops={count}")

text = "\n".join(lines)
out_path.write_text(text)

print(text)
print()
print("Saved metrics to:", out_path)
