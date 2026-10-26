import csv
from pathlib import Path
from collections import defaultdict, Counter, deque

import networkx as nx
import matplotlib.pyplot as plt

from make_recursive_graphs_and_quotient import read_exp, normal_form_hash_and_text

base = Path("rank8_recursive_graphs/quotient_normal_form_graph")
nodes_path = base / "quotient_nodes.csv"
edges_path = base / "quotient_edges.csv"

out_png = base / "final_labeled_quotient_graph.png"
out_svg = base / "final_labeled_quotient_graph.svg"
out_path_csv = base / "standard_to_strassen_path_labeled.csv"
out_self_loop_csv = base / "self_loop_nodes_labeled.csv"

# ------------------------------------------------------------
# 1. Load graph
# ------------------------------------------------------------

nodes = list(csv.DictReader(open(nodes_path)))
edges = list(csv.DictReader(open(edges_path)))

all_nodes = [row["node_id"] for row in nodes]
rank_by_node = {row["node_id"]: row["rank"] for row in nodes}

G = nx.Graph()
for node in all_nodes:
    G.add_node(node, rank=rank_by_node[node])

self_loop_count = Counter()
edge_type_between = {}

for row in edges:
    s = row["source"]
    t = row["target"]
    edge_type = row.get("edge_type", "")

    if s == t or row.get("is_self_loop", "").lower() in {"yes", "true", "1"}:
        self_loop_count[s] += 1
        continue

    G.add_edge(s, t, edge_type=edge_type)
    edge_type_between[frozenset([s, t])] = edge_type

# ------------------------------------------------------------
# 2. Find standard node
# ------------------------------------------------------------

standard_path = Path("standard_rank8.exp")
standard_path.write_text("""\
(a11)*(b11)*(c11)
(a12)*(b21)*(c11)
(a11)*(b12)*(c21)
(a12)*(b22)*(c21)
(a21)*(b11)*(c12)
(a22)*(b21)*(c12)
(a21)*(b12)*(c22)
(a22)*(b22)*(c22)
""")

standard_scheme = read_exp(standard_path)
standard_node, _ = normal_form_hash_and_text(standard_scheme)

rank7_nodes = [n for n in all_nodes if rank_by_node[n] == "7"]
if len(rank7_nodes) != 1:
    raise RuntimeError(f"Expected exactly one rank-7 node, found {len(rank7_nodes)}")

strassen_node = rank7_nodes[0]

# ------------------------------------------------------------
# 3. Find shortest path standard -> Strassen
# ------------------------------------------------------------

path = nx.shortest_path(G, standard_node, strassen_node)
path_edges = {frozenset([path[i], path[i+1]]) for i in range(len(path)-1)}

with open(out_path_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "source", "target", "source_rank", "target_rank", "edge_type"])
    writer.writeheader()

    for i in range(len(path)-1):
        s = path[i]
        t = path[i+1]
        etype = edge_type_between.get(frozenset([s, t]), "")
        writer.writerow({
            "step": i,
            "source": s,
            "target": t,
            "source_rank": rank_by_node[s],
            "target_rank": rank_by_node[t],
            "edge_type": etype,
        })

# ------------------------------------------------------------
# 4. Find isolated nodes
# ------------------------------------------------------------

isolated_nodes = [n for n in G.nodes if G.degree(n) == 0]

# ------------------------------------------------------------
# 5. Save self-loop report
# ------------------------------------------------------------

with open(out_self_loop_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["node_id", "rank", "self_loop_count", "is_standard", "is_strassen", "is_isolated"])
    writer.writeheader()

    for n, count in self_loop_count.most_common():
        writer.writerow({
            "node_id": n,
            "rank": rank_by_node.get(n, ""),
            "self_loop_count": count,
            "is_standard": "yes" if n == standard_node else "no",
            "is_strassen": "yes" if n == strassen_node else "no",
            "is_isolated": "yes" if n in isolated_nodes else "no",
        })

# ------------------------------------------------------------
# 6. Draw graph
# ------------------------------------------------------------

main_nodes = [n for n in G.nodes if G.degree(n) > 0]
main_subgraph = G.subgraph(main_nodes).copy()

pos = nx.spring_layout(main_subgraph, seed=7, k=0.35, iterations=300)

# Put isolated node manually to the side
for i, n in enumerate(isolated_nodes):
    pos[n] = (2.2, 1.2 - i * 0.15)

plt.figure(figsize=(18, 14))

# Normal flip edges
flip_edges = []
reduction_edges = []
path_flip_edges = []
path_reduction_edges = []

for u, v, data in G.edges(data=True):
    key = frozenset([u, v])
    etype = data.get("edge_type", "")

    if key in path_edges:
        if "reduction" in etype:
            path_reduction_edges.append((u, v))
        else:
            path_flip_edges.append((u, v))
    else:
        if "reduction" in etype:
            reduction_edges.append((u, v))
        else:
            flip_edges.append((u, v))

nx.draw_networkx_edges(G, pos, edgelist=flip_edges, width=0.6, alpha=0.22)
nx.draw_networkx_edges(G, pos, edgelist=reduction_edges, width=1.4, alpha=0.65, edge_color="green")

# Highlight shortest path
nx.draw_networkx_edges(G, pos, edgelist=path_flip_edges, width=3.0, alpha=0.95, edge_color="orange")
nx.draw_networkx_edges(G, pos, edgelist=path_reduction_edges, width=4.0, alpha=0.95, edge_color="red")

# Regular nodes
regular_rank8 = [
    n for n in G.nodes
    if rank_by_node[n] == "8"
    and n != standard_node
    and n not in isolated_nodes
    and n not in path
    and self_loop_count[n] == 0
]

rank8_self_loop_nodes = [
    n for n in G.nodes
    if rank_by_node[n] == "8"
    and n != standard_node
    and n not in isolated_nodes
    and n not in path
    and self_loop_count[n] > 0
]

path_middle_nodes = [
    n for n in path
    if n not in {standard_node, strassen_node}
]

nx.draw_networkx_nodes(G, pos, nodelist=regular_rank8, node_size=55, alpha=0.75)
nx.draw_networkx_nodes(G, pos, nodelist=rank8_self_loop_nodes, node_size=95, alpha=0.95, node_color="purple")
nx.draw_networkx_nodes(G, pos, nodelist=path_middle_nodes, node_size=150, alpha=0.95, node_color="orange")
nx.draw_networkx_nodes(G, pos, nodelist=[standard_node], node_size=320, node_color="blue")
nx.draw_networkx_nodes(G, pos, nodelist=[strassen_node], node_size=340, node_color="red")
nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_size=280, node_color="black")

# Labels
labels = {
    standard_node: "standard\nrank 8",
    strassen_node: "Strassen\nrank 7",
}

for i, n in enumerate(path):
    if n not in {standard_node, strassen_node}:
        labels[n] = str(i)

for n in isolated_nodes:
    labels[n] = "isolated\nrank 8"

for n, count in self_loop_count.items():
    if n not in labels and count > 0:
        labels[n] = f"loop×{count}"

nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

plt.title(
    "2×2 rank ≤ 8 quotient flip graph over F₂\n"
    "orange path = shortest standard-to-Strassen path, red = reduction, purple = self-loop classes",
    fontsize=16
)

plt.axis("off")
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.savefig(out_svg)
plt.close()

print("Final labeled quotient graph created.")
print()
print("standard node:", standard_node)
print("Strassen/rank-7 node:", strassen_node)
print("standard-to-Strassen distance:", len(path) - 1)
print("isolated nodes:", isolated_nodes)
print("nodes with self-loops:", len(self_loop_count))
print("total self-loop rows:", sum(self_loop_count.values()))
print()
print("Saved PNG:", out_png)
print("Saved SVG:", out_svg)
print("Saved path CSV:", out_path_csv)
print("Saved self-loop CSV:", out_self_loop_csv)
