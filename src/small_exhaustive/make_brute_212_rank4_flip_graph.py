import csv
import hashlib
import itertools
import re
from pathlib import Path
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt

exp_dir = Path("brute_212_rank4/valid_schemes_exp")
out_dir = Path("brute_212_rank4_graph")
out_dir.mkdir(exist_ok=True)

nodes_csv = out_dir / "nodes.csv"
edges_csv = out_dir / "edges.csv"
summary_txt = out_dir / "graph_summary.txt"
png_path = out_dir / "brute_212_rank4_flip_graph.png"
svg_path = out_dir / "brute_212_rank4_flip_graph.svg"
graphml_path = out_dir / "brute_212_rank4_flip_graph.graphml"
gexf_path = out_dir / "brute_212_rank4_flip_graph.gexf"

paren_re = re.compile(r"\(([^()]*)\)")

# ------------------------------------------------------------
# 1. Read variables from all .exp files
# ------------------------------------------------------------

files = sorted(exp_dir.glob("*.exp"))
if not files:
    raise RuntimeError(f"No .exp files found in {exp_dir}")

vars_by_kind = {"a": set(), "b": set(), "c": set()}

def split_terms(s):
    s = s.strip()
    if s in {"", "0"}:
        return []
    return [x.strip() for x in s.split("+") if x.strip()]

for path in files:
    for line in path.read_text().splitlines():
        groups = paren_re.findall(line)
        for group in groups:
            for var in split_terms(group):
                kind = var[0].lower()
                if kind in vars_by_kind:
                    vars_by_kind[kind].add(var)

var_maps = {}
for kind in ["a", "b", "c"]:
    ordered = sorted(vars_by_kind[kind])
    var_maps[kind] = {var: i for i, var in enumerate(ordered)}

print("Variable maps:")
for kind in ["a", "b", "c"]:
    print(kind, var_maps[kind])

A_count = len(var_maps["a"])
B_count = len(var_maps["b"])
C_count = len(var_maps["c"])

# ------------------------------------------------------------
# 2. Parse schemes
# ------------------------------------------------------------

def parse_form(group):
    mask = 0
    for var in split_terms(group):
        kind = var[0].lower()
        idx = var_maps[kind][var]
        mask ^= (1 << idx)
    return mask

def parse_scheme(path):
    triples = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        groups = paren_re.findall(line)
        if len(groups) != 3:
            raise ValueError(f"Could not parse line in {path}: {line}")

        a = parse_form(groups[0])
        b = parse_form(groups[1])
        c = parse_form(groups[2])

        if a == 0 or b == 0 or c == 0:
            raise ValueError(f"Zero factor in {path}: {line}")

        triples.append((a, b, c))

    return tuple(sorted(triples))

def scheme_hash(scheme):
    text = ";".join(f"{a},{b},{c}" for a, b, c in scheme)
    return hashlib.sha1(text.encode()).hexdigest()

def bit_positions(mask, count):
    for i in range(count):
        if mask & (1 << i):
            yield i

def tensor_bitset_of_term(term):
    a, b, c = term
    result = 0

    for ai in bit_positions(a, A_count):
        for bi in bit_positions(b, B_count):
            for ci in bit_positions(c, C_count):
                pos = (ai * B_count * C_count) + (bi * C_count) + ci
                result ^= (1 << pos)

    return result

def tensor_bitset_of_scheme(scheme):
    result = 0
    for term in scheme:
        result ^= tensor_bitset_of_term(term)
    return result

schemes = {}
file_by_hash = {}

for path in files:
    scheme = parse_scheme(path)
    h = scheme_hash(scheme)

    if h in schemes:
        raise RuntimeError(f"Duplicate scheme hash found: {h}")

    schemes[h] = scheme
    file_by_hash[h] = str(path)

target_bitset = tensor_bitset_of_scheme(next(iter(schemes.values())))

bad = []
for h, scheme in schemes.items():
    if tensor_bitset_of_scheme(scheme) != target_bitset:
        bad.append(h)

if bad:
    raise RuntimeError(f"{len(bad)} schemes did not match the target tensor")

print()
print("Loaded valid schemes:", len(schemes))
print("Target tensor bitset:", hex(target_bitset))

# ------------------------------------------------------------
# 3. Flip enumeration
# ------------------------------------------------------------

def replace_pair(scheme, i, j, new1, new2):
    out = []
    for k, term in enumerate(scheme):
        if k not in {i, j}:
            out.append(term)
    out.append(new1)
    out.append(new2)
    return tuple(sorted(out))

def is_rank4_clean(scheme):
    if len(scheme) != 4:
        return False
    if len(set(scheme)) != 4:
        return False
    for a, b, c in scheme:
        if a == 0 or b == 0 or c == 0:
            return False
    return True

edge_rows = []
edge_set = set()
directed_flip_attempts = 0
valid_flip_outputs = 0
known_valid_neighbors = 0

for source_hash, scheme in schemes.items():
    terms = list(scheme)

    for i, j in itertools.combinations(range(len(terms)), 2):
        a1, b1, c1 = terms[i]
        a2, b2, c2 = terms[j]

        candidate_flips = []

        # Same A factor: flip in B/C
        if a1 == a2:
            candidate_flips.append(("same_A", replace_pair(scheme, i, j, (a1, b1, c1 ^ c2), (a1, b1 ^ b2, c2))))
            candidate_flips.append(("same_A", replace_pair(scheme, i, j, (a1, b1 ^ b2, c1), (a1, b2, c1 ^ c2))))

        # Same B factor: flip in A/C
        if b1 == b2:
            candidate_flips.append(("same_B", replace_pair(scheme, i, j, (a1, b1, c1 ^ c2), (a1 ^ a2, b1, c2))))
            candidate_flips.append(("same_B", replace_pair(scheme, i, j, (a1 ^ a2, b1, c1), (a2, b1, c1 ^ c2))))

        # Same C factor: flip in A/B
        if c1 == c2:
            candidate_flips.append(("same_C", replace_pair(scheme, i, j, (a1, b1 ^ b2, c1), (a1 ^ a2, b2, c1))))
            candidate_flips.append(("same_C", replace_pair(scheme, i, j, (a1 ^ a2, b1, c1), (a2, b1 ^ b2, c1))))

        for flip_type, new_scheme in candidate_flips:
            directed_flip_attempts += 1

            if not is_rank4_clean(new_scheme):
                continue

            # Brent check
            if tensor_bitset_of_scheme(new_scheme) != target_bitset:
                continue

            valid_flip_outputs += 1

            target_hash = scheme_hash(new_scheme)

            if target_hash not in schemes:
                # Should not happen, because we brute-forced all valid rank-4 schemes.
                continue

            known_valid_neighbors += 1

            u, v = sorted([source_hash, target_hash])
            edge_key = (u, v, flip_type)

            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edge_rows.append({
                    "source": u,
                    "target": v,
                    "edge_type": flip_type,
                    "is_self_loop": "yes" if u == v else "no",
                })

# ------------------------------------------------------------
# 4. Build graph
# ------------------------------------------------------------

G = nx.Graph()

def bitcount(x):
    return bin(x).count("1")

for h, scheme in schemes.items():
    all_unit_terms = all(
        bitcount(a) == 1 and bitcount(b) == 1 and bitcount(c) == 1
        for a, b, c in scheme
    )

    G.add_node(
        h,
        file=file_by_hash[h],
        all_unit_terms="yes" if all_unit_terms else "no",
    )

for row in edge_rows:
    if row["source"] != row["target"]:
        G.add_edge(row["source"], row["target"], edge_type=row["edge_type"])

components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)

# ------------------------------------------------------------
# 5. Save CSVs
# ------------------------------------------------------------

with open(nodes_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["node_id", "file", "all_unit_terms", "degree"])
    writer.writeheader()

    for n in sorted(G.nodes):
        writer.writerow({
            "node_id": n,
            "file": G.nodes[n]["file"],
            "all_unit_terms": G.nodes[n]["all_unit_terms"],
            "degree": G.degree(n),
        })

with open(edges_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["source", "target", "edge_type", "is_self_loop"])
    writer.writeheader()
    writer.writerows(edge_rows)

# ------------------------------------------------------------
# 6. Draw graph
# ------------------------------------------------------------

plt.figure(figsize=(12, 10))

if G.number_of_edges() > 0:
    pos = nx.spring_layout(G, seed=3, k=0.8, iterations=300)
else:
    pos = nx.circular_layout(G)

standard_like = [n for n in G.nodes if G.nodes[n]["all_unit_terms"] == "yes"]
others = [n for n in G.nodes if n not in standard_like]

nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.45)
nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=140, alpha=0.8)
nx.draw_networkx_nodes(G, pos, nodelist=standard_like, node_size=280, node_color="red")

labels = {}
for n in standard_like:
    labels[n] = "standard-like"

nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

plt.title("Raw flip graph for all 2×1 × 1×2 rank-4 schemes over F₂")
plt.axis("off")
plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.savefig(svg_path)
plt.close()

nx.write_graphml(G, graphml_path)
nx.write_gexf(G, gexf_path)

# ------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------

degree_counts = Counter(dict(G.degree()).values())
edge_type_counts = Counter(row["edge_type"] for row in edge_rows)

summary = f"""Raw flip graph for 2x1 times 1x2 rank-4 schemes over F2

Input folder:
{exp_dir}

Valid schemes loaded:
{len(schemes)}

Target tensor bitset:
{hex(target_bitset)}

Flip enumeration:
Directed flip attempts:
{directed_flip_attempts}

Valid Brent-checked flip outputs:
{valid_flip_outputs}

Known valid neighbors among the 81 schemes:
{known_valid_neighbors}

Unique graph edge rows:
{len(edge_rows)}

Simple graph:
Nodes:
{G.number_of_nodes()}

Edges ignoring self-loops:
{G.number_of_edges()}

Connected components:
{len(components)}

Component sizes:
{[len(c) for c in components]}

Degree distribution:
{dict(sorted(degree_counts.items()))}

Edge type counts:
{dict(edge_type_counts)}

Output files:
{nodes_csv}
{edges_csv}
{summary_txt}
{png_path}
{svg_path}
{graphml_path}
{gexf_path}
"""

summary_txt.write_text(summary)
print(summary)
