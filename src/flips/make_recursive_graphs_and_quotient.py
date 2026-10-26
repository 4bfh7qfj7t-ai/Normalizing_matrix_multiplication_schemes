#!/usr/bin/env python3
"""
Build raw/sorted and normal-form quotient graphs from recursive_rank8_flip_explorer.py output.

Input folder must contain:
  rank8_flip_edges.csv
  rank8_to_rank7_reduction_edges.csv
  rank8_discovered_schemes/*.exp
  rank7_reduction_outputs/*.exp

Outputs:
  raw_sorted_graph/
      raw_nodes.csv
      raw_edges_combined.csv          (large: one row per valid directed flip/reduction event)
      raw_graph_summary.txt
  quotient_normal_form_graph/
      raw_to_quotient.csv             (can resume if script is interrupted)
      quotient_nodes.csv
      quotient_edges.csv
      normal_forms/*.txt
      graph_quotient_normal_form.{png,svg,graphml,gexf}
      quotient_graph_summary.txt

This script is intentionally for 2x2 over F2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from array import array
from collections import Counter, defaultdict, deque
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Mask = int
Row = Tuple[Mask, Mask, Mask]
Scheme = Tuple[Row, ...]
RowCode = int

# ---------------------------------------------------------------------------
# Basic 2x2 F2 matrix and row encoding helpers
# ---------------------------------------------------------------------------

def get(M: Mask, i: int) -> int:
    return (M >> i) & 1


def matmul(X: Mask, Y: Mask) -> Mask:
    a, b, c, d = get(X, 0), get(X, 1), get(X, 2), get(X, 3)
    e, f, g, h = get(Y, 0), get(Y, 1), get(Y, 2), get(Y, 3)
    return (
        ((a & e) ^ (b & g)) << 0
        | ((a & f) ^ (b & h)) << 1
        | ((c & e) ^ (d & g)) << 2
        | ((c & f) ^ (d & h)) << 3
    )


def det2(M: Mask) -> int:
    a, b, c, d = get(M, 0), get(M, 1), get(M, 2), get(M, 3)
    return (a & d) ^ (b & c)


def inv2(M: Mask) -> Mask:
    if det2(M) != 1:
        raise ValueError(f"not invertible: {M}")
    a, b, c, d = get(M, 0), get(M, 1), get(M, 2), get(M, 3)
    return (d << 0) | (b << 1) | (c << 2) | (a << 3)


def transpose(M: Mask) -> Mask:
    return (get(M, 0) << 0) | (get(M, 2) << 1) | (get(M, 1) << 2) | (get(M, 3) << 3)


TRANSPOSE = tuple(transpose(M) for M in range(16))
TOKEN = tuple("".join(str((M >> i) & 1) for i in range(4)) for M in range(16))


def encode_row(row: Row) -> RowCode:
    A, B, C = row
    return A | (B << 4) | (C << 8)


def decode_row(code: RowCode) -> Row:
    return (code & 15, (code >> 4) & 15, (code >> 8) & 15)


def colex_bits(M: Mask) -> Tuple[int, int, int, int]:
    # same comparison order as our previous normal-form script: [a b; c d] -> (b,d,a,c)
    return (get(M, 1), get(M, 3), get(M, 0), get(M, 2))


def bits_to_int(bits: Iterable[int]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


ROW_NORMAL_KEY: List[int] = [0] * 4096
for code in range(4096):
    A, B, C = decode_row(code)
    ROW_NORMAL_KEY[code] = bits_to_int(colex_bits(A) + colex_bits(B) + colex_bits(C))


def normal_row_sort_key(code: RowCode) -> int:
    return ROW_NORMAL_KEY[code]


# ---------------------------------------------------------------------------
# .exp parsing and raw/sorted hash matching recursive_rank8_flip_explorer.py
# ---------------------------------------------------------------------------

def bit_index(i: int, j: int) -> int:
    return (i - 1) * 2 + (j - 1)


def parse_factor(text: str, letter: str) -> Mask:
    s = text.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    s = s.strip()
    if not s or s == "0":
        raise ValueError(f"zero/empty {letter}-factor: {text!r}")
    mask = 0
    for term in s.split("+"):
        term = term.strip()
        if len(term) != 3 or term[0] != letter or not term[1:].isdigit():
            raise ValueError(f"bad term {term!r} in factor {text!r}")
        i, j = int(term[1]), int(term[2])
        if not (1 <= i <= 2 and 1 <= j <= 2):
            raise ValueError(f"index out of range in term {term!r}")
        mask ^= 1 << bit_index(i, j)
    if mask == 0:
        raise ValueError(f"zero {letter}-factor after parsing {text!r}")
    return mask


def read_exp(path: Path) -> Scheme:
    rows: List[Row] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("*")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected exactly two '*': {line!r}")
        rows.append((parse_factor(parts[0], "a"), parse_factor(parts[1], "b"), parse_factor(parts[2], "c")))
    if not rows:
        raise ValueError(f"{path}: no rows found")
    return tuple(rows)


def write_factor(mask: Mask, letter: str) -> str:
    terms = []
    for i in range(1, 3):
        for j in range(1, 3):
            if mask & (1 << bit_index(i, j)):
                terms.append(f"{letter}{i}{j}")
    if not terms:
        return "0"
    return "(" + "+".join(terms) + ")"


def row_to_exp(row: Row) -> str:
    A, B, C = row
    return f"{write_factor(A, 'a')}*{write_factor(B, 'b')}*{write_factor(C, 'c')}"


def raw_canonical_text(scheme: Scheme) -> str:
    # Must match recursive_rank8_flip_explorer.py exactly: sorted by numeric row tuple.
    return "\n".join(row_to_exp(row) for row in sorted(scheme)) + "\n"


def raw_sorted_hash(scheme: Scheme) -> str:
    return hashlib.sha1(raw_canonical_text(scheme).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fast brute-force 2x2 normal form over F2
# ---------------------------------------------------------------------------

def permutation_parity(perm: Tuple[int, int, int]) -> int:
    inv = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if perm[i] > perm[j]:
                inv += 1
    return inv % 2


def build_operation_row_maps() -> List[array]:
    GL2 = tuple(M for M in range(16) if det2(M) == 1)
    GL2_INV = {M: inv2(M) for M in GL2}

    # Sandwich action maps.
    action_maps = []
    for U in GL2:
        Uinv = GL2_INV[U]
        for V in GL2:
            Vinv = GL2_INV[V]
            for W in GL2:
                Winv = GL2_INV[W]
                Amap = tuple(matmul(matmul(U, M), Vinv) for M in range(16))
                Bmap = tuple(matmul(matmul(V, M), Winv) for M in range(16))
                Cmap = tuple(matmul(matmul(W, M), Uinv) for M in range(16))
                action_maps.append((Amap, Bmap, Cmap))

    op_maps: List[array] = []
    for perm in permutations((0, 1, 2)):
        odd = permutation_parity(perm) == 1
        for Amap, Bmap, Cmap in action_maps:
            arr = array("H", [0]) * 4096
            for code in range(4096):
                row = decode_row(code)
                x, y, z = row[perm[0]], row[perm[1]], row[perm[2]]
                if odd:
                    x, y, z = TRANSPOSE[x], TRANSPOSE[y], TRANSPOSE[z]
                arr[code] = encode_row((Amap[x], Bmap[y], Cmap[z]))
            op_maps.append(arr)
    return op_maps


OP_ROW_MAPS: Optional[List[array]] = None


def ensure_op_maps() -> List[array]:
    global OP_ROW_MAPS
    if OP_ROW_MAPS is None:
        print("Building 2x2 normal-form transformation table...", flush=True)
        t0 = time.time()
        OP_ROW_MAPS = build_operation_row_maps()
        print(f"Built {len(OP_ROW_MAPS)} transformation maps in {time.time() - t0:.1f}s", flush=True)
    return OP_ROW_MAPS


def scheme_to_codes(scheme: Scheme) -> Tuple[RowCode, ...]:
    return tuple(encode_row(row) for row in scheme)


def normal_form_codes(scheme: Scheme) -> Tuple[RowCode, ...]:
    ops = ensure_op_maps()
    codes = scheme_to_codes(scheme)
    best: Optional[Tuple[int, ...]] = None
    best_rows: Optional[Tuple[RowCode, ...]] = None

    for op in ops:
        transformed = [op[c] for c in codes]
        transformed.sort(key=normal_row_sort_key)
        key = tuple(ROW_NORMAL_KEY[c] for c in transformed)
        if best is None or key < best:
            best = key
            best_rows = tuple(transformed)

    assert best_rows is not None
    return best_rows


def normal_form_text_from_codes(codes: Tuple[RowCode, ...]) -> str:
    lines = []
    for code in codes:
        A, B, C = decode_row(code)
        lines.append(f"{TOKEN[A]} {TOKEN[B]} {TOKEN[C]}")
    return "\n".join(lines) + "\n"


def normal_form_hash_and_text(scheme: Scheme) -> Tuple[str, str]:
    nf_codes = normal_form_codes(scheme)
    text = normal_form_text_from_codes(nf_codes)
    return hashlib.sha1(text.encode()).hexdigest(), text


# ---------------------------------------------------------------------------
# CSV and union-find helpers
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self) -> None:
        self.parent: List[int] = []
        self.size: List[int] = []
        self.index: Dict[str, int] = {}

    def add(self, x: str) -> int:
        if x in self.index:
            return self.index[x]
        i = len(self.parent)
        self.index[x] = i
        self.parent.append(i)
        self.size.append(1)
        return i

    def find_i(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: str, b: str) -> None:
        ia, ib = self.add(a), self.add(b)
        ra, rb = self.find_i(ia), self.find_i(ib)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_sizes(self) -> List[int]:
        roots = Counter(self.find_i(i) for i in range(len(self.parent)))
        return sorted(roots.values(), reverse=True)


def open_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return path.open(newline="")


def write_dict_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Raw graph construction
# ---------------------------------------------------------------------------

def build_raw_graph(recursive_dir: Path, raw_out: Path) -> Dict[str, int]:
    raw_out.mkdir(parents=True, exist_ok=True)
    flip_csv = recursive_dir / "rank8_flip_edges.csv"
    red_csv = recursive_dir / "rank8_to_rank7_reduction_edges.csv"

    nodes: Dict[str, dict] = {}
    uf = UnionFind()
    edge_counts = Counter()
    self_loops = 0

    edges_path = raw_out / "raw_edges_combined.csv"
    with edges_path.open("w", newline="") as f_edges:
        edge_fields = ["source", "target", "edge_type", "source_rank", "target_rank"]
        writer = csv.DictWriter(f_edges, fieldnames=edge_fields)
        writer.writeheader()

        with open_csv(flip_csv) as f:
            for row in csv.DictReader(f):
                s = row["source_sorted_hash"]
                t = row["target_sorted_hash"]
                nodes.setdefault(s, {"node_id": s, "rank": "8"})
                nodes.setdefault(t, {"node_id": t, "rank": "8"})
                uf.union(s, t)
                writer.writerow({"source": s, "target": t, "edge_type": "flip_rank8_to_rank8", "source_rank": "8", "target_rank": "8"})
                edge_counts["flip_rank8_to_rank8"] += 1
                if s == t:
                    self_loops += 1

        with open_csv(red_csv) as f:
            for row in csv.DictReader(f):
                s = row["source_sorted_hash"]
                t = row["target_sorted_hash"]
                nodes.setdefault(s, {"node_id": s, "rank": "8"})
                nodes.setdefault(t, {"node_id": t, "rank": "7"})
                uf.union(s, t)
                writer.writerow({"source": s, "target": t, "edge_type": "reduction_rank8_to_rank7", "source_rank": "8", "target_rank": "7"})
                edge_counts["reduction_rank8_to_rank7"] += 1
                if s == t:
                    self_loops += 1

    node_rows = sorted(nodes.values(), key=lambda r: (r["rank"], r["node_id"]))
    write_dict_csv(raw_out / "raw_nodes.csv", node_rows, ["node_id", "rank"])

    rank_counts = Counter(row["rank"] for row in nodes.values())
    comp_sizes = uf.component_sizes()
    summary_lines = [
        "Recursive raw/sorted rank-7 / rank-8 flip graph summary",
        "",
        f"Input recursive folder: {recursive_dir}",
        f"Output folder: {raw_out}",
        "Node identity used here: sorted-summand hash",
        "",
        "Raw graph:",
        f"  total raw/sorted nodes: {len(nodes)}",
        f"  rank-8 raw/sorted nodes: {rank_counts.get('8', 0)}",
        f"  rank-7 raw/sorted nodes: {rank_counts.get('7', 0)}",
        f"  total edge rows: {sum(edge_counts.values())}",
        f"  directed flip edge rows: {edge_counts.get('flip_rank8_to_rank8', 0)}",
        f"  reduction edge rows: {edge_counts.get('reduction_rank8_to_rank7', 0)}",
        f"  self-loop edge rows: {self_loops}",
        "",
        "Connected components, treating all edges as undirected:",
        f"  number of components: {len(comp_sizes)}",
        f"  top 10 component sizes: {comp_sizes[:10]}",
        "",
        "Output files:",
        f"  raw nodes: {raw_out / 'raw_nodes.csv'}",
        f"  raw edges: {raw_out / 'raw_edges_combined.csv'}",
        "",
        "Note:",
        "  This raw graph is very large. The CSV edge list is the graph.",
        "  A full PNG/SVG drawing would be unreadable, so the drawing step is reserved for the quotient graph.",
    ]
    text = "\n".join(summary_lines)
    (raw_out / "raw_graph_summary.txt").write_text(text)
    print(text)
    return {
        "nodes": len(nodes),
        "rank8_nodes": rank_counts.get("8", 0),
        "rank7_nodes": rank_counts.get("7", 0),
        "edges": sum(edge_counts.values()),
        "flip_edges": edge_counts.get("flip_rank8_to_rank8", 0),
        "reduction_edges": edge_counts.get("reduction_rank8_to_rank7", 0),
        "components": len(comp_sizes),
    }


# ---------------------------------------------------------------------------
# Quotient graph construction
# ---------------------------------------------------------------------------

def load_existing_raw_to_q(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["raw_hash"]] = row
    return out


def compute_raw_to_quotient(recursive_dir: Path, quotient_out: Path, progress_every: int, resume: bool) -> Dict[str, dict]:
    nf_dir = quotient_out / "normal_forms"
    nf_dir.mkdir(parents=True, exist_ok=True)
    map_path = quotient_out / "raw_to_quotient.csv"
    existing = load_existing_raw_to_q(map_path) if resume else {}

    folders = [
        (recursive_dir / "rank8_discovered_schemes", 8),
        (recursive_dir / "rank7_reduction_outputs", 7),
    ]

    rows: Dict[str, dict] = dict(existing)
    errors: List[dict] = []
    started = time.time()
    processed_files = 0
    newly_computed = 0

    for folder, rank in folders:
        paths = sorted(folder.glob("*.exp"))
        print(f"Scanning {len(paths):,} rank-{rank} scheme files in {folder}", flush=True)
        for path in paths:
            processed_files += 1
            try:
                scheme = read_exp(path)
                raw_h = raw_sorted_hash(scheme)
                if raw_h in rows:
                    continue
                q_h, nf_text = normal_form_hash_and_text(scheme)
                nf_file = nf_dir / f"{q_h[:16]}_rank{rank}_normal_form.txt"
                if not nf_file.exists():
                    nf_file.write_text(nf_text)
                rows[raw_h] = {
                    "raw_hash": raw_h,
                    "rank": str(rank),
                    "representative_file": str(path),
                    "quotient_node": q_h,
                    "quotient_normal_form_file": str(nf_file),
                }
                newly_computed += 1
            except Exception as e:
                errors.append({"file": str(path), "rank": str(rank), "reason": str(e)})

            if progress_every > 0 and processed_files % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"normal-form progress: files_seen={processed_files:,} | "
                    f"raw_mapped={len(rows):,} | newly_computed={newly_computed:,} | "
                    f"errors={len(errors):,} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
                # Save checkpoint.
                write_dict_csv(
                    map_path,
                    sorted(rows.values(), key=lambda r: (r["rank"], r["raw_hash"])),
                    ["raw_hash", "rank", "representative_file", "quotient_node", "quotient_normal_form_file"],
                )

    write_dict_csv(
        map_path,
        sorted(rows.values(), key=lambda r: (r["rank"], r["raw_hash"])),
        ["raw_hash", "rank", "representative_file", "quotient_node", "quotient_normal_form_file"],
    )
    write_dict_csv(quotient_out / "normal_form_errors.csv", errors, ["file", "rank", "reason"])
    return rows


def connected_components_simple(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> List[int]:
    uf = UnionFind()
    for n in nodes:
        uf.add(n)
    for s, t in edges:
        if s != t:
            uf.union(s, t)
    return uf.component_sizes()


def build_quotient_graph(recursive_dir: Path, quotient_out: Path, progress_every: int, no_draw: bool, resume: bool) -> None:
    quotient_out.mkdir(parents=True, exist_ok=True)
    raw_to_q = compute_raw_to_quotient(recursive_dir, quotient_out, progress_every, resume)

    q_nodes: Dict[str, dict] = {}
    raw_count_by_q = Counter()
    for raw_h, row in raw_to_q.items():
        q = row["quotient_node"]
        rank = row["rank"]
        raw_count_by_q[q] += 1
        if q not in q_nodes:
            q_nodes[q] = {
                "node_id": q,
                "rank": rank,
                "label": f"rank {rank}",
                "representative_file": row["representative_file"],
                "normal_form_file": row["quotient_normal_form_file"],
                "raw_node_count": 0,
            }

    for q, c in raw_count_by_q.items():
        q_nodes[q]["raw_node_count"] = str(c)
        if q_nodes[q]["rank"] == "7":
            q_nodes[q]["label"] = f"rank 7 ({c} raw)"
        else:
            q_nodes[q]["label"] = q[:10]

    edge_counter: Counter[Tuple[str, str, str]] = Counter()
    missing_edges: List[dict] = []

    with open_csv(recursive_dir / "rank8_flip_edges.csv") as f:
        for row in csv.DictReader(f):
            s_raw = row["source_sorted_hash"]
            t_raw = row["target_sorted_hash"]
            qs = raw_to_q.get(s_raw, {}).get("quotient_node")
            qt = raw_to_q.get(t_raw, {}).get("quotient_node")
            if not qs or not qt:
                missing_edges.append({"source_raw": s_raw, "target_raw": t_raw, "edge_type": "flip_rank8_to_rank8", "reason": "raw_hash_missing_from_map"})
                continue
            a, b = sorted((qs, qt))
            edge_counter[(a, b, "flip_rank8_to_rank8")] += 1

    with open_csv(recursive_dir / "rank8_to_rank7_reduction_edges.csv") as f:
        for row in csv.DictReader(f):
            s_raw = row["source_sorted_hash"]
            t_raw = row["target_sorted_hash"]
            qs = raw_to_q.get(s_raw, {}).get("quotient_node")
            qt = raw_to_q.get(t_raw, {}).get("quotient_node")
            if not qs or not qt:
                missing_edges.append({"source_raw": s_raw, "target_raw": t_raw, "edge_type": "reduction_rank8_to_rank7", "reason": "raw_hash_missing_from_map"})
                continue
            edge_counter[(qs, qt, "reduction_rank8_to_rank7")] += 1

    q_edges = []
    for (s, t, etype), count in sorted(edge_counter.items()):
        q_edges.append({
            "source": s,
            "target": t,
            "edge_type": etype,
            "source_rank": q_nodes.get(s, {}).get("rank", ""),
            "target_rank": q_nodes.get(t, {}).get("rank", ""),
            "count": str(count),
            "is_self_loop": "yes" if s == t else "no",
        })

    write_dict_csv(
        quotient_out / "quotient_nodes.csv",
        sorted(q_nodes.values(), key=lambda r: (r["rank"], r["node_id"])),
        ["node_id", "rank", "label", "representative_file", "normal_form_file", "raw_node_count"],
    )
    write_dict_csv(
        quotient_out / "quotient_edges.csv",
        q_edges,
        ["source", "target", "edge_type", "source_rank", "target_rank", "count", "is_self_loop"],
    )
    write_dict_csv(
        quotient_out / "missing_edges.csv",
        missing_edges,
        ["source_raw", "target_raw", "edge_type", "reason"],
    )

    comp_sizes = connected_components_simple(q_nodes.keys(), [(e["source"], e["target"]) for e in q_edges if e["is_self_loop"] == "no"])
    rank_counts = Counter(row["rank"] for row in q_nodes.values())
    edge_counts_by_type = Counter(e["edge_type"] for e in q_edges)
    self_loops = sum(1 for e in q_edges if e["is_self_loop"] == "yes")

    draw_msg = draw_quotient_graph(q_nodes, q_edges, quotient_out, no_draw)

    summary_lines = [
        "Recursive normal-form quotient rank-7 / rank-8 flip graph summary",
        "",
        f"Input recursive folder: {recursive_dir}",
        f"Output folder: {quotient_out}",
        "Node identity used here: brute-force 2x2 normal form over F2",
        "",
        "Raw nodes collapsed:",
        f"  raw sorted nodes mapped: {len(raw_to_q)}",
        f"  raw rank-8 nodes mapped: {sum(1 for r in raw_to_q.values() if r['rank'] == '8')}",
        f"  raw rank-7 nodes mapped: {sum(1 for r in raw_to_q.values() if r['rank'] == '7')}",
        f"  missing edge endpoint mappings: {len(missing_edges)}",
        "",
        "Quotient graph:",
        f"  total quotient nodes: {len(q_nodes)}",
        f"  rank-8 quotient nodes: {rank_counts.get('8', 0)}",
        f"  rank-7 quotient nodes: {rank_counts.get('7', 0)}",
        f"  total quotient edges: {len(q_edges)}",
        f"  flip edges: {edge_counts_by_type.get('flip_rank8_to_rank8', 0)}",
        f"  reduction edges: {edge_counts_by_type.get('reduction_rank8_to_rank7', 0)}",
        f"  self-loops: {self_loops}",
        "",
        "Connected components, treating reductions as undirected and ignoring self-loops:",
        f"  number of components: {len(comp_sizes)}",
        f"  top 10 component sizes: {comp_sizes[:10]}",
        "",
        "Output files:",
        f"  raw-to-quotient map: {quotient_out / 'raw_to_quotient.csv'}",
        f"  quotient nodes: {quotient_out / 'quotient_nodes.csv'}",
        f"  quotient edges: {quotient_out / 'quotient_edges.csv'}",
        f"  normal forms folder: {quotient_out / 'normal_forms'}",
        "",
        "Drawing/export status:",
        f"  {draw_msg}",
        "",
        "Note:",
        "  This graph should show how many equivalence classes the large recursive raw component really contains.",
        "  Visual distances in the drawing are layout artifacts, not mathematical distances.",
    ]
    text = "\n".join(summary_lines)
    (quotient_out / "quotient_graph_summary.txt").write_text(text)
    print(text)


def draw_quotient_graph(nodes: Dict[str, dict], edges: List[dict], output_dir: Path, skip_draw: bool) -> str:
    try:
        import networkx as nx
    except Exception as e:
        return f"Graph export skipped because networkx is unavailable: {e}"

    G = nx.MultiDiGraph()
    for n, data in nodes.items():
        G.add_node(n, **{k: str(v) for k, v in data.items()})
    for e in edges:
        G.add_edge(e["source"], e["target"], edge_type=e["edge_type"], count=e["count"])

    graphml = output_dir / "graph_quotient_normal_form.graphml"
    gexf = output_dir / "graph_quotient_normal_form.gexf"
    try:
        nx.write_graphml(G, graphml)
        nx.write_gexf(G, gexf)
        export_msg = f"GraphML/GEXF written: {graphml}, {gexf}"
    except Exception as e:
        export_msg = f"GraphML/GEXF export issue: {e}"

    if skip_draw:
        return export_msg + "; drawing skipped by --no-draw"

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        return export_msg + f"; drawing skipped because matplotlib is unavailable: {e}"

    H = nx.Graph()
    for n, data in nodes.items():
        H.add_node(n, **data)

    flip_edges = []
    reduction_edges = []
    for e in edges:
        s, t = e["source"], e["target"]
        if s == t:
            continue
        H.add_edge(s, t)
        if e["edge_type"] == "flip_rank8_to_rank8":
            flip_edges.append((s, t))
        else:
            reduction_edges.append((s, t))

    rank8_nodes = [n for n, d in H.nodes(data=True) if str(d.get("rank")) == "8"]
    rank7_nodes = [n for n, d in H.nodes(data=True) if str(d.get("rank")) == "7"]

    try:
        pos = nx.spring_layout(H, seed=4, iterations=600)
    except Exception:
        pos = nx.circular_layout(H)

    plt.figure(figsize=(18, 12), dpi=250)
    nx.draw_networkx_edges(H, pos, edgelist=flip_edges, width=0.65, alpha=0.35)
    nx.draw_networkx_edges(H, pos, edgelist=reduction_edges, width=2.2, alpha=0.9, edge_color="green", arrows=True)
    nx.draw_networkx_nodes(H, pos, nodelist=rank8_nodes, node_size=24, alpha=0.85, edgecolors="black", linewidths=0.25)
    nx.draw_networkx_nodes(H, pos, nodelist=rank7_nodes, node_size=220, node_color="white", edgecolors="black", linewidths=1.8)
    labels = {n: nodes[n]["label"] for n in rank7_nodes}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)
    plt.title("Recursive 2x2 rank ≤ 8 quotient flip graph\nNodes collapsed by brute-force 2x2 normal form; reductions are green")
    plt.axis("off")
    plt.tight_layout()
    png = output_dir / "graph_quotient_normal_form.png"
    svg = output_dir / "graph_quotient_normal_form.svg"
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close()
    return export_msg + f"; drawing written: {png}, {svg}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recursive_dir", nargs="?", type=Path, default=Path("rank8_recursive_flip_exploration"))
    parser.add_argument("--output-dir", type=Path, default=Path("rank8_recursive_graphs"))
    parser.add_argument("--progress-every", type=int, default=1000, help="normal-form progress interval")
    parser.add_argument("--raw-only", action="store_true", help="only build raw/sorted CSV graph; skip normal-form quotient")
    parser.add_argument("--quotient-only", action="store_true", help="only build normal-form quotient; skip raw/sorted CSV graph")
    parser.add_argument("--no-draw", action="store_true", help="skip quotient PNG/SVG drawing; still writes CSV/GraphML/GEXF")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing raw_to_quotient.csv checkpoint")
    args = parser.parse_args()

    recursive_dir = args.recursive_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quotient_only:
        print("\n=== Building raw/sorted graph ===\n", flush=True)
        build_raw_graph(recursive_dir, out_dir / "raw_sorted_graph")

    if not args.raw_only:
        print("\n=== Building normal-form quotient graph ===\n", flush=True)
        build_quotient_graph(
            recursive_dir,
            out_dir / "quotient_normal_form_graph",
            progress_every=args.progress_every,
            no_draw=args.no_draw,
            resume=not args.no_resume,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
