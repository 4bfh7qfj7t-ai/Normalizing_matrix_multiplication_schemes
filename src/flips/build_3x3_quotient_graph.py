#!/usr/bin/env python3
"""Build raw and normal-form quotient graphs from 3x3 flip exploration output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import ModuleType
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_normal_form_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("normal_form_3x3_fast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        adjacency[node]
    for source, target in edges:
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen: set[str] = set()
    output: list[list[str]] = []
    for node in adjacency:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        output.append(component)
    return sorted(output, key=len, reverse=True)


def scheme_files(exploration_dir: Path) -> dict[str, tuple[int, Path]]:
    output: dict[str, tuple[int, Path]] = {}
    same_rank_files = sorted((exploration_dir / "same_rank_schemes").glob("*.txt"))
    reduced_files = sorted((exploration_dir / "reduction_schemes").glob("*.txt"))
    if not same_rank_files:
        raise RuntimeError("No same-rank schemes were found")

    def row_count(path: Path) -> int:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))

    for path in same_rank_files + reduced_files:
        digest = path.stem
        output[digest] = (row_count(path), path)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build raw and 3x3-normal-form quotient flip graphs")
    parser.add_argument("exploration_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--normal-form-script",
        type=Path,
        default=Path(__file__).with_name("normal_form_3x3_fast.py"),
        help="Path to normal_form_3x3_fast.py",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-draw", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    exploration_dir: Path = args.exploration_dir
    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw_graph"
    quotient_dir = output_dir / "quotient_graph"
    normal_dir = quotient_dir / "normal_forms"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)

    files = scheme_files(exploration_dir)
    flip_rows = read_csv(exploration_dir / "same_rank_flip_edges.csv")
    reduction_rows = read_csv(exploration_dir / "reduction_edges.csv")

    raw_nodes = [
        {"node_id": digest, "rank": str(rank), "representative_file": str(path)}
        for digest, (rank, path) in sorted(files.items(), key=lambda item: (item[1][0], item[0]))
    ]
    raw_edge_counts: Counter[tuple[str, str, str]] = Counter()
    for row in flip_rows:
        source, target = sorted((row["source_hash"], row["target_hash"]))
        raw_edge_counts[(source, target, "flip")] += 1
    for row in reduction_rows:
        raw_edge_counts[(row["source_hash"], row["target_hash"], "reduction")] += 1
    raw_edges = [
        {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "source_rank": str(files[source][0]),
            "target_rank": str(files[target][0]),
            "event_count": str(count),
            "self_loop": "yes" if source == target else "no",
        }
        for (source, target, edge_type), count in sorted(raw_edge_counts.items())
    ]
    write_csv(raw_dir / "nodes.csv", raw_nodes, ["node_id", "rank", "representative_file"])
    write_csv(
        raw_dir / "edges.csv",
        raw_edges,
        ["source", "target", "edge_type", "source_rank", "target_rank", "event_count", "self_loop"],
    )

    nf = load_normal_form_module(args.normal_form_script.resolve())
    print("Building optimized 3x3/F2 tables...", flush=True)
    tables = nf.Tables.build()
    raw_to_quotient: dict[str, str] = {}
    quotient_info: dict[str, dict] = {}
    mapping_rows: list[dict] = []
    started = time.time()

    for index, (raw_hash, (rank, path)) in enumerate(sorted(files.items()), start=1):
        scheme = nf.read_scheme(path)
        if not nf.brent_valid(scheme):
            raise RuntimeError(f"Raw node failed Brent verification: {path}")
        engine = nf.FastNormalForm3x3(tables)
        normal = engine.normal_form(scheme)
        if not nf.brent_valid(normal):
            raise RuntimeError(f"Normal form failed Brent verification: {path}")
        text = "".join(f"{nf.format_row(row)}\n" for row in normal)
        quotient_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        normal_path = normal_dir / f"{quotient_hash}.txt"
        if not normal_path.exists():
            normal_path.write_text(text, encoding="utf-8")
        raw_to_quotient[raw_hash] = quotient_hash
        mapping_rows.append({
            "raw_hash": raw_hash,
            "rank": str(rank),
            "representative_file": str(path),
            "quotient_node": quotient_hash,
            "normal_form_file": str(normal_path),
        })
        info = quotient_info.setdefault(quotient_hash, {
            "node_id": quotient_hash,
            "rank": str(rank),
            "representative_file": str(path),
            "normal_form_file": str(normal_path),
            "raw_node_count": 0,
        })
        if info["rank"] != str(rank):
            raise RuntimeError("A quotient node unexpectedly contains two ranks")
        info["raw_node_count"] += 1
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"canonicalized={index}/{len(files)} elapsed={time.time() - started:.1f}s", flush=True)

    write_csv(
        quotient_dir / "raw_to_quotient.csv",
        mapping_rows,
        ["raw_hash", "rank", "representative_file", "quotient_node", "normal_form_file"],
    )

    quotient_edge_counts: Counter[tuple[str, str, str]] = Counter()
    for (raw_source, raw_target, edge_type), count in raw_edge_counts.items():
        source = raw_to_quotient[raw_source]
        target = raw_to_quotient[raw_target]
        if edge_type == "flip":
            source, target = sorted((source, target))
        quotient_edge_counts[(source, target, edge_type)] += count

    quotient_nodes = [
        {
            **info,
            "raw_node_count": str(info["raw_node_count"]),
        }
        for _, info in sorted(quotient_info.items())
    ]
    quotient_edges = [
        {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "source_rank": quotient_info[source]["rank"],
            "target_rank": quotient_info[target]["rank"],
            "event_count": str(count),
            "self_loop": "yes" if source == target else "no",
        }
        for (source, target, edge_type), count in sorted(quotient_edge_counts.items())
    ]
    write_csv(
        quotient_dir / "nodes.csv",
        quotient_nodes,
        ["node_id", "rank", "representative_file", "normal_form_file", "raw_node_count"],
    )
    write_csv(
        quotient_dir / "edges.csv",
        quotient_edges,
        ["source", "target", "edge_type", "source_rank", "target_rank", "event_count", "self_loop"],
    )

    component_list = components(
        quotient_info,
        ((row["source"], row["target"]) for row in quotient_edges),
    )
    rank_counts = Counter(info["rank"] for info in quotient_info.values())
    edge_type_counts = Counter(row["edge_type"] for row in quotient_edges)
    self_loops = sum(row["self_loop"] == "yes" for row in quotient_edges)

    draw_status = "Drawing skipped"
    if not args.no_draw:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx

            graph = nx.Graph()
            for row in quotient_nodes:
                graph.add_node(row["node_id"], rank=int(row["rank"]), raw_node_count=int(row["raw_node_count"]))
            for row in quotient_edges:
                graph.add_edge(row["source"], row["target"], edge_type=row["edge_type"], event_count=int(row["event_count"]))
            nx.write_graphml(graph, quotient_dir / "graph.graphml")
            nx.write_gexf(graph, quotient_dir / "graph.gexf")
            positions = nx.spring_layout(graph, seed=7, iterations=300, method="force")
            flip_edges = [(row["source"], row["target"]) for row in quotient_edges if row["edge_type"] == "flip"]
            reduction_edges = [(row["source"], row["target"]) for row in quotient_edges if row["edge_type"] == "reduction"]
            same_rank = max(rank_counts, key=int)
            upper_nodes = [node for node, data in graph.nodes(data=True) if str(data["rank"]) == same_rank]
            lower_nodes = [node for node in graph if node not in upper_nodes]
            plt.figure(figsize=(14, 10))
            nx.draw_networkx_edges(graph, positions, edgelist=flip_edges, width=0.6, alpha=0.35)
            nx.draw_networkx_edges(graph, positions, edgelist=reduction_edges, width=1.8, edge_color="green", alpha=0.9)
            nx.draw_networkx_nodes(graph, positions, nodelist=upper_nodes, node_size=35, alpha=0.85)
            nx.draw_networkx_nodes(graph, positions, nodelist=lower_nodes, node_size=120, node_color="white", edgecolors="black")
            plt.title("3x3 F2 flip component quotient graph")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(quotient_dir / "graph.png", dpi=220, bbox_inches="tight")
            plt.savefig(quotient_dir / "graph.svg", bbox_inches="tight")
            plt.close()
            draw_status = "GraphML, GEXF, PNG, and SVG written"
        except Exception as error:
            draw_status = f"Drawing skipped because graph libraries failed: {error}"

    summary = "\n".join([
        "3x3 F2 quotient flip graph summary",
        "",
        f"Raw/sorted nodes: {len(files)}",
        f"Raw deduplicated edges: {len(raw_edges)}",
        f"Quotient nodes: {len(quotient_nodes)}",
        f"Quotient edges: {len(quotient_edges)}",
        f"Quotient flip edges: {edge_type_counts.get('flip', 0)}",
        f"Quotient reduction edges: {edge_type_counts.get('reduction', 0)}",
        f"Quotient self-loops: {self_loops}",
        f"Ranks: {dict(sorted(rank_counts.items(), key=lambda item: int(item[0])))}",
        f"Connected components: {len(component_list)}",
        f"Component sizes: {[len(component) for component in component_list]}",
        f"Drawing status: {draw_status}",
    ]) + "\n"
    (quotient_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
