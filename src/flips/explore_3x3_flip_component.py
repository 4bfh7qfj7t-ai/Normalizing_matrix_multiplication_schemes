#!/usr/bin/env python3
"""Explore a same-rank 3x3/F2 flip component and its one-step reductions.

Input schemes use the compact format accepted by normal_form_3x3_fast.py:
one row per multiplication and three nine-bit matrix tokens per row.

Discovery identity is the SHA-1 of the scheme after sorting its summands by
their numeric (A, B, C) bitmasks. This is deliberately only a raw/sorted
identity. Full equivalence is handled by build_3x3_quotient_graph.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Optional


Matrix = int
Row = tuple[Matrix, Matrix, Matrix]
Scheme = tuple[Row, ...]

PLUS1 = (1, 2, 0)
PLUS2 = (2, 0, 1)
FACTOR_NAME = ("A", "B", "C")


def parse_matrix(token: str) -> Matrix:
    token = token.strip()
    if len(token) != 9 or any(ch not in "01" for ch in token):
        raise ValueError(f"expected a nine-bit 0/1 token, got {token!r}")
    value = 0
    for position, ch in enumerate(token):
        if ch == "1":
            value |= 1 << position
    return value


def format_matrix(value: Matrix) -> str:
    return "".join("1" if (value >> i) & 1 else "0" for i in range(9))


def read_scheme(path: Path) -> Scheme:
    rows: list[Row] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: expected three matrix tokens")
        row = tuple(parse_matrix(part) for part in parts)
        if 0 in row:
            raise ValueError(f"{path}:{line_number}: zero factors are not supported")
        rows.append(row)  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"{path}: no scheme rows found")
    return tuple(rows)


def canonical_scheme(scheme: Iterable[Row]) -> Scheme:
    return tuple(sorted(tuple(scheme)))


def scheme_text(scheme: Iterable[Row]) -> str:
    rows = canonical_scheme(scheme)
    return "".join(
        f"{format_matrix(a)} {format_matrix(b)} {format_matrix(c)}\n"
        for a, b, c in rows
    )


def scheme_hash(scheme: Iterable[Row]) -> str:
    return hashlib.sha1(scheme_text(scheme).encode("utf-8")).hexdigest()


def save_scheme(scheme: Scheme, digest: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{digest}.txt"
    if not destination.exists():
        destination.write_text(scheme_text(scheme), encoding="utf-8")
    return destination


def tensor_bitset_of_row(row: Row) -> int:
    a_matrix, b_matrix, c_matrix = row
    output = 0
    for a in range(9):
        if not (a_matrix >> a) & 1:
            continue
        for b in range(9):
            if not (b_matrix >> b) & 1:
                continue
            for c in range(9):
                if (c_matrix >> c) & 1:
                    output ^= 1 << ((a * 9 + b) * 9 + c)
    return output


def multiplication_tensor() -> int:
    output = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = 3 * i + j
                b = 3 * j + k
                c = 3 * k + i
                output |= 1 << ((a * 9 + b) * 9 + c)
    return output


TARGET_TENSOR = multiplication_tensor()


def brent_valid(scheme: Scheme) -> bool:
    represented = 0
    for row in scheme:
        represented ^= tensor_bitset_of_row(row)
    return represented == TARGET_TENSOR


def has_zero_factor(scheme: Scheme) -> bool:
    return any(a == 0 or b == 0 or c == 0 for a, b, c in scheme)


def flip_candidate(scheme: Scheme, common_factor: int, first: int, second: int) -> Optional[Scheme]:
    if scheme[first][common_factor] != scheme[second][common_factor]:
        return None
    rows = [list(row) for row in scheme]
    b = PLUS1[common_factor]
    c = PLUS2[common_factor]
    rows[first][c] ^= rows[second][c]
    rows[second][b] ^= rows[first][b]
    return tuple(tuple(row) for row in rows)  # type: ignore[return-value]


def pair_reductions(scheme: Scheme) -> Iterator[tuple[str, int, int, Scheme]]:
    for first in range(len(scheme)):
        for second in range(first + 1, len(scheme)):
            ai, bi, ci = scheme[first]
            aj, bj, cj = scheme[second]
            if ai == aj and bi == bj and (ci ^ cj):
                rows = list(scheme)
                rows[first] = (ai, bi, ci ^ cj)
                del rows[second]
                yield "same_A_same_B_combine_C", first, second, tuple(rows)
            if ai == aj and ci == cj and (bi ^ bj):
                rows = list(scheme)
                rows[first] = (ai, bi ^ bj, ci)
                del rows[second]
                yield "same_A_same_C_combine_B", first, second, tuple(rows)
            if bi == bj and ci == cj and (ai ^ aj):
                rows = list(scheme)
                rows[first] = (ai ^ aj, bi, ci)
                del rows[second]
                yield "same_B_same_C_combine_A", first, second, tuple(rows)


def collect_seed_paths(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*.txt") if recursive else path.glob("*.txt")
    return sorted(candidate for candidate in iterator if candidate.is_file())


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["source_hash"] for row in csv.DictReader(handle)}


def open_writer(path: Path, fieldnames: list[str], resume: bool):
    append = resume and path.exists() and path.stat().st_size > 0
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
    return handle, writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore a reachable same-rank 3x3/F2 flip component and one-step reductions."
    )
    parser.add_argument("seed_path", type=Path, help="One compact nine-bit scheme or a folder of seeds")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive-seeds", action="store_true")
    parser.add_argument("--max-nodes", type=int, help="Stop after processing this many same-rank nodes")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Use --resume or a new directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    same_rank_dir = output_dir / "same_rank_schemes"
    reduction_dir = output_dir / "reduction_schemes"
    same_rank_dir.mkdir(exist_ok=True)
    reduction_dir.mkdir(exist_ok=True)

    flip_csv = output_dir / "same_rank_flip_edges.csv"
    reduction_csv = output_dir / "reduction_edges.csv"
    counts_csv = output_dir / "per_source_counts.csv"
    invalid_csv = output_dir / "invalid_or_skipped_seeds.csv"
    status_file = output_dir / "status_latest.txt"
    summary_file = output_dir / "summary.txt"

    seed_paths = collect_seed_paths(args.seed_path, args.recursive_seeds)
    if not seed_paths:
        raise SystemExit("No compact .txt seeds found")

    seen: dict[str, Scheme] = {}
    labels: dict[str, str] = {}
    reduction_seen: dict[str, Scheme] = {}
    invalid: list[tuple[str, str]] = []
    source_rank: Optional[int] = None

    if args.resume:
        for path in sorted(same_rank_dir.glob("*.txt")):
            scheme = canonical_scheme(read_scheme(path))
            digest = scheme_hash(scheme)
            seen[digest] = scheme
            labels[digest] = path.name
        for path in sorted(reduction_dir.glob("*.txt")):
            scheme = canonical_scheme(read_scheme(path))
            reduction_seen[scheme_hash(scheme)] = scheme

    seed_hashes: set[str] = set()
    for path in seed_paths:
        try:
            scheme = canonical_scheme(read_scheme(path))
            if source_rank is None:
                source_rank = len(scheme)
            if len(scheme) != source_rank:
                raise ValueError(f"rank {len(scheme)} does not match source rank {source_rank}")
            if not brent_valid(scheme):
                raise ValueError("failed the 3x3/F2 Brent check")
        except Exception as error:
            invalid.append((str(path), str(error)))
            continue
        digest = scheme_hash(scheme)
        seed_hashes.add(digest)
        if digest not in seen:
            seen[digest] = scheme
            labels[digest] = path.name
            save_scheme(scheme, digest, same_rank_dir)

    if source_rank is None or not seed_hashes:
        raise SystemExit("No valid seed schemes were loaded")
    if source_rank < 2:
        raise SystemExit("The source rank must be at least 2")

    with invalid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed_file", "reason"])
        writer.writerows(invalid)

    processed = load_processed(counts_csv) if args.resume else set()
    queue = deque(sorted(digest for digest in seen if digest not in processed))

    flip_fields = [
        "processed_order", "source_file", "source_hash", "source_rank", "edge_type",
        "shared_factor", "row_1", "row_2", "target_hash", "target_file", "target_status",
    ]
    reduction_fields = [
        "processed_order", "source_file", "source_hash", "source_rank", "edge_type",
        "reduction_kind", "row_1", "row_2", "target_hash", "target_rank", "target_file", "target_status",
    ]
    count_fields = [
        "processed_order", "source_file", "source_hash", "source_rank", "valid_flips",
        "new_neighbors", "valid_reductions", "new_reductions", "queue_size", "total_same_rank", "total_reduced_rank",
    ]

    flip_handle, flip_writer = open_writer(flip_csv, flip_fields, args.resume)
    reduction_handle, reduction_writer = open_writer(reduction_csv, reduction_fields, args.resume)
    counts_handle, counts_writer = open_writer(counts_csv, count_fields, args.resume)

    processed_this_run = 0
    processed_order = len(processed)
    started = time.time()

    print(f"Loaded {len(seed_hashes)} valid seed(s) at rank {source_rank}")
    print(f"Starting with {len(seen)} raw/sorted same-rank scheme(s)")
    print(f"Output: {output_dir}")

    try:
        while queue:
            if args.max_nodes is not None and processed_this_run >= args.max_nodes:
                break
            source_hash = queue.popleft()
            if source_hash in processed:
                continue
            scheme = seen[source_hash]
            processed_order += 1
            processed_this_run += 1
            valid_flips = new_neighbors = valid_reductions = new_reductions = 0

            for common_factor in range(3):
                for first in range(len(scheme)):
                    for second in range(first + 1, len(scheme)):
                        if scheme[first][common_factor] != scheme[second][common_factor]:
                            continue
                        for row_1, row_2 in ((first, second), (second, first)):
                            candidate = flip_candidate(scheme, common_factor, row_1, row_2)
                            if candidate is None or has_zero_factor(candidate):
                                continue
                            candidate = canonical_scheme(candidate)
                            if len(candidate) != source_rank or not brent_valid(candidate):
                                continue
                            target_hash = scheme_hash(candidate)
                            valid_flips += 1
                            status = "already_seen"
                            if target_hash not in seen:
                                seen[target_hash] = candidate
                                target_path = save_scheme(candidate, target_hash, same_rank_dir)
                                labels[target_hash] = target_path.name
                                queue.append(target_hash)
                                new_neighbors += 1
                                status = "new_enqueued"
                            target_path = same_rank_dir / f"{target_hash}.txt"
                            flip_writer.writerow({
                                "processed_order": processed_order,
                                "source_file": labels.get(source_hash, f"{source_hash}.txt"),
                                "source_hash": source_hash,
                                "source_rank": source_rank,
                                "edge_type": "flip",
                                "shared_factor": FACTOR_NAME[common_factor],
                                "row_1": row_1,
                                "row_2": row_2,
                                "target_hash": target_hash,
                                "target_file": str(target_path),
                                "target_status": status,
                            })

            for kind, first, second, candidate in pair_reductions(scheme):
                if has_zero_factor(candidate):
                    continue
                candidate = canonical_scheme(candidate)
                if len(candidate) != source_rank - 1 or not brent_valid(candidate):
                    continue
                target_hash = scheme_hash(candidate)
                valid_reductions += 1
                status = "already_seen"
                if target_hash not in reduction_seen:
                    reduction_seen[target_hash] = candidate
                    save_scheme(candidate, target_hash, reduction_dir)
                    new_reductions += 1
                    status = "new_saved"
                target_path = reduction_dir / f"{target_hash}.txt"
                reduction_writer.writerow({
                    "processed_order": processed_order,
                    "source_file": labels.get(source_hash, f"{source_hash}.txt"),
                    "source_hash": source_hash,
                    "source_rank": source_rank,
                    "edge_type": "reduction",
                    "reduction_kind": kind,
                    "row_1": first,
                    "row_2": second,
                    "target_hash": target_hash,
                    "target_rank": source_rank - 1,
                    "target_file": str(target_path),
                    "target_status": status,
                })

            processed.add(source_hash)
            counts_writer.writerow({
                "processed_order": processed_order,
                "source_file": labels.get(source_hash, f"{source_hash}.txt"),
                "source_hash": source_hash,
                "source_rank": source_rank,
                "valid_flips": valid_flips,
                "new_neighbors": new_neighbors,
                "valid_reductions": valid_reductions,
                "new_reductions": new_reductions,
                "queue_size": len(queue),
                "total_same_rank": len(seen),
                "total_reduced_rank": len(reduction_seen),
            })

            if args.progress_every > 0 and processed_this_run % args.progress_every == 0:
                elapsed = time.time() - started
                message = (
                    f"processed_this_run={processed_this_run} total_processed={len(processed)} "
                    f"same_rank_seen={len(seen)} queue={len(queue)} "
                    f"reductions_seen={len(reduction_seen)} elapsed={elapsed:.1f}s"
                )
                print(message, flush=True)
                flip_handle.flush()
                reduction_handle.flush()
                counts_handle.flush()
                status_file.write_text(message + "\n", encoding="utf-8")
    finally:
        flip_handle.close()
        reduction_handle.close()
        counts_handle.close()

    elapsed = time.time() - started
    complete = not queue
    summary = "\n".join([
        "Recursive 3x3 F2 flip exploration summary",
        "",
        f"Seed path: {args.seed_path}",
        f"Source rank: {source_rank}",
        f"Reduction rank: {source_rank - 1}",
        f"Valid unique seeds: {len(seed_hashes)}",
        f"Invalid or skipped seeds: {len(invalid)}",
        f"Processed this run: {processed_this_run}",
        f"Processed total: {len(processed)}",
        f"Queue empty: {complete}",
        f"Unprocessed queue size: {len(queue)}",
        f"Same-rank raw/sorted schemes reached: {len(seen)}",
        f"Reduced-rank raw/sorted schemes reached: {len(reduction_seen)}",
        f"Recorded directed flip events: {csv_row_count(flip_csv)}",
        f"Recorded reduction events: {csv_row_count(reduction_csv)}",
        f"Elapsed seconds this run: {elapsed:.3f}",
        "",
        "This is the component reachable from the supplied seeds. It does not prove that every",
        "3x3 scheme lies in this component.",
    ]) + "\n"
    summary_file.write_text(summary, encoding="utf-8")
    status_file.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
