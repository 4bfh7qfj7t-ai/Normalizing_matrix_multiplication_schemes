#!/usr/bin/env python3
"""
Recursive / BFS exploration of rank-8 flip neighbors for 2x2 matrix
multiplication schemes over F2 in .exp format.

What this script does:
  1. Starts from a folder of rank-8 .exp schemes.
  2. Uses sorted-summand hash for RAW discovery.
  3. Processes schemes in a queue.
  4. For each rank-8 scheme, enumerates every directed one-flip candidate.
  5. Brent-checks each candidate.
  6. New valid rank-8 schemes are saved and added to the queue.
  7. Also checks every rank-8 -> rank-7 pair reduction.
  8. Stops only when the queue is empty.

This explores the whole flip-connected component reachable from the seeds.
It does not prove that no disconnected rank-8 components exist elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

Row = Tuple[int, int, int]          # (A_factor, B_factor, C_factor), stored as bitmasks
Scheme = List[Row]

PLUS1 = [1, 2, 0]
PLUS2 = [2, 0, 1]
FACTOR_NAME = ["A", "B", "C"]


def bit_index(i: int, j: int, cols: int) -> int:
    """1-based matrix index (i,j) to row-major bit index."""
    return (i - 1) * cols + (j - 1)


def parse_factor(text: str, letter: str, rows: int, cols: int) -> int:
    """Parse a factor such as '(a11+a22)' into a bitmask over F2."""
    s = text.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    s = s.strip()
    if not s:
        raise ValueError(f"empty {letter}-factor")

    mask = 0
    for term in s.split("+"):
        term = term.strip()
        if len(term) != 3 or term[0] != letter or not term[1:].isdigit():
            raise ValueError(f"bad term {term!r} in factor {text!r}")
        i = int(term[1])
        j = int(term[2])
        if not (1 <= i <= rows and 1 <= j <= cols):
            raise ValueError(f"index out of range in term {term!r}")
        mask ^= 1 << bit_index(i, j, cols)   # XOR because we are over F2

    if mask == 0:
        raise ValueError(f"zero {letter}-factor after parsing {text!r}")
    return mask


def read_exp(path: Path, n: int, m: int, p: int) -> Scheme:
    """Read an .exp scheme file."""
    scheme: Scheme = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("*")
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_no}: expected exactly two '*': {line!r}")
        a = parse_factor(parts[0], "a", n, m)
        b = parse_factor(parts[1], "b", m, p)
        c = parse_factor(parts[2], "c", p, n)
        scheme.append((a, b, c))
    if not scheme:
        raise ValueError(f"{path}: no scheme rows found")
    return scheme


def write_factor(mask: int, letter: str, rows: int, cols: int) -> str:
    terms = []
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if mask & (1 << bit_index(i, j, cols)):
                terms.append(f"{letter}{i}{j}")
    if not terms:
        return "(0)"
    return "(" + "+".join(terms) + ")"


def row_to_exp(row: Row, n: int, m: int, p: int) -> str:
    a, b, c = row
    return (
        write_factor(a, "a", n, m)
        + "*"
        + write_factor(b, "b", m, p)
        + "*"
        + write_factor(c, "c", p, n)
    )


def scheme_to_exp(scheme: Scheme, n: int, m: int, p: int, sort_rows: bool = False) -> str:
    rows = list(scheme)
    if sort_rows:
        rows = sorted(rows)
    return "\n".join(row_to_exp(row, n, m, p) for row in rows) + "\n"


def canonical_text(scheme: Scheme, n: int, m: int, p: int) -> str:
    """Canonical only under summand sorting, not full equivalence."""
    return scheme_to_exp(scheme, n, m, p, sort_rows=True)


def canonical_hash(scheme: Scheme, n: int, m: int, p: int) -> str:
    return hashlib.sha1(canonical_text(scheme, n, m, p).encode()).hexdigest()


def has_zero_factor(scheme: Scheme) -> bool:
    return any(a == 0 or b == 0 or c == 0 for a, b, c in scheme)


def brent_check_f2(scheme: Scheme, n: int, m: int, p: int) -> bool:
    """GF(2) Brent/tensor correctness check."""
    a_size = n * m
    b_size = m * p
    c_size = p * n

    target = [0] * (a_size * b_size)

    # Target matrix multiplication tensor: a_{i,j} b_{j,k} c_{k,i}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = m * i + j
                b_idx = p * j + k
                c_idx = n * k + i
                target[a_idx * b_size + b_idx] ^= 1 << c_idx

    # XOR the tensor represented by the scheme.
    for a, b, c in scheme:
        for ai in range(a_size):
            if not (a & (1 << ai)):
                continue
            for bi in range(b_size):
                if b & (1 << bi):
                    target[ai * b_size + bi] ^= c

    return all(x == 0 for x in target)


def flip_candidate(scheme: Scheme, common_col: int, r1: int, r2: int) -> Optional[Scheme]:
    """
    Directed flip matching Tensor::flip(common_col, r1, r2, reduce_flag=false).

    The two rows must share factor common_col.
    Both orientations are obtained by calling this with (r1,r2) and (r2,r1).
    """
    if scheme[r1][common_col] != scheme[r2][common_col]:
        return None

    rows = [list(row) for row in scheme]
    b = PLUS1[common_col]
    c = PLUS2[common_col]

    rows[r1][c] ^= rows[r2][c]
    rows[r2][b] ^= rows[r1][b]

    return [tuple(row) for row in rows]  # type: ignore[list-item]


def pair_reduction_candidates(scheme: Scheme) -> Iterable[Tuple[str, int, int, Scheme]]:
    """
    Enumerate rank-1 pair reductions.

    Returns (reduction_kind, row_i, row_j, reduced_scheme).
    """
    r = len(scheme)
    for i in range(r):
        for j in range(i + 1, r):
            ai, bi, ci = scheme[i]
            aj, bj, cj = scheme[j]

            # Same A and same B: combine C factors, remove j.
            if ai == aj and bi == bj:
                new_c = ci ^ cj
                if new_c != 0:
                    rows = list(scheme)
                    rows[i] = (ai, bi, new_c)
                    del rows[j]
                    yield "same_A_same_B_combine_C", i, j, rows

            # Same A and same C: combine B factors, remove j.
            if ai == aj and ci == cj:
                new_b = bi ^ bj
                if new_b != 0:
                    rows = list(scheme)
                    rows[i] = (ai, new_b, ci)
                    del rows[j]
                    yield "same_A_same_C_combine_B", i, j, rows

            # Same B and same C: combine A factors, remove j.
            if bi == bj and ci == cj:
                new_a = ai ^ aj
                if new_a != 0:
                    rows = list(scheme)
                    rows[i] = (new_a, bi, ci)
                    del rows[j]
                    yield "same_B_same_C_combine_A", i, j, rows


def save_scheme_by_hash(
    scheme: Scheme,
    h: str,
    out_dir: Path,
    n: int,
    m: int,
    p: int,
    suffix: str = ".exp",
) -> str:
    """Save sorted/canonical .exp text using the sorted-summand hash as filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = h[:16] + suffix
    path = out_dir / name
    if not path.exists():
        path.write_text(canonical_text(scheme, n, m, p))
    return str(path)


def write_status(path: Path, lines: List[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank8_dir", type=Path, help="Seed folder containing rank-8 .exp schemes")
    parser.add_argument("--output-dir", type=Path, default=Path("rank8_recursive_flip_exploration"))
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=50,
                        help="Print progress after this many processed rank-8 schemes")
    parser.add_argument("--skip-brent", action="store_true",
                        help="Do not Brent-check generated candidates. Not recommended.")
    args = parser.parse_args()

    n, m, p = args.n, args.m, args.p
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rank8_scheme_dir = out_dir / "rank8_discovered_schemes"
    rank7_reduction_dir = out_dir / "rank7_reduction_outputs"
    rank8_scheme_dir.mkdir(parents=True, exist_ok=True)
    rank7_reduction_dir.mkdir(parents=True, exist_ok=True)

    flip_csv = out_dir / "rank8_flip_edges.csv"
    reduction_csv = out_dir / "rank8_to_rank7_reduction_edges.csv"
    per_source_csv = out_dir / "per_source_counts.csv"
    invalid_csv = out_dir / "invalid_or_skipped_sources.csv"
    status_txt = out_dir / "status_latest.txt"
    summary_txt = out_dir / "summary.txt"

    seed_paths = sorted(args.rank8_dir.glob("*.exp"))

    # RAW discovery state. Hashes are sorted-summand hashes.
    seen_rank8: Dict[str, Scheme] = {}
    seen_rank8_label: Dict[str, str] = {}
    seen_rank8_origin: Dict[str, str] = {}
    seen_rank7: Dict[str, str] = {}
    queue: Deque[str] = deque()

    invalid_sources: List[Tuple[str, str]] = []

    # Load and validate seeds.
    for source_path in seed_paths:
        try:
            scheme = read_exp(source_path, n, m, p)
        except Exception as e:
            invalid_sources.append((source_path.name, f"read_error: {e}"))
            continue

        if len(scheme) != 8:
            invalid_sources.append((source_path.name, f"rank_is_{len(scheme)}_not_8"))
            continue

        if not args.skip_brent and not brent_check_f2(scheme, n, m, p):
            invalid_sources.append((source_path.name, "source_failed_brent"))
            continue

        h = canonical_hash(scheme, n, m, p)
        if h not in seen_rank8:
            seen_rank8[h] = scheme
            # Save seeds too, so the output folder is self-contained.
            saved_path = save_scheme_by_hash(scheme, h, rank8_scheme_dir, n, m, p)
            seen_rank8_label[h] = source_path.name
            seen_rank8_origin[h] = f"seed:{source_path.name}; saved:{saved_path}"
            queue.append(h)

    with invalid_csv.open("w", newline="") as f_invalid:
        invalid_writer = csv.writer(f_invalid)
        invalid_writer.writerow(["source_file", "reason"])
        invalid_writer.writerows(invalid_sources)

    counters = {
        "processed_rank8_sources": 0,
        "raw_directed_flip_attempts": 0,
        "valid_directed_flip_outputs": 0,
        "new_rank8_schemes_discovered": 0,
        "raw_pair_reduction_attempts": 0,
        "valid_rank7_pair_reductions": 0,
        "new_rank7_schemes_discovered": 0,
        "flip_edge_rows_written": 0,
        "reduction_edge_rows_written": 0,
    }

    start_time = time.time()

    flip_fields = [
        "processed_order",
        "source_file",
        "source_sorted_hash",
        "edge_type",
        "shared_factor",
        "row_1",
        "row_2",
        "target_sorted_hash",
        "target_known_rank8_file",
        "target_output_file_if_new",
        "target_status",
    ]

    reduction_fields = [
        "processed_order",
        "source_file",
        "source_sorted_hash",
        "edge_type",
        "reduction_kind",
        "row_1",
        "row_2",
        "target_sorted_hash",
        "target_known_rank7_file",
        "target_output_file_if_new",
        "target_status",
    ]

    per_source_fields = [
        "processed_order",
        "source_file",
        "source_sorted_hash",
        "valid_directed_rank8_flips",
        "new_rank8_neighbors",
        "valid_rank7_pair_reductions",
        "new_rank7_reductions",
        "queue_size_after_processing",
        "total_seen_rank8_after_processing",
        "total_seen_rank7_after_processing",
    ]

    print("Recursive rank-8 flip exploration")
    print(f"Seed folder: {args.rank8_dir}")
    print(f"Output folder: {out_dir}")
    print(f"Seed .exp files seen: {len(seed_paths)}")
    print(f"Valid unique rank-8 seeds loaded: {len(seen_rank8)}")
    print(f"Invalid/skipped seeds: {len(invalid_sources)}")
    print(f"Progress interval: every {args.progress_every} processed rank-8 schemes")
    print("Starting BFS...\n")
    sys.stdout.flush()

    with flip_csv.open("w", newline="") as f_flip, \
         reduction_csv.open("w", newline="") as f_red, \
         per_source_csv.open("w", newline="") as f_counts:

        flip_writer = csv.DictWriter(f_flip, fieldnames=flip_fields)
        reduction_writer = csv.DictWriter(f_red, fieldnames=reduction_fields)
        per_source_writer = csv.DictWriter(f_counts, fieldnames=per_source_fields)
        flip_writer.writeheader()
        reduction_writer.writeheader()
        per_source_writer.writeheader()

        while queue:
            source_hash = queue.popleft()
            scheme = seen_rank8[source_hash]
            source_label = seen_rank8_label.get(source_hash, source_hash[:16])

            counters["processed_rank8_sources"] += 1
            processed_order = counters["processed_rank8_sources"]

            source_flip_count = 0
            source_new_rank8_count = 0
            source_reduction_count = 0
            source_new_rank7_count = 0

            # Same-rank flips.
            for common_col in range(3):
                for i in range(len(scheme)):
                    for j in range(i + 1, len(scheme)):
                        if scheme[i][common_col] != scheme[j][common_col]:
                            continue
                        for r1, r2 in ((i, j), (j, i)):
                            counters["raw_directed_flip_attempts"] += 1
                            candidate = flip_candidate(scheme, common_col, r1, r2)
                            if candidate is None:
                                continue
                            if has_zero_factor(candidate):
                                continue
                            if len(candidate) != 8:
                                continue
                            if not args.skip_brent and not brent_check_f2(candidate, n, m, p):
                                continue

                            target_hash = canonical_hash(candidate, n, m, p)
                            counters["valid_directed_flip_outputs"] += 1
                            source_flip_count += 1

                            target_known_label = ""
                            target_output_file = ""
                            target_status = "already_seen"

                            if target_hash not in seen_rank8:
                                # New raw/sorted rank-8 scheme found.
                                seen_rank8[target_hash] = candidate
                                target_output_file = save_scheme_by_hash(candidate, target_hash, rank8_scheme_dir, n, m, p)
                                seen_rank8_label[target_hash] = target_output_file
                                seen_rank8_origin[target_hash] = f"flip_from:{source_hash}; saved:{target_output_file}"
                                queue.append(target_hash)
                                counters["new_rank8_schemes_discovered"] += 1
                                source_new_rank8_count += 1
                                target_status = "new_enqueued"
                            else:
                                target_known_label = seen_rank8_label.get(target_hash, target_hash[:16])

                            flip_writer.writerow({
                                "processed_order": processed_order,
                                "source_file": source_label,
                                "source_sorted_hash": source_hash,
                                "edge_type": "flip_rank8_to_rank8",
                                "shared_factor": FACTOR_NAME[common_col],
                                "row_1": r1,
                                "row_2": r2,
                                "target_sorted_hash": target_hash,
                                "target_known_rank8_file": target_known_label,
                                "target_output_file_if_new": target_output_file,
                                "target_status": target_status,
                            })
                            counters["flip_edge_rows_written"] += 1

            # Rank-7 pair reductions.
            for kind, i, j, reduced in pair_reduction_candidates(scheme):
                counters["raw_pair_reduction_attempts"] += 1
                if has_zero_factor(reduced):
                    continue
                if len(reduced) != 7:
                    continue
                if not args.skip_brent and not brent_check_f2(reduced, n, m, p):
                    continue

                target_hash = canonical_hash(reduced, n, m, p)
                counters["valid_rank7_pair_reductions"] += 1
                source_reduction_count += 1

                target_known_label = ""
                target_output_file = ""
                target_status = "already_seen"

                if target_hash not in seen_rank7:
                    target_output_file = save_scheme_by_hash(reduced, target_hash, rank7_reduction_dir, n, m, p)
                    seen_rank7[target_hash] = target_output_file
                    counters["new_rank7_schemes_discovered"] += 1
                    source_new_rank7_count += 1
                    target_status = "new_saved"
                else:
                    target_known_label = seen_rank7[target_hash]

                reduction_writer.writerow({
                    "processed_order": processed_order,
                    "source_file": source_label,
                    "source_sorted_hash": source_hash,
                    "edge_type": "reduction_rank8_to_rank7",
                    "reduction_kind": kind,
                    "row_1": i,
                    "row_2": j,
                    "target_sorted_hash": target_hash,
                    "target_known_rank7_file": target_known_label,
                    "target_output_file_if_new": target_output_file,
                    "target_status": target_status,
                })
                counters["reduction_edge_rows_written"] += 1

            per_source_writer.writerow({
                "processed_order": processed_order,
                "source_file": source_label,
                "source_sorted_hash": source_hash,
                "valid_directed_rank8_flips": source_flip_count,
                "new_rank8_neighbors": source_new_rank8_count,
                "valid_rank7_pair_reductions": source_reduction_count,
                "new_rank7_reductions": source_new_rank7_count,
                "queue_size_after_processing": len(queue),
                "total_seen_rank8_after_processing": len(seen_rank8),
                "total_seen_rank7_after_processing": len(seen_rank7),
            })

            if args.progress_every > 0 and processed_order % args.progress_every == 0:
                elapsed = time.time() - start_time
                msg = (
                    f"processed={processed_order:,} | "
                    f"rank8_seen={len(seen_rank8):,} | "
                    f"queue={len(queue):,} | "
                    f"rank7_seen={len(seen_rank7):,} | "
                    f"valid_flip_edges={counters['valid_directed_flip_outputs']:,} | "
                    f"valid_reductions={counters['valid_rank7_pair_reductions']:,} | "
                    f"elapsed={elapsed:.1f}s"
                )
                print(msg)
                sys.stdout.flush()
                f_flip.flush()
                f_red.flush()
                f_counts.flush()
                write_status(status_txt, [
                    "Recursive rank-8 flip exploration status",
                    "",
                    f"Seed folder: {args.rank8_dir}",
                    f"Output folder: {out_dir}",
                    f"Processed rank-8 schemes: {processed_order}",
                    f"Rank-8 raw/sorted schemes seen: {len(seen_rank8)}",
                    f"Queue size: {len(queue)}",
                    f"Rank-7 raw/sorted reductions seen: {len(seen_rank7)}",
                    f"Valid directed flip outputs: {counters['valid_directed_flip_outputs']}",
                    f"Valid rank-7 reductions: {counters['valid_rank7_pair_reductions']}",
                    f"Elapsed seconds: {elapsed:.1f}",
                ])

    elapsed = time.time() - start_time

    summary = f"""Recursive rank-8 flip exploration summary

Seed folder: {args.rank8_dir}
Output folder: {out_dir}
Format: ({n},{m},{p}) over F2
Discovery identity: sorted-summand hash

Seed .exp files seen: {len(seed_paths)}
Valid unique rank-8 seeds loaded: {len([h for h, origin in seen_rank8_origin.items() if origin.startswith('seed:')])}
Invalid/skipped seeds: {len(invalid_sources)}

Final BFS status:
Processed rank-8 schemes: {counters['processed_rank8_sources']}
Queue empty: {len(queue) == 0}
Rank-8 raw/sorted schemes reached: {len(seen_rank8)}
Rank-7 raw/sorted reduction outputs reached: {len(seen_rank7)}

Flip enumeration totals:
Raw directed same-rank flip attempts: {counters['raw_directed_flip_attempts']}
Valid directed same-rank flip outputs: {counters['valid_directed_flip_outputs']}
New rank-8 schemes discovered after seeds: {counters['new_rank8_schemes_discovered']}
Flip edge CSV: {flip_csv}
Rank-8 discovered schemes folder: {rank8_scheme_dir}

Reduction totals:
Raw pair-reduction attempts: {counters['raw_pair_reduction_attempts']}
Valid rank-8 -> rank-7 pair reductions: {counters['valid_rank7_pair_reductions']}
New rank-7 sorted-summand outputs discovered: {counters['new_rank7_schemes_discovered']}
Reduction edge CSV: {reduction_csv}
Rank-7 reduction outputs folder: {rank7_reduction_dir}

Other output files:
Per-source counts CSV: {per_source_csv}
Invalid/skipped seed CSV: {invalid_csv}
Latest status file: {status_txt}

Elapsed seconds: {elapsed:.1f}

Important interpretation:
This exhausts the rank-8 flip-connected component reachable from the seed folder.
It does not prove that there are no disconnected rank-8 components elsewhere.
"""

    summary_txt.write_text(summary)
    write_status(status_txt, summary.splitlines())
    print("\n" + summary)


if __name__ == "__main__":
    main()
