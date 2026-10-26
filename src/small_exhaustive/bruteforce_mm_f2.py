#!/usr/bin/env python3
"""
Brute-force matrix-multiplication schemes over F2 for small formats.

For format (m,n,p), this searches rank-r unordered sets of rank-one tensors
for multiplication A_{m x n} * B_{n x p} -> output_{m x p}.

Scheme identity checked by Brent equations over F2:
    sum_t alpha_t(a) beta_t(b) gamma_t(c) = sum_{i,j,k} a_ij b_jk c_ki

Output is saved in .exp style:
    (a11+a12)*(b21)*(c11+c12)

Important dimensions:
  --dims 1 2 1  means 1x2 times 2x1 -> 1x1 dot product.
                 Over F2, there are 3*3*1 = 9 possible rank-one tensors;
                 rank 2 has C(9,2)=36 candidates.

  --dims 2 1 2  means 2x1 times 1x2 -> 2x2 outer product.
                 Over F2, there are 3*3*15 = 135 possible rank-one tensors;
                 rank 4 has C(135,4)=13,232,835 candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Tuple


Tensor = Tuple[int, int, int, int]  # (a_mask, b_mask, c_mask, full_tensor_bitset)


def bit_positions(mask: int) -> Iterable[int]:
    """Yield zero-based bit positions set in mask."""
    x = mask
    pos = 0
    while x:
        if x & 1:
            yield pos
        x >>= 1
        pos += 1


def var_names(prefix: str, rows: int, cols: int) -> List[str]:
    """Names in row-major order: a11, a12, ..., a21, ..."""
    return [f"{prefix}{i}{j}" for i in range(1, rows + 1) for j in range(1, cols + 1)]


def linear_form(mask: int, names: List[str]) -> str:
    terms = [names[i] for i in bit_positions(mask)]
    if not terms:
        return "0"
    return "+".join(terms)


def term_to_string(t: Tensor, a_names: List[str], b_names: List[str], c_names: List[str]) -> str:
    a, b, c, _ = t
    return f"({linear_form(a, a_names)})*({linear_form(b, b_names)})*({linear_form(c, c_names)})"


def scheme_to_text(tensors: List[Tensor], a_names: List[str], b_names: List[str], c_names: List[str]) -> str:
    lines = [term_to_string(t, a_names, b_names, c_names) for t in tensors]
    # Sorted-summand canonical text. This removes only product-row order.
    lines.sort()
    return "\n".join(lines) + "\n"


def tensor_bitset(a_mask: int, b_mask: int, c_mask: int, a_dim: int, b_dim: int, c_dim: int) -> int:
    """Return full coefficient tensor as a Python integer bitset."""
    out = 0
    # coordinate index = ((a_index * b_dim) + b_index) * c_dim + c_index
    for ai in bit_positions(a_mask):
        base_ab = ai * b_dim
        for bi in bit_positions(b_mask):
            base = (base_ab + bi) * c_dim
            for ci in bit_positions(c_mask):
                out ^= 1 << (base + ci)  # XOR over F2
    return out


def build_rank_one_tensors(m: int, n: int, p: int) -> List[Tensor]:
    a_dim = m * n
    b_dim = n * p
    c_dim = p * m  # c_{k,i}: rows p, cols m

    tensors: List[Tensor] = []
    for a_mask in range(1, 1 << a_dim):
        for b_mask in range(1, 1 << b_dim):
            for c_mask in range(1, 1 << c_dim):
                full = tensor_bitset(a_mask, b_mask, c_mask, a_dim, b_dim, c_dim)
                tensors.append((a_mask, b_mask, c_mask, full))
    return tensors


def target_tensor(m: int, n: int, p: int) -> int:
    """Target sum_{i,j,k} a_ij b_jk c_ki as bitset."""
    a_dim = m * n
    b_dim = n * p
    c_dim = p * m

    out = 0
    # zero-based i in [0,m), j in [0,n), k in [0,p)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                a_index = i * n + j      # a_{i+1,j+1}
                b_index = j * p + k      # b_{j+1,k+1}
                c_index = k * m + i      # c_{k+1,i+1}
                coord = ((a_index * b_dim) + b_index) * c_dim + c_index
                out ^= 1 << coord
    return out


def xor_scheme(tensors: List[Tensor], combo: Tuple[int, ...]) -> int:
    acc = 0
    for idx in combo:
        acc ^= tensors[idx][3]
    return acc


def brent_check(tensors: List[Tensor], combo: Tuple[int, ...], target: int) -> bool:
    return xor_scheme(tensors, combo) == target


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def save_tensor_dictionary(path: Path, tensors: List[Tensor], a_names: List[str], b_names: List[str], c_names: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tensor_id", "a_mask", "b_mask", "c_mask", "term", "tensor_bitset_hex"],
        )
        writer.writeheader()
        for i, t in enumerate(tensors):
            writer.writerow({
                "tensor_id": i,
                "a_mask": t[0],
                "b_mask": t[1],
                "c_mask": t[2],
                "term": term_to_string(t, a_names, b_names, c_names),
                "tensor_bitset_hex": hex(t[3]),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force small matrix multiplication schemes over F2.")
    parser.add_argument("--dims", nargs=3, type=int, metavar=("M", "N", "P"), required=True,
                        help="Matrix format: A is MxN, B is NxP, output is MxP.")
    parser.add_argument("--rank", type=int, required=True, help="Rank/number of products in each candidate scheme.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where results will be saved.")
    parser.add_argument("--progress-every", type=int, default=1_000_000,
                        help="Print progress after this many candidates checked.")
    parser.add_argument("--save-exp", action="store_true",
                        help="Save each valid scheme as its own .exp file. CSV is always saved.")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Optional safety stop after this many candidates checked.")
    args = parser.parse_args()

    m, n, p = args.dims
    r = args.rank
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_dir = out_dir / "valid_schemes_exp"
    if args.save_exp:
        valid_dir.mkdir(parents=True, exist_ok=True)

    a_names = var_names("a", m, n)
    b_names = var_names("b", n, p)
    c_names = var_names("c", p, m)

    tensors = build_rank_one_tensors(m, n, p)
    target = target_tensor(m, n, p)
    total_candidates = math.comb(len(tensors), r)

    save_tensor_dictionary(out_dir / "tensor_dictionary.csv", tensors, a_names, b_names, c_names)

    valid_csv_path = out_dir / "valid_schemes.csv"
    summary_path = out_dir / "summary.txt"

    print("Brute-force matrix multiplication over F2")
    print(f"Format: ({m},{n},{p}) = {m}x{n} times {n}x{p}")
    print(f"Rank: {r}")
    print(f"A variables: {len(a_names)}")
    print(f"B variables: {len(b_names)}")
    print(f"C variables: {len(c_names)}")
    print(f"Rank-one tensors: {len(tensors)}")
    print(f"Candidate unordered rank-{r} schemes: {total_candidates:,}")
    print(f"Output folder: {out_dir}")
    print()

    start = time.time()
    checked = 0
    valid = 0
    valid_hashes = set()

    with valid_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scheme_index", "sorted_summand_hash", "tensor_ids", "scheme_file"],
        )
        writer.writeheader()

        for combo in combinations(range(len(tensors)), r):
            checked += 1

            if brent_check(tensors, combo, target):
                scheme_tensors = [tensors[i] for i in combo]
                text = scheme_to_text(scheme_tensors, a_names, b_names, c_names)
                h = sha1_text(text)
                valid_hashes.add(h)

                scheme_file = ""
                if args.save_exp:
                    scheme_file = str(valid_dir / f"scheme_{valid:06d}_{h[:12]}.exp")
                    Path(scheme_file).write_text(text)

                writer.writerow({
                    "scheme_index": valid,
                    "sorted_summand_hash": h,
                    "tensor_ids": " ".join(map(str, combo)),
                    "scheme_file": scheme_file,
                })
                valid += 1

            if args.progress_every and checked % args.progress_every == 0:
                elapsed = time.time() - start
                rate = checked / elapsed if elapsed else 0.0
                print(
                    f"checked={checked:,}/{total_candidates:,} | "
                    f"valid={valid:,} | unique_hashes={len(valid_hashes):,} | "
                    f"rate={rate:,.0f}/s | elapsed={elapsed:.1f}s"
                )

            if args.max_candidates is not None and checked >= args.max_candidates:
                print("Stopped early because --max-candidates was reached.")
                break

    elapsed = time.time() - start

    lines = []
    def add(x=""):
        lines.append(str(x))

    add("Brute-force matrix multiplication over F2 summary")
    add()
    add(f"Format: ({m},{n},{p}) = {m}x{n} times {n}x{p}")
    add(f"Rank: {r}")
    add(f"A variables: {len(a_names)}")
    add(f"B variables: {len(b_names)}")
    add(f"C variables: {len(c_names)}")
    add(f"Rank-one tensors: {len(tensors)}")
    add(f"Candidate unordered rank-{r} schemes total: {total_candidates}")
    add(f"Candidates checked: {checked}")
    add(f"Valid Brent-checked schemes: {valid}")
    add(f"Unique sorted-summand hashes among valid schemes: {len(valid_hashes)}")
    add(f"Target tensor bitset: {hex(target)}")
    add(f"Elapsed seconds: {elapsed:.3f}")
    add()
    add(f"Tensor dictionary: {out_dir / 'tensor_dictionary.csv'}")
    add(f"Valid schemes CSV: {valid_csv_path}")
    if args.save_exp:
        add(f"Valid .exp folder: {valid_dir}")
    else:
        add("Valid .exp files were not saved. Re-run with --save-exp if needed.")

    summary = "\n".join(lines)
    summary_path.write_text(summary)

    print()
    print(summary)
    print()
    print("Finished.")


if __name__ == "__main__":
    main()
