#!/usr/bin/env python3
"""Fast normal forms for 3x3 matrix-multiplication schemes over F_2.

This is a drop-in companion to the earlier ``normal_form_one.py`` project.
It reads and writes the same compact 9-bit row-major format::

    AAAAAAAAA BBBBBBBBB CCCCCCCCC

one multiplication term per line.

The normal form is the same Algorithm-1 normal form used by Kauers and
Moosbauer and by the earlier project:

* maximize the sorted rank pattern over the six tensor-factor permutations;
* among those candidates, choose the lexicographically smallest orbit element;
* compare matrices in colexicographic column order;
* allow row permutations, factor permutations (with transpose for odd
  permutations), and the GL(3,F_2)^3 sandwich action.

The speedup is in the first row.  The older implementation minimized every
candidate first row by scanning all 168^3 = 4,741,632 triples (U,V,W).
Here the special first-row treatment (Algorithm 2 in the paper) is used:

* if A and B are invertible, only 168 conjugations are examined;
* if A is invertible and B is singular, at most 168^2 pairs are examined;
* if A is singular and nonzero, only the transporter pairs mapping A to its
  canonical rank matrix are combined with W.

After the first row, the usual stabilizer-based Algorithm 1 continues.

The implementation is specialized to n=3 and F_2 so it can use compact
9-bit integers and precomputed multiplication/action tables.  This is both
faster and substantially lighter than repeatedly constructing Python tuples of
matrix entries inside the main search loops.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Matrix = int                 # nine row-major bits, positions 0..8
Row = tuple[Matrix, Matrix, Matrix]
Scheme = tuple[Row, ...]
GroupElement = tuple[int, int, int]  # indices of U,V,W in GL(3,F_2)


# ---------------------------------------------------------------------------
# 3x3 matrices over F_2
# ---------------------------------------------------------------------------

N = 3
MATRIX_COUNT = 1 << 9
IDENTITY: Matrix = (1 << 0) | (1 << 4) | (1 << 8)
COLEX_POSITIONS = (2, 5, 8, 1, 4, 7, 0, 3, 6)


def mat_mul_raw(left: Matrix, right: Matrix) -> Matrix:
    """Multiply two 3x3 binary matrices stored as nine-bit integers."""
    output = 0
    for i in range(3):
        for j in range(3):
            value = 0
            for k in range(3):
                value ^= (
                    ((left >> (3 * i + k)) & 1)
                    & ((right >> (3 * k + j)) & 1)
                )
            if value:
                output |= 1 << (3 * i + j)
    return output


def mat_transpose_raw(matrix: Matrix) -> Matrix:
    output = 0
    for i in range(3):
        for j in range(3):
            if (matrix >> (3 * i + j)) & 1:
                output |= 1 << (3 * j + i)
    return output


def mat_rank_raw(matrix: Matrix) -> int:
    """Rank over F_2 using row XOR elimination."""
    rows = [
        (matrix >> (3 * row)) & 0b111
        for row in range(3)
    ]
    rank = 0
    for column in range(3):
        pivot = None
        for row in range(rank, 3):
            if (rows[row] >> column) & 1:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row in range(3):
            if row != rank and ((rows[row] >> column) & 1):
                rows[row] ^= rows[rank]
        rank += 1
    return rank


def mat_inverse_raw(matrix: Matrix) -> Matrix:
    """Inverse of an invertible 3x3 binary matrix."""
    # Each augmented row contains 3 left bits and 3 right bits.
    rows = []
    for i in range(3):
        left = (matrix >> (3 * i)) & 0b111
        right = 1 << i
        rows.append(left | (right << 3))

    for column in range(3):
        pivot = None
        for row in range(column, 3):
            if (rows[row] >> column) & 1:
                pivot = row
                break
        if pivot is None:
            raise ValueError("Matrix is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(3):
            if row != column and ((rows[row] >> column) & 1):
                rows[row] ^= rows[column]

    output = 0
    for i in range(3):
        right = (rows[i] >> 3) & 0b111
        output |= right << (3 * i)
    return output


def colex_key_raw(matrix: Matrix) -> int:
    """Binary integer whose order equals the paper's matrix colex order."""
    key = 0
    for position in COLEX_POSITIONS:
        key = (key << 1) | ((matrix >> position) & 1)
    return key


def canonical_rank_matrix(rank: int) -> Matrix:
    """The Proposition-2 minimal 3x3 matrix of the requested rank."""
    if rank < 0 or rank > 3:
        raise ValueError(f"Bad rank: {rank}")
    output = 0
    # Block form [0 0; I_r 0]: ones at (3-r+i, i).
    for i in range(rank):
        row = 3 - rank + i
        column = i
        output |= 1 << (3 * row + column)
    return output


@dataclass(frozen=True)
class Tables:
    """All small F_2/GL(3,2) tables used by the search."""

    rank: tuple[int, ...]
    colex: tuple[int, ...]
    transpose: tuple[int, ...]
    gl_matrices: tuple[Matrix, ...]
    gl_index: dict[Matrix, int]
    inverse_index: tuple[int, ...]
    left: tuple[tuple[Matrix, ...], ...]
    right: tuple[tuple[Matrix, ...], ...]
    compose: tuple[tuple[int, ...], ...]
    identity_index: int

    @classmethod
    def build(cls) -> "Tables":
        ranks = tuple(mat_rank_raw(m) for m in range(MATRIX_COUNT))
        colex = tuple(colex_key_raw(m) for m in range(MATRIX_COUNT))
        transpose = tuple(mat_transpose_raw(m) for m in range(MATRIX_COUNT))

        gl_matrices = tuple(m for m in range(MATRIX_COUNT) if ranks[m] == 3)
        gl_index = {m: i for i, m in enumerate(gl_matrices)}
        if len(gl_matrices) != 168:
            raise RuntimeError(f"Expected |GL(3,2)|=168, got {len(gl_matrices)}")

        inverse_index = tuple(
            gl_index[mat_inverse_raw(m)]
            for m in gl_matrices
        )

        # left[g][M] = G*M, right[g][M] = M*G
        left = tuple(
            tuple(mat_mul_raw(g, m) for m in range(MATRIX_COUNT))
            for g in gl_matrices
        )
        right = tuple(
            tuple(mat_mul_raw(m, g) for m in range(MATRIX_COUNT))
            for g in gl_matrices
        )

        compose = tuple(
            tuple(gl_index[left[i][gl_matrices[j]]] for j in range(168))
            for i in range(168)
        )

        return cls(
            rank=ranks,
            colex=colex,
            transpose=transpose,
            gl_matrices=gl_matrices,
            gl_index=gl_index,
            inverse_index=inverse_index,
            left=left,
            right=right,
            compose=compose,
            identity_index=gl_index[IDENTITY],
        )


# ---------------------------------------------------------------------------
# Input/output and Brent checking
# ---------------------------------------------------------------------------


def parse_matrix_token(token: str) -> Matrix:
    token = token.strip()
    if len(token) != 9 or any(ch not in "01" for ch in token):
        raise ValueError(
            f"Expected a 9-character 0/1 matrix token, got: {token!r}"
        )
    matrix = 0
    for position, ch in enumerate(token):
        if ch == "1":
            matrix |= 1 << position
    return matrix


def format_matrix_token(matrix: Matrix) -> str:
    return "".join("1" if (matrix >> i) & 1 else "0" for i in range(9))


def read_scheme(path: str | Path, *, transpose_c: bool = False) -> Scheme:
    rows: list[Row] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{path}, line {line_number}: expected exactly three "
                    f"9-bit tokens, got {line!r}"
                )
            A, B, C = (parse_matrix_token(part) for part in parts)
            if transpose_c:
                C = mat_transpose_raw(C)
            if A == 0 or B == 0 or C == 0:
                raise ValueError(
                    f"{path}, line {line_number}: zero factors are not "
                    "supported in a multiplication-scheme row"
                )
            rows.append((A, B, C))
    if not rows:
        raise ValueError(f"No scheme rows found in {path}")
    return tuple(rows)


def write_scheme(path: str | Path, scheme: Scheme) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for A, B, C in scheme:
            handle.write(
                f"{format_matrix_token(A)} "
                f"{format_matrix_token(B)} "
                f"{format_matrix_token(C)}\n"
            )


def tensor_bitset_of_row(row: Row) -> int:
    A, B, C = row
    output = 0
    a_positions = [i for i in range(9) if (A >> i) & 1]
    b_positions = [i for i in range(9) if (B >> i) & 1]
    c_positions = [i for i in range(9) if (C >> i) & 1]
    for a in a_positions:
        for b in b_positions:
            for c in c_positions:
                output ^= 1 << ((a * 9 + b) * 9 + c)
    return output


def multiplication_tensor_bitset() -> int:
    output = 0
    for i in range(3):
        for j in range(3):
            for ell in range(3):
                a = 3 * i + j
                b = 3 * j + ell
                c = 3 * ell + i
                output |= 1 << ((a * 9 + b) * 9 + c)
    return output


TARGET_TENSOR = multiplication_tensor_bitset()


def brent_valid(scheme: Scheme) -> bool:
    tensor = 0
    for row in scheme:
        tensor ^= tensor_bitset_of_row(row)
    return tensor == TARGET_TENSOR


# ---------------------------------------------------------------------------
# Normal-form ordering and symmetries
# ---------------------------------------------------------------------------


def row_key(row: Row, tables: Tables) -> int:
    A, B, C = row
    return (tables.colex[A] << 18) | (tables.colex[B] << 9) | tables.colex[C]


def canonical_tail(rows: Iterable[Row], tables: Tables) -> Scheme:
    return tuple(sorted(rows, key=lambda r: row_key(r, tables)))


def rank_vector(row: Row, tables: Tables) -> tuple[int, int, int]:
    A, B, C = row
    return (tables.rank[A], tables.rank[B], tables.rank[C])


def rank_pattern(scheme: Scheme, tables: Tables) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((rank_vector(row, tables) for row in scheme), reverse=True))


def permutation_parity(permutation: Sequence[int]) -> int:
    inversions = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            inversions += permutation[i] > permutation[j]
    return inversions & 1


def permute_row(row: Row, permutation: Sequence[int], tables: Tables) -> Row:
    out = tuple(row[i] for i in permutation)
    if permutation_parity(permutation):
        out = tuple(tables.transpose[m] for m in out)
    return out  # type: ignore[return-value]


def permute_scheme(scheme: Scheme, permutation: Sequence[int], tables: Tables) -> Scheme:
    return tuple(permute_row(row, permutation, tables) for row in scheme)


def remove_one(rows: Scheme, selected: Row) -> Scheme:
    output: list[Row] = []
    removed = False
    for row in rows:
        if not removed and row == selected:
            removed = True
        else:
            output.append(row)
    if not removed:
        raise RuntimeError("Selected row was not present in transformed tail")
    return tuple(output)


# ---------------------------------------------------------------------------
# Optimized Algorithm 1 + Algorithm 2 first-row treatment
# ---------------------------------------------------------------------------


class FastNormalForm3x3:
    def __init__(self, tables: Tables, *, verbose: bool = False) -> None:
        self.t = tables
        self.verbose = verbose
        self._first_row_cache: dict[Row, tuple[Row, tuple[GroupElement, ...]]] = {}
        self._a_transporter_cache: dict[Matrix, tuple[tuple[int, int], ...]] = {}
        self.action_calls = 0

    def apply_row(self, row: Row, g: GroupElement) -> Row:
        self.action_calls += 1
        A, B, C = row
        u, v, w = g
        t = self.t
        A1 = t.left[u][t.right[t.inverse_index[v]][A]]
        B1 = t.left[v][t.right[t.inverse_index[w]][B]]
        C1 = t.left[w][t.right[t.inverse_index[u]][C]]
        return (A1, B1, C1)

    def apply_scheme(self, scheme: Scheme, g: GroupElement) -> Scheme:
        return tuple(self.apply_row(row, g) for row in scheme)

    def compose(self, left_g: GroupElement, right_g: GroupElement) -> GroupElement:
        t = self.t
        return (
            t.compose[left_g[0]][right_g[0]],
            t.compose[left_g[1]][right_g[1]],
            t.compose[left_g[2]][right_g[2]],
        )

    def inverse_group(self, g: GroupElement) -> GroupElement:
        inv = self.t.inverse_index
        return (inv[g[0]], inv[g[1]], inv[g[2]])

    def a_transporters(self, A: Matrix) -> tuple[tuple[int, int], ...]:
        """All (U,V) sending A to its Proposition-2 canonical matrix."""
        cached = self._a_transporter_cache.get(A)
        if cached is not None:
            return cached

        t = self.t
        target = canonical_rank_matrix(t.rank[A])
        output: list[tuple[int, int]] = []
        for u in range(168):
            ua = t.left[u][A]
            for v in range(168):
                image = t.right[t.inverse_index[v]][ua]
                if image == target:
                    output.append((u, v))
        result = tuple(output)
        self._a_transporter_cache[A] = result
        return result

    def minimize_first_row_algorithm_2(
        self, row: Row
    ) -> tuple[Row, tuple[GroupElement, ...]]:
        """Return the minimal GL(3,2)^3 image and every transporter attaining it."""
        cached = self._first_row_cache.get(row)
        if cached is not None:
            return cached

        A, B, C = row
        t = self.t
        rank_a = t.rank[A]
        rank_b = t.rank[B]
        if rank_a == 0 or rank_b == 0 or t.rank[C] == 0:
            raise ValueError("Algorithm 2 requires nonzero factors")

        best_key: int | None = None
        best_row: Row | None = None
        transporters: list[GroupElement] = []

        def consider(g: GroupElement, image: Row | None = None) -> None:
            nonlocal best_key, best_row, transporters
            if image is None:
                image = self.apply_row(row, g)
            key = row_key(image, t)
            if best_key is None or key < best_key:
                best_key = key
                best_row = image
                transporters = [g]
            elif key == best_key:
                transporters.append(g)

        if rank_a == 3:
            # U A V^-1 = I  iff  U = V A^-1.
            a_index = t.gl_index[A]
            a_inverse_index = t.inverse_index[a_index]

            if rank_b == 3:
                # Also force B to I.  W = V B, leaving only conjugation of C.
                b_index = t.gl_index[B]
                for v in range(168):
                    u = t.compose[v][a_inverse_index]
                    w = t.compose[v][b_index]
                    consider((u, v, w))
            else:
                # A is I after normalization; search only V,W rather than U,V,W.
                for v in range(168):
                    u = t.compose[v][a_inverse_index]
                    for w in range(168):
                        consider((u, v, w))
        else:
            # Enumerate only the (U,V) pairs that map A to its canonical rank form.
            for u, v in self.a_transporters(A):
                for w in range(168):
                    consider((u, v, w))

        if best_row is None:
            raise RuntimeError("Algorithm 2 found no image of the first row")

        # Remove duplicates while keeping deterministic order.
        unique_transporters = tuple(sorted(set(transporters)))
        result = (best_row, unique_transporters)
        self._first_row_cache[row] = result
        return result

    def stabilizer_from_transporters(
        self,
        minimum_row: Row,
        transporters: Sequence[GroupElement],
    ) -> tuple[GroupElement, ...]:
        """Recover Stab(minimum_row) from one complete transporter coset."""
        if not transporters:
            raise RuntimeError("Cannot derive a stabilizer from an empty transporter set")
        base_inverse = self.inverse_group(transporters[0])
        stabilizer = {
            self.compose(g, base_inverse)
            for g in transporters
        }
        result = tuple(sorted(stabilizer))
        # This check is cheap because the stabilizer after Algorithm 2 is small.
        for h in result:
            if self.apply_row(minimum_row, h) != minimum_row:
                raise RuntimeError("Internal error while deriving first-row stabilizer")
        return result

    def _first_iteration(
        self, seed_scheme: Scheme
    ) -> tuple[Row, set[Scheme], tuple[GroupElement, ...]]:
        """Algorithm 1's first iteration, accelerated by Algorithm 2."""
        t = self.t
        maximal = max(rank_vector(row, t) for row in seed_scheme)
        rows_to_try = sorted(
            {row for row in seed_scheme if rank_vector(row, t) == maximal},
            key=lambda r: row_key(r, t),
        )

        global_min: Row | None = None
        global_key: int | None = None
        new_tails: set[Scheme] = set()
        transporter_source: tuple[GroupElement, ...] | None = None

        for row in rows_to_try:
            local_min, transporters = self.minimize_first_row_algorithm_2(row)
            local_key = row_key(local_min, t)

            if global_key is None or local_key < global_key:
                global_key = local_key
                global_min = local_min
                new_tails = set()
                transporter_source = transporters

            if local_key == global_key:
                if transporter_source is None:
                    transporter_source = transporters
                for g in transporters:
                    transformed = self.apply_scheme(seed_scheme, g)
                    remaining = remove_one(transformed, local_min)
                    new_tails.add(canonical_tail(remaining, t))

        if global_min is None or transporter_source is None:
            raise RuntimeError("Could not select the first normal-form row")

        stabilizer = self.stabilizer_from_transporters(
            global_min, transporter_source
        )
        return global_min, new_tails, stabilizer

    def _later_iteration(
        self,
        tails: set[Scheme],
        stabilizer: tuple[GroupElement, ...],
    ) -> tuple[Row, set[Scheme], tuple[GroupElement, ...]]:
        """One ordinary stabilizer-restricted Algorithm-1 iteration."""
        t = self.t
        global_min: Row | None = None
        global_key: int | None = None
        new_tails: set[Scheme] = set()

        for tail in tails:
            maximal = max(rank_vector(row, t) for row in tail)
            rows_to_try = {
                row for row in tail if rank_vector(row, t) == maximal
            }

            for row in rows_to_try:
                local_key: int | None = None
                local_min: Row | None = None
                best_elements: list[GroupElement] = []

                for g in stabilizer:
                    image = self.apply_row(row, g)
                    key = row_key(image, t)
                    if local_key is None or key < local_key:
                        local_key = key
                        local_min = image
                        best_elements = [g]
                    elif key == local_key:
                        best_elements.append(g)

                if local_min is None or local_key is None:
                    raise RuntimeError("Current stabilizer was empty")

                if global_key is None or local_key < global_key:
                    global_key = local_key
                    global_min = local_min
                    new_tails = set()

                if local_key == global_key:
                    for g in best_elements:
                        transformed = self.apply_scheme(tail, g)
                        remaining = remove_one(transformed, local_min)
                        new_tails.add(canonical_tail(remaining, t))

        if global_min is None:
            raise RuntimeError("Could not select the next normal-form row")

        new_stabilizer = tuple(
            g for g in stabilizer if self.apply_row(global_min, g) == global_min
        )
        if not new_stabilizer:
            raise RuntimeError("Stabilizer unexpectedly became empty")
        return global_min, new_tails, new_stabilizer

    def normalize_seed(self, seed_scheme: Scheme) -> Scheme:
        first, tails, stabilizer = self._first_iteration(seed_scheme)
        candidate: list[Row] = [first]

        if self.verbose:
            print(
                f"  row 1: {format_row(first)} | tails={len(tails)} "
                f"| stabilizer={len(stabilizer)}"
            )

        row_number = 1
        while tails != {()}:
            row_number += 1
            next_row, tails, stabilizer = self._later_iteration(tails, stabilizer)
            candidate.append(next_row)
            if self.verbose:
                print(
                    f"  row {row_number}: {format_row(next_row)} | "
                    f"tails={len(tails)} | stabilizer={len(stabilizer)}"
                )
        return tuple(candidate)

    def normal_form(self, scheme: Scheme) -> Scheme:
        t = self.t
        permuted: list[tuple[tuple[tuple[int, int, int], ...], Scheme]] = []
        for permutation in itertools.permutations((0, 1, 2)):
            candidate = canonical_tail(
                permute_scheme(scheme, permutation, t), t
            )
            permuted.append((rank_pattern(candidate, t), candidate))

        maximal_pattern = max(pattern for pattern, _ in permuted)
        seeds = sorted(
            {
                candidate
                for pattern, candidate in permuted
                if pattern == maximal_pattern
            },
            key=lambda s: tuple(row_key(row, t) for row in s),
        )

        if self.verbose:
            print(f"Maximal rank pattern: {maximal_pattern}")
            print(f"Column-permutation seeds: {len(seeds)}")

        best: Scheme | None = None
        best_key: tuple[int, ...] | None = None
        for index, seed in enumerate(seeds, start=1):
            if self.verbose:
                print(f"Seed {index}/{len(seeds)}")
            candidate = self.normalize_seed(seed)
            key = tuple(row_key(row, t) for row in candidate)
            if best_key is None or key < best_key:
                best = candidate
                best_key = key

        if best is None:
            raise RuntimeError("Normal-form computation produced no candidate")
        return best


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def format_row(row: Row) -> str:
    return " ".join(format_matrix_token(m) for m in row)


def default_output_file(input_file: Path, output_dir: Path | None) -> Path:
    name = input_file.stem + "_normal_form.txt"
    return (output_dir / name) if output_dir is not None else input_file.with_name(name)


def normalize_one_file(
    input_file: Path,
    output_file: Path,
    tables: Tables,
    *,
    transpose_c: bool,
    verify: bool,
    verbose: bool,
) -> tuple[int, float, int]:
    scheme = read_scheme(input_file, transpose_c=transpose_c)
    if verify and not brent_valid(scheme):
        raise ValueError("Input scheme failed the 3x3 F_2 Brent check")

    engine = FastNormalForm3x3(tables, verbose=verbose)
    started = time.perf_counter()
    normal = engine.normal_form(scheme)
    elapsed = time.perf_counter() - started

    if verify and not brent_valid(normal):
        raise RuntimeError("Output normal form failed the Brent check")
    write_scheme(output_file, normal)
    return len(scheme), elapsed, engine.action_calls


def collect_files(path: Path, extension: str, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    if not extension.startswith("."):
        extension = "." + extension
    iterator: Iterator[Path]
    iterator = path.rglob(f"*{extension}") if recursive else path.glob(f"*{extension}")
    return sorted(
        p for p in iterator
        if p.is_file() and not p.stem.endswith("_normal_form")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Algorithm-1 normal forms for 3x3 matrix multiplication "
            "schemes over F_2, using Algorithm 2 to accelerate the first row."
        )
    )
    parser.add_argument(
        "input_path",
        help="One 9-bit scheme file or a folder of scheme files.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file when input_path is one file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output folder. Required when input_path is a folder.",
    )
    parser.add_argument(
        "--extension", default="txt",
        help="Extension used for folder input. Default: txt.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search subfolders when input_path is a folder.",
    )
    parser.add_argument(
        "--transpose-c", action="store_true",
        help="Transpose every C matrix while reading, matching the older tool.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip Brent checks before and after normalization.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print row-by-row stabilizer and tail information.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="For folder input, skip outputs that already exist.",
    )
    parser.add_argument(
        "--max-files", type=int,
        help="Optional limit for testing folder mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path)

    if args.output and args.output_dir:
        raise SystemExit("Use either --output or --output-dir, not both")
    if input_path.is_dir() and args.output:
        raise SystemExit("--output can only be used with one input file")
    if input_path.is_dir() and not args.output_dir:
        raise SystemExit("Folder input requires --output-dir")

    print("Building optimized 3x3 F_2 matrix/GL tables...")
    table_start = time.perf_counter()
    tables = Tables.build()
    print(
        f"Built |GL(3,2)|={len(tables.gl_matrices)} tables in "
        f"{time.perf_counter() - table_start:.3f}s"
    )
    print("Algorithm 2 replaces the 168^3 first-row brute-force scan.")
    print()

    files = collect_files(input_path, args.extension, args.recursive)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise SystemExit("No input files found")

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failures = 0
    total_time = 0.0

    for index, source in enumerate(files, start=1):
        if args.output:
            destination = Path(args.output)
        else:
            destination = default_output_file(source, output_dir)

        if args.skip_existing and destination.exists():
            print(f"[{index}/{len(files)}] skip existing: {destination}")
            continue

        print(f"[{index}/{len(files)}] {source.name}")
        try:
            rows, elapsed, action_calls = normalize_one_file(
                source,
                destination,
                tables,
                transpose_c=args.transpose_c,
                verify=not args.no_verify,
                verbose=args.verbose,
            )
            print(
                f"  rows={rows} time={elapsed:.3f}s "
                f"row-actions={action_calls:,}"
            )
            print(f"  written: {destination}")
            success += 1
            total_time += elapsed
        except Exception as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            failures += 1
        print()

    print("Finished.")
    print(f"Successful files: {success}")
    print(f"Files with errors: {failures}")
    if success:
        print(f"Total normalization time: {total_time:.3f}s")
        print(f"Average normalization time: {total_time / success:.3f}s")


if __name__ == "__main__":
    main()
