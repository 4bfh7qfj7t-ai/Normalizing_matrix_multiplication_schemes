import csv
import hashlib
import itertools
import re
from pathlib import Path
from collections import Counter

exp_dir = Path("brute_212_rank4/valid_schemes_exp")
out_dir = Path("brute_212_rank4_reductions")
out_dir.mkdir(exist_ok=True)

rank3_out_dir = out_dir / "valid_rank3_reduction_outputs"
rank3_out_dir.mkdir(exist_ok=True)

summary_path = out_dir / "reduction_summary.txt"
attempts_csv = out_dir / "reduction_attempts.csv"
valid_csv = out_dir / "valid_rank3_reductions.csv"

paren_re = re.compile(r"\(([^()]*)\)")

files = sorted(exp_dir.glob("*.exp"))
if not files:
    raise RuntimeError(f"No .exp files found in {exp_dir}")

# ------------------------------------------------------------
# 1. Parse variable names
# ------------------------------------------------------------

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

var_lists = {}
var_maps = {}

for kind in ["a", "b", "c"]:
    ordered = sorted(vars_by_kind[kind])
    var_lists[kind] = ordered
    var_maps[kind] = {var: i for i, var in enumerate(ordered)}

A_count = len(var_lists["a"])
B_count = len(var_lists["b"])
C_count = len(var_lists["c"])

print("Variable maps:")
print("A:", var_lists["a"])
print("B:", var_lists["b"])
print("C:", var_lists["c"])
print()

# ------------------------------------------------------------
# 2. Scheme parsing and tensor checking
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

        triples.append((a, b, c))

    return tuple(sorted(triples))

def scheme_hash(scheme):
    text = ";".join(f"{a},{b},{c}" for a, b, c in sorted(scheme))
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

def clean_scheme(scheme, expected_rank):
    scheme = tuple(sorted(scheme))

    if len(scheme) != expected_rank:
        return None

    if len(set(scheme)) != expected_rank:
        return None

    for a, b, c in scheme:
        if a == 0 or b == 0 or c == 0:
            return None

    return scheme

def form_to_text(mask, kind):
    names = []

    for i, name in enumerate(var_lists[kind]):
        if mask & (1 << i):
            names.append(name)

    if not names:
        return "0"

    return "+".join(names)

def scheme_to_exp_text(scheme):
    lines = []

    for a, b, c in sorted(scheme):
        lines.append(
            f"({form_to_text(a, 'a')})*({form_to_text(b, 'b')})*({form_to_text(c, 'c')})"
        )

    return "\n".join(lines) + "\n"

schemes = {}

for path in files:
    scheme = parse_scheme(path)
    h = scheme_hash(scheme)
    schemes[h] = {
        "scheme": scheme,
        "file": str(path),
    }

target_bitset = tensor_bitset_of_scheme(next(iter(schemes.values()))["scheme"])

bad_sources = []

for h, info in schemes.items():
    if tensor_bitset_of_scheme(info["scheme"]) != target_bitset:
        bad_sources.append(h)

if bad_sources:
    raise RuntimeError(f"{len(bad_sources)} input schemes do not match the target tensor")

print("Loaded valid rank-4 schemes:", len(schemes))
print("Target tensor bitset:", hex(target_bitset))
print()

# ------------------------------------------------------------
# 3. Try reductions
# ------------------------------------------------------------

attempt_rows = []
valid_rows = []
valid_rank3_hashes = set()

total_pairs_checked = 0
reduction_patterns_found = 0
rank3_candidates_clean = 0
valid_rank3_reductions = 0

for source_hash, info in schemes.items():
    scheme = info["scheme"]
    terms = list(scheme)

    for i, j in itertools.combinations(range(len(terms)), 2):
        total_pairs_checked += 1

        a1, b1, c1 = terms[i]
        a2, b2, c2 = terms[j]

        candidates = []

        # Same A and same B:
        # A*B*C1 + A*B*C2 = A*B*(C1+C2)
        if a1 == a2 and b1 == b2:
            candidates.append(("same_A_same_B", (a1, b1, c1 ^ c2)))

        # Same A and same C:
        # A*B1*C + A*B2*C = A*(B1+B2)*C
        if a1 == a2 and c1 == c2:
            candidates.append(("same_A_same_C", (a1, b1 ^ b2, c1)))

        # Same B and same C:
        # A1*B*C + A2*B*C = (A1+A2)*B*C
        if b1 == b2 and c1 == c2:
            candidates.append(("same_B_same_C", (a1 ^ a2, b1, c1)))

        for reduction_type, merged_term in candidates:
            reduction_patterns_found += 1

            new_terms = []

            for k, term in enumerate(terms):
                if k not in {i, j}:
                    new_terms.append(term)

            new_terms.append(merged_term)

            rank3_scheme = clean_scheme(new_terms, expected_rank=3)

            status = "not_clean_rank3"
            rank3_hash = ""

            if rank3_scheme is not None:
                rank3_candidates_clean += 1
                rank3_hash = scheme_hash(rank3_scheme)

                if tensor_bitset_of_scheme(rank3_scheme) == target_bitset:
                    status = "valid_rank3_reduction"
                    valid_rank3_reductions += 1
                    valid_rank3_hashes.add(rank3_hash)

                    out_file = rank3_out_dir / f"rank3_{rank3_hash[:12]}.exp"
                    out_file.write_text(scheme_to_exp_text(rank3_scheme))

                    valid_rows.append({
                        "source_hash": source_hash,
                        "source_file": info["file"],
                        "reduction_type": reduction_type,
                        "rank3_hash": rank3_hash,
                        "rank3_file": str(out_file),
                    })
                else:
                    status = "brent_failed"

            attempt_rows.append({
                "source_hash": source_hash,
                "source_file": info["file"],
                "term_i": i,
                "term_j": j,
                "reduction_type": reduction_type,
                "status": status,
                "rank3_hash": rank3_hash,
            })

# ------------------------------------------------------------
# 4. Save files
# ------------------------------------------------------------

with open(attempts_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_hash",
            "source_file",
            "term_i",
            "term_j",
            "reduction_type",
            "status",
            "rank3_hash",
        ],
    )
    writer.writeheader()
    writer.writerows(attempt_rows)

with open(valid_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "source_hash",
            "source_file",
            "reduction_type",
            "rank3_hash",
            "rank3_file",
        ],
    )
    writer.writeheader()
    writer.writerows(valid_rows)

status_counts = Counter(row["status"] for row in attempt_rows)
type_counts = Counter(row["reduction_type"] for row in attempt_rows)

summary = f"""Rank-4 to rank-3 reduction check for 2x1 times 1x2 over F2

Input folder:
{exp_dir}

Valid rank-4 schemes loaded:
{len(schemes)}

Target tensor bitset:
{hex(target_bitset)}

Pair checks:
{total_pairs_checked}

Reduction patterns found:
{reduction_patterns_found}

Clean rank-3 candidates:
{rank3_candidates_clean}

Valid Brent-checked rank-3 reductions:
{valid_rank3_reductions}

Unique valid rank-3 outputs:
{len(valid_rank3_hashes)}

Status counts:
{dict(status_counts)}

Reduction type counts:
{dict(type_counts)}

Output files:
{attempts_csv}
{valid_csv}
{rank3_out_dir}
{summary_path}
"""

summary_path.write_text(summary)

print(summary)
