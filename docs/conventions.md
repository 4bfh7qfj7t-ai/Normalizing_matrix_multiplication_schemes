# Mathematical and Compatibility Conventions

## Matrix and factor order

Matrices are flattened in row-major order. Scheme rows are ordered `(A, B, C)`. Permuting tensor factors is part of canonicalization and, for odd permutations, the generic implementation transposes all three matrices. These details are compatibility contracts.

## C-factor convention

Raw `.exp` sources occur in both Brent-C and output-C conventions. The converters preserve the literal `c` variables. The normal-form programs expose `--transpose-c`; use it only when the input's recorded convention requires it.

The historical 273-file compact 2x2 corpus reproduces its stored representatives with transposition disabled. The raw `strassen_test.exp` fixture uses the output-C convention and needs C transposition to match `test_nf.txt`. These facts are both correct because they apply at different format boundaries.

## Two separate 2x2 canonicalizers

The repository intentionally retains two definitions:

1. `src/normal_forms/normal_form_one.py` implements generic Algorithm 1. It maximizes the rank pattern across factor permutations, then minimizes within the admissible candidates.
2. `src/flips/make_recursive_graphs_and_quotient.py` computes a global lexicographic minimum across all six factor permutations and GL(2,2)^3.

They select different text representatives for 247 of the audited 273 quotient classes, while preserving the same partition on that corpus. Their output hashes belong to different namespaces. Do not consolidate them without a separately reviewed equivalence proof and migration map.

## Hashes

Existing SHA-1 and content-hash identifiers are preserved exactly in data, filenames, and graph nodes. New repository manifests use SHA-256 only as an outer integrity layer; they do not replace research identifiers.

## Ordering and reproducibility

Do not change row sorting, permutation enumeration, GL enumeration, lexicographic keys, rank-pattern comparison, or factor transposition order merely for style or speed. Such changes require byte-level regression evidence.

