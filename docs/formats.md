# File Formats

## Raw `.exp` schemes

Each nonblank, noncomment line is one rank-one tensor term with three `*`-separated linear factors:

```text
(a11+a22)*(b11-b21)*(c11+c12)
```

Variables are ordered by factor (`a`, `b`, `c`) and then row-major matrix index. Signs are reduced modulo the selected field. The raw format does not by itself settle whether the stored C factor represents Brent's C factor or the output-coordinate convention; that convention must be supplied by the workflow.

## Compact schemes

Each nonblank, noncomment line contains exactly three matrix tokens:

```text
1001 0101 0011
```

Each token stores one matrix in row-major order. A 2x2 token has four entries; a 3x3 token has nine. For moduli at most 10, entries may be adjacent digits. Comma-separated entries are accepted by the generic matrix parser and are required for larger moduli.

Rows are rank-one terms. Their multiplicity matters until a specific normal-form algorithm sorts or transforms them.

## Graph records

The authoritative quotient files are under `results/2x2/recursive_quotient`:

- `quotient_nodes.csv` identifies node hashes, rank, representative file, normal-form file, and raw-node multiplicity.
- `quotient_edges.csv` records source, target, edge type, endpoint ranks, multiplicity, and self-loop status.
- `standard_to_strassen_path.csv` records the eight-edge standard-to-rank-7 path.

Hash values and filenames are preserved historical identifiers. Do not recalculate them with a different digest or canonicalizer in place.

