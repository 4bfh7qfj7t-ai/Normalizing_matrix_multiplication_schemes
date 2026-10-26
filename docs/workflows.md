# Authoritative Workflows

Commands below assume the repository root as the current directory. Programs remain byte-identical to the archive, so some are individual scripts rather than a unified package.

## 3x3 conversion and optimized normal form

```bash
python3 src/converters/convert_3x3_exp_directory.py data/3x3/raw /tmp/compact --flat
python3 src/verification/verify_brent_general.py /tmp/compact --report /tmp/compact-verification.csv
python3 src/normal_forms/normal_form_3x3_fast.py /tmp/compact --output-dir /tmp/normal-forms
python3 src/verification/verify_brent_general.py /tmp/normal-forms --report /tmp/nf-verification.csv
```

`data/3x3/compact_unique` is the canonical deduplicated compact input set. The original full converted set had one exact duplicate pair.

## Generic Algorithm 1 normal form

Keep `gl_cube.py` beside `normal_form_one.py` or add `src/normal_forms` to `PYTHONPATH`:

```bash
PYTHONPATH=src/normal_forms python3 src/normal_forms/normal_form_one.py INPUT.txt -o OUTPUT.txt --matrix-size 2 --modulus 2
```

Only add `--transpose-c` when the input convention requires it. For fields other than F2, the implementation does not add independent nonzero row-scaling normalization.

## Recursive rank-8 flips and rank-7 reductions

The full exploration is intentionally not launched by the regression suite because it creates hundreds of thousands of files. Run it only in a dedicated output location:

```bash
python3 src/flips/recursive_rank8_flip_explorer.py data/2x2/rank8_seeds --output-dir /large/output/exploration
python3 src/flips/make_recursive_graphs_and_quotient.py /large/output/exploration --output-dir /large/output/graphs
```

The stored completed run is summarized in `results/2x2/recursive_exploration_summary`; its compact quotient is in `results/2x2/recursive_quotient`.

The stats and final-plot scripts retain historical working-directory assumptions. Until a later tested CLI-only refactor, run them in a temporary workspace containing `rank8_recursive_graphs/quotient_normal_form_graph` and `standard_rank8.exp` at the expected relative paths.

## Corrected C++ flip search

Build outside the repository or remove the resulting binary afterward:

```bash
cp -R cpp/flip_search /tmp/flip-search-build
make -C /tmp/flip-search-build -B
```

The batch wrappers are quarantined. Do not use them as supported commands in this pass.

## Small exhaustive 2x1-by-1x2/F2 experiment

```bash
python3 src/small_exhaustive/bruteforce_mm_f2.py --dims 2 1 2 --rank 4 --output-dir /tmp/brute_212_rank4 --save-exp
```

The two postprocessors retain fixed relative paths. The regression runner creates the expected temporary layout, then checks the 81 schemes, 216-edge connected graph, and absence of rank-3 reductions.

