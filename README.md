# Matrix-Multiplication Scheme Research Code

This folder contains the Python code and data used to convert, verify, normalize, and study matrix-multiplication schemes. The main dataset contains 3x3 schemes over F2. A smaller 2x2 dataset and older 2x2 flip-graph programs are also included for reference.

## What is included

- `src/normal_forms/` contains the normal-form programs.
  - `normal_form_3x3_fast.py` is the recommended program for 3x3 schemes over F2.
  - `normal_form_one.py` is the slower generic implementation. Keep it separate from the optimized 3x3 program.
  - `gl_cube.py` supplies supporting linear-group operations.
- `src/converters/` converts raw scheme files into the compact text format used by the normal-form programs.
- `src/verification/` checks schemes against the Brent equations.
- `src/reporting/` counts distinct normal-form files.
- `src/flips/` contains both flip-graph workflows.
  - `explore_3x3_flip_component.py` and `build_3x3_quotient_graph.py` are the new 3x3/F2 exploration and graph programs.
  - `recursive_rank8_flip_explorer.py` and `make_recursive_graphs_and_quotient.py` are the established 2x2 rank-8 programs and should not be used for the 3x3 run.
- `src/small_exhaustive/` contains the small 2x1-by-1x2 exhaustive F2 experiment.
- `data/3x3/raw/` contains the raw 3x3 schemes.
- `data/3x3/compact_unique/` contains 17,375 distinct compact 3x3 schemes.
- `data/2x2/rank8_seeds/` contains the 273 historical rank-8 seed schemes.
- `docs/` explains file formats, conventions, workflows, provenance, and the Python environment.

The external C++ flip-search project, generated result folders, tests, caches, and compiled programs are not included in this sharing copy.

## Installation

Python 3 is required. From this folder, install the graph and plotting dependencies with:

```bash
python3 -m pip install -r requirements-lock.txt
```

`networkx` is used to construct and analyze graphs. `matplotlib` is used to draw them.

## Recommended 3x3 workflow

Read `docs/formats.md` and `docs/conventions.md` before processing the data. The compact files contain one tensor term per line, written as three 9-bit matrix factors.

Verify the compact 3x3 dataset:

```bash
python3 src/verification/verify_brent_general.py \
  data/3x3/compact_unique \
  --matrix-size 3 \
  --modulus 2 \
  --report verification_report.csv
```

Compute optimized 3x3 normal forms into a new output folder:

```bash
python3 src/normal_forms/normal_form_3x3_fast.py \
  data/3x3/compact_unique \
  --output-dir normal_forms_3x3 \
  --skip-existing
```

Count the distinct normal forms:

```bash
python3 src/reporting/count_unique_normal_forms.py \
  normal_forms_3x3 \
  --report unique_normal_forms_report.txt
```

For a quick test, add `--max-files 10` to the verification or optimized normal-form command before running the entire dataset.

Instructions for the 3x3 flip-graph workflow are in `docs/three_by_three_flip_graph.md`.

## Important conventions

- Do not replace `normal_form_one.py` with `normal_form_3x3_fast.py`; they are separate programs with different scopes.
- Do not mix raw `.exp` files with compact 9-bit `.txt` files.
- Do not change row ordering, factor ordering, transpose conventions, or identifiers without documenting the change.
- Write generated files to a new output folder instead of overwriting the supplied data.
- The generic Algorithm 1 normal form and the historical 2x2 quotient-graph representative are separate conventions.

For additional details, see `docs/workflows.md`, `docs/formats.md`, `docs/conventions.md`, and `docs/environment.md`.
