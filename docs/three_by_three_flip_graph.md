# 3x3 F2 Flip Graph Tools

These two scripts adapt the earlier 2x2 exploration workflow to compact 3x3 schemes over F2.

`explore_3x3_flip_component.py` follows every valid same-rank flip reachable from the supplied seeds. It Brent-checks every saved scheme and records all valid one-step reductions to the next lower rank. A run from one seed finds that seed's reachable component; it does not prove that the component is the entire 3x3 search space.

`build_3x3_quotient_graph.py` builds the raw graph and collapses its nodes with `normal_form_3x3_fast.py`. It preserves SHA-1 graph identifiers because that is the existing project convention.

Keep these scripts beside `normal_form_3x3_fast.py`, or pass its location explicitly with `--normal-form-script`.

## Small test

```bash
python3 explore_3x3_flip_component.py SEED.txt \
  --output-dir flip3x3_test \
  --max-nodes 100

python3 build_3x3_quotient_graph.py flip3x3_test \
  --output-dir flip3x3_test_graphs \
  --normal-form-script normal_form_3x3_fast.py
```

If the first run stops at the node limit, continue it with:

```bash
python3 explore_3x3_flip_component.py SEED.txt \
  --output-dir flip3x3_test \
  --max-nodes 100 \
  --resume
```

Remove `--max-nodes` only after the limited run has been checked.
