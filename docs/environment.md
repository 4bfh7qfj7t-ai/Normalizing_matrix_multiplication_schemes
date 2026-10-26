# Audited Environment

The successful cleanup regression was run with:

- Python 3.14.2
- NetworkX 3.6.1
- Matplotlib 3.11.1
- Apple Clang 21.0.0
- GNU-compatible `make`

These are the exact audited versions, not claims about the oldest supported versions. `requirements-lock.txt` pins the two non-standard Python libraries used by graph and plotting programs.

For headless or restricted environments, set `MPLCONFIGDIR` to a writable temporary directory. Set `PYTHONDONTWRITEBYTECODE=1` when the working tree must remain free of interpreter caches.

The Python normal-form and verification programs otherwise use the standard library. The C++ source expects a compiler with modern C++ support as selected by its retained makefile.

