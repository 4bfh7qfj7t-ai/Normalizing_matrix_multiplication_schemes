# Provenance

## Immutable source

The source archive is `Archive 2.zip` with SHA-256:

```text
bbd0b0958836a45c4f62dadc223360505d5d7fb7778300136a6b2c5cda464fd0
```

The original ZIP was not edited or repacked. The first-pass cleanup was constructed as a new sibling folder.

## Embedded `flips` Git state

Before research files were copied, the exact archived dirty Git state was captured under `provenance/flips_dirty_state` and as `provenance/flips_dirty_state_provenance.tar.gz`.

The capture contains:

- HEAD `e31a0a0f0d2577cee5da047ca7dcae0c61992e40`;
- remote `https://github.com/jakobmoosbauer/flips.git`;
- human and complete machine-readable status records;
- a full-index binary-capable tracked diff;
- changed tracked-file checksums;
- the complete untracked-path list;
- all untracked source files and their checksums;
- a checksum covering every provenance artifact.

The corrected C++ source in `cpp/flip_search` is copied from this dirty worktree, not reconstructed from upstream.

## Disposition records

`manifests/copied_files.csv` maps every byte-identical copied archive file to its repository path and SHA-256. `manifests/archive_entry_disposition.csv.gz` assigns every one of the 751,269 ZIP entries a final disposition and identifies any clean-copy target paths.

Historical bulk outputs marked `ARCHIVED_IN_ORIGINAL_ONLY` remain recoverable from the immutable ZIP. They were not silently deleted from the archive of record.

