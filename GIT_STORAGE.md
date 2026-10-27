# Git and model artifacts

This repository uses branch `main` and the candidate tag `v0.1.0-rc1`. Model weights, fitted
state arrays and response fixtures use Git LFS. After cloning a hosted version:

```sh
git lfs install
git lfs pull
```

The companion delivery ZIP contains the actual working-tree model bytes,
not LFS pointer placeholders. The Python package wheel may require companion
assets as explained in the project README. `RELEASE_MANIFEST.json` checks the
complete delivered file set, excluding itself and `.git/`.

Source and data hashes deliberately preserve original line endings. Do not
automatically normalize archived source files or frozen evidence. Source
bindings and original experiment contracts remain provenance, while the README
documents portable runtime interfaces. The public repository is identified in `PUBLICATION.json`; no new license
grant is implied by the candidate label.
