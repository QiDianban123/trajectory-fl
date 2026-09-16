# Three-mode full-training archive

> **Final-result decision:** the centralized, local-only, and federated models
> and metrics in this archive are the project's accepted final training results.
> This acceptance includes an explicit provenance exception: the manifests recorded Git HEAD
> `da360341570941e38f532069c1aac76e477553bb`, while source and configuration
> changes needed by the full-data run were first committed together with the
> archive in `b3d465cbc6ce80cbf6409e7560684d39ed10bd1b`. The historical worktree
> state cannot be proven retroactively. This limitation remains disclosed in
> `PROVENANCE.json`, but does not disqualify this archive from final-result use.

This directory contains the retained artifacts from the comparable 20-pass
highD experiments. All three runs used seed 42, the same processed split,
model configuration, initialization identity, and 10,726,580 training-sample
visits.

`FINAL_MANIFEST.json` is the authoritative release index. It lists only files
that are actually retained in this archive and records their SHA-256 digests.
The manifests under `results/` remain historical run records: their original
training-time artifact paths are preserved for provenance, but omitted
intermediate checkpoints, RNG states, predictions, and logs are not part of
the final archive.

Verify the complete retained scope from the repository root with:

```text
python scripts/verify_final_archive.py
```

## Final comparison

| Mode | ADE (m) | FDE (m) | Runtime (s) | Test samples |
| --- | ---: | ---: | ---: | ---: |
| Centralized | 1.588678 | 3.521342 | 1926.552 | 115301 |
| Local only | 3.047601 | 7.612483 | 2835.506 | 115301 |
| Federated (FedAvg) | 1.273001 | 2.902582 | 2706.408 | 115301 |

Lower ADE and FDE are better. The federated model produced the best metrics
in this run.

## Models

- `models/centralized/best.pt`: best centralized checkpoint.
- `models/local_only/rsu_01.pt` through `rsu_05.pt`: final independent RSU
  models.
- `models/federated/global_round_0019.pt`: global model after federated round
  20 (the zero-based checkpoint name is round 19).

The `.pt` files contain PyTorch state dictionaries. Instantiate the project
model with the archived configuration before calling `load_state_dict`.

## Results and figures

- `comparison.csv`: compact machine-readable comparison.
- `FINAL_MANIFEST.json`: authoritative final models, results, configuration,
  figures, retention scope, and per-file integrity hashes.
- `results/`: original metrics, manifests, histories, and per-mode result
  tables.
- `figures/three_mode_comparison.png`: unified ADE/FDE comparison.
- `figures/loss_curve.png` and trajectory images: centralized diagnostic
  figures produced by the runner.
- `configs/`: snapshots of the data, model, and three experiment configs.
- `SHA256SUMS`: integrity hashes for all retained model files.
- `PROVENANCE.json`: machine-readable final-result decision and provenance exception.

Large per-sample predictions, intermediate recovery checkpoints, RNG states,
and logs remain excluded by `.gitignore`; they are not required for inference
from the final models.
