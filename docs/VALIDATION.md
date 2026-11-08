# Release validation — 2026-11-08

Validation ran against the packaged source in this repository.

| Check | Environment | Result |
|---|---|---|
| Python parsing and shell syntax | Local macOS and zjc-hope-8 | 131 Python files, 8 shell files passed |
| Frozen model/data/training hashes | Local macOS and zjc-hope-8 | All 21 core Python files match the frozen training input source |
| Evaluation scheduling and camera-color tests | Local macOS | 9 tests passed |
| Model, query memory, cache, routing, padding and TBPTT tests | zjc-hope-8 starvla, CPU | 64 tests passed |
| Existing 80k checkpoint recipe/tensor layout | zjc-hope-8 starvla, metadata inspection | Passed: 752 tensors, zero-state gated 64Q memory, absolute [0], relative [-15,0], 9 images, stride 50 |
| Formal evaluation launcher dry-run | zjc-hope-8 | Checkpoint precheck passed; generated all 9 task commands, 25 episodes/task, cover_blocks-only R/B swapping |
| Release artifact scan | Local macOS | No checkpoint, dataset/video files, private keys, or credential-pattern matches |

The 188/225 (83.56%) task scores in `results.json` are the recorded formal experiment results. Release validation did not start another training run or execute another full 225-episode evaluation.
