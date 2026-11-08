# Final-model historical KV retrieval budget ablations

Top-6 and Top-8 retain the final 64Q zero-state slot Gate memory M, current-only detached visual KV writes, all-token query scope, three-camera first/t−15/current prompt (9 images), reset-every-decision native S/conv, seed42, 80k updates, 8 GPUs, B2/GPU, TBPTT2 and ZeRO1. Each starts from the same Qwen3.5-4B base; no VLA checkpoint resume.

Only `framework.moba_memory.topk_history_images` changes from 4 to 6 or 8. K counts individual camera images, selected per query/head; it is not a number of physical times and does not expand to all views of a selected time. Effective K is bounded by available valid history images.

Run `bash scripts/train_topk.sh 6` or `bash scripts/train_topk.sh 8` with the same environment overrides as `train_final.sh`. Use 8 GPUs sequentially on zjc-hope-8-2 to preserve the original global batch and optimizer sharding.

The original implementation rejected K > 4 and its exact attention mask explicitly read only four route choices. The extension admits K up to 8 and includes all selected choices in the mask. For K <= 4 the four-choice path is retained. Dataset, query memory, native-state policy, optimizer and training implementation are unchanged. `topk_extension_sha256.json` records the two audited core changes relative to the preserved original frozen hashes.

Startup verification must use resolved `runtime.json` and actual metrics (`moba_topk_history_images`, `moba_topk_effective`, cache deltas, memory state), since original shared informational strings may still mention Top-4. Top-8 must reach effective K=8 once three prior decisions have provided nine historical camera images. The second run starts only after the first run exits successfully with an 80k checkpoint; a failed run blocks the queue for inspection.
