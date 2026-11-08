# Corrected final-model M-only ablation

This experiment is **M-only (64Q Gate query memory)**, not native DeltaNet S-only.

| Component | Final model | M-only |
|---|---|---|
| Cross-decision query memory | 64Q, zero initial state, slot Gate | Same |
| First memory write | Complete | Same |
| Native DeltaNet S / convolution state | Reset each decision | Same |
| Historical visual KV write deltas | `[0]` | `[]` |
| Historical visual KV available for retrieval | Current images accumulated over decisions | Empty at all decisions |
| Temporal prompt | First, t−15, current; three cameras, nine images | Same |
| Data / seed / budget | RMBench9, seed 42, 80k | Same |
| Training | 8 GPUs, B2/GPU, TBPTT2, ZeRO1 | Same |

Use `scripts/train_m_only.sh`. This uses the exact final model and training
implementation. The existing supported `cache_delta_indices: []` setting keeps
visual history empty while preserving the same per-episode position bookkeeping.
The MoBA context/hooks remain installed but have no historical keys to read.
No model architecture is replaced with a native-S recurrent backbone.

Start a fresh run from Qwen3.5-4B; do not resume the previously trained native
S-only checkpoint. That checkpoint remains an extra architecture baseline,
not the final model's M-only ablation. Original training runtime strings such
as `full_attention_kv_policy` describe the shared implementation; the resolved
`moba_memory.cache_delta_indices: []`, ablation launch contract and observed
zero history counts define this run's actual KV availability.
