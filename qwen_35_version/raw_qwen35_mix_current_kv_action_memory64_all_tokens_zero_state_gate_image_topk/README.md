Qwen3.5 64-query zero-state gate with Top-4 individual-image KV retrieval

This independent copy is based on the frozen Top-4 zero_state_gate training source.
Historical routing uses (physical timestep, camera) blocks. Every full-attention
query/head independently selects at most four images across all historical views.
Image means are cached per layer. Persistent K/V remain detached and stored once;
physical-time grouping is retained only for transactional writes and time capacity.
Retrieval packing exposes each camera independently without expanding selections
to its sibling views. Relative image spans survive insertion of memory tokens.

Default training: pretrained Qwen3.5-4B, all parameters, 80k updates, 8 GPUs,
B2/GPU, TBPTT=2, 64 recurrent queries with zero initial memory and slot gates,
absolute [0] plus relative [-15,0], all-token KV reads, cache_delta_indices=[0].
Native DeltaNet S resets each decision. Query-memory gradients cross decisions
inside the TBPTT window, while KV remains detached at capture.

New configuration: block_unit=image, topk_history_images=4.
The inherited frame_time mode is available only for reference checks and requires
topk_history_frames instead; conflicting Top-k field names are rejected.

Verification: tests.test_image_routing (CPU and --gpu), inherited memory/cache
checks, and tests.validate_real_memory64 on recorded 9-image decisions.
