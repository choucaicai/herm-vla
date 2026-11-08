# RMBench integration

See the root README for the final training and evaluation commands.

Data conversion: `train_files/convert_rmbench_to_lerobot.py` rewrites prepared StarVLA LeRobot-v2 metadata for the EventVLA reader, preserving parquet/MP4 files through symlinks.

`eval_files/model2robotwin_mem_interface.py` supplies shared client utilities. The final 64Q/Image-Top4 policy adapter lives in the Qwen model package. `scripts/eval_final.sh` uses the formal nine-task protocol and cover_blocks-only R/B correction.
