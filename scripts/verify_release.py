#!/usr/bin/env python3
"""Validate the release without loading model weights or importing PyTorch."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk"


def main():
    python_files = list(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    shell_files = list(ROOT.rglob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)
    scheduler = (ROOT / "scripts/eval_rmbench_v2.sh").read_text()
    body = scheduler.split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
    ast.parse(body, filename="eval_rmbench_v2 embedded Python")

    frozen = json.loads((ROOT / "docs/frozen_source_sha256.json").read_text())
    extension = json.loads((ROOT / "docs/topk_extension_sha256.json").read_text())["core_changes"]
    checked = []
    for relative, expected in frozen.items():
        if relative.startswith(("model/", "data/", "training/")) and relative.endswith(".py"):
            path = PACKAGE / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                change = extension.get(relative, {})
                if change.get("frozen_sha256") != expected or change.get("extended_sha256") != actual:
                    raise AssertionError(f"Frozen model/data/trainer hash mismatch: {relative}")
            checked.append(relative)

    results = json.loads((ROOT / "docs/results.json").read_text())
    assert len(results["tasks"]) == 9
    assert sum(row["success"] for row in results["tasks"]) == results["success"] == 188
    assert sum(row["episodes"] for row in results["tasks"]) == results["episodes"] == 225
    assert {row["task"] for row in results["tasks"] if row["swap_rb"]} == {"cover_blocks"}
    assert all(row["episodes"] == 25 for row in results["tasks"])

    required = [
        "scripts/train_final.sh", "scripts/eval_final.sh", "scripts/run_policy_server.sh",
        "scripts/validate_checkpoint.sh", "scripts/rmbench_color_swap_interface.py",
        "examples/RMBench/train_files/convert_rmbench_to_lerobot.py",
        "eventvla/dataloader/lerobot_datasets.py", "requirements.txt", "requirements-cuda.txt",
    ]
    assert all((ROOT / name).is_file() for name in required)
    forbidden = []
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts or not path.is_file():
            continue
        if path.suffix in {".pt", ".pth", ".safetensors", ".mp4", ".parquet", ".pem", ".key"}:
            forbidden.append(str(path.relative_to(ROOT)))
        if path.stat().st_size > 5_000_000:
            forbidden.append(str(path.relative_to(ROOT)))
    assert not forbidden, forbidden
    print(json.dumps({
        "passed": True,
        "python_files_parsed": len(python_files),
        "shell_files_parsed": len(shell_files),
        "frozen_core_python_files_verified": len(checked),
        "recorded_result": "188/225 (83.56%)",
        "checkpoint_data_and_key_files_in_release": forbidden,
    }, indent=2))


if __name__ == "__main__":
    main()
