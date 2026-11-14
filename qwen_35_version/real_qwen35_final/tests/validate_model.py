"""Exercise real Qwen3.5 forward/backward and serving; no optimizer updates."""

import argparse
import json
from pathlib import Path
import socket
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from ..data.temporal_profile import build_recurrent_dataset
from ..data.episode_windows import RememEpisodeWindowDataset, RememWindow
from ..eval.policy import RecurrentServingPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/train.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cfg = OmegaConf.load(args.config)
    mixture = build_recurrent_dataset(cfg)
    source = mixture.datasets[0]
    windows = RememEpisodeWindowDataset(mixture, decision_stride=50, reset_each_window=False)
    # Exercise a trajectory whose videos use the second shard.
    ep = next(int(ep) for ep in source.trajectory_ids
              if source.get_video_path(int(ep), "cam_high").name == "file-001.mp4")
    samples = []
    for timestep in (0, 50):
        sample, mask = windows._load_unit(RememWindow(0, ep, timestep))
        sample.update(episode_id="piper-model-probe", slot_id=0, view_id=0, stream_visit_id=0,
                      timestep=timestep, reset=timestep == 0, is_new_episode=timestep == 0,
                      done=False, is_last=False, action_valid_mask=mask)
        samples.append(sample)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = output_path.with_name("model_dataset_statistics.json")
    mixture.save_dataset_statistics(stats_path)
    print("Loading Qwen3.5-4B for two-decision validation", flush=True)
    model = RecurrentServingPolicy(cfg).to(device="cuda", dtype=torch.bfloat16).train()
    model.norm_stats = json.loads(stats_path.read_text())
    assert model.action_model.model.fc2.out_features == 7
    state, losses, reports = None, [], []
    for step, sample in enumerate(samples):
        result, state = model([sample], recurrent_state=state, return_recurrent_state=True)
        assert tuple(state.shape) == (1, 64, 2560) and state.requires_grad
        assert torch.isfinite(state).all() and torch.isfinite(result["total_loss"])
        if step == 0:
            assert (model.query_memory.last_gate == 1).all()
        pending = model.moba_memory.pending
        effective = int(pending.stats()["moba_topk_effective"])
        assert effective == (0 if step == 0 else 2)
        model.commit_moba_memory()
        for slot in model.moba_memory.slots:
            for frame in slot.frames.values():
                assert len(frame.image_spans) == 2
                for key, value, mean in frame.layers.values():
                    assert not key.requires_grad and not value.requires_grad
        losses.append(result["total_loss"])
        reports.append({"timestep": sample["timestep"], "loss": float(losses[-1].detach()),
                        "effective_history_images": effective, "memory_shape": list(state.shape)})
        print("Validated forward", reports[-1], flush=True)
    sum(losses).backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Non-finite gradient: {name}")
    grad_names = ["action_model.model.fc2.weight", "query_memory.write_queries",
                  "query_memory.feedback_projection.weight"]
    parameters = dict(model.named_parameters())
    for name in grad_names:
        assert parameters[name].grad is not None and parameters[name].grad.abs().sum() > 0, name
    print("Validated backward", flush=True)
    model.zero_grad(set_to_none=True)
    del losses, state, result
    model.eval()
    model.reset_memory()
    response = model.predict_action([samples[0]], episode_id="serving-probe", decision_id=0)
    assert response["actions"].shape == (1, 50, 7)
    assert np.isfinite(response["actions"]).all()
    repeated = model.predict_action([samples[0]], episode_id="serving-probe", decision_id=0)
    np.testing.assert_array_equal(repeated["actions"], response["actions"])
    model.reset_memory()
    assert model._query_memory_state is None and model._active_episode is None
    report = {"passed": True, "hostname": socket.gethostname(), "episode": ep,
              "forward_backward": reports, "gradient_checks": grad_names,
              "serving_shape": [1, 50, 7], "retry_idempotent": True,
              "episode_reset": True, "optimizer_updates": 0, "checkpoints_written": 0,
              "max_cuda_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
              "elapsed_seconds": time.monotonic() - start}
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
