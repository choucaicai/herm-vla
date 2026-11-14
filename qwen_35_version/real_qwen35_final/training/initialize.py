"""Transfer the original final model while initializing a new 7-D output layer."""

from pathlib import Path
import torch

OUTPUT_KEYS = {"action_model.model.fc2.weight", "action_model.model.fc2.bias"}


def transfer_final_weights(model, state):
    for wrapper in ("module", "state_dict"):
        if wrapper in state and isinstance(state[wrapper], dict):
            state = state[wrapper]
            break
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"Final checkpoint keys differ: missing={set(expected)-set(state)}, "
                         f"unexpected={set(state)-set(expected)}")
    for key, value in state.items():
        if key in OUTPUT_KEYS:
            if value.shape[0] != 14 or expected[key].shape[0] != 7 or value.shape[1:] != expected[key].shape[1:]:
                raise ValueError(f"Expected original 14-D output and new 7-D output: {key}")
        elif value.shape != expected[key].shape:
            raise ValueError(f"Incompatible final checkpoint tensor: {key}")
    missing, unexpected = model.load_state_dict(
        {key: value for key, value in state.items() if key not in OUTPUT_KEYS}, strict=False,
    )
    if set(missing) != OUTPUT_KEYS or unexpected:
        raise RuntimeError(f"Unexpected transfer result: {missing}, {unexpected}")
    return {"loaded_tensors": len(state) - 2, "reinitialized": sorted(OUTPUT_KEYS)}


def initialize_from_final(model, checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return {"checkpoint": str(path), **transfer_final_weights(model, state)}
