"""Configurable temporal observations for the mixed recurrent policy."""

from __future__ import annotations

ABSOLUTE_INDICES = (0,)
DELTA_INDICES = (-15, 0)
SUPPORTED_PROFILES = {
    (ABSOLUTE_INDICES, DELTA_INDICES),
    ((), (-25, 0)),
    ((), (-30, -15, 0)),
    # Retain the old profile when reading existing checkpoint configurations.
    ((), (25, 0)),
}


def temporal_indices(cfg):
    image = cfg.datasets.vla_data.temporal.image
    absolute = tuple(int(x) for x in image.get("absolute_indices", []))
    delta = tuple(int(x) for x in image.delta_indices)
    if (absolute, delta) not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported mix temporal profile: absolute={absolute}, delta={delta}"
        )
    return absolute, delta


def apply_recurrent_temporal_profile(cfg):
    absolute, delta = temporal_indices(cfg)
    if any(offset > 0 for offset in delta):
        raise ValueError(
            "Fresh mix training requires nonpositive historical deltas, not future frames"
        )
    data = cfg.datasets.vla_data
    data.temporal.enabled = True
    image = data.temporal.image
    image.absolute_indices = list(absolute)
    image.delta_indices = list(delta)
    image.include_initial = bool(absolute)
    image.include_current = 0 in delta
    data.pack_tbptt_windows = True
    return cfg


def build_recurrent_dataset(cfg):
    from eventvla.dataloader.lerobot_datasets import get_vla_dataset

    apply_recurrent_temporal_profile(cfg)
    expected_absolute, expected_delta = temporal_indices(cfg)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    sources = getattr(dataset, "datasets", None) or [dataset]
    for source in sources:
        modalities = getattr(source, "modality_configs", None)
        if not modalities or "video" not in modalities:
            raise ValueError("The recurrent dataset must expose its video modality")
        video = modalities["video"]
        absolute = tuple(int(x) for x in getattr(video, "absolute_indices", ()))
        delta = tuple(int(x) for x in getattr(video, "delta_indices", ()))
        if absolute != expected_absolute or delta != expected_delta:
            raise ValueError(
                f"Unexpected video history: absolute={absolute}, delta={delta}"
            )
    return dataset
