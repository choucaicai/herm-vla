from types import SimpleNamespace
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from ..model.eventvla_qwen35_mix import RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK
from ..model.query_memory import QueryMemory
from ..model.padding_safe_delta import install_padding_safe_deltanet

def tiny_text_model():
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3],
        },
        max_position_embeddings=4096, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    fused_norm = qwen.FusedRMSNormGated
    qwen.FusedRMSNormGated = None
    try:
        model = qwen.Qwen3_5TextModel(config)
    finally:
        qwen.FusedRMSNormGated = fused_norm
    for layer in model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.causal_conv1d_fn = None
            layer.linear_attn.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
    return model


def tiny_vision_policy(padding_safe=True):
    config = Qwen3_5Config(
        text_config=tiny_text_model().config.to_dict(),
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "in_channels": 3, "patch_size": 2,
            "temporal_patch_size": 1, "spatial_merge_size": 2,
            "out_hidden_size": 64, "num_position_embeddings": 16,
        },
        image_token_id=100, video_token_id=101,
        vision_start_token_id=102, vision_end_token_id=103,
    )
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    fused_norm = qwen.FusedRMSNormGated
    qwen.FusedRMSNormGated = None
    try:
        backbone = qwen.Qwen3_5Model(config)
    finally:
        qwen.FusedRMSNormGated = fused_norm
    for layer in backbone.language_model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.causal_conv1d_fn = None
            layer.linear_attn.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
            layer.linear_attn.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
    policy = RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK.__new__(RawQwen35MixCurrentKVActionMemory64AllTokensZeroStateGateImageTopK)
    torch.nn.Module.__init__(policy)
    policy.qwen_vl_interface = SimpleNamespace(
        model=SimpleNamespace(config=config, model=backbone, device=torch.device("cpu"))
    )
    policy._layer_types = tuple(config.text_config.layer_types)
    policy.action_token_id = 110
    policy.chunk_len = 2
    policy._memory_pad_token_id = 0
    policy.query_memory = QueryMemory(64)
    policy._moba_examples = None
    policy._moba_frame_specs = None
    policy._last_memory64_report = {}
    if padding_safe:
        install_padding_safe_deltanet(backbone.language_model)
    return policy, backbone


def tiny_image_inputs(lengths, images_per_sample=9):
    rows = [
        [102, 100, 100, 100, 100, 103] * images_per_sample + [7] * length + [110, 110]
        for length in lengths
    ]
    longest = max(map(len, rows))
    ids = torch.tensor([[0] * (longest - len(row)) + row for row in rows])
    return {
        "input_ids": ids,
        "attention_mask": (ids != 0).long(),
        "mm_token_type_ids": (ids == 100).long(),
        "image_grid_thw": torch.tensor([[1, 4, 4]] * (images_per_sample * len(rows))),
        "pixel_values": torch.randn(images_per_sample * len(rows) * 16, 12),
    }



def sample(timestep=0, slot=0, episode="a", **kwargs):
    return {"episode_id":episode,"slot_id":slot,"view_id":0,"stream_visit_id":0,
            "timestep":timestep,"image_metas":[
                {"time_index":group,"time_role":role,"delta":delta,"absolute_index":absolute,
                 "view":f"camera_{camera}"}
                for group,(role,delta,absolute) in enumerate([
                    ("first",None,0),("history",-15,None),("current",0,None)])
                for camera in range(3)], **kwargs}


def history3_sample(timestep=0, **kwargs):
    result=sample(timestep,**kwargs)
    result["image_metas"]=[
        {"time_index":group,"time_role":"history" if delta else "current","delta":delta,
         "absolute_index":None,"view":f"camera_{camera}"}
        for group,delta in enumerate([-30,-15,0]) for camera in range(3)]
    result["trajectory_length"]=1000
    return result


def forward(policy, inputs, examples, state=None):
    policy._moba_examples=examples
    policy._moba_frame_specs=None
    try:return policy._forward_qwen_with_state(inputs,state)
    finally:policy._moba_examples=None;policy._moba_frame_specs=None
