"""Dataset mixture definitions for EventVLA."""

from typing import Dict, List, Tuple


DATASET_NAMED_MIXTURES: Dict[str, List[Tuple[str, float, str]]] = {
    "robotwin_mem": [
        ("cover_blocks_hard", 1.0, "robotwin_mem"),
        ("put_back_block_hard", 1.0, "robotwin_mem"),
        ("rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("observe_and_pickup_hard", 1.0, "robotwin_mem"),
        ("find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("observe_and_pickup_object", 1.0, "robotwin_mem"),
        ("reproduct_route", 1.0, "robotwin_mem"),
        ("press_button_keyframe", 1.0, "robotwin_mem"),
    ],
    "robotwin_mem8": [
        ("/path/to/RoboTwin-Mem/lerobot_2.1/cover_blocks_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/pick_objects_in_order", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/pick_the_unhidden_block", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/press_button_keyframe", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/put_back_block_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/reproduce_route", 1.0, "robotwin_mem"),
    ], 
    "robotwin_mem9": [
        ("/path/to/RoboTwin-Mem/lerobot_2.1/cover_blocks_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/find_seal_and_seal_stamp", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/pick_objects_in_order", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/pick_the_unhidden_block", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/press_button_keyframe", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/put_back_block_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/rearrange_blocks_hard", 1.0, "robotwin_mem"),
        ("/path/to/RoboTwin-Mem/lerobot_2.1/reproduce_route", 1.0, "robotwin_mem"),
        ("/path/to/RMBench/lerobot_2.1/swap_T_hard", 1.0, "robotwin_mem"),
    ], 
}


# Local RMBench conversion: nine official tasks in data/lerobot_2.1.
DATASET_NAMED_MIXTURES["rmbench9"] = [
    ("battery_try", 1.0, "robotwin_mem"),
    ("blocks_ranking_try", 1.0, "robotwin_mem"),
    ("cover_blocks", 1.0, "robotwin_mem"),
    ("observe_and_pickup", 1.0, "robotwin_mem"),
    ("press_button", 1.0, "robotwin_mem"),
    ("put_back_block", 1.0, "robotwin_mem"),
    ("rearrange_blocks", 1.0, "robotwin_mem"),
    ("swap_T", 1.0, "robotwin_mem"),
    ("swap_blocks", 1.0, "robotwin_mem"),
]
