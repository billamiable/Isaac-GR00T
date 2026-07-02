# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""External data configuration module for GR1 arms-only simulation (ego + wrist views).

Arm-relative / hand-absolute variant of ``gr1_arms_only_wrist_data_config``: the
``left_arm`` and ``right_arm`` action keys use ``ActionRepresentation.RELATIVE``
(delta w.r.t. the current state), while ``left_hand`` and ``right_hand`` remain
``ActionRepresentation.ABSOLUTE`` (the hand DoFs are near-binary open/close grasps
plus degenerate constant dims, for which relative normalization is ill-conditioned).

State modality (including ``sin_cos_embedding_keys``) is unchanged: sin/cos
embedding only affects state and is orthogonal to the action representation.

The underlying lerobot dataset must provide both ``observation.images.ego_view``
and ``observation.images.wrist_view`` (see ``meta/modality.json``). Use exactly
one of these config files per run -- both register for ``EmbodimentTag.GR1`` and
``register_modality_config`` rejects a double registration.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

gr1_arms_only_wrist_relative_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        sin_cos_embedding_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(gr1_arms_only_wrist_relative_config, embodiment_tag=EmbodimentTag.GR1)
