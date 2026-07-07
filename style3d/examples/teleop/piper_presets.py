# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot-specific teleoperation specs for AgileX PiPER examples."""

from __future__ import annotations

from .newton_ik_teleop import ArmSpec


def piper_single_arm_spec(
    *,
    name: str = "piper",
    joint_prefix: str = "joint",
    joint_suffix: str = "",
    ee_body_name: str = "link6",
    finger_joint_names: tuple[str, str] | None = None,
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.13503),
    joint_search_start: int = 0,
    joint_search_stop: int | None = None,
    body_search_start: int = 0,
    body_search_stop: int | None = None,
    home_q: tuple[float, ...] = (0.0, 1.2, -1.6, 0.0, 0.8, 0.0),
    home_gripper_opening: float = 0.035,
    min_gripper_opening: float = 0.0,
    max_gripper_opening: float = 0.035,
) -> ArmSpec:
    """Return an ``ArmSpec`` for the AgileX PiPER MJCF convention."""

    joint_names = tuple(f"{joint_prefix}{i}{joint_suffix}" for i in range(1, 7))
    if finger_joint_names is None:
        finger_joint_names = (f"{joint_prefix}7{joint_suffix}", f"{joint_prefix}8{joint_suffix}")
    return ArmSpec(
        name=name,
        joint_names=joint_names,
        ee_body_name=ee_body_name,
        ee_offset=ee_offset,
        gripper_joint_names=finger_joint_names,
        gripper_joint_signs=(1.0, -1.0),
        joint_search_start=joint_search_start,
        joint_search_stop=joint_search_stop,
        body_search_start=body_search_start,
        body_search_stop=body_search_stop,
        home_q=home_q,
        home_gripper_opening=home_gripper_opening,
        min_gripper_opening=min_gripper_opening,
        max_gripper_opening=max_gripper_opening,
    )


def piper_dual_arm_specs(
    *,
    left_suffix: str = "",
    right_suffix: str = "_arm2",
    left_ee_body_name: str = "gripper_base_left",
    right_ee_body_name: str = "gripper_base_right",
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.1358),
    left_joint_search_start: int = 0,
    left_joint_search_stop: int | None = None,
    left_body_search_start: int = 0,
    left_body_search_stop: int | None = None,
    right_joint_search_start: int = 0,
    right_joint_search_stop: int | None = None,
    right_body_search_start: int = 0,
    right_body_search_stop: int | None = None,
    left_home_q: tuple[float, ...] = (-0.1240, 0.7980, -1.1250, -0.2253, 0.9856, 0.0829),
    right_home_q: tuple[float, ...] = (0.1240, 0.7980, -1.1250, 0.2253, 0.9856, -0.0829),
    home_gripper_opening: float = 0.035,
    min_gripper_opening: float = 0.0,
    max_gripper_opening: float = 0.035,
) -> tuple[ArmSpec, ArmSpec]:
    """Return left/right PiPER specs for dual-arm MJCF naming conventions."""

    left = piper_single_arm_spec(
        name="left",
        joint_suffix=left_suffix,
        ee_body_name=left_ee_body_name,
        ee_offset=ee_offset,
        joint_search_start=left_joint_search_start,
        joint_search_stop=left_joint_search_stop,
        body_search_start=left_body_search_start,
        body_search_stop=left_body_search_stop,
        home_q=left_home_q,
        home_gripper_opening=home_gripper_opening,
        min_gripper_opening=min_gripper_opening,
        max_gripper_opening=max_gripper_opening,
    )
    right = piper_single_arm_spec(
        name="right",
        joint_suffix=right_suffix,
        ee_body_name=right_ee_body_name,
        ee_offset=ee_offset,
        joint_search_start=right_joint_search_start,
        joint_search_stop=right_joint_search_stop,
        body_search_start=right_body_search_start,
        body_search_stop=right_body_search_stop,
        home_q=right_home_q,
        home_gripper_opening=home_gripper_opening,
        min_gripper_opening=min_gripper_opening,
        max_gripper_opening=max_gripper_opening,
    )
    return left, right
