# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton IK based teleoperation utilities for examples.

The helpers in this module intentionally depend only on Newton, Warp, NumPy,
and optionally pygame for joystick input.  They are meant to be used by example
scripts that already own the simulation loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.ik as ik


PoseWxyz = np.ndarray


@wp.kernel
def copy_body_poses_kernel(
    src_body_q: wp.array[wp.transform],
    dst_body_q: wp.array[wp.transform],
    src_body_start: int,
    dst_body_start: int,
):
    body_offset = wp.tid()
    dst_body_q[dst_body_start + body_offset] = src_body_q[src_body_start + body_offset]


@wp.kernel
def copy_body_poses_with_previous_kernel(
    src_body_q: wp.array[wp.transform],
    dst_body_q: wp.array[wp.transform],
    dst_body_q_prev: wp.array[wp.transform],
    src_body_start: int,
    dst_body_start: int,
):
    body_offset = wp.tid()
    pose = src_body_q[src_body_start + body_offset]
    dst_body_q[dst_body_start + body_offset] = pose
    dst_body_q_prev[dst_body_start + body_offset] = pose


@dataclass(frozen=True)
class ArmSpec:
    """Labels and geometry needed to control one arm in a Newton model."""

    name: str
    joint_names: tuple[str, ...]
    ee_body_name: str
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_joint_names: tuple[str, ...] = ()
    gripper_joint_signs: tuple[float, ...] = ()
    joint_search_start: int = 0
    joint_search_stop: int | None = None
    body_search_start: int = 0
    body_search_stop: int | None = None
    home_q: tuple[float, ...] | None = None
    home_gripper_opening: float = 0.035
    min_gripper_opening: float = 0.0
    max_gripper_opening: float = 0.035

    def __post_init__(self):
        if not self.joint_names:
            raise ValueError("ArmSpec.joint_names must contain at least one joint")
        if self.gripper_joint_names and self.gripper_joint_signs:
            if len(self.gripper_joint_names) != len(self.gripper_joint_signs):
                raise ValueError("gripper_joint_names and gripper_joint_signs must have the same length")


@dataclass
class Button:
    last_state: bool = False

    def update(self, current_state: bool) -> bool:
        triggered = bool(current_state) and not self.last_state
        self.last_state = bool(current_state)
        return triggered


@dataclass
class TeleopState:
    joint_q: np.ndarray
    gripper_opening: float
    home_q: np.ndarray
    home_gripper_opening: float


def find_label_index(labels: Sequence[object], name: str, start: int = 0, stop: int | None = None) -> int:
    """Find an exact or suffix label match inside ``labels[start:stop]``."""

    stop = len(labels) if stop is None else min(stop, len(labels))
    for index in range(start, stop):
        label = str(labels[index])
        if label == name or label.endswith(f"/{name}") or label.endswith(name):
            return index
    available = [str(labels[index]) for index in range(start, stop)]
    raise ValueError(f"Could not find label {name!r}. Available labels: {available}")


def resolve_joint_q_indices(
    model: newton.Model,
    joint_names: Sequence[str],
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Resolve one-DoF joint labels to their joint_q coordinate indices."""

    joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int32)
    indices: list[int] = []
    for joint_name in joint_names:
        joint_index = find_label_index(model.joint_label, joint_name, start, stop)
        q_start = int(joint_q_start[joint_index])
        q_stop = int(joint_q_start[joint_index + 1])
        q_dim = q_stop - q_start
        if q_dim != 1:
            raise ValueError(
                f"Teleop expects one-DoF joints, but {joint_name!r} has {q_dim} joint_q coordinates"
            )
        indices.append(q_start)
    return np.asarray(indices, dtype=np.int32)


def current_joint_q(model: newton.Model, state: newton.State | None = None) -> np.ndarray:
    if state is not None and getattr(state, "joint_q", None) is not None:
        return np.asarray(state.joint_q.numpy(), dtype=np.float64).copy()
    return np.asarray(model.joint_q.numpy(), dtype=np.float64).copy()


def quat_rotate_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    u = np.array([x, y, z], dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def quat_angle_xyzw(q_a_xyzw: np.ndarray, q_b_xyzw: np.ndarray) -> float:
    qa = q_a_xyzw / max(np.linalg.norm(q_a_xyzw), 1.0e-12)
    qb = q_b_xyzw / max(np.linalg.norm(q_b_xyzw), 1.0e-12)
    dot = abs(float(np.dot(qa, qb)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    q = np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return q / max(np.linalg.norm(q), 1.0e-12)


def quat_from_euler_xyz_wxyz(angles_rad: np.ndarray) -> np.ndarray:
    rx, ry, rz = angles_rad
    sx, cx = math.sin(0.5 * rx), math.cos(0.5 * rx)
    sy, cy = math.sin(0.5 * ry), math.cos(0.5 * ry)
    sz, cz = math.sin(0.5 * rz), math.cos(0.5 * rz)
    qx = np.asarray([cx, sx, 0.0, 0.0], dtype=np.float64)
    qy = np.asarray([cy, 0.0, sy, 0.0], dtype=np.float64)
    qz = np.asarray([cz, 0.0, 0.0, sz], dtype=np.float64)
    return quat_mul_wxyz(quat_mul_wxyz(qx, qy), qz)


class NewtonIKArm:
    """Small FK/IK adapter around ``newton.ik.IKSolver`` for one arm."""

    def __init__(
        self,
        model: newton.Model,
        spec: ArmSpec,
        *,
        state: newton.State | None = None,
        full_q_getter: Callable[[], np.ndarray] | None = None,
        iterations: int = 24,
        position_tolerance: float = 2.5e-3,
        rotation_tolerance: float = math.radians(8.0),
        include_rotation_objective: bool = True,
        joint_limit_weight: float = 10.0,
        lambda_initial: float = 0.1,
    ) -> None:
        self.model = model
        self.spec = spec
        self.state = state
        self.full_q_getter = full_q_getter
        self.iterations = int(iterations)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.joint_q_indices = resolve_joint_q_indices(
            model,
            spec.joint_names,
            start=spec.joint_search_start,
            stop=spec.joint_search_stop,
        )
        self.gripper_q_indices = (
            resolve_joint_q_indices(
                model,
                spec.gripper_joint_names,
                start=spec.joint_search_start,
                stop=spec.joint_search_stop,
            )
            if spec.gripper_joint_names
            else np.empty(0, dtype=np.int32)
        )
        if spec.gripper_joint_signs:
            self.gripper_signs = np.asarray(spec.gripper_joint_signs, dtype=np.float64)
        else:
            self.gripper_signs = np.ones(len(self.gripper_q_indices), dtype=np.float64)
        self.ee_body = find_label_index(
            model.body_label,
            spec.ee_body_name,
            spec.body_search_start,
            spec.body_search_stop,
        )
        self.ee_offset_np = np.asarray(spec.ee_offset, dtype=np.float64)
        self.ee_offset_wp = wp.vec3(*map(float, self.ee_offset_np))

        self._fk_state = model.state()
        self._fk_q = wp.array(self.current_full_q().astype(np.float32), dtype=wp.float32, device=model.device)
        self._fk_qd = wp.zeros_like(model.joint_qd)

        start_pose = self.solve_fk(self.current_arm_q())
        target_pos = wp.vec3(*map(float, start_pose[:3]))
        target_q_xyzw = quat_wxyz_to_xyzw(start_pose[3:7])
        self.pos_obj = ik.IKObjectivePosition(
            link_index=int(self.ee_body),
            link_offset=self.ee_offset_wp,
            target_positions=wp.array([target_pos], dtype=wp.vec3, device=model.device),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=int(self.ee_body),
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*map(float, target_q_xyzw))], dtype=wp.vec4, device=model.device),
        )
        self.limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
            weight=float(joint_limit_weight),
        )
        objectives = [self.pos_obj, self.limit_obj]
        if include_rotation_objective:
            objectives.insert(1, self.rot_obj)
        self.solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=float(lambda_initial),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._ik_q = wp.array(
            self.current_full_q()[None, :].astype(np.float32),
            dtype=wp.float32,
            device=model.device,
        )

    def current_full_q(self) -> np.ndarray:
        if self.full_q_getter is not None:
            return np.asarray(self.full_q_getter(), dtype=np.float64).copy()
        return current_joint_q(self.model, self.state)

    def current_arm_q(self) -> np.ndarray:
        return self.current_full_q()[self.joint_q_indices].copy()

    def joint_limits(self) -> np.ndarray:
        lower = self.model.joint_limit_lower.numpy()[self.joint_q_indices]
        upper = self.model.joint_limit_upper.numpy()[self.joint_q_indices]
        return np.stack([lower, upper], axis=1).astype(np.float64)

    def full_q_with_arm(self, arm_q: Sequence[float], *, full_q: np.ndarray | None = None) -> np.ndarray:
        q = self.current_full_q() if full_q is None else np.asarray(full_q, dtype=np.float64).copy()
        q[self.joint_q_indices] = np.asarray(arm_q, dtype=np.float64)[: len(self.joint_q_indices)]
        return q

    def full_q_with_gripper(self, opening: float, *, full_q: np.ndarray | None = None) -> np.ndarray:
        q = self.current_full_q() if full_q is None else np.asarray(full_q, dtype=np.float64).copy()
        if len(self.gripper_q_indices):
            opening = float(np.clip(opening, self.spec.min_gripper_opening, self.spec.max_gripper_opening))
            q[self.gripper_q_indices] = self.gripper_signs * opening
        return q

    def solve_fk(self, joint_angles: Sequence[float], *, full_q: np.ndarray | None = None) -> PoseWxyz:
        """Return TCP pose as ``[x, y, z, qw, qx, qy, qz]``."""

        q = self.full_q_with_arm(joint_angles, full_q=full_q)
        self._fk_q.assign(q.astype(np.float32))
        self._fk_qd.zero_()
        newton.eval_fk(self.model, self._fk_q, self._fk_qd, self._fk_state)

        body_pose = self._fk_state.body_q.numpy()[int(self.ee_body)]
        q_xyzw = body_pose[3:7].astype(np.float64)
        pos = body_pose[:3].astype(np.float64) + quat_rotate_xyzw(q_xyzw, self.ee_offset_np)
        return np.asarray([pos[0], pos[1], pos[2], q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

    def solve_ik(
        self,
        target_pose: Sequence[float],
        *,
        seed_joints: Sequence[float] | None = None,
        seed_full_q: np.ndarray | None = None,
        threshold: float = 1.0e-3,
    ) -> np.ndarray | None:
        """Solve IK and return this arm's joint coordinates."""

        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape[0] < 7:
            raise ValueError("target_pose must be [x, y, z, qw, qx, qy, qz]")

        target_pos = target[:3]
        target_q_xyzw = quat_wxyz_to_xyzw(target[3:7])
        target_q_xyzw /= max(np.linalg.norm(target_q_xyzw), 1.0e-12)

        self.pos_obj.set_target_position(0, wp.vec3(*map(float, target_pos)))
        self.rot_obj.set_target_rotation(0, wp.vec4(*map(float, target_q_xyzw)))

        seed = self.current_full_q() if seed_full_q is None else np.asarray(seed_full_q, dtype=np.float64).copy()
        if seed_joints is not None:
            seed[self.joint_q_indices] = np.asarray(seed_joints, dtype=np.float64)[: len(self.joint_q_indices)]
        self._ik_q.assign(seed[None, :].astype(np.float32))
        self.solver.step(self._ik_q, self._ik_q, iterations=self.iterations)

        solved_full = self._ik_q.numpy()[0].astype(np.float64)
        candidate = solved_full[self.joint_q_indices].copy()
        fk = self.solve_fk(candidate, full_q=solved_full)
        pos_err = float(np.linalg.norm(fk[:3] - target_pos))
        rot_err = quat_angle_xyzw(quat_wxyz_to_xyzw(fk[3:7]), target_q_xyzw)
        if pos_err > max(float(threshold), self.position_tolerance) or rot_err > self.rotation_tolerance:
            return None
        return candidate


class KinematicArticulationMirror:
    """Evaluate FK on a source model and copy body poses into a target state."""

    def __init__(
        self,
        source_model: newton.Model,
        *,
        source_body_start: int = 0,
        target_body_start: int = 0,
        body_count: int | None = None,
    ) -> None:
        self.source_model = source_model
        self.source_body_start = int(source_body_start)
        self.target_body_start = int(target_body_start)
        self.body_count = int(source_model.body_count - source_body_start if body_count is None else body_count)
        if self.body_count <= 0:
            raise ValueError("body_count must be positive")
        self.source_state = source_model.state()
        self._source_q = wp.array(source_model.joint_q.numpy(), dtype=wp.float32, device=source_model.device)
        self._source_qd = wp.zeros_like(source_model.joint_qd)

    def apply(
        self,
        full_q: Sequence[float],
        target_state: newton.State,
        *,
        target_body_q_prev: wp.array | None = None,
    ) -> None:
        self._source_q.assign(np.asarray(full_q, dtype=np.float32))
        self._source_qd.zero_()
        newton.eval_fk(self.source_model, self._source_q, self._source_qd, self.source_state)
        if target_body_q_prev is None:
            wp.launch(
                copy_body_poses_kernel,
                dim=self.body_count,
                inputs=[
                    self.source_state.body_q,
                    target_state.body_q,
                    self.source_body_start,
                    self.target_body_start,
                ],
                device=self.source_model.device,
            )
        else:
            wp.launch(
                copy_body_poses_with_previous_kernel,
                dim=self.body_count,
                inputs=[
                    self.source_state.body_q,
                    target_state.body_q,
                    target_body_q_prev,
                    self.source_body_start,
                    self.target_body_start,
                ],
                device=self.source_model.device,
            )


class GamepadTeleopController:
    """Gamepad controller that updates one or more ``NewtonIKArm`` targets."""

    def __init__(
        self,
        arms: Sequence[NewtonIKArm],
        *,
        base_full_q: Sequence[float] | None = None,
        active_arm: str | None = None,
        deadzone: float = 0.1,
        step_joint: float = 0.04,
        step_pos: float = 0.004,
        step_rot_deg: float = 2.0,
        step_gripper: float = 0.0015,
        speed_factors: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 3.0),
        speed_index: int = 2,
        joystick_index: int = 0,
        print_help: bool = True,
    ) -> None:
        if not arms:
            raise ValueError("GamepadTeleopController requires at least one arm")
        self.arms = {arm.spec.name: arm for arm in arms}
        self.arm_order = [arm.spec.name for arm in arms]
        self.active_arm = active_arm or self.arm_order[0]
        if self.active_arm not in self.arms:
            raise ValueError(f"Unknown active_arm {self.active_arm!r}; available={self.arm_order}")

        self.full_q_target = (
            np.asarray(base_full_q, dtype=np.float64).copy()
            if base_full_q is not None
            else arms[0].current_full_q()
        )
        self.states: dict[str, TeleopState] = {}
        for arm in arms:
            current = self.full_q_target[arm.joint_q_indices].copy()
            home_q = np.asarray(arm.spec.home_q, dtype=np.float64) if arm.spec.home_q is not None else current.copy()
            if len(home_q) != len(arm.joint_q_indices):
                raise ValueError(f"Home q for arm {arm.spec.name!r} must have {len(arm.joint_q_indices)} values")
            self.states[arm.spec.name] = TeleopState(
                joint_q=current,
                gripper_opening=float(arm.spec.home_gripper_opening),
                home_q=home_q,
                home_gripper_opening=float(arm.spec.home_gripper_opening),
            )
            self.full_q_target = arm.full_q_with_gripper(
                self.states[arm.spec.name].gripper_opening,
                full_q=arm.full_q_with_arm(current, full_q=self.full_q_target),
            )

        self.control_mode = "joint"
        self.deadzone = float(deadzone)
        self.step_joint = float(step_joint)
        self.step_pos = float(step_pos)
        self.step_rot_deg = float(step_rot_deg)
        self.step_gripper = float(step_gripper)
        self.speed_factors = tuple(float(v) for v in speed_factors)
        self.speed_index = int(np.clip(speed_index, 0, len(self.speed_factors) - 1))
        self.axis_map = {"lx": 0, "ly": 1, "rx": 3, "ry": 4, "lt": 2, "rt": 5}
        self.btn_map = {"a": 0, "b": 1, "x": 2, "y": 3, "lb": 4, "rb": 5, "back": 6, "start": 7, "home": 8}
        self.buttons = {name: Button() for name in self.btn_map}

        try:
            import pygame
        except ImportError:  # pragma: no cover
            pygame = None
        self.pygame = pygame
        if pygame is None:
            self.joystick = None
            print("[NewtonTeleop] pygame is not installed; gamepad input is disabled.", flush=True)
            return

        pygame.init()
        pygame.joystick.init()
        self.joystick = (
            pygame.joystick.Joystick(joystick_index) if pygame.joystick.get_count() > joystick_index else None
        )
        if self.joystick is not None:
            self.joystick.init()
            print(f"[NewtonTeleop] Joystick: {self.joystick.get_name()}", flush=True)
        else:
            print("[NewtonTeleop] No joystick detected. Connect one and restart teleop.", flush=True)
        if print_help:
            self.print_instructions()

    @property
    def current_arm(self) -> NewtonIKArm:
        return self.arms[self.active_arm]

    @property
    def current_state(self) -> TeleopState:
        return self.states[self.active_arm]

    def close(self) -> None:
        if self.pygame is not None:
            self.pygame.quit()

    def _get_axis(self, name: str) -> float:
        if self.joystick is None:
            return 0.0
        value = float(self.joystick.get_axis(self.axis_map[name]))
        if name in ("lt", "rt"):
            value = (value + 1.0) / 2.0
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def _get_hat(self) -> tuple[int, int]:
        if self.joystick is None or self.joystick.get_numhats() == 0:
            return (0, 0)
        return self.joystick.get_hat(0)

    def update(self) -> np.ndarray:
        """Poll the gamepad and return the current full joint_q target."""

        if self.pygame is not None:
            self.pygame.event.pump()
        if self.joystick is None:
            return self.full_q_target

        events = {}
        for name, button in self.buttons.items():
            if button.update(bool(self.joystick.get_button(self.btn_map[name]))):
                events[name] = True
        self._handle_buttons(events)

        speed = self.speed_factors[self.speed_index]
        if self.control_mode == "joint":
            self._update_joint_mode(speed)
        else:
            self._update_pose_mode(speed)
        self._update_gripper(speed)
        self._write_current_state_to_full_q()
        return self.full_q_target

    def _handle_buttons(self, events: dict[str, bool]) -> None:
        if "back" in events and len(self.arm_order) > 1:
            index = (self.arm_order.index(self.active_arm) + 1) % len(self.arm_order)
            self.active_arm = self.arm_order[index]
            print(f"[NewtonTeleop] Active arm: {self.active_arm}", flush=True)
        if "start" in events:
            self.control_mode = "pose" if self.control_mode == "joint" else "joint"
            print(f"[NewtonTeleop] Mode: {self.control_mode}", flush=True)
        if "lb" in events:
            self.speed_index = max(0, self.speed_index - 1)
            print(f"[NewtonTeleop] Speed: {self.speed_factors[self.speed_index]}x", flush=True)
        if "rb" in events:
            self.speed_index = min(len(self.speed_factors) - 1, self.speed_index + 1)
            print(f"[NewtonTeleop] Speed: {self.speed_factors[self.speed_index]}x", flush=True)
        if "y" in events:
            for state in self.states.values():
                state.joint_q = state.home_q.copy()
                state.gripper_opening = state.home_gripper_opening
            print("[NewtonTeleop] Home", flush=True)

    def _update_joint_mode(self, speed: float) -> None:
        arm = self.current_arm
        state = self.current_state
        lx = self._get_axis("lx")
        ly = self._get_axis("ly")
        rx = self._get_axis("rx")
        ry = self._get_axis("ry")
        hat = self._get_hat()
        delta = np.zeros(len(arm.joint_q_indices), dtype=np.float64)
        if len(delta) > 0:
            delta[0] = -lx * self.step_joint * speed
        if len(delta) > 1:
            delta[1] = -ly * self.step_joint * speed
        if len(delta) > 2:
            delta[2] = ry * self.step_joint * speed
        if len(delta) > 3:
            delta[3] = hat[0] * self.step_joint * speed
        if len(delta) > 4:
            delta[4] = -hat[1] * self.step_joint * speed
        if len(delta) > 5:
            delta[5] = rx * self.step_joint * speed
        limits = arm.joint_limits()
        state.joint_q = np.clip(state.joint_q + delta, limits[:, 0], limits[:, 1])

    def _update_pose_mode(self, speed: float) -> None:
        arm = self.current_arm
        state = self.current_state
        lx = self._get_axis("lx")
        ly = self._get_axis("ly")
        rx = self._get_axis("rx")
        ry = self._get_axis("ry")
        hat = self._get_hat()
        if abs(lx) + abs(ly) + abs(rx) + abs(ry) + abs(hat[0]) + abs(hat[1]) < 1.0e-3:
            return

        current_pose = arm.solve_fk(state.joint_q, full_q=self.full_q_target)
        d_world = np.asarray([-ly, -lx, -ry], dtype=np.float64) * self.step_pos * speed
        d_euler_rad = np.radians(np.asarray([hat[0], -hat[1], -rx], dtype=np.float64) * self.step_rot_deg * speed)
        target_quat = quat_mul_wxyz(current_pose[3:7], quat_from_euler_xyz_wxyz(d_euler_rad))
        target_pose = np.concatenate([current_pose[:3] + d_world, target_quat])
        solved = arm.solve_ik(target_pose, seed_joints=state.joint_q, seed_full_q=self.full_q_target)
        if solved is not None:
            limits = arm.joint_limits()
            state.joint_q = np.clip(solved, limits[:, 0], limits[:, 1])

    def _update_gripper(self, speed: float) -> None:
        arm = self.current_arm
        state = self.current_state
        lt = self._get_axis("lt")
        rt = self._get_axis("rt")
        delta = (rt - lt) * self.step_gripper * speed
        state.gripper_opening = float(
            np.clip(
                state.gripper_opening + delta,
                arm.spec.min_gripper_opening,
                arm.spec.max_gripper_opening,
            )
        )

    def _write_current_state_to_full_q(self) -> None:
        for name in self.arm_order:
            arm = self.arms[name]
            state = self.states[name]
            self.full_q_target = arm.full_q_with_arm(state.joint_q, full_q=self.full_q_target)
            self.full_q_target = arm.full_q_with_gripper(state.gripper_opening, full_q=self.full_q_target)

    def print_instructions(self) -> None:
        print("\n" + "=" * 60)
        print("NEWTON GAMEPAD TELEOP")
        print("=" * 60)
        print("[BACK] switch arm   [START] joint/pose mode   [Y] home")
        print("[LB/RB] speed       LT close gripper           RT open gripper")
        print("Joint: left stick J1/J2, right stick J3/J6, d-pad J4/J5")
        print("Pose: left stick XY, right stick Z/yaw, d-pad roll/pitch")
        print(f"arms={self.arm_order}, active={self.active_arm}, mode={self.control_mode}")
        print("=" * 60 + "\n")


class KeyboardTeleopController:
    """Keyboard controller that updates one or two ``NewtonIKArm`` targets.

    The default bindings use the SIM1 translation layout, with separate
    hold-to-move gripper keys: left arm on ``WASD/QE/RF/ZC/XV`` and right arm on
    ``UJHK/YI/OL/BM/NP``.  For a single arm, the left-arm layout is used.
    """

    @dataclass(frozen=True)
    class Layout:
        """Keyboard bindings for one arm."""

        name: str
        x_pos: str
        x_neg: str
        y_pos: str
        y_neg: str
        z_pos: str
        z_neg: str
        pitch_pos: str
        pitch_neg: str
        yaw_pos: str
        yaw_neg: str
        gripper_close: str
        gripper_open: str

    DEFAULT_LEFT_LAYOUT = Layout(
        name="left",
        x_pos="w",
        x_neg="s",
        y_pos="a",
        y_neg="d",
        z_pos="e",
        z_neg="q",
        pitch_pos="r",
        pitch_neg="f",
        yaw_pos="z",
        yaw_neg="c",
        gripper_close="x",
        gripper_open="v",
    )
    DEFAULT_RIGHT_LAYOUT = Layout(
        name="right",
        x_pos="u",
        x_neg="j",
        y_pos="h",
        y_neg="k",
        z_pos="i",
        z_neg="y",
        pitch_pos="o",
        pitch_neg="l",
        yaw_pos="b",
        yaw_neg="m",
        gripper_close="n",
        gripper_open="p",
    )

    def __init__(
        self,
        arms: Sequence[NewtonIKArm],
        viewer,
        *,
        base_full_q: Sequence[float] | None = None,
        step_pos: float = 0.004,
        step_rot_deg: float = 2.0,
        step_gripper: float = 0.0015,
        shift_speed_factor: float = 2.0,
        layouts: Sequence[KeyboardTeleopController.Layout] | None = None,
        resolve_viewer_conflicts: bool = True,
        print_help: bool = True,
    ) -> None:
        if not arms:
            raise ValueError("KeyboardTeleopController requires at least one arm")
        if viewer is None or not hasattr(viewer, "is_key_down"):
            raise ValueError("KeyboardTeleopController requires a viewer with is_key_down()")

        self.viewer = viewer
        self.arms = {arm.spec.name: arm for arm in arms}
        self.arm_order = [arm.spec.name for arm in arms]
        self.full_q_target = (
            np.asarray(base_full_q, dtype=np.float64).copy()
            if base_full_q is not None
            else arms[0].current_full_q()
        )
        self.states: dict[str, TeleopState] = {}
        for arm in arms:
            current = self.full_q_target[arm.joint_q_indices].copy()
            home_q = np.asarray(arm.spec.home_q, dtype=np.float64) if arm.spec.home_q is not None else current.copy()
            if len(home_q) != len(arm.joint_q_indices):
                raise ValueError(f"Home q for arm {arm.spec.name!r} must have {len(arm.joint_q_indices)} values")
            self.states[arm.spec.name] = TeleopState(
                joint_q=current,
                gripper_opening=float(arm.spec.home_gripper_opening),
                home_q=home_q,
                home_gripper_opening=float(arm.spec.home_gripper_opening),
            )
            self.full_q_target = arm.full_q_with_gripper(
                self.states[arm.spec.name].gripper_opening,
                full_q=arm.full_q_with_arm(current, full_q=self.full_q_target),
            )

        if layouts is None:
            layouts = (self.DEFAULT_LEFT_LAYOUT,) if len(arms) == 1 else (
                self.DEFAULT_LEFT_LAYOUT,
                self.DEFAULT_RIGHT_LAYOUT,
            )
        if len(layouts) < len(arms):
            raise ValueError("KeyboardTeleopController needs at least one layout per arm")
        self.layouts = {arm.spec.name: layouts[index] for index, arm in enumerate(arms)}

        self.step_pos = float(step_pos)
        self.step_rot_deg = float(step_rot_deg)
        self.step_gripper = float(step_gripper)
        self.shift_speed_factor = float(shift_speed_factor)
        self._viewer_conflict_restore: Callable[[], None] | None = None
        if resolve_viewer_conflicts:
            self._viewer_conflict_restore = self._resolve_viewer_conflicts(viewer)
        if print_help:
            self.print_instructions()

    @property
    def current_arm(self) -> NewtonIKArm:
        return self.arms[self.arm_order[0]]

    @property
    def current_state(self) -> TeleopState:
        return self.states[self.arm_order[0]]

    def close(self) -> None:
        if self._viewer_conflict_restore is not None:
            self._viewer_conflict_restore()
            self._viewer_conflict_restore = None

    def update(self) -> np.ndarray:
        """Poll the keyboard and return the current full joint_q target."""

        speed = self.shift_speed_factor if self._key_down("shift") else 1.0
        for arm_name in self.arm_order:
            self._update_arm_pose(arm_name, speed)
            self._update_arm_gripper(arm_name, speed)
        self._write_states_to_full_q()
        return self.full_q_target

    def _key_down(self, key: str) -> bool:
        try:
            return bool(self.viewer.is_key_down(key))
        except Exception:
            return False

    def _axis(self, positive: str, negative: str) -> float:
        return float(self._key_down(positive)) - float(self._key_down(negative))

    def _update_arm_pose(self, arm_name: str, speed: float) -> None:
        arm = self.arms[arm_name]
        state = self.states[arm_name]
        layout = self.layouts[arm_name]

        d_world = np.asarray(
            [
                self._axis(layout.x_pos, layout.x_neg),
                self._axis(layout.y_pos, layout.y_neg),
                self._axis(layout.z_pos, layout.z_neg),
            ],
            dtype=np.float64,
        )
        pitch = self._axis(layout.pitch_pos, layout.pitch_neg)
        yaw = self._axis(layout.yaw_pos, layout.yaw_neg)
        if float(np.linalg.norm(d_world, ord=1)) + abs(pitch) + abs(yaw) < 1.0e-6:
            return

        current_pose = arm.solve_fk(state.joint_q, full_q=self.full_q_target)
        d_world *= self.step_pos * speed
        d_euler_rad = np.radians(np.asarray([0.0, pitch, yaw], dtype=np.float64) * self.step_rot_deg * speed)
        target_quat = quat_mul_wxyz(current_pose[3:7], quat_from_euler_xyz_wxyz(d_euler_rad))
        target_pose = np.concatenate([current_pose[:3] + d_world, target_quat])
        solved = arm.solve_ik(target_pose, seed_joints=state.joint_q, seed_full_q=self.full_q_target)
        if solved is not None:
            limits = arm.joint_limits()
            state.joint_q = np.clip(solved, limits[:, 0], limits[:, 1])

    def _update_arm_gripper(self, arm_name: str, speed: float) -> None:
        arm = self.arms[arm_name]
        state = self.states[arm_name]
        layout = self.layouts[arm_name]
        delta = self._axis(layout.gripper_open, layout.gripper_close) * self.step_gripper * speed
        if abs(delta) < 1.0e-12:
            return
        state.gripper_opening = float(
            np.clip(
                state.gripper_opening + delta,
                arm.spec.min_gripper_opening,
                arm.spec.max_gripper_opening,
            )
        )

    def _write_states_to_full_q(self) -> None:
        for name in self.arm_order:
            arm = self.arms[name]
            state = self.states[name]
            self.full_q_target = arm.full_q_with_arm(state.joint_q, full_q=self.full_q_target)
            self.full_q_target = arm.full_q_with_gripper(state.gripper_opening, full_q=self.full_q_target)

    def print_instructions(self) -> None:
        print("\n" + "=" * 60)
        print("NEWTON KEYBOARD TELEOP")
        print("=" * 60)
        print("Left arm:  W/S X, A/D Y, E/Q Z, R/F pitch, Z/C yaw, X/V close/open")
        print("Right arm: U/J X, H/K Y, I/Y Z, O/L pitch, B/M yaw, N/P close/open")
        print("Gripper keys are hold-to-move; release to stop")
        print("Hold Shift for 2x motion speed")
        print("Viewer conflicts: camera arrows only, G toggles UI, 0 frames camera")
        print(f"arms={self.arm_order}")
        print("=" * 60 + "\n")

    @staticmethod
    def _resolve_viewer_conflicts(viewer) -> Callable[[], None] | None:
        renderer = getattr(viewer, "renderer", None)
        gui = getattr(viewer, "gui", None)
        if renderer is None or gui is None:
            return None

        original_update_camera = getattr(viewer, "_update_camera", None)
        original_callbacks = (
            list(getattr(renderer, "_key_callbacks", [])) if hasattr(renderer, "_key_callbacks") else None
        )

        def update_camera_arrows_only(dt: float) -> None:
            try:
                import pyglet
            except Exception:
                return

            key = pyglet.window.key

            def is_camera_key_down(symbol: int) -> bool:
                return bool(symbol in (key.UP, key.DOWN, key.LEFT, key.RIGHT) and renderer.is_key_down(symbol))

            gui.update_camera_from_keys(dt, is_camera_key_down)

        viewer._update_camera = update_camera_arrows_only

        if original_callbacks is not None:
            original_on_key_press = getattr(viewer, "on_key_press", None)

            def on_key_press_no_teleop_conflicts(symbol: int, modifiers: int) -> None:
                del modifiers
                if getattr(gui, "is_keyboard_capturing", lambda: False)():
                    return
                try:
                    import pyglet
                except Exception:
                    return

                if symbol == pyglet.window.key.SPACE:
                    viewer._paused = not viewer._paused
                elif symbol == pyglet.window.key.PERIOD and getattr(viewer, "_paused", False):
                    viewer._step_requested = True
                elif symbol == pyglet.window.key.G:
                    gui.show_ui = not gui.show_ui
                elif symbol == pyglet.window.key._0:
                    gui.frame_camera_on_model()
                elif symbol == pyglet.window.key.ESCAPE and hasattr(renderer, "close"):
                    renderer.close()

            for index, callback in enumerate(renderer._key_callbacks):
                if callback is original_on_key_press or getattr(callback, "__self__", None) is viewer:
                    renderer._key_callbacks[index] = on_key_press_no_teleop_conflicts
                    break

        def restore() -> None:
            if original_update_camera is not None:
                viewer._update_camera = original_update_camera
            if original_callbacks is not None and hasattr(renderer, "_key_callbacks"):
                renderer._key_callbacks[:] = original_callbacks

        return restore


def make_gamepad_teleop(
    model: newton.Model,
    specs: Sequence[ArmSpec],
    *,
    state: newton.State | None = None,
    full_q_getter: Callable[[], np.ndarray] | None = None,
    base_full_q: Sequence[float] | None = None,
    ik_iterations: int = 24,
    include_rotation_objective: bool = True,
    **controller_kwargs,
) -> GamepadTeleopController:
    """Create ``NewtonIKArm`` objects and wrap them in a gamepad controller."""

    arms = [
        NewtonIKArm(
            model,
            spec,
            state=state,
            full_q_getter=full_q_getter,
            iterations=ik_iterations,
            include_rotation_objective=include_rotation_objective,
        )
        for spec in specs
    ]
    return GamepadTeleopController(arms, base_full_q=base_full_q, **controller_kwargs)


def make_keyboard_teleop(
    model: newton.Model,
    specs: Sequence[ArmSpec],
    viewer,
    *,
    state: newton.State | None = None,
    full_q_getter: Callable[[], np.ndarray] | None = None,
    base_full_q: Sequence[float] | None = None,
    ik_iterations: int = 24,
    include_rotation_objective: bool = True,
    **controller_kwargs,
) -> KeyboardTeleopController:
    """Create ``NewtonIKArm`` objects and wrap them in a keyboard controller."""

    arms = [
        NewtonIKArm(
            model,
            spec,
            state=state,
            full_q_getter=full_q_getter,
            iterations=ik_iterations,
            include_rotation_objective=include_rotation_objective,
        )
        for spec in specs
    ]
    return KeyboardTeleopController(arms, viewer, base_full_q=base_full_q, **controller_kwargs)


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
