# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the PiPER arm in the fixed-mouth canvas bag VBD scene.

This example deliberately reuses ``cloth_vbd_piper_bag`` for scene construction
and keeps the teleoperation glue separate from the scripted pick planner.

Command:
    python -m newton.examples cloth_vbd_piper_bag_teleop --device cuda:0
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cloth import example_cloth_vbd_piper_bag as piper_bag
from newton.examples.teleop import (
    GamepadTeleopController,
    KinematicArticulationMirror,
    NewtonIKArm,
    piper_single_arm_spec,
)


class Example(piper_bag.Example):
    def __init__(self, viewer, args):
        if bool(args.no_robot):
            raise ValueError("cloth_vbd_piper_bag_teleop requires the PiPER robot; do not pass --no-robot.")

        self.teleop_full_q = None
        self.teleop = None
        self.teleop_mirror = None
        self.prev_robot_tool_pos = None
        self.robot_tool_velocity = np.zeros(3, dtype=np.float64)

        args.enable_robot_planner = False
        args.no_cuda_graph = True
        super().__init__(viewer, args)
        self._setup_teleop()

    @staticmethod
    def create_parser():
        return create_parser()

    def _setup_teleop(self) -> None:
        assert self.piper_info is not None

        ik_builder = newton.ModelBuilder()
        ik_info = piper_bag.add_piper(ik_builder, self.args, parse_visuals=False, parse_meshes=False, announce=False)
        if ik_info is None:
            raise ValueError("Unable to build PiPER IK model for teleop.")

        self.ik_model = ik_builder.finalize()
        self.teleop_full_q = self.ik_model.joint_q.numpy().astype(np.float64).copy()
        self.robot_ee_body_main = piper_bag.find_label_index(
            self.model.body_label,
            self.args.robot_ee_body,
            self.piper_info.body_start,
            self.piper_info.body_end,
        )
        self.robot_ee_offset = np.asarray(self.args.robot_ee_offset, dtype=np.float64)

        spec = piper_single_arm_spec(
            name="piper",
            ee_body_name=self.args.robot_ee_body,
            finger_joint_names=(self.args.robot_finger0_joint, self.args.robot_finger1_joint),
            ee_offset=tuple(float(v) for v in self.robot_ee_offset),
            home_q=tuple(float(v) for v in tuple(self.args.robot_home_q or piper_bag.DEFAULT_ROBOT_HOME_Q)[:6]),
            home_gripper_opening=float(self.args.gripper_opening),
            min_gripper_opening=float(self.args.gripper_closed_opening),
            max_gripper_opening=float(self.args.gripper_opening),
        )
        arm = NewtonIKArm(
            self.ik_model,
            spec,
            full_q_getter=lambda: self.teleop_full_q,
            iterations=max(1, int(self.args.robot_ik_iterations)),
            include_rotation_objective=not bool(self.args.teleop_position_only),
            joint_limit_weight=float(self.args.robot_ik_joint_limit_weight),
            lambda_initial=float(self.args.robot_ik_lambda),
        )
        self.teleop = GamepadTeleopController(
            [arm],
            base_full_q=self.teleop_full_q,
            step_joint=float(self.args.teleop_step_joint),
            step_pos=float(self.args.teleop_step_pos),
            step_rot_deg=float(self.args.teleop_step_rot_deg),
            step_gripper=float(self.args.teleop_step_gripper),
            joystick_index=max(0, int(self.args.teleop_joystick)),
            print_help=not bool(self.args.quiet),
        )
        self.teleop_mirror = KinematicArticulationMirror(
            self.ik_model,
            source_body_start=int(ik_info.body_start),
            target_body_start=int(self.piper_info.body_start),
            body_count=int(ik_info.body_end - ik_info.body_start),
        )
        self._update_teleop_robot_pose(self.state_0)
        self.prev_robot_tool_pos = self.robot_tool_pos.copy()
        print(
            "[PiperBagTeleop] "
            f"ee_body={self.args.robot_ee_body}, ee_offset={self.robot_ee_offset.astype(float).tolist()}, "
            f"attach_max_gripper={self._attach_gripper_threshold():g}, "
            f"release_gripper={self._release_gripper_threshold():g}",
            flush=True,
        )

    def _attach_gripper_threshold(self) -> float:
        if self.args.teleop_attach_max_gripper_opening is not None:
            return float(self.args.teleop_attach_max_gripper_opening)
        return float(self.args.attach_max_gripper_opening)

    def _release_gripper_threshold(self) -> float:
        if self.args.teleop_release_gripper_opening is not None:
            return float(self.args.teleop_release_gripper_opening)
        return max(1.25 * self._attach_gripper_threshold(), 0.7 * float(self.args.gripper_opening))

    def _update_teleop_robot_pose(self, state: newton.State) -> None:
        assert self.teleop_mirror is not None
        assert self.teleop_full_q is not None

        previous_tool_pos = self.robot_tool_pos.copy() if hasattr(self, "robot_tool_pos") else None
        self.teleop_mirror.apply(self.teleop_full_q, state)
        body_q = state.body_q.numpy()
        self.robot_tool_pos = piper_bag.transform_point_np(body_q[self.robot_ee_body_main], self.robot_ee_offset)
        if previous_tool_pos is None:
            self.robot_tool_velocity = np.zeros(3, dtype=np.float64)
        else:
            self.robot_tool_velocity = (self.robot_tool_pos - previous_tool_pos) / max(self.sim_dt, 1.0e-8)
        self.robot_gripper_opening = float(self.teleop.current_state.gripper_opening)

    def _should_attach_ball_to_gripper(self, state: newton.State, time: float) -> tuple[bool, float]:
        del time
        if self.robot_gripper_opening > self._attach_gripper_threshold():
            return False, float("inf")

        tool_ball_distance = self._tool_ball_distance(state)
        if tool_ball_distance > self._attach_tool_ball_distance_limit():
            return False, tool_ball_distance
        return True, tool_ball_distance

    def _should_release_ball_from_gripper(self) -> bool:
        return self.robot_gripper_opening >= self._release_gripper_threshold()

    def _attached_ball_pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        del time
        return self.robot_tool_pos + self.ball_attach_offset, self.robot_tool_velocity.copy()

    def simulate(self):
        for substep in range(self.sim_substeps):
            time = self.sim_time + substep * self.sim_dt
            self._update_teleop_robot_pose(self.state_0)
            if self.ball_attached and self._should_release_ball_from_gripper():
                self._release_attached_ball(self.state_0, time)
            elif self.ball_attached:
                self._drive_attached_ball(self.state_0, time)

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)

            if not self.ball_attached:
                should_attach, tool_ball_distance = self._should_attach_ball_to_gripper(self.state_0, time)
                if should_attach:
                    self._attach_ball_to_gripper(self.state_0, time, tool_ball_distance)
                    self._drive_attached_ball(self.state_0, time)
                    self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.ball_attached:
                self._drive_attached_ball(self.state_1, time + self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        assert self.teleop is not None
        self.teleop_full_q = self.teleop.update().astype(np.float64).copy()
        super().step()
        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
            print(
                "[PiperBagTeleop] "
                f"tool={self.robot_tool_pos.astype(float).round(4).tolist()} "
                f"gripper={self.robot_gripper_opening:.4f} "
                f"tool_ball_distance={self._tool_ball_distance(self.state_0):.4f}",
                flush=True,
            )


def create_parser() -> argparse.ArgumentParser:
    parser = piper_bag.create_parser()
    parser.description = __doc__
    parser.set_defaults(enable_robot_planner=False, no_cuda_graph=True)
    parser.add_argument("--teleop-joystick", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--teleop-step-joint", type=float, default=0.04, help="Joint mode step per control tick [rad]")
    parser.add_argument("--teleop-step-pos", type=float, default=0.004, help="Pose mode translation step [m]")
    parser.add_argument("--teleop-step-rot-deg", type=float, default=2.0, help="Pose mode rotation step [deg]")
    parser.add_argument("--teleop-step-gripper", type=float, default=0.0015, help="Gripper opening step [m]")
    parser.add_argument(
        "--teleop-position-only",
        action="store_true",
        help="Use a position-only IK objective for easier coarse teleop near singular configurations",
    )
    parser.add_argument(
        "--teleop-attach-max-gripper-opening",
        type=float,
        default=None,
        help="Attach the ball only when gripper opening is at or below this value [m]",
    )
    parser.add_argument(
        "--teleop-release-gripper-opening",
        type=float,
        default=None,
        help="Release the attached ball when gripper opening reaches this value [m]",
    )
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
