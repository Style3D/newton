# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate a kinematic PiPER arm in the WAIC Style3D Pro scene.

The robot is driven with Newton IK/FK and does not use a rigid-body or joint
dynamics solver. Style3D Pro continues to simulate the cloth and tennis ball.

Command:
    python -m newton.examples waic_kinematic_teleop --device cuda:0
"""

import numpy as np
import synreal_sim as sim
import warp as wp

import newton
import newton.examples
from style3d.examples import example_waic_pick_and_place as waic
from style3d.examples.teleop import GamepadTeleopController, NewtonIKArm, TeleopCameraRig, piper_single_arm_spec


class Example(waic.Example):
    """WAIC scene with a gamepad-driven kinematic PiPER arm."""

    def __init__(self, viewer, args):
        super().__init__(viewer, args)

        self.teleop_full_q = self.ik_model.joint_q.numpy().astype(np.float64).copy()
        ee_offset = tuple(float(self.piper_gripper_center_offset[i]) for i in range(3))
        spec = piper_single_arm_spec(
            name="piper",
            ee_body_name="gripper_base_left",
            ee_offset=ee_offset,
            home_q=tuple(float(value) for value in self.teleop_full_q[:6]),
            home_gripper_opening=0.5 * self.gripper_opening,
            min_gripper_opening=0.5 * self.gripper_closed_opening,
            max_gripper_opening=0.5 * self.gripper_opening,
        )
        self.teleop_arm = NewtonIKArm(
            self.ik_model,
            spec,
            full_q_getter=lambda: self.teleop_full_q,
            iterations=max(1, int(args.teleop_ik_iterations)),
            include_rotation_objective=not bool(args.teleop_position_only),
        )
        self.teleop = GamepadTeleopController(
            [self.teleop_arm],
            base_full_q=self.teleop_full_q,
            deadzone=float(args.teleop_deadzone),
            step_joint=float(args.teleop_step_joint),
            step_pos=float(args.teleop_step_pos),
            step_rot_deg=float(args.teleop_step_rot_deg),
            step_gripper=float(args.teleop_step_gripper),
            speed_index=0,
            joystick_index=max(0, int(args.teleop_joystick)),
            print_help=not bool(args.quiet),
        )
        self.teleop_full_q = self.teleop.full_q_target.copy()
        self.kinematic_joint_q = wp.array(
            self.teleop_full_q.astype(np.float32),
            dtype=wp.float32,
            device=self.ik_model.device,
        )
        wrist_body = next(
            i
            for i in range(self.front_piper_body_start, self.front_piper_body_start + self.front_piper_body_count)
            if self.model.body_label[i] == "link6" or self.model.body_label[i].endswith("/link6")
        )
        self.teleop_camera = TeleopCameraRig(
            self.viewer,
            base_body_index=self.front_piper_body_start,
            wrist_body_index=wrist_body,
            tool_offset=ee_offset,
            initial_mode=args.teleop_camera,
            rear_position_offset=(-0.90, 0.0, 1.15),
            rear_target_offset=(0.30, 0.0, 0.25),
            wrist_position_offset=(-0.0735, 0.0078, 0.0384),
            wrist_quat_wxyz=(0.1228, 0.6964, -0.6964, -0.1228),
            wrist_fov=43.23,
            print_help=not bool(args.quiet),
        )
        self._update_teleop_camera()

    @staticmethod
    def create_parser():
        return create_parser()

    def simulate_piper(self):
        self.teleop_full_q = self.teleop.update().astype(np.float64).copy()
        self.kinematic_joint_q.assign(self.teleop_full_q.astype(np.float32))

        piper_body_q_in = self.ik_state.body_q.numpy()
        wp.copy(self.ik_state.joint_q, self.kinematic_joint_q)
        self.ik_state.joint_qd.zero_()
        newton.eval_fk(
            self.ik_model,
            self.kinematic_joint_q,
            self.ik_model.joint_qd,
            self.ik_state,
        )
        piper_body_q_out = self.ik_state.body_q.numpy()

        for body_idx, _shape_idx, shape_label in self.piper_rigid_body_shapes:
            trans_0 = piper_body_q_in[body_idx]
            trans_1 = piper_body_q_out[body_idx]
            begin_trans = sim.Transform(
                sim.Vec3f(trans_0[0], trans_0[1], trans_0[2]),
                sim.Quat(trans_0[3], trans_0[4], trans_0[5], trans_0[6]),
                sim.Vec3f(1.0, 1.0, 1.0),
            )
            end_trans = sim.Transform(
                sim.Vec3f(trans_1[0], trans_1[1], trans_1[2]),
                sim.Quat(trans_1[3], trans_1[4], trans_1[5], trans_1[6]),
                sim.Vec3f(1.0, 1.0, 1.0),
            )
            self.body_entities[shape_label].move(begin_trans, end_trans)

        for body_idx, shape_idx, shape_label in self.piper_mesh_collider_shapes:
            begin_pos = self._piper_shape_world_vertices_np(shape_idx, piper_body_q_in[body_idx])
            end_pos = self._piper_shape_world_vertices_np(shape_idx, piper_body_q_out[body_idx])
            self.mesh_collider_entities[shape_label].move_verts(begin_pos, end_pos)

        self.sync_frontend_piper_display_state()

    def _update_teleop_camera(self):
        self.teleop_camera.update(self.state_0, gamepad_controller=self.teleop)

    def step(self):
        super().step()
        self._update_teleop_camera()

    def render(self):
        self._update_teleop_camera()
        super().render()


def create_parser():
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--teleop-joystick", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--teleop-deadzone", type=float, default=0.1, help="Gamepad stick deadzone")
    parser.add_argument("--teleop-step-joint", type=float, default=0.04, help="Joint-mode step per tick [rad]")
    parser.add_argument("--teleop-step-pos", type=float, default=0.004, help="Pose-mode translation step [m]")
    parser.add_argument("--teleop-step-rot-deg", type=float, default=2.0, help="Pose-mode rotation step [deg]")
    parser.add_argument("--teleop-step-gripper", type=float, default=0.0015, help="Gripper joint step [m]")
    parser.add_argument("--teleop-ik-iterations", type=int, default=24, help="IK iterations per pose-mode tick")
    parser.add_argument(
        "--teleop-camera",
        choices=TeleopCameraRig.MODES,
        default="rear",
        help="Initial viewer camera mode",
    )
    parser.add_argument(
        "--teleop-position-only",
        action="store_true",
        help="Use position-only IK in pose mode",
    )
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
