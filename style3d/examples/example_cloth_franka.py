# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# Coupled robot-cloth simulation: Featherstone for the Franka arm and
# Style3D Pro for the T-shirt. The simulation runs in centimeter scale
# for better numerical behavior; visualization is converted back to
# meters via a separate viz_state and pre-scaled shape arrays.
#
# Command: python -m newton.examples cloth_franka
###########################################################################

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
#from newton.solvers import SolverFeatherstone
from newton.solvers import SolverFeatherstone, SolverVBD
from style3d import style3d_pro

# ----------------------------------------------------------------------
# Robot key-pose sequence
# ----------------------------------------------------------------------

# Orientations reused across the key-pose groups.
_QUAT_LEFT = (0.8536, -0.3536, 0.3536, -0.1464)   # top-left + bottom-left groups
_QUAT_RIGHT = (0.9239, -0.3827, 0.0, 0.0)         # top-right, bottom-right, bottom groups

# Gripper finger activation (later multiplied by 4.0 to drive finger position [cm]).
CLAMP_OPEN = 0.8
CLAMP_CLOSE = 0.1


@dataclasses.dataclass(frozen=True)
class KeyPose:
    """A single waypoint for the end-effector."""

    duration: float                              # transition time [s]
    pos_cm: tuple[float, float, float]           # gripper position [cm]
    quat: tuple[float, float, float, float]      # gripper orientation
    gripper: float                               # CLAMP_OPEN or CLAMP_CLOSE


ROBOT_KEY_POSES: list[KeyPose] = [
    # Descend to working height before approaching the cloth.
    KeyPose(4.0, (31.0, -60.0, 40.0), _QUAT_LEFT, CLAMP_OPEN),
    # Top-left corner: approach, grasp, lift, drag toward center, release.
    KeyPose(2.0, (31.0, -60.0, 20.0), _QUAT_LEFT, CLAMP_OPEN),
    KeyPose(2.0, (31.0, -60.0, 20.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(2.0, (26.0, -60.0, 26.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(2.0, (12.0, -60.0, 31.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(3.0, (-6.0, -60.0, 31.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(1.0, (-6.0, -60.0, 31.0), _QUAT_LEFT, CLAMP_OPEN),
    # Bottom-left corner.
    KeyPose(2.0, (15.0, -33.0, 31.0), _QUAT_LEFT, CLAMP_OPEN),
    KeyPose(3.0, (15.0, -33.0, 21.0), _QUAT_LEFT, CLAMP_OPEN),
    KeyPose(3.0, (15.0, -33.0, 21.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(2.0, (15.0, -33.0, 28.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(3.0, (-2.0, -33.0, 28.0), _QUAT_LEFT, CLAMP_CLOSE),
    KeyPose(1.0, (-2.0, -33.0, 28.0), _QUAT_LEFT, CLAMP_OPEN),
    # Top-right corner.
    KeyPose(2.0, (-28.0, -60.0, 28.0), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(2.0, (-28.0, -60.0, 20.0), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(2.0, (-28.0, -60.0, 20.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(2.0, (-18.0, -60.0, 31.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(3.0, (5.0, -60.0, 31.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(1.0, (5.0, -60.0, 31.0), _QUAT_RIGHT, CLAMP_OPEN),
    # Bottom-right corner.
    KeyPose(3.0, (-18.0, -30.0, 20.5), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(3.0, (-18.0, -30.0, 20.5), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(2.0, (-3.0, -30.0, 31.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(3.0, (-3.0, -30.0, 31.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(2.0, (-3.0, -30.0, 31.0), _QUAT_RIGHT, CLAMP_OPEN),
    # Bottom edge: pick up, fold, release.
    KeyPose(2.0, (0.0, -20.0, 30.0), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(2.0, (0.0, -20.0, 19.5), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(2.0, (0.0, -20.0, 19.5), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(2.0, (0.0, -20.0, 35.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(1.0, (0.0, -30.0, 35.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(1.5, (0.0, -30.0, 35.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(1.5, (0.0, -40.0, 35.0), _QUAT_RIGHT, CLAMP_CLOSE),
    KeyPose(1.5, (0.0, -40.0, 35.0), _QUAT_RIGHT, CLAMP_OPEN),
    KeyPose(2.0, (-28.0, -60.0, 28.0), _QUAT_RIGHT, CLAMP_OPEN),
]


# ----------------------------------------------------------------------
# Warp kernels
# ----------------------------------------------------------------------

@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


@wp.kernel
def compute_ee_tip_velocity(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    ee_id: int,
    ee_offset: wp.vec3,
    body_out: wp.array[float],
):
    # body_qd is COM-referenced (linear velocity at body COM, world frame).
    # Compute EE tip velocity in world frame, consistent with compute_ee_delta
    # which measures the tip position as transform_point(body_q, ee_offset).
    X_wb = body_q[ee_id]
    r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
    qd = body_qd[ee_id]
    omega = wp.spatial_bottom(qd)
    v_com = wp.spatial_top(qd)
    v_tip = v_com + wp.cross(omega, r_world)
    body_out[0] = v_tip[0]
    body_out[1] = v_tip[1]
    body_out[2] = v_tip[2]
    body_out[3] = omega[0]
    body_out[4] = omega[1]
    body_out[5] = omega[2]


# ----------------------------------------------------------------------
# Example
# ----------------------------------------------------------------------

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self._init_params()
        self._build_model()
        self._init_simulation_state()
        self._init_visualization()
        self._init_runtime()

    # ----- initialization -------------------------------------------------

    def _init_params(self):
        # Toggle simulation modules.
        self.add_cloth = True
        self.add_robot = True

        # Substepping and timing.
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # Visualization: simulation in cm, viewer in meters.
        self.viz_scale = 0.01

        # Contact parameters (cm scale).
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2
        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e-2
        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 1e-3
        self.robot_contact_mu = 1.5
        self.self_contact_friction = 0.25

        # Cloth elasticity.
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-6
        self.bending_ke = 5
        self.bending_kd = 1e-2

        # Table geometry (cm).
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)

    def _build_model(self):
        """Build the scene, finalize the model, and apply post-finalize fixups."""
        self.scene = ModelBuilder(gravity=-981.0)

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)
            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        self._add_table()
        self._add_cloth()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        self._hide_table_from_viewer()
        self._apply_contact_materials()
        self._prepare_table_viz_arrays()

    def _add_table(self):
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(self.table_pos_cm, wp.quat_identity()),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

    def _add_cloth(self):
        if not self.add_cloth:
            return
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/shirt"))
        self.scene.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in shirt_mesh.vertices],
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 70.0, 30.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=1.0,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )
        self.scene.color()

    def _hide_table_from_viewer(self):
        # The GL viewer bakes primitive dimensions into the mesh and ignores
        # shape_scale, so we hide the table and re-render it manually at
        # meter scale in render().
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _apply_contact_materials(self):
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        for attr, value in (
            ("shape_material_ke", self.robot_contact_ke),
            ("shape_material_kd", self.robot_contact_kd),
            ("shape_material_mu", self.robot_contact_mu),
        ):
            arr = getattr(self.model, attr)
            np_arr = arr.numpy()
            np_arr[...] = value
            setattr(self.model, attr, wp.array(np_arr, dtype=arr.dtype, device=arr.device))

    def _prepare_table_viz_arrays(self):
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

    def _init_simulation_state(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        # Cloth-body collisions need a custom margin, so we drive collision
        # detection explicitly rather than letting the cloth solver do it.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self._init_jacobian_controller()

        self.use_style3d_pro = False
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            if self.use_style3d_pro:
                self.cloth_solver = style3d_pro.SolverStyle3DPro(self.model)
            else:
                self.cloth_solver = SolverVBD(
                    self.model,
                    iterations=self.iterations,
                    integrate_with_external_rigid_solver=True,
                    particle_self_contact_radius=self.particle_self_contact_radius,
                    particle_self_contact_margin=self.particle_self_contact_margin,
                    particle_topological_contact_filter_threshold=1,
                    particle_rest_shape_contact_exclusion_radius=0.5,
                    particle_enable_self_contact=True,
                    particle_vertex_contact_buffer_size=16,
                    particle_edge_contact_buffer_size=20,
                    particle_collision_detection_interval=-1,
                )

    def _init_jacobian_controller(self):
        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i):
            return wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)

        self.Jacobian_one_hots = [onehot(i) for i in range(out_dim)]
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()
        self.endeffector_offset_pos = wp.vec3(*self.endeffector_offset.p)

    def _init_visualization(self):
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Holds meter-scale copies of particle/body data for render().
        self.viz_state = self.model.state()

        # Pre-compute meter-scale shape arrays. Two viewer paths need handling:
        #   1) GL viewer's CUDA path reads model.shape_transform / shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) Base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)
                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

    def _init_runtime(self):
        # Gravity arrays swapped during the substep loop: zero gravity while
        # the robot solver steps so it sees rigid-only dynamics.
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3)

        # Initial FK so state_0 reflects the URDF rest configuration.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # CUDA graph capture for the substep loop.
        self.graph = None
        if self.add_cloth:
            self._capture()

    def _capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ----- articulation ---------------------------------------------------

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Flatten the dataclass list into the numpy arrays the rest of the
        # control code expects: targets[i] = (px, py, pz, qx, qy, qz, qw, gripper).
        self.robot_key_poses = ROBOT_KEY_POSES
        self.transition_duration = np.array([p.duration for p in ROBOT_KEY_POSES], dtype=np.float32)
        self.robot_key_poses_time = np.cumsum(self.transition_duration)
        self.targets = np.array(
            [(*p.pos_cm, *p.quat, p.gripper) for p in ROBOT_KEY_POSES],
            dtype=np.float32,
        )
        self.target = self.targets[0]

        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform((0.0, 0.0, 22.0), wp.quat_identity())

    # ----- control --------------------------------------------------------

    def compute_body_jacobian(self, model: Model, joint_q: wp.array, joint_qd: wp.array):
        """Compute the Jacobian of the end-effector velocity w.r.t. joint_q."""
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        out_dim = 6
        in_dim = model.joint_dof_count

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                compute_ee_tip_velocity,
                dim=1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                    self.endeffector_id,
                    self.endeffector_offset_pos,
                ],
                outputs=[self.body_out],
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(self, state_in: State):
        # After the key-pose sequence ends, hold position with zero velocity.
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(self.model, state_in.joint_q, state_in.joint_qd)

        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        # Null-space projector keeps the arm near its initial pose while
        # following the EE target.
        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Gripper finger control (finger positions in cm).
        delta_q[-2] = self.target[-1] * 4.0 - q[-2]
        delta_q[-1] = self.target[-1] * 4.0 - q[-1]

        self.target_joint_qd.assign(delta_q)

    # ----- stepping -------------------------------------------------------

    @contextlib.contextmanager
    def _robot_only_physics(self):
        """Disable particles and gravity while the robot solver steps.

        Note: shape_contact_pair_count is set to 0 and not restored, matching
        the original code — it is recomputed by the collision pipeline in the
        same substep.
        """
        saved_particle_count = self.model.particle_count
        self.model.particle_count = 0
        self.model.shape_contact_pair_count = 0
        self.model.gravity.assign(self.gravity_zero)
        try:
            yield
        finally:
            self.model.particle_count = saved_particle_count
            self.model.gravity.assign(self.gravity_earth)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                with self._robot_only_physics():
                    self.state_0.joint_qd.assign(self.target_joint_qd)
                    self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                    self.state_0.particle_f.zero_()

            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    # ----- rendering ------------------------------------------------------

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters.
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap to meter-scale shape data, render, restore.
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    # ----- tests ----------------------------------------------------------

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
