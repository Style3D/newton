# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

sys.path.append("D:/Desktop/synreal-sim/build_vs/lib/RelWithDebInfo")

import numpy as np
import synreal_sim as sim
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.usd
from newton import Mesh
from newton.solvers import style3d


def sim_log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
    if level == sim.LogLevel.INFO:
        print("[info]: ", message)
    elif level == sim.LogLevel.ERROR:
        print("[error]: ", message)
    elif level == sim.LogLevel.WARNING:
        print("[warning]: ", message)
    elif level == sim.LogLevel.DEBUG:
        print("[debug]: ", message)


def add_piper(builder: newton.ModelBuilder, piper_mjcf_path: Path):
    piper_body_start = builder.body_count
    piper_joint_q_start = builder.joint_coord_count
    piper_joint_qd_start = builder.joint_dof_count
    piper_xform = wp.transform(p=wp.vec3(-0.3, 0.555, 0), q=wp.quat_identity())
    builder.add_mjcf(
        str(piper_mjcf_path),
        xform=piper_xform,
        floating=False,
        parse_visuals=True,
        parse_meshes=True,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
    )
    return (
        piper_body_start,
        builder.body_count - piper_body_start,
        piper_joint_q_start,
        builder.joint_coord_count - piper_joint_q_start,
        piper_joint_qd_start,
        builder.joint_dof_count - piper_joint_qd_start,
    )


PIPER_GRIPPER_BODY_SUFFIXES = ("/link7", "/link8")
PIPER_IK_EE_BODY_SUFFIX = "/gripper_base_left"
PIPER_FINGER_JOINT_SUFFIXES = ("/link7/joint7", "/link8/joint8")
GRIPPER_AS_RIGIDBODY = False


@wp.kernel
def _set_finger_opening(
    joint_q: wp.array[float],
    finger0_q: int,
    finger1_q: int,
    finger_opening: float,
):
    half_opening = 0.5 * finger_opening
    if finger0_q >= 0:
        joint_q[finger0_q] = half_opening
    if finger1_q >= 0:
        joint_q[finger1_q] = -half_opening


def smoothstep(alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def vec3_to_np(value) -> np.ndarray:
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer
        self.viewer._paused = True
        self._last_paused = self.viewer._paused

        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        ASSET_ROOT = Path(__file__).resolve().parent / "assets"

        from pxr import Usd

        # Bag
        garment_name = "Style3D_Bag5"
        usd_stage = Usd.Stage.Open(str(ASSET_ROOT / (garment_name + ".usd")))
        usd_prim_garment = usd_stage.GetPrimAtPath(str("/Root/" + garment_name + "/Root_Garment"))
        garment_mesh, garment_mesh_uv_indices = newton.usd.get_mesh(
            usd_prim_garment,
            load_uvs=True,
            preserve_facevarying_uvs=True,
            return_uv_indices=True,
        )
        garment_mesh_uv = garment_mesh.uvs * 1.0e-3
        style3d.add_cloth_mesh(
            builder,
            label="bag",
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            panel_verts=garment_mesh_uv.tolist(),
            panel_indices=garment_mesh_uv_indices.tolist(),
            vertices=garment_mesh.vertices.tolist(),
            indices=garment_mesh.indices.tolist(),
            density=0.3,
            scale=1.0,
            particle_radius=5.0e-3,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
            edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
        )

        # tennis
        tennis_name = "wangqiu"
        usd_stage = Usd.Stage.Open(str(ASSET_ROOT / (tennis_name + ".usd")))
        usd_prim_tennis = usd_stage.GetPrimAtPath(str("/root/Trim_1/" + tennis_name))
        tennis_mesh = newton.usd.get_mesh(usd_prim_tennis)
        tennis_body = builder.add_body()
        builder.add_shape_mesh(
            label=tennis_name,
            body=tennis_body,
            xform=wp.transform(
                p=wp.vec3(0, 0.555 + 0.4, 0),
                q=wp.quat_identity(),
            ),
            mesh=Mesh(tennis_mesh.vertices, tennis_mesh.indices),
        )


        # Desk
        desk_name = "desk"
        usd_stage = Usd.Stage.Open(str(ASSET_ROOT / (desk_name + ".usd")))
        usd_prim_desk = usd_stage.GetPrimAtPath(str("/Root/" + desk_name + "/Root_SkinnedMesh_Avatar_0_Sub_0"))
        desk_mesh = newton.usd.get_mesh(usd_prim_desk)
        builder.add_shape_mesh(
            label=desk_name,
            body=builder.add_body(),
            xform=wp.transform(
                p=wp.vec3(0, 0, 0),
                q=wp.quat_identity(),
            ),
            mesh=Mesh(desk_mesh.vertices, desk_mesh.indices),
        )


        # Rack
        rack_name = "rack"
        usd_stage = Usd.Stage.Open(str(ASSET_ROOT / (rack_name + ".usd")))
        usd_prim_rack = usd_stage.GetPrimAtPath(str("/Root/" + rack_name + "/Root_SkinnedMesh_Avatar_0_Sub_0"))
        rack_mesh = newton.usd.get_mesh(usd_prim_rack)
        rack_body = builder.add_body()
        builder.add_shape_mesh(
            label=rack_name,
            body=rack_body,
            xform=wp.transform(
                p=wp.vec3(0, 0, 0),
                q=wp.quat_identity(),
            ),
            mesh=Mesh(rack_mesh.vertices, rack_mesh.indices),
        )

        piper_mjcf_path = ASSET_ROOT / "piper" / "piper_with_texture.xml"
        (
            self.front_piper_body_start,
            self.front_piper_body_count,
            self.front_piper_joint_q_start,
            self.front_piper_joint_q_count,
            self.front_piper_joint_qd_start,
            self.front_piper_joint_qd_count,
        ) = add_piper(builder, piper_mjcf_path)

        # add a table
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.shape_scale_np = self.model.shape_scale.numpy()
        self.shape_transform_np = self.model.shape_transform.numpy()
        self.tennis_shape_idx = next(
            shape_idx
            for shape_idx in self.model.body_shapes[tennis_body]
            if self.model.shape_label[shape_idx] == tennis_name
        )
        tennis_vertices_np = np.asarray(tennis_mesh.vertices, dtype=np.float32).reshape(-1, 3)
        tennis_local_center_np = 0.5 * (tennis_vertices_np.min(axis=0) + tennis_vertices_np.max(axis=0))
        self.tennis_local_center = wp.vec3(*map(float, tennis_local_center_np))
        body_trans_np = self.model.body_q.numpy()
        tennis_body_xform = wp.transform(
            p=wp.vec3(body_trans_np[tennis_body][0], body_trans_np[tennis_body][1], body_trans_np[tennis_body][2]),
            q=wp.quat(
                body_trans_np[tennis_body][3],
                body_trans_np[tennis_body][4],
                body_trans_np[tennis_body][5],
                body_trans_np[tennis_body][6],
            ),
        )
        tennis_shape_trans = self.shape_transform_np[self.tennis_shape_idx]
        tennis_shape_xform = wp.transform(
            p=wp.vec3(tennis_shape_trans[0], tennis_shape_trans[1], tennis_shape_trans[2]),
            q=wp.quat(tennis_shape_trans[3], tennis_shape_trans[4], tennis_shape_trans[5], tennis_shape_trans[6]),
        )
        self.tennis_shape_xform = tennis_shape_xform
        self._hide_frontend_piper_collision_shapes()
        self.sim_rigid_body_indices = (tennis_body, rack_body)
        self.dynamic_sim_rigid_body_shapes = []

        ik_builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        add_piper(ik_builder, piper_mjcf_path)
        self.ik_model = ik_builder.finalize()
        self.ik_state = self.ik_model.state()
        newton.eval_fk(
            self.ik_model,
            self.ik_model.joint_q,
            self.ik_model.joint_qd,
            self.ik_state,
        )

        self.ik_piper_ee_body = next(
            i for i, label in enumerate(self.ik_model.body_label) if label.endswith(PIPER_IK_EE_BODY_SUFFIX)
        )
        self.piper_gripper_body_indices_backend = tuple(
            i for i, label in enumerate(self.ik_model.body_label) if label.endswith(PIPER_GRIPPER_BODY_SUFFIXES)
        )
        self.piper_finger_q_indices = tuple(
            int(self.ik_model.joint_q_start.numpy()[i])
            for i, label in enumerate(self.ik_model.joint_label)
            if label.endswith(PIPER_FINGER_JOINT_SUFFIXES)
        )
        self.ik_joint_q = self.ik_model.joint_q.reshape((1, self.ik_model.joint_coord_count))
        ik_body_q_np = self.ik_state.body_q.numpy()
        ee_xform = wp.transform(*ik_body_q_np[self.ik_piper_ee_body])
        initial_gripper_center = self._piper_gripper_center_np(ik_body_q_np)
        self.piper_gripper_center_offset = wp.transform_point(
            wp.transform_inverse(ee_xform),
            wp.vec3(*map(float, initial_gripper_center)),
        )
        self.piper_initial_tool_pos = initial_gripper_center
        initial_tennis_shape_xform = wp.transform_multiply(tennis_body_xform, tennis_shape_xform)
        self.current_tennis_center = vec3_to_np(
            wp.transform_point(initial_tennis_shape_xform, self.tennis_local_center)
        )
        self.ik_target_pos = wp.array(
            [wp.vec3(*map(float, self.piper_initial_tool_pos))],
            dtype=wp.vec3,
        )
        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=self.ik_piper_ee_body,
            link_offset=self.piper_gripper_center_offset,
            target_positions=self.ik_target_pos,
        )
        self.ik_joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_joint_limits_obj],
            lambda_initial=0.08,
            jacobian_mode=ik.IKJacobianType.MIXED,
        )
        self.ik_iters = 18
        self.piper_shape_scale_np = self.ik_model.shape_scale.numpy()
        self.piper_shape_transform_np = self.ik_model.shape_transform.numpy()
        self.piper_shape_body_vertices_np = {}
        self.gripper_opening = 0.07
        self.gripper_closed_opening = 0.05
        self.pick_start_time = 0.4
        self.approach_end_time = 1.4
        self.descend_end_time = 2.0
        self.grasp_end_time = 2.35
        self.lift_end_time = 3.3
        self.release_time = 4.4
        self.retract_end_time = 5.2
        self.home_end_time = 6.0
        self.piper_ik_keyframe_offsets = {
            "above_tennis": np.array([0.0, 0.20, 0.0], dtype=np.float64),
            "grasp_tennis": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "lift_tennis": np.array([0.0, 0.48, 0.0], dtype=np.float64),
            "transfer": np.array([-0.05, 0.40, -0.40], dtype=np.float64),
            "release": np.array([-0.05, 0.40, -0.40], dtype=np.float64),
            "retract": np.array([-0.05, 0.35, -0.35], dtype=np.float64),
        }
        self.tennis_pick_center = self.current_tennis_center.copy()

        # Set log callback
        sim.set_log_callback(sim_log_callback)
        sim.login("simsdk003", "xSXiaCMd", True, None)

        # Create world
        self.world = sim.World()
        world_attrib = sim.WorldAttrib()
        world_attrib.time_step = 0.01
        world_attrib.enable_gpu = True
        world_attrib.ground_height = 0.555
        world_attrib.iterations = 50
        world_attrib.nonlinear_iterations = 1
        world_attrib.enable_rigid_self_collision = False

        if self.model.up_axis == newton.Axis.Z:
            world_attrib.gravity = sim.Vec3f(0, 0, -9.8)
            world_attrib.ground_direction = sim.Vec3f(0, 0, 1)
        self.world.set_attrib(world_attrib)

        # Create Cloth
        verts_np = self.model.particle_q.numpy()
        faces_np = self.model.tri_indices.numpy()
        self.is_fixed = [False] * len(verts_np)
        self.fixed_indices = list(range(len(verts_np)))

        cloth_attrib = sim.ClothAttrib()
        cloth_attrib.thickness = 2e-3
        cloth_attrib.bend_stiff = sim.Vec3f(5e-7, 5e-7, 5e-7)
        self.cloth = sim.Cloth(faces_np, verts_np, [], False)
        self.cloth.set_pin(self.is_fixed, self.fixed_indices)
        self.cloth.set_attrib(cloth_attrib)
        self.cloth.attach(self.world)

        # Create rigid bodies
        self.body_entities = {}
        self.mesh_collider_entities = {}
        body_trans_np = self.model.body_q.numpy()
        self.shape_flags = self.model.shape_flags.numpy()
        self.shape_body_vertices_np = {}

        # # Register body entity
        collide_shapes_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for i in self.sim_rigid_body_indices:
            shape_indices = self.model.body_shapes[i]
            for shape_idx in shape_indices:
                if isinstance(self.model.shape_source[shape_idx], newton.Mesh):
                    if self.shape_flags[shape_idx] & collide_shapes_flag == 0:
                        continue

                    trans = body_trans_np[i]
                    if i == tennis_body:
                        shape_vertices_np = np.array(self.model.shape_source[shape_idx].vertices, dtype=np.float32)
                        shape_vertices_np *= self.shape_scale_np[shape_idx]
                        shape_trans = self.shape_transform_np[shape_idx]
                        body_xform = wp.transform(
                            p=wp.vec3(trans[0], trans[1], trans[2]),
                            q=wp.quat(trans[3], trans[4], trans[5], trans[6]),
                        )
                        shape_xform = wp.transform(
                            p=wp.vec3(shape_trans[0], shape_trans[1], shape_trans[2]),
                            q=wp.quat(shape_trans[3], shape_trans[4], shape_trans[5], shape_trans[6]),
                        )
                        rigid_xform = wp.transform_multiply(body_xform, shape_xform)
                    else:
                        shape_vertices_np = self._shape_body_vertices_np(shape_idx)
                        rigid_xform = trans

                    translation = sim.Vec3f(rigid_xform[0], rigid_xform[1], rigid_xform[2])
                    rotation = sim.Quat(rigid_xform[3], rigid_xform[4], rigid_xform[5], rigid_xform[6])
                    scaling = sim.Vec3f(1.0, 1.0, 1.0)
                    transform = sim.Transform(translation, rotation, scaling)
                    static_mesh = sim.Mesh(self.model.shape_source[shape_idx].indices.flatten(), shape_vertices_np)
                    rigid_body = sim.RigidBody(static_mesh, transform)
                    rigid_body.attach(self.world)
                    rigid_body.set_pin(i != tennis_body)
                    attrib = sim.RigidBodyAttrib()
                    attrib.mass = 10.0 if i != tennis_body else 0.058
                    rigid_body.set_attrib(attrib)
                    self.body_entities[self.model.shape_label[shape_idx]] = rigid_body
                    if i == tennis_body:
                        self.dynamic_sim_rigid_body_shapes.append((i, shape_idx, self.model.shape_label[shape_idx]))

        self.piper_rigid_body_shapes = []
        self.piper_mesh_collider_shapes = []
        piper_body_q_np = self.ik_state.body_q.numpy()
        piper_shape_flags_np = self.ik_model.shape_flags.numpy()
        for body_idx in self.piper_gripper_body_indices_backend:
            for shape_idx in self.ik_model.body_shapes[body_idx]:
                if piper_shape_flags_np[shape_idx] & collide_shapes_flag == 0:
                    continue
                if not isinstance(self.ik_model.shape_source[shape_idx], newton.Mesh):
                    continue

                shape_label = self.ik_model.shape_label[shape_idx]
                shape_tris_np = np.asarray(
                    self.ik_model.shape_source[shape_idx].indices,
                    dtype=np.int32,
                ).reshape(-1, 3)
                if GRIPPER_AS_RIGIDBODY:
                    trans = piper_body_q_np[body_idx]
                    translation = sim.Vec3f(trans[0], trans[1], trans[2])
                    rotation = sim.Quat(trans[3], trans[4], trans[5], trans[6])
                    transform = sim.Transform(translation, rotation, sim.Vec3f(1.0, 1.0, 1.0))
                    static_mesh = sim.Mesh(
                        self.ik_model.shape_source[shape_idx].indices.flatten(),
                        self._piper_shape_body_vertices_np(shape_idx),
                    )
                    rigid_body = sim.RigidBody(static_mesh, transform)
                    rigid_body.attach(self.world)
                    rigid_body.set_pin(True)
                    self.body_entities[shape_label] = rigid_body
                    self.piper_rigid_body_shapes.append((body_idx, shape_idx, shape_label))
                else:
                    shape_vertices_np = self._piper_shape_world_vertices_np(shape_idx, piper_body_q_np[body_idx])
                    mesh_collider = sim.MeshCollider(shape_tris_np, shape_vertices_np)
                    collider_attrib = sim.ColliderAttrib()
                    collider_attrib.collision_gap = 5e-3
                    collider_attrib.static_friction = 1.0
                    collider_attrib.dynamic_friction = 1.0
                    mesh_collider.set_attrib(collider_attrib)
                    mesh_collider.attach(self.world)
                    self.mesh_collider_entities[shape_label] = mesh_collider
                    self.piper_mesh_collider_shapes.append((body_idx, shape_idx, shape_label))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -1.7, 1.4), 0.0, -270.0)

    def _hide_frontend_piper_collision_shapes(self):
        shape_flags = self.model.shape_flags.numpy()
        collide_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible_flag = int(newton.ShapeFlags.VISIBLE)
        for body_idx in range(
            self.front_piper_body_start,
            self.front_piper_body_start + self.front_piper_body_count,
        ):
            for shape_idx in self.model.body_shapes[body_idx]:
                if shape_flags[shape_idx] & collide_flag:
                    shape_flags[shape_idx] &= ~visible_flag
        self.model.shape_flags.assign(shape_flags)

    def sync_frontend_piper_display_state(self):
        wp.copy(
            self.state_0.body_q,
            self.ik_state.body_q,
            dest_offset=self.front_piper_body_start,
            count=self.front_piper_body_count,
        )
        wp.copy(
            self.state_0.body_qd,
            self.ik_state.body_qd,
            dest_offset=self.front_piper_body_start,
            count=self.front_piper_body_count,
        )
        wp.copy(
            self.state_1.body_q,
            self.ik_state.body_q,
            dest_offset=self.front_piper_body_start,
            count=self.front_piper_body_count,
        )
        wp.copy(
            self.state_1.body_qd,
            self.ik_state.body_qd,
            dest_offset=self.front_piper_body_start,
            count=self.front_piper_body_count,
        )
        wp.copy(
            self.state_0.joint_q,
            self.ik_state.joint_q,
            dest_offset=self.front_piper_joint_q_start,
            count=self.front_piper_joint_q_count,
        )
        wp.copy(
            self.state_0.joint_qd,
            self.ik_state.joint_qd,
            dest_offset=self.front_piper_joint_qd_start,
            count=self.front_piper_joint_qd_count,
        )
        wp.copy(
            self.state_1.joint_q,
            self.ik_state.joint_q,
            dest_offset=self.front_piper_joint_q_start,
            count=self.front_piper_joint_q_count,
        )
        wp.copy(
            self.state_1.joint_qd,
            self.ik_state.joint_qd,
            dest_offset=self.front_piper_joint_qd_start,
            count=self.front_piper_joint_qd_count,
        )

    def _tennis_center_from_body_xform(self, body_xform):
        shape_world_xform = wp.transform_multiply(body_xform, self.tennis_shape_xform)
        return vec3_to_np(wp.transform_point(shape_world_xform, self.tennis_local_center))

    def _transform_points_np(self, points: np.ndarray, xform_np) -> np.ndarray:
        trans = np.asarray(xform_np[:3], dtype=np.float32)
        rot = np.asarray(wp.quat_to_matrix(wp.quat(*xform_np[3:7])), dtype=np.float32).reshape(3, 3)
        return np.ascontiguousarray(points @ rot.T + trans, dtype=np.float32)

    def _shape_body_vertices_np(self, shape_idx: int) -> np.ndarray:
        cached = self.shape_body_vertices_np.get(shape_idx)
        if cached is not None:
            return cached

        vertices = np.asarray(self.model.shape_source[shape_idx].vertices, dtype=np.float32).reshape(-1, 3)
        shape_vertices = self._transform_points_np(vertices, self.shape_transform_np[shape_idx])
        shape_vertices *= np.asarray(self.shape_scale_np[shape_idx], dtype=np.float32)
        shape_vertices = np.ascontiguousarray(shape_vertices, dtype=np.float32)
        self.shape_body_vertices_np[shape_idx] = shape_vertices
        return shape_vertices

    def _piper_shape_body_vertices_np(self, shape_idx: int) -> np.ndarray:
        cached = self.piper_shape_body_vertices_np.get(shape_idx)
        if cached is not None:
            return cached

        vertices = np.asarray(self.ik_model.shape_source[shape_idx].vertices, dtype=np.float32).reshape(-1, 3)
        shape_vertices = self._transform_points_np(vertices, self.piper_shape_transform_np[shape_idx])
        shape_vertices *= np.asarray(self.piper_shape_scale_np[shape_idx], dtype=np.float32)
        shape_vertices = np.ascontiguousarray(shape_vertices, dtype=np.float32)
        self.piper_shape_body_vertices_np[shape_idx] = shape_vertices
        return shape_vertices

    def _piper_shape_world_vertices_np(self, shape_idx: int, body_xform_np) -> np.ndarray:
        return self._transform_points_np(self._piper_shape_body_vertices_np(shape_idx), body_xform_np)

    def _piper_gripper_center_np(self, piper_body_q_np: np.ndarray) -> np.ndarray:
        shape_scale_np = self.ik_model.shape_scale.numpy()
        shape_transform_np = self.ik_model.shape_transform.numpy()
        shape_flags_np = self.ik_model.shape_flags.numpy()
        collide_shapes_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)

        centers = []
        for body_idx in self.piper_gripper_body_indices_backend:
            body_points = []
            for shape_idx in self.ik_model.body_shapes[body_idx]:
                if shape_flags_np[shape_idx] & collide_shapes_flag == 0:
                    continue
                if not isinstance(self.ik_model.shape_source[shape_idx], newton.Mesh):
                    continue

                vertices = np.asarray(
                    self.ik_model.shape_source[shape_idx].vertices,
                    dtype=np.float32,
                ).reshape(-1, 3)
                shape_points = self._transform_points_np(vertices, shape_transform_np[shape_idx])
                shape_points *= np.asarray(shape_scale_np[shape_idx], dtype=np.float32)
                body_points.append(self._transform_points_np(shape_points, piper_body_q_np[body_idx]))

            if body_points:
                points = np.concatenate(body_points, axis=0)
                centers.append(0.5 * (points.min(axis=0) + points.max(axis=0)))
            else:
                centers.append(np.asarray(piper_body_q_np[body_idx][:3], dtype=np.float32))

        if not centers:
            return vec3_to_np(wp.transform_get_translation(wp.transform(*piper_body_q_np[self.ik_piper_ee_body])))

        return np.asarray(np.mean(centers, axis=0), dtype=np.float64)

    def _segment(self, time: float, t0: float, t1: float, p0: np.ndarray, p1: np.ndarray):
        if t1 <= t0:
            return p1.copy()
        alpha = smoothstep((time - t0) / (t1 - t0))
        return (1.0 - alpha) * p0 + alpha * p1

    def _piper_keyframes(self):
        tennis = self.tennis_pick_center
        offsets = self.piper_ik_keyframe_offsets
        return {
            "initial": self.piper_initial_tool_pos.copy(),
            "above_tennis": tennis + offsets["above_tennis"],
            "grasp_tennis": tennis + offsets["grasp_tennis"],
            "lift_tennis": tennis + offsets["lift_tennis"],
            "transfer": tennis + offsets["transfer"],
            "release": tennis + offsets["release"],
            "retract": tennis + offsets["retract"],
            "home": self.piper_initial_tool_pos.copy(),
        }

    def _piper_plan(self, time: float):
        keyframes = self._piper_keyframes()

        if time < self.pick_start_time:
            tool_pos = keyframes["initial"]
        elif time < self.approach_end_time:
            tool_pos = self._segment(
                time,
                self.pick_start_time,
                self.approach_end_time,
                keyframes["initial"],
                keyframes["above_tennis"],
            )
        elif time < self.descend_end_time:
            tool_pos = self._segment(
                time,
                self.approach_end_time,
                self.descend_end_time,
                keyframes["above_tennis"],
                keyframes["grasp_tennis"],
            )
        elif time < self.grasp_end_time:
            tool_pos = keyframes["grasp_tennis"]
        elif time < self.lift_end_time:
            tool_pos = self._segment(
                time,
                self.grasp_end_time,
                self.lift_end_time,
                keyframes["grasp_tennis"],
                keyframes["lift_tennis"],
            )
        elif time < self.release_time:
            tool_pos = self._segment(
                time,
                self.lift_end_time,
                self.release_time,
                keyframes["lift_tennis"],
                keyframes["transfer"],
            )
        elif time < self.retract_end_time:
            tool_pos = self._segment(
                time,
                self.release_time,
                self.retract_end_time,
                keyframes["release"],
                keyframes["retract"],
            )
        else:
            tool_pos = self._segment(
                time,
                self.retract_end_time,
                self.home_end_time,
                keyframes["retract"],
                keyframes["home"],
            )

        if time < self.descend_end_time:
            gripper = self.gripper_opening
        elif time < self.grasp_end_time:
            alpha = smoothstep(
                (time - self.descend_end_time) / max(self.grasp_end_time - self.descend_end_time, 1.0e-6)
            )
            gripper = (1.0 - alpha) * self.gripper_opening + alpha * self.gripper_closed_opening
        elif time < self.release_time:
            gripper = self.gripper_closed_opening
        else:
            gripper = self.gripper_opening

        return tool_pos, float(gripper)

    def update_tennis_body_xform(self, body_xform):
        self.current_tennis_center = self._tennis_center_from_body_xform(body_xform)

    def simulate_piper(self):
        if self.sim_time <= self.grasp_end_time:
            self.tennis_pick_center = self.current_tennis_center.copy()

        tool_pos, gripper = self._piper_plan(self.sim_time)
        self.ik_pos_obj.set_target_position(0, wp.vec3(*map(float, tool_pos)))
        piper_body_q_in = self.ik_state.body_q.numpy()
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        finger0_q = self.piper_finger_q_indices[0] if len(self.piper_finger_q_indices) > 0 else -1
        finger1_q = self.piper_finger_q_indices[1] if len(self.piper_finger_q_indices) > 1 else -1
        wp.launch(
            _set_finger_opening,
            dim=1,
            inputs=[self.ik_joint_q.flatten(), int(finger0_q), int(finger1_q), float(gripper)],
            device=self.ik_model.device,
        )
        newton.eval_fk(
            self.ik_model,
            self.ik_joint_q.flatten(),
            self.ik_model.joint_qd,
            self.ik_state,
        )

        piper_body_q_out = self.ik_state.body_q.numpy()
        for body_idx, _shape_idx, shape_label in self.piper_rigid_body_shapes:
            trans_0 = piper_body_q_in[body_idx]
            trans_1 = piper_body_q_out[body_idx]
            translation_0 = sim.Vec3f(trans_0[0], trans_0[1], trans_0[2])
            translation_1 = sim.Vec3f(trans_1[0], trans_1[1], trans_1[2])
            rotation_0 = sim.Quat(trans_0[3], trans_0[4], trans_0[5], trans_0[6])
            rotation_1 = sim.Quat(trans_1[3], trans_1[4], trans_1[5], trans_1[6])
            begin_trans = sim.Transform(translation_0, rotation_0, sim.Vec3f(1.0, 1.0, 1.0))
            end_trans = sim.Transform(translation_1, rotation_1, sim.Vec3f(1.0, 1.0, 1.0))
            self.body_entities[shape_label].move(begin_trans, end_trans)

        for body_idx, shape_idx, shape_label in self.piper_mesh_collider_shapes:
            begin_pos = self._piper_shape_world_vertices_np(shape_idx, piper_body_q_in[body_idx])
            end_pos = self._piper_shape_world_vertices_np(shape_idx, piper_body_q_out[body_idx])
            self.mesh_collider_entities[shape_label].move_verts(begin_pos, end_pos)

        self.sync_frontend_piper_display_state()

    def simulate(self):
        self.simulate_piper()

        self.world.step_sim()
        #self.world.begin_sim_loop()
        if self.world.fetch_sim(0):
            verts = self.cloth.get_positions()
            self.state_1.particle_q.assign(verts)
            body_q = self.state_1.body_q.numpy()
            for body_idx, shape_idx, shape_label in self.dynamic_sim_rigid_body_shapes:
                transform = self.body_entities[shape_label].get_transform()
                rigid_xform = wp.transform(
                    p=wp.vec3(transform.translation.x, transform.translation.y, transform.translation.z),
                    q=wp.quat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w),
                )
                shape_trans = self.shape_transform_np[shape_idx]
                shape_xform = wp.transform(
                    p=wp.vec3(shape_trans[0], shape_trans[1], shape_trans[2]),
                    q=wp.quat(shape_trans[3], shape_trans[4], shape_trans[5], shape_trans[6]),
                )
                body_xform = wp.transform_multiply(rigid_xform, wp.transform_inverse(shape_xform))
                body_q[body_idx] = (
                    body_xform[0],
                    body_xform[1],
                    body_xform[2],
                    body_xform[3],
                    body_xform[4],
                    body_xform[5],
                    body_xform[6],
                )
                self.update_tennis_body_xform(body_xform)
            self.state_1.body_q.assign(body_q)
        (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        ik_body_q = self.ik_state.body_q.numpy()
        ik_joint_q = self.ik_state.joint_q.numpy()
        if not np.isfinite(ik_body_q).all() or not np.isfinite(ik_joint_q).all():
            raise ValueError("Piper IK state is not finite.")

        front_body_q = self.state_0.body_q.numpy()[
            self.front_piper_body_start : self.front_piper_body_start + self.front_piper_body_count
        ]
        front_joint_q = self.state_0.joint_q.numpy()[
            self.front_piper_joint_q_start : self.front_piper_joint_q_start + self.front_piper_joint_q_count
        ]
        if not np.allclose(front_body_q, ik_body_q, atol=1.0e-6):
            raise ValueError("Frontend Piper body transforms do not match the IK state.")
        if not np.allclose(front_joint_q, ik_joint_q, atol=1.0e-6):
            raise ValueError("Frontend Piper joint coordinates do not match the IK state.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
