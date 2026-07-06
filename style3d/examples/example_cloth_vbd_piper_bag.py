# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate a fixed-mouth canvas bag beside an AgileX PiPER arm with VBD.

Command:
    python -m newton.examples cloth_vbd_piper_bag --device cuda:0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import BodyFlags, ParticleFlags
from style3d.examples._style3d_asset_probe import analyze_mesh, load_mesh, resolve_path


MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_ROOT_ENV = "NEWTON_MENAGERIE_PATH"
PIPER_FOLDER = "agilex_piper"
PIPER_MJCF = "piper.xml"
DEFAULT_BAG_ASSET = "style3d/examples/assets/style3d_probe/bag/canvas_bag/FBD_03.usd"
DEFAULT_ROBOT_HOME_Q = (0.0, 1.2, -1.6, 0.0, 0.8, 0.0, 0.028, -0.028)
DEFAULT_BALL_XY = (0.42, -0.16)
DEFAULT_BALL_MASS = float(4.0 / 3.0 * np.pi * 0.03**3 * 250.0)
BAG_COLOR = (0.35, 0.62, 0.88)
MOUTH_COLOR = (0.05, 0.95, 0.25)
MOUTH_LOOP_COLOR = (1.0, 0.75, 0.05)


@dataclass(frozen=True)
class PiperInfo:
    body_start: int
    body_end: int


def convert_y_up_to_z_up(vertices: np.ndarray) -> np.ndarray:
    """Rotate a Y-up asset into Newton's Z-up world frame."""
    converted = vertices.copy()
    converted[:, 1] = -vertices[:, 2]
    converted[:, 2] = vertices[:, 1]
    return converted


def rotate_vertices_z(vertices: np.ndarray, angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rotated = vertices.copy()
    rotated[:, 0] = c * vertices[:, 0] - s * vertices[:, 1]
    rotated[:, 1] = s * vertices[:, 0] + c * vertices[:, 1]
    return rotated


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    q_w = q[3]
    return v + 2.0 * np.cross(q_xyz, np.cross(q_xyz, v) + q_w * v)


def transform_point_np(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3] + quat_rotate_np(transform[3:7], point)


def parse_vec2_arg(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers, e.g. 0.45,0.0")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_vec3_arg(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 0.0,0.0,0.0")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_float_tuple_arg(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in text.split(",")]
    try:
        return tuple(float(part) for part in parts if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def find_label_index(labels, name: str, start: int = 0, stop: int | None = None) -> int:
    stop = len(labels) if stop is None else stop
    for index in range(start, stop):
        label = str(labels[index])
        if label == name or label.endswith(f"/{name}") or label.endswith(name):
            return index
    available = [str(labels[index]) for index in range(start, stop)]
    raise ValueError(f"Could not find label '{name}'. Available labels: {available}")


def get_ball_start_pos(args: argparse.Namespace) -> np.ndarray:
    if args.ball_pos is not None:
        return np.asarray(args.ball_pos, dtype=np.float64)
    return np.asarray(
        (
            float(args.ball_xy[0]),
            float(args.ball_xy[1]),
            float(args.ball_radius) + float(args.ball_ground_clearance),
        ),
        dtype=np.float64,
    )


def get_ball_density(args: argparse.Namespace) -> float:
    if args.ball_density is not None:
        return float(args.ball_density)
    radius = float(args.ball_radius)
    volume = 4.0 / 3.0 * np.pi * radius**3
    if volume <= 0.0:
        raise ValueError(f"Ball radius must be positive, got {radius}.")
    return float(args.ball_mass) / volume


def mesh_boundary_vertices(faces: np.ndarray) -> np.ndarray:
    components = boundary_components(faces)
    return np.concatenate(components) if components else np.empty(0, dtype=np.int32)


def boundary_components(faces: np.ndarray) -> list[np.ndarray]:
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            edge = (min(int(i), int(j)), max(int(i), int(j)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    adjacency: dict[int, list[int]] = {}
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    components: list[np.ndarray] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in adjacency[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int32))
    return components


def select_mouth_vertices(vertices: np.ndarray, faces: np.ndarray, band: float) -> np.ndarray:
    boundary = mesh_boundary_vertices(faces)
    if len(boundary) == 0:
        return np.empty(0, dtype=np.int32)
    boundary_z = vertices[boundary, 2]
    min_z = float(np.max(boundary_z) - band)
    return boundary[boundary_z >= min_z]


def select_mouth_loop_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    components = boundary_components(faces)
    if not components:
        return np.empty(0, dtype=np.int32)

    def component_score(component: np.ndarray) -> float:
        points = vertices[component]
        size_xy = np.ptp(points[:, :2], axis=0)
        return float(len(component)) * float(size_xy[0]) * float(size_xy[1])

    return max(components, key=component_score)


@wp.kernel
def set_ik_finger_opening_kernel(
    ik_joint_q: wp.array2d[float],
    finger0_q: int,
    finger1_q: int,
    finger_opening: float,
):
    if finger0_q >= 0:
        ik_joint_q[0, finger0_q] = finger_opening
    if finger1_q >= 0:
        ik_joint_q[0, finger1_q] = -finger_opening


@wp.kernel
def copy_robot_body_poses_kernel(
    src_body_q: wp.array[wp.transform],
    dst_body_q: wp.array[wp.transform],
    dst_body_start: int,
):
    body_id = wp.tid()
    dst_body_q[dst_body_start + body_id] = src_body_q[body_id]


@wp.kernel
def drive_body_pose_kernel(
    body_id: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    position: wp.vec3,
    velocity: wp.vec3,
    rotation: wp.quat,
):
    pose = wp.transform(position, rotation)
    body_q[body_id] = pose
    body_q_prev[body_id] = pose
    body_qd[body_id] = wp.spatial_vector(velocity, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def release_body_kernel(
    body_id: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    velocity: wp.vec3,
):
    body_q_prev[body_id] = body_q[body_id]
    body_qd[body_id] = wp.spatial_vector(velocity, wp.vec3(0.0, 0.0, 0.0))


def smoothstep(alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def resolve_piper_mjcf(args: argparse.Namespace) -> Path | None:
    if args.no_robot:
        return None
    if args.piper_mjcf:
        path = Path(args.piper_mjcf).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    local_root = os.environ.get(MENAGERIE_ROOT_ENV)
    if local_root:
        path = Path(local_root).expanduser() / PIPER_FOLDER / PIPER_MJCF
        if path.exists():
            return path.resolve()

    folder = newton.examples.download_external_git_folder(MENAGERIE_URL, PIPER_FOLDER)
    return (Path(folder) / PIPER_MJCF).resolve()


def add_fixed_mouth_bag(
    builder: newton.ModelBuilder,
    args: argparse.Namespace,
) -> tuple[slice, slice, np.ndarray, np.ndarray, np.ndarray]:
    asset_path = resolve_path(args.bag_asset)
    mesh = load_mesh(asset_path)
    report = analyze_mesh(asset_path, mesh)
    if report.status != "ok":
        raise ValueError(f"Cannot load bag asset {asset_path}: {report.warnings}")

    scale = float(args.bag_scale) if args.bag_scale is not None else float(report.recommended_scale or 1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if args.bag_up_axis == "y":
        vertices = convert_y_up_to_z_up(vertices)
    vertices = rotate_vertices_z(vertices, float(args.bag_yaw))
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1, 3)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
    bag_center_xy = np.asarray(args.bag_center, dtype=np.float64)
    pos = wp.vec3(
        float(bag_center_xy[0] - bbox_center_xy[0] * scale),
        float(bag_center_xy[1] - bbox_center_xy[1] * scale),
        float(args.bag_start_height - bbox_min[2] * scale),
    )

    start_particle = len(builder.particle_q)
    start_tri = len(builder.tri_indices)
    builder.add_cloth_mesh(
        pos=pos,
        rot=wp.quat_identity(),
        scale=scale,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices.tolist(),
        indices=faces.reshape(-1).tolist(),
        density=float(args.bag_density),
        tri_ke=float(args.bag_tri_ke),
        tri_ka=float(args.bag_tri_ka),
        tri_kd=float(args.bag_tri_kd),
        edge_ke=float(args.bag_edge_ke),
        edge_kd=float(args.bag_edge_kd),
        add_springs=bool(args.bag_springs),
        spring_ke=float(args.bag_spring_ke),
        spring_kd=float(args.bag_spring_kd),
        particle_radius=float(args.bag_particle_radius),
        validate_mesh=False,
        label="canvas_bag",
    )
    bag_slice = slice(start_particle, len(builder.particle_q))
    bag_tri_slice = slice(start_tri, len(builder.tri_indices))

    mouth_local = select_mouth_vertices(vertices, faces, float(args.mouth_band) / scale) if args.fix_mouth else []
    mouth_indices = np.asarray([bag_slice.start + int(i) for i in mouth_local], dtype=np.int32)
    for particle in mouth_indices:
        index = int(particle)
        builder.particle_flags[index] = builder.particle_flags[index] & ~ParticleFlags.ACTIVE
        builder.particle_mass[index] = 0.0

    mouth_loop_local = select_mouth_loop_vertices(vertices, faces)
    mouth_loop_indices = np.asarray([bag_slice.start + int(i) for i in mouth_loop_local], dtype=np.int32)
    pos_np = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
    mouth_loop_points = vertices[mouth_loop_local] * scale + pos_np
    mouth_center = mouth_loop_points.mean(axis=0) if len(mouth_loop_points) else np.zeros(3, dtype=np.float64)

    print(
        "[PiperBagVBD] "
        f"bag={asset_path}, prim={report.usd_prim_path}, up_axis={args.bag_up_axis}, "
        f"yaw={float(args.bag_yaw):g}, scale={scale:g}, "
        f"particles={bag_slice.stop - bag_slice.start}, triangles={faces.shape[0]}, "
        f"fixed_mouth_particles={len(mouth_indices)}, bbox_size={(bbox_max - bbox_min).astype(float).tolist()}",
        flush=True,
    )
    print(
        "[PiperBagVBD] "
        f"bag_mouth_center={mouth_center.astype(float).tolist()}, mouth_loop_particles={len(mouth_loop_indices)}",
        flush=True,
    )
    for warning in report.warnings[:4]:
        print(f"[PiperBagVBD] bag warning: {warning}", flush=True)
    return bag_slice, bag_tri_slice, mouth_indices, mouth_loop_indices, mouth_center


def add_piper(
    builder: newton.ModelBuilder,
    args: argparse.Namespace,
    *,
    parse_visuals: bool = True,
    parse_meshes: bool = True,
    announce: bool = True,
) -> PiperInfo | None:
    piper_mjcf = resolve_piper_mjcf(args)
    if piper_mjcf is None:
        return None
    if not piper_mjcf.exists():
        raise FileNotFoundError(f"AgileX PiPER MJCF not found: {piper_mjcf}")

    base_pos = wp.vec3(*map(float, args.robot_pos))
    base_yaw = float(args.robot_yaw)
    base_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), base_yaw)
    body_start = builder.body_count
    dof_start = len(builder.joint_q)
    builder.add_mjcf(
        str(piper_mjcf),
        xform=wp.transform(base_pos, base_rot),
        floating=False,
        parse_visuals=parse_visuals,
        parse_meshes=parse_meshes,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
    )
    for body in range(body_start, builder.body_count):
        builder.body_flags[body] = int(BodyFlags.KINEMATIC)

    robot_home_q = tuple(args.robot_home_q or DEFAULT_ROBOT_HOME_Q)
    robot_q_count = len(builder.joint_q) - dof_start
    for local_q, value in enumerate(robot_home_q[:robot_q_count]):
        builder.joint_q[dof_start + local_q] = float(value)

    info = PiperInfo(
        body_start=body_start,
        body_end=builder.body_count,
    )
    if announce:
        print(
            "[PiperBagVBD] "
            f"piper={piper_mjcf}, robot_dofs={robot_q_count}, "
            f"robot_bodies={info.body_end - info.body_start}, robot_shapes={builder.shape_count}",
            flush=True,
        )
    return info


def add_grasp_ball(builder: newton.ModelBuilder, args: argparse.Namespace) -> int:
    radius = float(args.ball_radius)
    ball_pos = get_ball_start_pos(args)
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*map(float, ball_pos)), q=wp.quat_identity()),
        label="grasp_ball",
    )
    cfg = newton.ModelBuilder.ShapeConfig(
        density=get_ball_density(args),
        ke=float(args.ball_contact_ke),
        kd=float(args.ball_contact_kd),
        mu=float(args.ball_contact_mu),
    )
    builder.add_shape_sphere(body=body, radius=radius, cfg=cfg, label="grasp_ball_shape")
    return body


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.use_cuda_graph = not bool(args.no_cuda_graph)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.piper_info = add_piper(builder, args)
        self.ball_body = add_grasp_ball(builder, args)
        (
            self.bag_slice,
            self.bag_tri_slice,
            self.mouth_indices_np,
            self.mouth_loop_indices_np,
            self.bag_mouth_center,
        ) = add_fixed_mouth_bag(builder, args)
        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = float(args.soft_contact_ke)
        self.model.soft_contact_kd = float(args.soft_contact_kd)
        self.model.soft_contact_mu = float(args.soft_contact_mu)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_qd.zero_()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self.model.contacts()
        self.robot_planner_enabled = (
            bool(args.enable_robot_planner) and self.piper_info is not None and not bool(args.no_robot)
        )
        self.use_cuda_graph = self.use_cuda_graph and not self.robot_planner_enabled

        bag_tri_indices = np.asarray(builder.tri_indices[self.bag_tri_slice], dtype=np.int32).reshape(-1)
        self.bag_tri_indices = wp.array(bag_tri_indices, dtype=wp.int32, device=self.model.device)
        self.mouth_indices = wp.array(self.mouth_indices_np, dtype=wp.int32, device=self.model.device)
        self.mouth_loop_indices = wp.array(self.mouth_loop_indices_np, dtype=wp.int32, device=self.model.device)
        particle_colors = np.tile(np.asarray(BAG_COLOR, dtype=np.float32), (self.model.particle_count, 1))
        if len(self.mouth_loop_indices_np):
            particle_colors[self.mouth_loop_indices_np] = np.asarray(MOUTH_LOOP_COLOR, dtype=np.float32)
        if len(self.mouth_indices_np):
            particle_colors[self.mouth_indices_np] = np.asarray(MOUTH_COLOR, dtype=np.float32)
        self.particle_render_colors = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)

        self.ball_released = False
        self.ball_attached = False
        self.ball_attach_offset = np.zeros(3, dtype=np.float64)
        self.ball_attach_opening = float(args.gripper_closed_opening)
        self.ball_attach_time_actual = -1.0
        self.ball_rotation = wp.quat_identity()
        self.ball_start_pos = get_ball_start_pos(args)
        self.ball_density = get_ball_density(args)
        self.ball_pick_pos = self.ball_start_pos.copy()
        self.ball_release_pos = np.asarray(self.bag_mouth_center, dtype=np.float64)
        self.ball_release_pos[2] += float(args.ball_release_height)
        self.ball_release_velocity = np.asarray(args.ball_release_velocity, dtype=np.float64)
        self.effective_release_time = float(args.ball_release_time)
        self.ball_attach_time = float(args.ball_grasp_start_time)
        self.robot_plan_initialized = False
        self.robot_tool_pos = self.ball_start_pos.copy()
        self.robot_gripper_opening = float(args.gripper_opening)
        if self.robot_planner_enabled:
            self._setup_robot_planner()

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=max(1, int(args.solver_iterations)),
            particle_enable_self_contact=not bool(args.no_self_contact),
            particle_self_contact_radius=float(args.self_contact_radius),
            particle_self_contact_margin=float(args.self_contact_margin),
            particle_topological_contact_filter_threshold=max(0, int(args.topological_contact_filter_threshold)),
            particle_enable_tile_solve=not bool(args.no_tile_solve),
            rigid_body_particle_contact_buffer_size=max(1, int(args.rigid_body_particle_contact_buffer_size)),
            rigid_body_contact_buffer_size=max(1, int(args.rigid_body_contact_buffer_size)),
            friction_epsilon=float(args.friction_epsilon),
        )

        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.65, -1.05, 0.65), pitch=-22.0, yaw=118.0)

        print(
            "[PiperBagVBD] "
            f"frame_dt={self.frame_dt:g}s, substeps={self.sim_substeps}, sim_dt={self.sim_dt:g}s, "
            f"bodies={self.model.body_count}, shapes={self.model.shape_count}, particles={self.model.particle_count}, "
            f"ball_radius={float(args.ball_radius):g}, ball_density={self.ball_density:g}, "
            f"ball_start_pos={self.ball_start_pos.astype(float).tolist()}, robot_planner={self.robot_planner_enabled}, "
            f"ball_release_pos={self.ball_release_pos.astype(float).tolist()}",
            flush=True,
        )
        self.capture()

    @staticmethod
    def create_parser():
        return create_parser()

    def _setup_robot_planner(self) -> None:
        assert self.piper_info is not None

        ik_builder = newton.ModelBuilder()
        ik_info = add_piper(ik_builder, self.args, parse_visuals=False, parse_meshes=False, announce=False)
        if ik_info is None:
            self.robot_planner_enabled = False
            return

        self.ik_model = ik_builder.finalize()
        self.ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, self.ik_state)

        self.robot_ee_body = find_label_index(self.ik_model.body_label, self.args.robot_ee_body)
        self.robot_ee_body_main = find_label_index(
            self.model.body_label,
            self.args.robot_ee_body,
            self.piper_info.body_start,
            self.piper_info.body_end,
        )
        self.robot_ee_offset = np.asarray(self.args.robot_ee_offset, dtype=np.float64)
        ee_body_q = self.ik_state.body_q.numpy()
        ee_tf = wp.transform(*ee_body_q[self.robot_ee_body])
        ee_pos = wp.transform_point(ee_tf, wp.vec3(*map(float, self.robot_ee_offset)))
        self.robot_initial_tool_pos = np.asarray([float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])])

        ik_finger0_joint = find_label_index(self.ik_model.joint_label, self.args.robot_finger0_joint)
        ik_finger1_joint = find_label_index(self.ik_model.joint_label, self.args.robot_finger1_joint)
        ik_joint_q_start = self.ik_model.joint_q_start.numpy()
        self.ik_finger0_q = int(ik_joint_q_start[ik_finger0_joint])
        self.ik_finger1_q = int(ik_joint_q_start[ik_finger1_joint])

        self.ik_joint_q = wp.array(self.ik_model.joint_q, shape=(1, self.ik_model.joint_coord_count))
        self.ik_pos_obj = ik.IKObjectivePosition(
            link_index=self.robot_ee_body,
            link_offset=wp.vec3(*map(float, self.robot_ee_offset)),
            target_positions=wp.array([wp.vec3(*map(float, self.robot_initial_tool_pos))], dtype=wp.vec3),
        )
        self.ik_joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=float(self.args.robot_ik_joint_limit_weight),
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.ik_pos_obj, self.ik_joint_limits_obj],
            lambda_initial=float(self.args.robot_ik_lambda),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = max(1, int(self.args.robot_ik_iterations))

        self.robot_start_time = max(float(self.args.ball_grasp_start_time), float(self.args.robot_settle_time))
        self._initialize_robot_pick_plan(self.ball_start_pos)

        print(
            "[PiperBagVBD] "
            f"robot_ee_body={self.args.robot_ee_body}({self.robot_ee_body}), "
            f"ee_offset={self.robot_ee_offset.astype(float).tolist()}, "
            f"start_time={self.robot_start_time:g}, release_time={self.effective_release_time:g}",
            flush=True,
        )

    def _initialize_robot_pick_plan(self, ball_pos: np.ndarray) -> None:
        self.ball_pick_pos = np.asarray(ball_pos, dtype=np.float64).copy()

        start = self.robot_start_time
        self.robot_approach_end_time = start + max(float(self.args.robot_approach_time), 0.0)
        self.robot_descend_end_time = self.robot_approach_end_time + max(float(self.args.robot_descend_time), 0.0)
        self.ball_attach_time = self.robot_descend_end_time + max(float(self.args.robot_grasp_time), 0.0)
        self.robot_lift_end_time = self.ball_attach_time + max(float(self.args.robot_lift_time), 0.0)
        self.effective_release_time = max(
            float(self.args.ball_release_time),
            self.robot_lift_end_time + max(float(self.args.robot_transfer_min_time), 0.0),
        )

        self.robot_pre_grasp_pos = self.ball_pick_pos + np.array([0.0, 0.0, float(self.args.robot_approach_height)])
        self.robot_grasp_pos = self.ball_pick_pos + np.array([0.0, 0.0, float(self.args.robot_grasp_height_offset)])
        self.robot_lift_pos = self.ball_pick_pos + np.array([0.0, 0.0, float(self.args.robot_lift_height)])
        self.robot_retract_pos = self.ball_release_pos + np.array([0.0, 0.0, float(self.args.robot_retract_height)])

    def _segment(self, time: float, t0: float, t1: float, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        if t1 <= t0:
            return p1.copy()
        alpha = smoothstep((time - t0) / (t1 - t0))
        return (1.0 - alpha) * p0 + alpha * p1

    def _robot_plan(self, time: float) -> tuple[np.ndarray, float]:
        if time < self.robot_start_time or not self.robot_plan_initialized:
            return self.robot_initial_tool_pos.copy(), float(self.args.gripper_opening)

        start = self.robot_start_time
        if time < self.robot_approach_end_time:
            tool_pos = self._segment(
                time, start, self.robot_approach_end_time, self.robot_initial_tool_pos, self.robot_pre_grasp_pos
            )
        elif time < self.robot_descend_end_time:
            tool_pos = self._segment(
                time,
                self.robot_approach_end_time,
                self.robot_descend_end_time,
                self.robot_pre_grasp_pos,
                self.robot_grasp_pos,
            )
        elif time < self.ball_attach_time:
            tool_pos = self.robot_grasp_pos.copy()
        elif time < self.robot_lift_end_time:
            tool_pos = self._segment(
                time, self.ball_attach_time, self.robot_lift_end_time, self.robot_grasp_pos, self.robot_lift_pos
            )
        elif time < self.effective_release_time:
            tool_pos = self._segment(
                time, self.robot_lift_end_time, self.effective_release_time, self.robot_lift_pos, self.ball_release_pos
            )
        else:
            tool_pos = self._segment(
                time,
                self.effective_release_time,
                self.effective_release_time + max(float(self.args.robot_retract_time), 1.0e-6),
                self.ball_release_pos,
                self.robot_retract_pos,
            )

        if time < self.robot_descend_end_time:
            gripper = float(self.args.gripper_opening)
        elif time < self.ball_attach_time:
            alpha = smoothstep((time - self.robot_descend_end_time) / max(float(self.args.robot_grasp_time), 1.0e-6))
            gripper = (1.0 - alpha) * float(self.args.gripper_opening) + alpha * float(self.args.gripper_closed_opening)
        elif time < self.effective_release_time:
            gripper = float(self.args.gripper_closed_opening)
        else:
            gripper = float(self.args.gripper_opening)
        if self.ball_attached and time < self.effective_release_time:
            gripper = self.ball_attach_opening

        return tool_pos, gripper

    def _update_robot_planner(self, state: newton.State, time: float) -> None:
        if not self.robot_planner_enabled:
            return
        if time >= self.robot_start_time and not self.robot_plan_initialized:
            body_q = state.body_q.numpy()
            self._initialize_robot_pick_plan(body_q[self.ball_body, :3])
            self.robot_plan_initialized = True
            print(
                "[PiperBagVBD] "
                f"planning grasp from settled ball_pos={self.ball_pick_pos.astype(float).tolist()}, "
                f"attach_time={self.ball_attach_time:g}, release_time={self.effective_release_time:g}",
                flush=True,
            )

        tool_pos, gripper = self._robot_plan(time)
        self.robot_tool_pos = tool_pos
        self.robot_gripper_opening = gripper
        self.ik_pos_obj.set_target_position(0, wp.vec3(*map(float, tool_pos)))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_ik_finger_opening_kernel,
            dim=1,
            inputs=[
                self.ik_joint_q,
                int(self.ik_finger0_q),
                int(self.ik_finger1_q),
                float(gripper),
            ],
            device=self.model.device,
        )
        newton.eval_fk(self.ik_model, self.ik_joint_q.flatten(), self.ik_model.joint_qd, self.ik_state)
        wp.launch(
            copy_robot_body_poses_kernel,
            dim=self.piper_info.body_end - self.piper_info.body_start,
            inputs=[
                self.ik_state.body_q,
                state.body_q,
                int(self.piper_info.body_start),
            ],
            device=self.model.device,
        )

    def _attach_tool_ball_distance_limit(self) -> float:
        if self.args.attach_max_tool_ball_distance is not None:
            return float(self.args.attach_max_tool_ball_distance)
        return max(1.5 * float(self.args.ball_radius), 0.02)

    def _tool_ball_distance(self, state: newton.State) -> float:
        body_q = state.body_q.numpy()
        ball_pos = np.asarray(body_q[self.ball_body, :3], dtype=np.float64)
        return float(np.linalg.norm(ball_pos - self.robot_tool_pos))

    def _should_attach_ball_to_gripper(self, state: newton.State, time: float) -> tuple[bool, float]:
        if time < self.robot_descend_end_time or time >= float(self.effective_release_time):
            return False, float("inf")
        if self.robot_gripper_opening > float(self.args.attach_max_gripper_opening):
            return False, float("inf")

        tool_ball_distance = self._tool_ball_distance(state)
        if tool_ball_distance > self._attach_tool_ball_distance_limit():
            return False, tool_ball_distance
        return True, tool_ball_distance

    def _attached_ball_pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        tool_pos, _gripper = self._robot_plan(time)
        position = tool_pos + self.ball_attach_offset
        eps = max(1.0e-4, self.sim_dt)
        prev_time = max(time - eps, self.ball_attach_time_actual if self.ball_attach_time_actual >= 0.0 else time - eps)
        next_time = min(time + eps, self.effective_release_time)
        prev_tool, _ = self._robot_plan(prev_time)
        next_tool, _ = self._robot_plan(next_time)
        velocity = (next_tool - prev_tool) / max(next_time - prev_time, eps)
        return position, velocity

    def _drive_attached_ball(self, state: newton.State, time: float) -> None:
        position, velocity = self._attached_ball_pose(time)
        wp.launch(
            drive_body_pose_kernel,
            dim=1,
            inputs=[
                int(self.ball_body),
                state.body_q,
                state.body_qd,
                self.solver.body_q_prev,
                wp.vec3(*map(float, position)),
                wp.vec3(*map(float, velocity)),
                self.ball_rotation,
            ],
            device=self.model.device,
        )

    def _attach_ball_to_gripper(
        self,
        state: newton.State,
        time: float,
        tool_ball_distance: float,
    ) -> None:
        body_q = state.body_q.numpy()
        ball_pos = np.asarray(body_q[self.ball_body, :3], dtype=np.float64)
        self.ball_attach_offset = ball_pos - self.robot_tool_pos
        self.ball_attach_opening = float(self.robot_gripper_opening)
        self.ball_attach_time_actual = time
        self.ball_attached = True
        self.ball_released = False
        print(
            "[PiperBagVBD] "
            f"ball attached at time={time:.3f}, "
            f"gripper_opening={self.ball_attach_opening:g}, "
            f"tool_ball_distance={tool_ball_distance:g}, "
            f"attach_offset={self.ball_attach_offset.astype(float).tolist()}",
            flush=True,
        )

    def _release_attached_ball(self, state: newton.State, time: float) -> None:
        _position, velocity = self._attached_ball_pose(float(self.effective_release_time))
        release_velocity = velocity + self.ball_release_velocity
        wp.launch(
            release_body_kernel,
            dim=1,
            inputs=[
                int(self.ball_body),
                state.body_q,
                state.body_qd,
                self.solver.body_q_prev,
                wp.vec3(*map(float, release_velocity)),
            ],
            device=self.model.device,
        )
        self.ball_attached = False
        self.ball_released = True
        print(
            "[PiperBagVBD] "
            f"ball detached at time={time:.3f}, "
            f"velocity={release_velocity.astype(float).tolist()}",
            flush=True,
        )

    def capture(self):
        self.graph = None
        if not self.use_cuda_graph or not wp.get_device().is_cuda:
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print(f"[PiperBagVBD] CUDA graph capture enabled for {self.sim_substeps} substeps.", flush=True)
        except Exception as exc:
            print(f"[PiperBagVBD] CUDA graph capture failed; falling back to Python stepping: {exc!r}", flush=True)

    def simulate(self):
        for _ in range(self.sim_substeps):
            time = self.sim_time + _ * self.sim_dt
            self._update_robot_planner(self.state_0, time)
            if self.ball_attached and time >= float(self.effective_release_time):
                self._release_attached_ball(self.state_0, time)
            elif self.ball_attached:
                self._drive_attached_ball(self.state_0, time)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            if (
                self.robot_planner_enabled
                and not self.ball_attached
                and not self.ball_released
            ):
                should_attach, tool_ball_distance = self._should_attach_ball_to_gripper(self.state_0, time)
                if should_attach:
                    self._attach_ball_to_gripper(
                        self.state_0,
                        time,
                        tool_ball_distance,
                    )
                    self._update_robot_planner(self.state_0, time)
                    self._drive_attached_ball(self.state_0, time)
                    self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.ball_attached:
                self._drive_attached_ball(self.state_1, time + self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1

        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
            q = self.state_0.particle_q.numpy()
            qd = self.state_0.particle_qd.numpy()
            body_q = self.state_0.body_q.numpy()
            robot_status = ""
            if self.robot_planner_enabled:
                tool_pos = transform_point_np(body_q[self.robot_ee_body_main], self.robot_ee_offset)
                tool_err = float(np.linalg.norm(tool_pos - self.robot_tool_pos))
                tool_ball_distance = float(np.linalg.norm(body_q[self.ball_body, :3] - tool_pos))
                robot_status = (
                    f" tool={tool_pos.astype(float).round(4).tolist()}"
                    f" target={self.robot_tool_pos.astype(float).round(4).tolist()}"
                    f" tool_ball_distance={tool_ball_distance:.4f}"
                    f" tool_err={tool_err:.4f} gripper={self.robot_gripper_opening:.4f} "
                )
            print(
                "[PiperBagVBD] "
                f"frame={self.frame} time={self.sim_time:.3f} "
                f"ball_z={float(body_q[self.ball_body, 2]):.4f} "
                f"attached={self.ball_attached} released={self.ball_released} "
                f"{robot_status}"
                f"bag_z=[{float(np.min(q[:, 2])):.4f}, {float(np.max(q[:, 2])):.4f}] "
                f"max_particle_speed={float(np.max(np.linalg.norm(qd, axis=1))):.4f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/piper_bag/canvas_bag",
            self.state_0.particle_q,
            self.bag_tri_indices,
            hidden=bool(self.args.no_colored_bag),
            backface_culling=False,
            color=BAG_COLOR,
        )
        self.viewer.log_points(
            "/piper_bag/mouth_particles",
            self.state_0.particle_q,
            self.model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not bool(self.args.show_mouth_particles),
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if len(q) == 0:
            raise ValueError("PiPER bag VBD example has no particles.")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("PiPER bag VBD particle state is not finite.")
        max_abs_position = float(np.max(np.abs(q)))
        max_speed = float(np.max(np.linalg.norm(qd, axis=1)))
        if max_abs_position > 20.0 or max_speed > 250.0:
            raise ValueError(
                f"PiPER bag VBD simulation appears unstable: "
                f"max_abs_position={max_abs_position:.3f}, max_speed={max_speed:.3f}"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--bag-asset", default=DEFAULT_BAG_ASSET, help="USD/OBJ bag mesh to import")
    parser.add_argument("--bag-scale", type=float, default=None, help="Bag asset scale override")
    parser.add_argument(
        "--bag-up-axis",
        choices=["y", "z"],
        default="y",
        help="Up axis authored in the bag asset; Y-up assets are rotated into Newton's Z-up frame",
    )
    parser.add_argument("--bag-yaw", type=float, default=float(np.pi / 2.0), help="Bag yaw applied after up-axis conversion [rad]")
    parser.add_argument("--bag-center", type=parse_vec2_arg, default=(0.50, 0.12), help="Bag center XY position [m]")
    parser.add_argument("--bag-start-height", type=float, default=0.15, help="Lowest bag vertex height [m]")
    parser.add_argument("--fix-mouth", action=argparse.BooleanOptionalAction, default=True, help="Fix the bag mouth")
    parser.add_argument(
        "--mouth-band",
        type=float,
        default=0.035,
        help="Fix boundary vertices within this distance below the highest boundary vertex [m]",
    )

    parser.add_argument("--piper-mjcf", default=None, help="Optional local path to agilex_piper/piper.xml")
    parser.add_argument("--no-robot", action="store_true", help="Skip importing the AgileX PiPER MJCF")
    parser.add_argument("--robot-pos", type=parse_vec3_arg, default=(0.0, 0.0, 0.0), help="Robot base position [m]")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot base yaw [rad]")
    parser.add_argument(
        "--robot-home-q",
        type=parse_float_tuple_arg,
        default=DEFAULT_ROBOT_HOME_Q,
        help="Comma-separated PiPER home joint coordinates",
    )
    parser.add_argument(
        "--enable-robot-planner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PiPER IK to plan a pick-transfer-release motion for the ball",
    )
    parser.add_argument("--robot-ee-body", default="link6", help="PiPER body used as the IK end-effector")
    parser.add_argument(
        "--robot-ee-offset",
        type=parse_vec3_arg,
        default=(0.0, 0.0, 0.13503),
        help="End-effector tool point offset in the selected body frame [m]",
    )
    parser.add_argument("--robot-finger0-joint", default="joint7", help="Positive-opening PiPER finger joint")
    parser.add_argument("--robot-finger1-joint", default="joint8", help="Negative-opening PiPER finger joint")
    parser.add_argument("--robot-ik-iterations", type=int, default=18)
    parser.add_argument("--robot-ik-lambda", type=float, default=0.1)
    parser.add_argument("--robot-ik-joint-limit-weight", type=float, default=10.0)
    parser.add_argument("--robot-settle-time", type=float, default=0.5, help="Let the ball settle before planning grasp [s]")
    parser.add_argument("--robot-approach-time", type=float, default=0.8)
    parser.add_argument("--robot-descend-time", type=float, default=0.6)
    parser.add_argument("--robot-grasp-time", type=float, default=0.35)
    parser.add_argument("--robot-lift-time", type=float, default=0.6)
    parser.add_argument("--robot-transfer-min-time", type=float, default=0.9)
    parser.add_argument("--robot-retract-time", type=float, default=0.7)
    parser.add_argument("--robot-approach-height", type=float, default=0.22)
    parser.add_argument("--robot-grasp-height-offset", type=float, default=0.0)
    parser.add_argument("--robot-lift-height", type=float, default=0.30)
    parser.add_argument("--robot-retract-height", type=float, default=0.10)
    parser.add_argument("--gripper-opening", type=float, default=0.035)
    parser.add_argument("--gripper-closed-opening", type=float, default=0.0)
    parser.add_argument(
        "--attach-max-gripper-opening",
        type=float,
        default=0.025,
        help="Only attach after the gripper has closed at least this far [m]",
    )
    parser.add_argument(
        "--attach-max-tool-ball-distance",
        type=float,
        default=None,
        help="Only attach when the tool point is near the ball center [m]",
    )

    parser.add_argument(
        "--ball-pos",
        type=parse_vec3_arg,
        default=None,
        help="Optional ball center position [m]; if omitted, z is ball radius plus ground clearance",
    )
    parser.add_argument("--ball-xy", type=parse_vec2_arg, default=DEFAULT_BALL_XY, help="Ball center XY when --ball-pos is omitted [m]")
    parser.add_argument("--ball-ground-clearance", type=float, default=5.0e-4, help="Initial ball clearance above ground [m]")
    parser.add_argument("--ball-radius", type=float, default=0.025, help="Free rigid sphere radius [m]")
    parser.add_argument(
        "--ball-density",
        type=float,
        default=None,
        help="Free rigid sphere density [kg/m^3]; overrides --ball-mass when set",
    )
    parser.add_argument("--ball-mass", type=float, default=DEFAULT_BALL_MASS, help="Free rigid sphere target mass [kg]")
    parser.add_argument("--ball-contact-ke", type=float, default=5.0e4)
    parser.add_argument("--ball-contact-kd", type=float, default=20.0)
    parser.add_argument("--ball-contact-mu", type=float, default=0.65)
    parser.add_argument("--ball-grasp-start-time", type=float, default=0.0, help="Time when the carry trajectory starts [s]")
    parser.add_argument("--ball-release-time", type=float, default=3.2, help="Time when the ball is released [s]")
    parser.add_argument(
        "--ball-release-height",
        type=float,
        default=0.15,
        help="Move the gripper and release the ball this far above the detected bag mouth center [m]",
    )
    parser.add_argument(
        "--ball-release-velocity",
        type=parse_vec3_arg,
        default=(0.0, 0.0, 0.0),
        help="Additional velocity assigned when releasing the ball [m/s]",
    )

    parser.add_argument("--view-fps", type=float, default=60.0)
    parser.add_argument("--view-substeps", type=int, default=8)
    parser.add_argument("--solver-iterations", type=int, default=16)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--no-colored-bag", action="store_true")
    parser.add_argument(
        "--show-mouth-particles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay fixed mouth particles",
    )

    parser.add_argument("--bag-density", type=float, default=0.02)
    parser.add_argument("--bag-particle-radius", type=float, default=0.005)
    parser.add_argument("--bag-tri-ke", type=float, default=8.0e4)
    parser.add_argument("--bag-tri-ka", type=float, default=8.0e4)
    parser.add_argument("--bag-tri-kd", type=float, default=8.0e-1)
    parser.add_argument("--bag-edge-ke", type=float, default=5.0e-2)
    parser.add_argument("--bag-edge-kd", type=float, default=8.0e-2)
    parser.add_argument("--bag-springs", action="store_true", help="Add structural springs to the bag mesh")
    parser.add_argument("--bag-spring-ke", type=float, default=2.0e3)
    parser.add_argument("--bag-spring-kd", type=float, default=1.0e-2)

    parser.add_argument("--soft-contact-ke", type=float, default=5.0e4)
    parser.add_argument("--soft-contact-kd", type=float, default=2.0)
    parser.add_argument("--soft-contact-mu", type=float, default=0.5)
    parser.add_argument("--friction-epsilon", type=float, default=1.0e-2)
    parser.add_argument("--self-contact-radius", type=float, default=0.003)
    parser.add_argument("--self-contact-margin", type=float, default=0.006)
    parser.add_argument("--topological-contact-filter-threshold", type=int, default=2)
    parser.add_argument("--rigid-body-particle-contact-buffer-size", type=int, default=32768)
    parser.add_argument("--rigid-body-contact-buffer-size", type=int, default=4096)
    parser.add_argument("--no-self-contact", action="store_true")
    parser.add_argument("--no-tile-solve", action="store_true")
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
