#!/usr/bin/env python3
"""
Style3D multi-layer fold probe for the green T-shirt asset.

The script drives four kinematic grasp regions on the garment:
  1. one side edge folds toward the center;
  2. the other side edge folds toward the center;
  3. bottom hem folds toward the center;
  4. collar/neck region folds toward the center;
then holds all folds so the Style3D self-collision solver has to maintain a
stack of cloth layers.

Examples:
    python -m newton.examples cloth_style3d_fold_probe --viewer null --num-frames 2
    python -m style3d.examples.example_cloth_style3d_fold_probe \
        style3d/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt.obj \
        --device cuda:0 --view-substeps 10 --solver-iterations 4 --linear-iterations 10 \
        --cloth-density 0.3 --particle-radius 0.005
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

import newton.examples
from style3d.examples.tools.mesh_asset_utils import analyze_mesh, load_mesh, resolve_path
from style3d.examples.tools.style3d_cloth_utils import (
    build_style3d_cloth_model,
    cloth_params_from_args,
    collide_model,
    configure_style3d_solver_collision,
    parse_vec3_arg,
)


def define_drive_kernel(wp):
    @wp.func
    def clamp01(x: float):
        return wp.min(1.0, wp.max(0.0, x))

    @wp.func
    def smootherstep(x: float):
        t = clamp01(x)
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @wp.func
    def fold_progress(time: float, start: float, duration: float):
        return smootherstep((time - start) / duration)

    @wp.kernel
    def drive_grip_points(
        time: float,
        dt: float,
        first_side_indices: wp.array[wp.int32],
        first_side_rest: wp.array[wp.vec3],
        second_side_indices: wp.array[wp.int32],
        second_side_rest: wp.array[wp.vec3],
        bottom_indices: wp.array[wp.int32],
        bottom_rest: wp.array[wp.vec3],
        collar_indices: wp.array[wp.int32],
        collar_rest: wp.array[wp.vec3],
        q0: wp.array[wp.vec3],
        qd0: wp.array[wp.vec3],
        q1: wp.array[wp.vec3],
        qd1: wp.array[wp.vec3],
        center_x: float,
        first_side_target_y: float,
        second_side_target_y: float,
        bottom_target_x: float,
        collar_target_x: float,
        first_side_start: float,
        second_side_start: float,
        bottom_start: float,
        collar_start: float,
        fold_duration: float,
        side_fold_duration: float,
        lift_height: float,
        base_z: float,
        layer_gap: float,
        ground_clearance: float,
    ):
        tid = wp.tid()
        first_side_count = first_side_indices.shape[0]
        second_side_count = second_side_indices.shape[0]
        bottom_count = bottom_indices.shape[0]
        collar_count = collar_indices.shape[0]

        if tid < first_side_count:
            particle = first_side_indices[tid]
            rest = first_side_rest[tid]
            p = fold_progress(time, first_side_start, side_fold_duration)
            y = rest[1] * (1.0 - p) + first_side_target_y * p
            z = rest[2] * (1.0 - p) + (base_z + layer_gap) * p + lift_height * wp.sin(wp.pi * p)
            z = wp.max(z, ground_clearance)
            target = wp.vec3(rest[0], y, z)
            q0[particle] = target
            q1[particle] = target
            vel = wp.vec3(0.0)
            qd0[particle] = vel
            qd1[particle] = vel
        elif tid < first_side_count + second_side_count:
            j = tid - first_side_count
            particle = second_side_indices[j]
            rest = second_side_rest[j]
            p = fold_progress(time, second_side_start, side_fold_duration)
            y = rest[1] * (1.0 - p) + second_side_target_y * p
            z = rest[2] * (1.0 - p) + (base_z + 2.0 * layer_gap) * p + lift_height * wp.sin(wp.pi * p)
            z = wp.max(z, ground_clearance)
            target = wp.vec3(rest[0], y, z)
            q0[particle] = target
            q1[particle] = target
            vel = wp.vec3(0.0)
            qd0[particle] = vel
            qd1[particle] = vel
        elif tid < first_side_count + second_side_count + bottom_count:
            j = tid - first_side_count - second_side_count
            particle = bottom_indices[j]
            rest = bottom_rest[j]
            p = fold_progress(time, bottom_start, fold_duration)
            x = rest[0] * (1.0 - p) + bottom_target_x * p
            z = rest[2] * (1.0 - p) + (base_z + 3.0 * layer_gap) * p + lift_height * wp.sin(wp.pi * p)
            z = wp.max(z, ground_clearance)
            target = wp.vec3(x, rest[1], z)
            q0[particle] = target
            q1[particle] = target
            vel = wp.vec3(0.0)
            qd0[particle] = vel
            qd1[particle] = vel
        else:
            j = tid - first_side_count - second_side_count - bottom_count
            particle = collar_indices[j]
            rest = collar_rest[j]
            p = fold_progress(time, collar_start, fold_duration)
            x = rest[0] * (1.0 - p) + collar_target_x * p
            y = rest[1] * (1.0 - 0.08 * p)
            z = rest[2] * (1.0 - p) + (base_z + 4.0 * layer_gap) * p + lift_height * wp.sin(wp.pi * p)
            z = wp.max(z, ground_clearance)
            target = wp.vec3(x, y, z)
            q0[particle] = target
            q1[particle] = target
            vel = wp.vec3(0.0)
            qd0[particle] = vel
            qd1[particle] = vel

    return drive_grip_points


def define_ground_clearance_kernel(wp):
    @wp.kernel
    def enforce_ground_clearance(
        q: wp.array[wp.vec3],
        qd: wp.array[wp.vec3],
        ground_clearance: float,
    ):
        tid = wp.tid()
        pos = q[tid]
        if pos[2] < ground_clearance:
            q[tid] = wp.vec3(pos[0], pos[1], ground_clearance)
            vel = qd[tid]
            if vel[2] < 0.0:
                qd[tid] = wp.vec3(vel[0], vel[1], 0.0)

    return enforce_ground_clearance


def boundary_components(faces: np.ndarray) -> list[list[int]]:
    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b, c in faces:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if i > j:
                i, j = j, i
            edge_counts[(i, j)] += 1

    adjacency: dict[int, list[int]] = defaultdict(list)
    for (i, j), count in edge_counts.items():
        if count == 1:
            adjacency[i].append(j)
            adjacency[j].append(i)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            item = stack.pop()
            component.append(item)
            for nxt in adjacency[item]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)
    return components


def select_grips(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    first_side: str,
    side_grip_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loops = boundary_components(faces)
    if not loops:
        raise RuntimeError("No boundary loops found; cannot infer bottom/collar grips.")

    loop_stats = []
    center_y = float(0.5 * (vertices[:, 1].min() + vertices[:, 1].max()))
    for loop in loops:
        pts = vertices[loop]
        extent = np.ptp(pts, axis=0)
        mean = pts.mean(axis=0)
        loop_stats.append((loop, extent, mean))

    bottom_loop = max(loop_stats, key=lambda item: float(item[2][0]))[0]

    # The neck opening is the small, central boundary loop near the minimum-x side.
    collar_candidates = [
        item
        for item in loop_stats
        if item[0] is not bottom_loop and abs(float(item[2][1] - center_y)) < 0.18 * float(np.ptp(vertices[:, 1]))
    ]
    if not collar_candidates:
        collar_candidates = [item for item in loop_stats if item[0] is not bottom_loop]
    collar_loop = min(collar_candidates, key=lambda item: float(item[2][0]))[0]

    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    length = x_max - x_min
    width = y_max - y_min

    bottom_mask = vertices[:, 0] >= x_max - 0.05 * length
    collar_mask = (vertices[:, 0] <= x_min + 0.12 * length) & (np.abs(vertices[:, 1] - center_y) <= 0.18 * width)
    negative_side_mask = vertices[:, 1] <= y_min + side_grip_width * width
    positive_side_mask = vertices[:, 1] >= y_max - side_grip_width * width

    bottom = np.unique(np.concatenate([np.asarray(bottom_loop, dtype=np.int32), np.nonzero(bottom_mask)[0]])).astype(
        np.int32
    )
    collar = np.unique(np.concatenate([np.asarray(collar_loop, dtype=np.int32), np.nonzero(collar_mask)[0]])).astype(
        np.int32
    )
    collar = np.setdiff1d(collar, bottom, assume_unique=False).astype(np.int32)
    reserved = np.union1d(bottom, collar)
    negative_side = np.setdiff1d(np.nonzero(negative_side_mask)[0], reserved, assume_unique=False).astype(np.int32)
    positive_side = np.setdiff1d(np.nonzero(positive_side_mask)[0], reserved, assume_unique=False).astype(np.int32)
    if first_side == "negative-y":
        first_side_indices = negative_side
        second_side_indices = positive_side
    else:
        first_side_indices = positive_side
        second_side_indices = negative_side
    if len(bottom) == 0 or len(collar) == 0 or len(first_side_indices) == 0 or len(second_side_indices) == 0:
        raise RuntimeError(
            f"Grip selection failed: bottom={len(bottom)} collar={len(collar)} "
            f"first_side={len(first_side_indices)} second_side={len(second_side_indices)}"
        )
    return bottom, collar, first_side_indices, second_side_indices


def place_cloth_near_ground(wp, model, state, *, ground_clearance: float) -> np.ndarray:
    positions = state.particle_q.numpy()
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    xy_center = 0.5 * (bbox_min[:2] + bbox_max[:2])
    positions[:, 0] -= xy_center[0]
    positions[:, 1] -= xy_center[1]
    positions[:, 2] += ground_clearance - bbox_min[2]
    array = wp.array(positions, dtype=wp.vec3)
    state.particle_q = array
    model.particle_q = wp.array(positions, dtype=wp.vec3)
    if state.particle_qd is not None:
        state.particle_qd.zero_()
    return positions


def setup_fold_sim(args: argparse.Namespace) -> SimpleNamespace:
    mesh_path = resolve_path(args.asset)
    mesh = load_mesh(mesh_path)
    report = analyze_mesh(mesh_path, mesh)
    if report.status != "ok":
        raise RuntimeError(f"Mesh report failed: {report.warnings}")

    cloth_params = cloth_params_from_args(args)
    scale = float(report.recommended_scale or 1.0)
    newton, wp, model, state_0, _builder = build_style3d_cloth_model(
        mesh,
        scale=scale,
        device=args.device,
        warp_cache_dir=args.warp_cache_dir,
        style3d_panel_axes=args.style3d_panel_axes,
        start_height=float(args.start_height),
        style3d_fix_panel_winding=not args.no_style3d_fix_panel_winding,
        style3d_clean_nonmanifold=bool(args.style3d_clean_nonmanifold),
        style3d_sew_distance=float(args.style3d_sew_distance),
        style3d_sew_ke=float(args.style3d_sew_ke),
        style3d_sew_kd=float(args.style3d_sew_kd),
        cloth_params=cloth_params,
    )
    if hasattr(model, "set_gravity"):
        model.set_gravity((0.0, 0.0, -9.81))

    requested_clearance = float(args.start_height if args.ground_clearance is None else args.ground_clearance)
    ground_clearance = max(
        requested_clearance,
        float(cloth_params.particle_radius) + float(cloth_params.soft_contact_margin),
    )
    positioned = place_cloth_near_ground(wp, model, state_0, ground_clearance=ground_clearance)
    selection_faces = mesh.style3d_faces if mesh.style3d_faces is not None and len(positioned) == len(mesh.style3d_vertices) else mesh.faces
    bottom_indices, collar_indices, first_side_indices, second_side_indices = select_grips(
        positioned,
        selection_faces,
        first_side=args.side,
        side_grip_width=float(args.side_grip_width),
    )

    state_1 = model.state()
    state_1.particle_q = wp.array(positioned, dtype=wp.vec3)
    state_1.particle_qd.zero_()
    control = model.control()
    contacts = collide_model(model, state_0, soft_contact_margin=cloth_params.soft_contact_margin)

    solver = newton.solvers.SolverStyle3D(
        model,
        iterations=max(1, int(cloth_params.solver_iterations)),
        linear_iterations=max(1, int(cloth_params.linear_iterations)),
    )
    configure_style3d_solver_collision(solver, state_0, cloth_params)

    drive_grip_points = define_drive_kernel(wp)
    enforce_ground_clearance = define_ground_clearance_kernel(wp)
    bottom_rest_np = positioned[bottom_indices].astype(np.float32)
    collar_rest_np = positioned[collar_indices].astype(np.float32)
    first_side_rest_np = positioned[first_side_indices].astype(np.float32)
    second_side_rest_np = positioned[second_side_indices].astype(np.float32)
    bottom_indices_wp = wp.array(bottom_indices, dtype=wp.int32)
    collar_indices_wp = wp.array(collar_indices, dtype=wp.int32)
    first_side_indices_wp = wp.array(first_side_indices, dtype=wp.int32)
    second_side_indices_wp = wp.array(second_side_indices, dtype=wp.int32)
    bottom_rest_wp = wp.array(bottom_rest_np, dtype=wp.vec3)
    collar_rest_wp = wp.array(collar_rest_np, dtype=wp.vec3)
    first_side_rest_wp = wp.array(first_side_rest_np, dtype=wp.vec3)
    second_side_rest_wp = wp.array(second_side_rest_np, dtype=wp.vec3)

    x_min = float(positioned[:, 0].min())
    x_max = float(positioned[:, 0].max())
    y_min = float(positioned[:, 1].min())
    y_max = float(positioned[:, 1].max())
    center_x = 0.5 * (x_min + x_max)
    center_y = 0.5 * (y_min + y_max)
    garment_len = x_max - x_min
    garment_width = y_max - y_min
    z_min = float(positioned[:, 2].min())
    z_max = float(positioned[:, 2].max())
    base_z = z_min + 0.25 * (z_max - z_min)
    layer_gap = max(float(args.layer_gap), 2.2 * float(cloth_params.particle_radius))
    lift_height = float(args.lift_height)
    bottom_target_x = center_x - float(args.center_offset) * garment_len
    collar_target_x = center_x + float(args.center_offset) * garment_len
    first_side_sign = -1.0 if args.side == "negative-y" else 1.0
    second_side_sign = -first_side_sign
    first_side_target_y = center_y - first_side_sign * float(args.side_target_offset) * garment_width
    second_side_target_y = center_y - second_side_sign * float(args.side_target_offset) * garment_width

    print(
        "[FoldProbe] "
        f"first_side={args.side} first_side_grip={len(first_side_indices)} "
        f"second_side_grip={len(second_side_indices)} bottom_grip={len(bottom_indices)} "
        f"collar_grip={len(collar_indices)} center_x={center_x:.4f} "
        f"first_side_target_y={first_side_target_y:.4f} second_side_target_y={second_side_target_y:.4f} "
        f"bottom_target_x={bottom_target_x:.4f} collar_target_x={collar_target_x:.4f} "
        f"layer_gap={layer_gap:.4f} ground_clearance={ground_clearance:.4f}",
        flush=True,
    )

    return SimpleNamespace(
        newton=newton,
        wp=wp,
        model=model,
        state_0=state_0,
        state_1=state_1,
        control=control,
        contacts=contacts,
        cloth_params=cloth_params,
        solver=solver,
        drive_grip_points=drive_grip_points,
        enforce_ground_clearance=enforce_ground_clearance,
        bottom_indices=bottom_indices,
        collar_indices=collar_indices,
        first_side_indices=first_side_indices,
        second_side_indices=second_side_indices,
        bottom_indices_wp=bottom_indices_wp,
        collar_indices_wp=collar_indices_wp,
        first_side_indices_wp=first_side_indices_wp,
        second_side_indices_wp=second_side_indices_wp,
        bottom_rest_wp=bottom_rest_wp,
        collar_rest_wp=collar_rest_wp,
        first_side_rest_wp=first_side_rest_wp,
        second_side_rest_wp=second_side_rest_wp,
        center_x=center_x,
        first_side_target_y=first_side_target_y,
        second_side_target_y=second_side_target_y,
        bottom_target_x=bottom_target_x,
        collar_target_x=collar_target_x,
        lift_height=lift_height,
        base_z=base_z,
        layer_gap=layer_gap,
        ground_clearance=ground_clearance,
        sim_time=0.0,
    )


def apply_grip_trajectory(sim: SimpleNamespace, args: argparse.Namespace, time_value: float, sim_dt: float) -> None:
    wp = sim.wp
    wp.launch(
        sim.drive_grip_points,
        dim=(
            len(sim.first_side_indices)
            + len(sim.second_side_indices)
            + len(sim.bottom_indices)
            + len(sim.collar_indices)
        ),
        inputs=[
            time_value,
            sim_dt,
            sim.first_side_indices_wp,
            sim.first_side_rest_wp,
            sim.second_side_indices_wp,
            sim.second_side_rest_wp,
            sim.bottom_indices_wp,
            sim.bottom_rest_wp,
            sim.collar_indices_wp,
            sim.collar_rest_wp,
            sim.state_0.particle_q,
            sim.state_0.particle_qd,
            sim.state_1.particle_q,
            sim.state_1.particle_qd,
            sim.center_x,
            sim.first_side_target_y,
            sim.second_side_target_y,
            sim.bottom_target_x,
            sim.collar_target_x,
            float(args.side_start),
            float(args.second_side_start),
            float(args.bottom_start),
            float(args.collar_start),
            float(args.fold_duration),
            float(args.side_fold_duration),
            sim.lift_height,
            sim.base_z,
            sim.layer_gap,
            sim.ground_clearance,
        ],
    )


def enforce_fold_ground_clearance(sim: SimpleNamespace) -> None:
    for state in (sim.state_0, sim.state_1):
        sim.wp.launch(
            sim.enforce_ground_clearance,
            dim=state.particle_q.shape[0],
            inputs=[state.particle_q, state.particle_qd, sim.ground_clearance],
        )


def step_fold_sim(sim: SimpleNamespace, args: argparse.Namespace, sim_dt: float) -> None:
    t_next = sim.sim_time + sim_dt
    sim.state_0.clear_forces()
    apply_grip_trajectory(sim, args, t_next, sim_dt)
    if hasattr(sim.solver, "rebuild_bvh"):
        sim.solver.rebuild_bvh(sim.state_0)
    sim.solver.step(sim.state_0, sim.state_1, sim.control, sim.contacts, sim_dt)
    sim.state_0, sim.state_1 = sim.state_1, sim.state_0
    sim.sim_time = t_next
    enforce_fold_ground_clearance(sim)
    apply_grip_trajectory(sim, args, sim.sim_time, sim_dt)


def step_fold_frame(sim: SimpleNamespace, args: argparse.Namespace, substeps: int, sim_dt: float) -> None:
    apply_grip_trajectory(sim, args, sim.sim_time, sim_dt)
    sim.contacts = collide_model(
        sim.model,
        sim.state_0,
        sim.contacts,
        soft_contact_margin=sim.cloth_params.soft_contact_margin,
    )
    for _ in range(substeps):
        step_fold_sim(sim, args, sim_dt)


def run_headless(args: argparse.Namespace) -> None:
    sim = setup_fold_sim(args)
    frame_dt = 1.0 / float(args.view_fps)
    substeps = max(1, int(args.view_substeps))
    sim_dt = frame_dt / substeps
    for _ in range(max(0, int(args.headless_frames))):
        step_fold_frame(sim, args, substeps, sim_dt)
    sim.wp.synchronize()
    q = sim.state_0.particle_q.numpy()
    print(
        "[FoldProbe] headless_done "
        f"frames={int(args.headless_frames)} time={sim.sim_time:.3f} "
        f"finite={bool(np.isfinite(q).all())} bbox_min={q.min(axis=0).tolist()} bbox_max={q.max(axis=0).tolist()}",
        flush=True,
    )


def run_viewer(args: argparse.Namespace) -> None:
    sim = setup_fold_sim(args)
    newton = sim.newton
    wp = sim.wp

    import newton.viewer  # noqa: PLC0415

    viewer = newton.viewer.ViewerGL()
    viewer.show_particles = bool(args.show_particles)
    viewer.show_triangles = True
    viewer.set_model(sim.model)
    viewer.set_camera(wp.vec3(0.0, -1.85, 0.75), pitch=-28.0, yaw=90.0)
    print("[FoldProbe] Viewer opened. Close the window or press ESC to exit.", flush=True)

    frame_dt = 1.0 / float(args.view_fps)
    substeps = max(1, int(args.view_substeps))
    sim_dt = frame_dt / substeps

    try:
        while viewer.is_running():
            if not viewer.is_paused():
                step_fold_frame(sim, args, substeps, sim_dt)

            viewer.begin_frame(sim.sim_time)
            viewer.log_state(sim.state_0)
            viewer.log_contacts(sim.contacts, sim.state_0)
            viewer.end_frame()
    finally:
        viewer.close()


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim = setup_fold_sim(args)

        self.viewer.show_particles = bool(args.show_particles)
        self.viewer.show_triangles = True
        self.viewer.set_model(self.sim.model)
        self.viewer.set_camera(self.sim.wp.vec3(0.0, -1.85, 0.75), pitch=-28.0, yaw=90.0)

    @staticmethod
    def create_parser():
        return create_parser()

    def step(self):
        step_fold_frame(self.sim, self.args, self.sim_substeps, self.sim_dt)

    def render(self):
        self.viewer.begin_frame(self.sim.sim_time)
        self.viewer.log_state(self.sim.state_0)
        self.viewer.log_contacts(self.sim.contacts, self.sim.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.sim.state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            raise ValueError("Fold probe particle positions are not finite.")
        if float(q[:, 2].min()) < self.sim.ground_clearance - 1.0e-5:
            raise ValueError("Fold probe particle positions fell below ground clearance.")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument(
        "asset",
        nargs="?",
        default="style3d/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt.obj",
    )
    parser.add_argument("--warp-cache-dir", default="/tmp/warp_cache")
    parser.add_argument("--view-fps", type=float, default=60.0)
    parser.add_argument("--view-substeps", type=int, default=10)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument("--linear-iterations", type=int, default=10)
    parser.add_argument("--cloth-density", type=float, default=0.3)
    parser.add_argument("--particle-radius", type=float, default=5.0e-3)
    parser.add_argument("--tri-ka", type=float, default=100.0)
    parser.add_argument("--tri-kd", type=float, default=1.5e-6)
    parser.add_argument("--tri-aniso-ke", type=parse_vec3_arg, default=(100.0, 100.0, 10.0))
    parser.add_argument("--edge-aniso-ke", type=parse_vec3_arg, default=(2.0e-5, 1.0e-5, 5.0e-6))
    parser.add_argument("--edge-kd", type=float, default=1.0e-3)
    parser.add_argument("--soft-contact-margin", type=float, default=3.5e-3)
    parser.add_argument("--soft-contact-ke", type=float, default=10.0)
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6)
    parser.add_argument("--soft-contact-mu", type=float, default=0.2)
    parser.add_argument("--step-dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--no-style3d-self-collision", action="store_true")
    parser.add_argument("--style3d-collision-radius", type=float, default=3.0e-3)
    parser.add_argument("--style3d-collision-stiff-vf", type=float, default=0.5)
    parser.add_argument("--style3d-collision-stiff-ee", type=float, default=0.1)
    parser.add_argument("--style3d-collision-stiff-ef", type=float, default=1.0)
    parser.add_argument("--style3d-panel-axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--start-height", type=float, default=0.05)
    parser.add_argument("--ground-clearance", type=float, default=None)
    parser.add_argument("--no-style3d-fix-panel-winding", action="store_true")
    parser.add_argument("--style3d-clean-nonmanifold", action="store_true")
    parser.add_argument("--style3d-sew-distance", type=float, default=0.0)
    parser.add_argument("--style3d-sew-ke", type=float, default=100.0)
    parser.add_argument("--style3d-sew-kd", type=float, default=1.0e-3)
    parser.add_argument("--side-start", type=float, default=0.25)
    parser.add_argument("--second-side-start", type=float, default=2.25)
    parser.add_argument("--bottom-start", type=float, default=4.25)
    parser.add_argument("--collar-start", type=float, default=6.25)
    parser.add_argument("--fold-duration", type=float, default=1.55)
    parser.add_argument("--side-fold-duration", type=float, default=1.75)
    parser.add_argument("--lift-height", type=float, default=0.18)
    parser.add_argument("--center-offset", type=float, default=0.04)
    parser.add_argument("--side", choices=["positive-y", "negative-y"], default="positive-y")
    parser.add_argument("--side-grip-width", type=float, default=0.12)
    parser.add_argument("--side-target-offset", type=float, default=0.28)
    parser.add_argument("--layer-gap", type=float, default=0.0)
    parser.add_argument("--show-particles", action="store_true")
    parser.add_argument("--headless-frames", type=int, default=0, help="Run this many viewer frames without opening GL")
    return parser


def parse_args() -> argparse.Namespace:
    return create_parser().parse_args()


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if int(args.headless_frames) > 0:
        run_headless(args)
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
