# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a local Style3D garment asset for material tuning.

Command:
    python -m newton.examples cloth_style3d_garment_drop --device cuda:0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np
import warp as wp

import newton
import newton.examples
from style3d.examples.tools.mesh_asset_utils import analyze_mesh, load_mesh, resolve_path
from style3d.examples.tools.style3d_cloth_utils import (
    build_style3d_cloth_model,
    cloth_params_from_args,
    configure_style3d_solver_collision,
    parse_vec3_arg,
)


DEFAULT_GARMENT_ASSET = "style3d/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt.obj"


def define_fold_kernel(wp):
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
    def drive_fold_grips(
        time: float,
        positive_indices: wp.array[wp.int32],
        positive_rest: wp.array[wp.vec3],
        negative_indices: wp.array[wp.int32],
        negative_rest: wp.array[wp.vec3],
        bottom_indices: wp.array[wp.int32],
        bottom_rest: wp.array[wp.vec3],
        q0: wp.array[wp.vec3],
        qd0: wp.array[wp.vec3],
        q1: wp.array[wp.vec3],
        qd1: wp.array[wp.vec3],
        center_x: float,
        positive_target_y: float,
        negative_target_y: float,
        bottom_target_x: float,
        positive_start: float,
        negative_start: float,
        bottom_start: float,
        duration: float,
        lift_height: float,
        base_z: float,
        layer_gap: float,
        ground_clearance: float,
    ):
        tid = wp.tid()
        positive_count = positive_indices.shape[0]
        negative_count = negative_indices.shape[0]
        bottom_count = bottom_indices.shape[0]

        if tid < positive_count:
            particle = positive_indices[tid]
            rest = positive_rest[tid]
            if time >= positive_start and time <= positive_start + duration:
                p = fold_progress(time, positive_start, duration)
                y = rest[1] * (1.0 - p) + positive_target_y * p
                z = rest[2] * (1.0 - p) + (base_z + layer_gap) * p + lift_height * wp.sin(wp.pi * p)
                z = wp.max(z, ground_clearance + layer_gap)
                target = wp.vec3(rest[0], y, z)
                q0[particle] = target
                q1[particle] = target
                qd0[particle] = wp.vec3(0.0)
                qd1[particle] = wp.vec3(0.0)
        elif tid < positive_count + negative_count:
            j = tid - positive_count
            particle = negative_indices[j]
            rest = negative_rest[j]
            if time >= negative_start and time <= negative_start + duration:
                p = fold_progress(time, negative_start, duration)
                y = rest[1] * (1.0 - p) + negative_target_y * p
                z = rest[2] * (1.0 - p) + (base_z + 2.0 * layer_gap) * p + lift_height * wp.sin(wp.pi * p)
                z = wp.max(z, ground_clearance + 2.0 * layer_gap)
                target = wp.vec3(rest[0], y, z)
                q0[particle] = target
                q1[particle] = target
                qd0[particle] = wp.vec3(0.0)
                qd1[particle] = wp.vec3(0.0)
        else:
            j = tid - positive_count - negative_count
            particle = bottom_indices[j]
            rest = bottom_rest[j]
            if time >= bottom_start and time <= bottom_start + duration:
                p = fold_progress(time, bottom_start, duration)
                x = rest[0] * (1.0 - p) + bottom_target_x * p
                z = rest[2] * (1.0 - p) + (base_z + 3.0 * layer_gap) * p + lift_height * wp.sin(wp.pi * p)
                z = wp.max(z, ground_clearance + 3.0 * layer_gap)
                target = wp.vec3(x, rest[1], z)
                q0[particle] = target
                q1[particle] = target
                qd0[particle] = wp.vec3(0.0)
                qd1[particle] = wp.vec3(0.0)

    return drive_fold_grips


def maybe_use_expanded_topology(mesh, enabled: bool):
    if not enabled:
        return mesh
    if mesh.style3d_vertices is None or mesh.style3d_faces is None or mesh.style3d_panel_vertices is None:
        raise ValueError("This asset does not have OBJ face-varying UV topology to expand.")
    return replace(
        mesh,
        vertices=mesh.style3d_vertices,
        faces=mesh.style3d_faces,
        face_counts=[3] * int(mesh.style3d_faces.shape[0]),
        panel_vertices=mesh.style3d_panel_vertices,
        panel_faces=mesh.style3d_faces,
        style3d_vertices=None,
        style3d_faces=None,
        style3d_panel_vertices=None,
        panel_coordinate_source=f"{mesh.panel_coordinate_source}_physical",
    )


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


def select_fold_grips(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    side_grip_width: float,
    bottom_grip_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min = float(vertices[:, 0].min())
    x_max = float(vertices[:, 0].max())
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    length = x_max - x_min
    width = y_max - y_min

    loops = boundary_components(faces)
    if loops:
        bottom_loop = max(loops, key=lambda loop: float(vertices[loop, 0].mean()))
        bottom = np.asarray(bottom_loop, dtype=np.int32)
    else:
        bottom = np.asarray([], dtype=np.int32)

    bottom_mask = vertices[:, 0] >= x_max - bottom_grip_length * length
    bottom = np.unique(np.concatenate([bottom, np.nonzero(bottom_mask)[0]])).astype(np.int32)

    positive_mask = vertices[:, 1] >= y_max - side_grip_width * width
    negative_mask = vertices[:, 1] <= y_min + side_grip_width * width
    positive = np.setdiff1d(np.nonzero(positive_mask)[0], bottom, assume_unique=False).astype(np.int32)
    negative = np.setdiff1d(np.nonzero(negative_mask)[0], bottom, assume_unique=False).astype(np.int32)
    if len(positive) == 0 or len(negative) == 0 or len(bottom) == 0:
        raise RuntimeError(
            f"Fold grip selection failed: positive={len(positive)} negative={len(negative)} bottom={len(bottom)}"
        )
    return positive, negative, bottom


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.collide_every_substeps = max(1, int(args.collide_every_substeps))
        self.last_stats: dict[str, float] = {}
        self.fold_enabled = not bool(args.no_fold)
        self.fold_initialized = False
        self.fold_stage = "settle"

        asset_path = resolve_path(args.asset)
        mesh = load_mesh(asset_path)
        source_report = analyze_mesh(asset_path, mesh)
        if source_report.status != "ok":
            raise ValueError(f"Cannot load garment asset {asset_path}: {source_report.warnings}")

        mesh = maybe_use_expanded_topology(mesh, bool(args.use_expanded_topology))
        self.fold_faces = mesh.faces.copy()
        report = analyze_mesh(asset_path, mesh)
        if report.status != "ok":
            raise ValueError(f"Cannot build garment asset {asset_path}: {report.warnings}")

        self.cloth_params = cloth_params_from_args(args)
        scale = float(args.scale) if args.scale is not None else float(report.recommended_scale or 1.0)
        print(
            "[GarmentDrop] "
            f"asset={source_report.path}, scale={scale:g}, verts={report.vertex_count}, "
            f"tris={report.triangle_count}, components={report.component_count}, "
            f"boundary_edges={report.boundary_edge_count}, nonmanifold_edges={report.nonmanifold_edge_count}, "
            f"panel_source={report.panel_coordinate_source}",
            flush=True,
        )
        if args.use_expanded_topology:
            print("[GarmentDrop] using expanded OBJ (vertex, uv) topology as physical cloth.", flush=True)
        for warning in report.warnings[:6]:
            print(f"[GarmentDrop] asset warning: {warning}", flush=True)

        self.newton, self.wp, self.model, self.state_0, _builder = build_style3d_cloth_model(
            mesh,
            scale=scale,
            device=args.device,
            warp_cache_dir=args.warp_cache_dir,
            style3d_panel_axes=args.style3d_panel_axes,
            start_height=float(args.start_height),
            style3d_fix_panel_winding=not bool(args.no_style3d_fix_panel_winding),
            style3d_clean_nonmanifold=bool(args.style3d_clean_nonmanifold),
            style3d_min_triangle_area_ratio=float(args.style3d_min_triangle_area_ratio),
            style3d_sew_distance=float(args.style3d_sew_distance),
            style3d_sew_ke=float(args.style3d_sew_ke),
            style3d_sew_kd=float(args.style3d_sew_kd),
            cloth_params=self.cloth_params,
        )

        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = self.newton.solvers.SolverStyle3D(
            self.model,
            iterations=max(1, int(self.cloth_params.solver_iterations)),
            linear_iterations=max(1, int(self.cloth_params.linear_iterations)),
            enable_translation_preconditioner=bool(args.style3d_translation_preconditioner),
        )
        configure_style3d_solver_collision(self.solver, self.state_0, self.cloth_params)

        self.collision_pipeline = self.newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
            soft_contact_margin=float(self.cloth_params.soft_contact_margin),
        )
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.show_particles = bool(args.show_particles)
        self.viewer.show_triangles = True
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -1.3, 0.85), pitch=-25.0, yaw=90.0)
        self.drive_fold_grips = define_fold_kernel(self.wp)

        print(
            "[GarmentDrop] "
            f"substeps={self.sim_substeps}, sim_dt={self.sim_dt:g}s, "
            f"soft_contact_margin={self.cloth_params.soft_contact_margin:g}, "
            f"self_collision_radius={self.cloth_params.style3d_collision_radius:g}, "
            f"fold={'on' if self.fold_enabled else 'off'}",
            flush=True,
        )
        self.update_stats()

    @staticmethod
    def create_parser():
        return create_parser()

    def initialize_fold(self):
        positions = self.state_0.particle_q.numpy().astype(np.float32)
        positive, negative, bottom = select_fold_grips(
            positions,
            self.fold_faces,
            side_grip_width=float(self.args.fold_side_grip_width),
            bottom_grip_length=float(self.args.fold_bottom_grip_length),
        )

        x_min = float(positions[:, 0].min())
        x_max = float(positions[:, 0].max())
        y_min = float(positions[:, 1].min())
        y_max = float(positions[:, 1].max())
        z_min = float(positions[:, 2].min())
        z_max = float(positions[:, 2].max())
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        width = y_max - y_min
        length = x_max - x_min

        self.fold_positive_indices = positive
        self.fold_negative_indices = negative
        self.fold_bottom_indices = bottom
        self.fold_positive_indices_wp = self.wp.array(positive, dtype=self.wp.int32)
        self.fold_negative_indices_wp = self.wp.array(negative, dtype=self.wp.int32)
        self.fold_bottom_indices_wp = self.wp.array(bottom, dtype=self.wp.int32)
        self.fold_positive_rest_wp = self.wp.array(positions[positive], dtype=self.wp.vec3)
        self.fold_negative_rest_wp = self.wp.array(positions[negative], dtype=self.wp.vec3)
        self.fold_bottom_rest_wp = self.wp.array(positions[bottom], dtype=self.wp.vec3)
        self.fold_center_x = center_x
        self.fold_positive_target_y = center_y - float(self.args.fold_side_target_offset) * width
        self.fold_negative_target_y = center_y + float(self.args.fold_side_target_offset) * width
        self.fold_bottom_target_x = center_x - float(self.args.fold_bottom_target_offset) * length
        self.fold_ground_clearance = max(
            float(self.args.fold_ground_clearance),
            float(self.cloth_params.particle_radius) + float(self.cloth_params.soft_contact_margin),
        )
        self.fold_base_z = max(
            z_min + float(self.args.fold_base_height_ratio) * (z_max - z_min),
            self.fold_ground_clearance,
        )
        self.fold_layer_gap = max(
            float(self.args.fold_layer_gap),
            2.5 * float(self.cloth_params.particle_radius),
            2.5 * float(self.cloth_params.style3d_collision_radius),
        )
        self.fold_lift_height = float(self.args.fold_lift_height)
        self.fold_t0 = self.sim_time
        self.fold_initialized = True
        self.fold_stage = "fold_positive_side"
        print(
            "[GarmentDrop] fold initialized "
            f"time={self.fold_t0:.3f} positive={len(positive)} negative={len(negative)} bottom={len(bottom)} "
            f"positive_target_y={self.fold_positive_target_y:.4f} "
            f"negative_target_y={self.fold_negative_target_y:.4f} bottom_target_x={self.fold_bottom_target_x:.4f} "
            f"layer_gap={self.fold_layer_gap:.4f} ground_clearance={self.fold_ground_clearance:.4f}",
            flush=True,
        )

    def maybe_update_fold_stage(self):
        if not self.fold_enabled:
            self.fold_stage = "drop"
            return
        if not self.fold_initialized:
            self.fold_stage = "settle"
            if self.sim_time >= float(self.args.fold_settle_time):
                self.initialize_fold()
            return
        elapsed = self.sim_time - self.fold_t0
        duration = float(self.args.fold_duration)
        hold = float(self.args.fold_hold_time)
        if elapsed < duration:
            self.fold_stage = "fold_positive_side"
        elif elapsed < duration + hold:
            self.fold_stage = "hold_positive_side"
        elif elapsed < 2.0 * duration + hold:
            self.fold_stage = "fold_negative_side"
        elif elapsed < 2.0 * duration + 2.0 * hold:
            self.fold_stage = "hold_negative_side"
        elif elapsed < 3.0 * duration + 2.0 * hold:
            self.fold_stage = "fold_bottom"
        else:
            self.fold_stage = "hold_fold"

    def apply_fold_trajectory(self):
        if not self.fold_initialized:
            return
        duration = float(self.args.fold_duration)
        hold = float(self.args.fold_hold_time)
        positive_start = self.fold_t0
        negative_start = self.fold_t0 + duration + hold
        bottom_start = self.fold_t0 + 2.0 * duration + 2.0 * hold
        self.wp.launch(
            self.drive_fold_grips,
            dim=len(self.fold_positive_indices) + len(self.fold_negative_indices) + len(self.fold_bottom_indices),
            inputs=[
                self.sim_time,
                self.fold_positive_indices_wp,
                self.fold_positive_rest_wp,
                self.fold_negative_indices_wp,
                self.fold_negative_rest_wp,
                self.fold_bottom_indices_wp,
                self.fold_bottom_rest_wp,
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.state_1.particle_q,
                self.state_1.particle_qd,
                self.fold_center_x,
                self.fold_positive_target_y,
                self.fold_negative_target_y,
                self.fold_bottom_target_x,
                positive_start,
                negative_start,
                bottom_start,
                duration,
                self.fold_lift_height,
                self.fold_base_z,
                self.fold_layer_gap,
                self.fold_ground_clearance,
            ],
        )

    def simulate(self):
        self.maybe_update_fold_stage()
        self.apply_fold_trajectory()
        for substep in range(self.sim_substeps):
            if substep % self.collide_every_substeps == 0:
                self.collision_pipeline.collide(
                    self.state_0,
                    self.contacts,
                    soft_contact_margin=float(self.cloth_params.soft_contact_margin),
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.apply_fold_trajectory()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.apply_fold_trajectory()

    def update_stats(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        speed = np.linalg.norm(qd, axis=1) if len(qd) else np.array([0.0])
        self.last_stats = {
            "z_min": float(np.min(q[:, 2])) if len(q) else 0.0,
            "z_max": float(np.max(q[:, 2])) if len(q) else 0.0,
            "max_speed": float(np.max(speed)),
        }

    def log_stats(self):
        s = self.last_stats
        print(
            "[GarmentDrop] "
            f"frame={self.frame} time={self.sim_time:.3f} "
            f"stage={self.fold_stage} z=[{s['z_min']:.4f}, {s['z_max']:.4f}] "
            f"max_speed={s['max_speed']:.4f}",
            flush=True,
        )

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
            self.wp.synchronize()
            self.update_stats()
            self.log_stats()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        s = self.last_stats
        ui.text(f"Frame: {self.frame}")
        ui.text(f"Time: {self.sim_time:.3f} s")
        ui.text(f"Stage: {self.fold_stage}")
        ui.text(f"z min/max: {s.get('z_min', 0.0):.4f} / {s.get('z_max', 0.0):.4f}")
        ui.text(f"max speed: {s.get('max_speed', 0.0):.4f} m/s")
        ui.separator()
        ui.text(f"density: {self.cloth_params.density:g}")
        ui.text(f"tri aniso ke: {self.cloth_params.tri_aniso_ke}")
        ui.text(f"edge aniso ke: {self.cloth_params.edge_aniso_ke}")
        ui.text(f"collision radius: {self.cloth_params.style3d_collision_radius:g}")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if len(q) == 0:
            raise ValueError("Garment drop model has no particles.")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Garment drop particle state is not finite.")
        if float(np.min(q[:, 2])) < -0.25:
            raise ValueError(f"Garment penetrated too far below ground: z_min={float(np.min(q[:, 2])):.4f} m")
        max_speed = float(np.max(np.linalg.norm(qd, axis=1)))
        if max_speed > 100.0:
            raise ValueError(f"Garment simulation appears unstable: max_speed={max_speed:.3f} m/s")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    newton.examples.add_broad_phase_arg(parser)
    parser.add_argument("asset", nargs="?", default=DEFAULT_GARMENT_ASSET, help="OBJ/USD garment asset to import")
    parser.add_argument("--warp-cache-dir", default="/tmp/warp_cache")
    parser.add_argument("--scale", type=float, default=None, help="Asset scale override; default uses probe estimate")
    parser.add_argument("--use-expanded-topology", action="store_true", help="Use OBJ (vertex, uv) pairs as physical particles")
    parser.add_argument("--view-fps", type=float, default=60.0, help="Simulation frames advanced per viewer frame")
    parser.add_argument("--view-substeps", type=int, default=8, help="Style3D solver substeps per viewer frame")
    parser.add_argument("--collide-every-substeps", type=int, default=1, help="Refresh body/ground contacts every N substeps")
    parser.add_argument("--start-height", type=float, default=0.35, help="Lowest garment vertex height above the ground [m]")
    parser.add_argument("--show-particles", action="store_true", help="Render cloth particles")
    parser.add_argument("--log-interval", type=int, default=30, help="Print state stats every N frames; 0 disables logging")
    parser.add_argument("--no-fold", action="store_true", help="Disable scripted folding and only drop the garment")
    parser.add_argument("--fold-settle-time", type=float, default=2.0, help="Free-fall/settle time before folding starts [s]")
    parser.add_argument("--fold-duration", type=float, default=2.0, help="Duration of each scripted lift-and-fold segment [s]")
    parser.add_argument(
        "--fold-hold-time",
        type=float,
        default=0.5,
        help="Free-fall wait time between scripted fold segments after releasing the previous grip [s]",
    )
    parser.add_argument("--fold-side-grip-width", type=float, default=0.10, help="Fraction of garment width used as each side grip")
    parser.add_argument("--fold-bottom-grip-length", type=float, default=0.06, help="Fraction of garment length used as the bottom grip")
    parser.add_argument("--fold-side-target-offset", type=float, default=0.25, help="Side fold target offset as fraction of garment width")
    parser.add_argument("--fold-bottom-target-offset", type=float, default=0.04, help="Bottom fold target offset as fraction of garment length")
    parser.add_argument("--fold-lift-height", type=float, default=0.22, help="Arc lift height for scripted grips [m]")
    parser.add_argument("--fold-layer-gap", type=float, default=0.02, help="Vertical spacing between scripted fold layers [m]")
    parser.add_argument("--fold-ground-clearance", type=float, default=0.015, help="Minimum scripted grip height above ground [m]")
    parser.add_argument("--fold-base-height-ratio", type=float, default=0.25, help="Base fold height within current garment bbox")

    parser.add_argument("--solver-iterations", type=int, default=5, help="Style3D nonlinear iterations per substep")
    parser.add_argument("--linear-iterations", type=int, default=10, help="Style3D linear iterations per substep")
    parser.add_argument(
        "--style3d-translation-preconditioner",
        action="store_true",
        help="Enable the Style3D coarse translation PCG preconditioner",
    )
    parser.set_defaults(step_dt=1.0 / 120.0)

    parser.add_argument("--cloth-density", type=float, default=0.3, help="Cloth areal density")
    parser.add_argument("--particle-radius", type=float, default=0.003, help="Cloth particle/contact radius [m]")
    parser.add_argument("--tri-ka", type=float, default=200.0, help="Triangle area preservation stiffness")
    parser.add_argument("--tri-kd", type=float, default=1.0e-5, help="Triangle damping")
    parser.add_argument(
        "--tri-aniso-ke",
        type=parse_vec3_arg,
        default=(200.0, 200.0, 50.0),
        help="Style3D anisotropic stretch/shear stiffness as weft,warp,shear",
    )
    parser.add_argument(
        "--edge-aniso-ke",
        type=parse_vec3_arg,
        default=(5.0e-6, 5.0e-6, 5.0e-6),
        help="Style3D anisotropic bending stiffness as weft,warp,shear",
    )
    parser.add_argument("--edge-kd", type=float, default=1.0e-4, help="Edge/bending damping")

    parser.add_argument("--soft-contact-margin", type=float, default=0.003, help="Ground/body soft contact margin [m]")
    parser.add_argument("--soft-contact-ke", type=float, default=10.0, help="Ground/body soft contact stiffness")
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6, help="Ground/body soft contact damping")
    parser.add_argument("--soft-contact-mu", type=float, default=0.2, help="Ground/body soft contact friction")

    parser.add_argument("--no-style3d-self-collision", action="store_true", help="Disable Style3D cloth self-collision")
    parser.add_argument("--style3d-collision-radius", type=float, default=0.002, help="Style3D self-collision radius [m]")
    parser.add_argument("--style3d-collision-stiff-vf", type=float, default=0.05, help="Vertex-face self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ee", type=float, default=0.02, help="Edge-edge self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ef", type=float, default=0.1, help="Edge-face untangling stiffness")

    parser.add_argument("--style3d-panel-axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--no-style3d-fix-panel-winding", action="store_true")
    parser.add_argument(
        "--style3d-clean-nonmanifold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop duplicate triangles and tiny area triangles before Style3D build",
    )
    parser.add_argument(
        "--style3d-min-triangle-area-ratio",
        type=float,
        default=0.05,
        help="Drop triangles with 3D area below this fraction of the median positive triangle area",
    )
    parser.add_argument("--style3d-sew-distance", type=float, default=0.0)
    parser.add_argument("--style3d-sew-ke", type=float, default=100.0)
    parser.add_argument("--style3d-sew-kd", type=float, default=1.0e-3)
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
