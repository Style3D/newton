# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop the local Style3D silk cloth asset for material tuning.

Command:
    python -m newton.examples cloth_style3d_silk_drop --device cuda:0
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import warp as wp

import newton
import newton.examples
from style3d.examples._style3d_asset_probe import (
    analyze_mesh,
    build_newton_cloth_model,
    cloth_params_from_args,
    configure_style3d_solver_collision,
    load_mesh,
    parse_vec3_arg,
    resolve_path,
)


DEFAULT_SILK_ASSET = "style3d/examples/assets/style3d_probe/cloth/silk_35/silk_35_aligned.obj"


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

        asset_path = resolve_path(args.asset)
        mesh = load_mesh(asset_path)
        panel_scale = float(args.style3d_panel_scale)
        if mesh.panel_vertices is not None and panel_scale != 1.0:
            mesh = replace(
                mesh,
                panel_vertices=mesh.panel_vertices * panel_scale,
                style3d_panel_vertices=(
                    None if mesh.style3d_panel_vertices is None else mesh.style3d_panel_vertices * panel_scale
                ),
                panel_coordinate_source=f"{mesh.panel_coordinate_source}*{panel_scale:g}",
            )
        report = analyze_mesh(asset_path, mesh)
        if report.status != "ok":
            raise ValueError(f"Cannot load silk asset {asset_path}: {report.warnings}")

        self.cloth_params = cloth_params_from_args(args)
        scale = float(args.scale) if args.scale is not None else float(report.recommended_scale or 1.0)
        print(
            "[SilkDrop] "
            f"asset={report.path}, scale={scale:g}, verts={report.vertex_count}, tris={report.triangle_count}, "
            f"components={report.component_count}, boundary_edges={report.boundary_edge_count}, "
            f"nonmanifold_edges={report.nonmanifold_edge_count}, panel_source={report.panel_coordinate_source}",
            flush=True,
        )
        for warning in report.warnings[:6]:
            print(f"[SilkDrop] asset warning: {warning}", flush=True)

        self.newton, self.wp, self.model, self.state_0, _builder = build_newton_cloth_model(
            mesh,
            backend="style3d",
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
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.85, 0.7), pitch=-28.0, yaw=90.0)

        print(
            "[SilkDrop] "
            f"substeps={self.sim_substeps}, sim_dt={self.sim_dt:g}s, "
            f"soft_contact_margin={self.cloth_params.soft_contact_margin:g}, "
            f"self_collision_radius={self.cloth_params.style3d_collision_radius:g}",
            flush=True,
        )
        self.update_stats()

    @staticmethod
    def create_parser():
        return create_parser()

    def simulate(self):
        for substep in range(self.sim_substeps):
            if substep % self.collide_every_substeps == 0:
                self.collision_pipeline.collide(
                    self.state_0,
                    self.contacts,
                    soft_contact_margin=float(self.cloth_params.soft_contact_margin),
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

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
            "[SilkDrop] "
            f"frame={self.frame} time={self.sim_time:.3f} "
            f"z=[{s['z_min']:.4f}, {s['z_max']:.4f}] max_speed={s['max_speed']:.4f}",
            flush=True,
        )

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
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
            raise ValueError("Silk drop model has no particles.")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Silk drop particle state is not finite.")
        if float(np.min(q[:, 2])) < -0.25:
            raise ValueError(f"Silk penetrated too far below ground: z_min={float(np.min(q[:, 2])):.4f} m")
        max_speed = float(np.max(np.linalg.norm(qd, axis=1)))
        if max_speed > 100.0:
            raise ValueError(f"Silk simulation appears unstable: max_speed={max_speed:.3f} m/s")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    newton.examples.add_broad_phase_arg(parser)
    parser.add_argument("asset", nargs="?", default=DEFAULT_SILK_ASSET, help="OBJ/USD silk asset to import")
    parser.add_argument("--warp-cache-dir", default="/tmp/warp_cache")
    parser.add_argument("--scale", type=float, default=None, help="Asset scale override; default uses probe estimate")
    parser.add_argument("--view-fps", type=float, default=60.0, help="Simulation frames advanced per viewer frame")
    parser.add_argument("--view-substeps", type=int, default=8, help="Style3D solver substeps per viewer frame")
    parser.add_argument("--collide-every-substeps", type=int, default=1, help="Refresh body/ground contacts every N substeps")
    parser.add_argument("--start-height", type=float, default=0.45, help="Lowest silk vertex height above the ground [m]")
    parser.add_argument("--show-particles", action="store_true", help="Render cloth particles")
    parser.add_argument("--log-interval", type=int, default=30, help="Print state stats every N frames; 0 disables logging")

    parser.add_argument("--solver-iterations", type=int, default=5, help="Style3D nonlinear iterations per substep")
    parser.add_argument("--linear-iterations", type=int, default=10, help="Style3D linear iterations per substep")
    parser.set_defaults(step_dt=1.0 / 120.0)

    parser.add_argument("--cloth-density", type=float, default=0.093, help="Cloth areal density")
    parser.add_argument("--particle-radius", type=float, default=0.002, help="Cloth particle/contact radius [m]")
    parser.add_argument("--tri-ka", type=float, default=40.0, help="Triangle area preservation stiffness")
    parser.add_argument("--tri-kd", type=float, default=1.0e-5, help="Triangle damping")
    parser.add_argument(
        "--tri-aniso-ke",
        type=parse_vec3_arg,
        default=(40.0, 25.0, 3.0),
        help="Style3D anisotropic stretch/shear stiffness as weft,warp,shear",
    )
    parser.add_argument(
        "--edge-aniso-ke",
        type=parse_vec3_arg,
        default=(2.5e-7, 1.25e-7, 2.6e-7),
        help="Style3D anisotropic bending stiffness as weft,warp,shear",
    )
    parser.add_argument("--edge-kd", type=float, default=1.0e-4, help="Edge/bending damping")

    parser.add_argument("--soft-contact-margin", type=float, default=0.002, help="Ground/body soft contact margin [m]")
    parser.add_argument("--soft-contact-ke", type=float, default=10.0, help="Ground/body soft contact stiffness")
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6, help="Ground/body soft contact damping")
    parser.add_argument("--soft-contact-mu", type=float, default=0.35, help="Ground/body soft contact friction")

    parser.add_argument("--no-style3d-self-collision", action="store_true", help="Disable Style3D cloth self-collision")
    parser.add_argument("--style3d-collision-radius", type=float, default=0.0015, help="Style3D self-collision radius [m]")
    parser.add_argument("--style3d-collision-stiff-vf", type=float, default=0.05, help="Vertex-face self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ee", type=float, default=0.02, help="Edge-edge self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ef", type=float, default=0.1, help="Edge-face untangling stiffness")

    parser.add_argument("--style3d-panel-axes", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument(
        "--style3d-panel-scale",
        type=float,
        default=0.001,
        help="Scale applied to OBJ vt panel coordinates before building Style3D cloth",
    )
    parser.add_argument("--no-style3d-fix-panel-winding", action="store_true")
    parser.add_argument(
        "--style3d-clean-nonmanifold",
        action=argparse.BooleanOptionalAction,
        default=True,
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
