# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a local Style3D bag asset for material and contact tuning.

Command:
    python -m newton.examples cloth_style3d_bag_drop --device cuda:0
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cloth._style3d_asset_probe import (
    analyze_mesh,
    build_newton_cloth_model,
    cloth_params_from_args,
    configure_style3d_solver_collision,
    load_mesh,
    parse_vec3_arg,
    resolve_path,
)

DEFAULT_BAG_ASSET = "newton/examples/assets/style3d_probe/bag/nonwoven_small_6/nonwoven_small_6.obj"


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.use_cuda_graph = not bool(args.no_cuda_graph)

        asset_path = resolve_path(args.asset)
        mesh = load_mesh(asset_path)
        report = analyze_mesh(asset_path, mesh)
        if report.status != "ok":
            raise ValueError(f"Cannot load bag asset {asset_path}: {report.warnings}")

        self.cloth_params = cloth_params_from_args(args)
        scale = float(args.scale) if args.scale is not None else float(report.recommended_scale or 1.0)
        print(
            "[BagDrop] "
            f"asset={report.path}, scale={scale:g}, verts={report.vertex_count}, tris={report.triangle_count}, "
            f"components={report.component_count}, nonmanifold_edges={report.nonmanifold_edge_count}",
            flush=True,
        )
        for warning in report.warnings[:4]:
            print(f"[BagDrop] asset warning: {warning}", flush=True)

        (
            self.newton,
            self.wp,
            self.model,
            self.state_0,
            _builder,
        ) = build_newton_cloth_model(
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
        self.viewer.set_camera(pos=wp.vec3(0.0, -1.4, 0.75), pitch=-22.0, yaw=90.0)

        print(
            "[BagDrop] "
            f"frame_dt={self.frame_dt:g}s, substeps={self.sim_substeps}, sim_dt={self.sim_dt:g}s, "
            f"soft_contact_margin={self.cloth_params.soft_contact_margin:g}",
            flush=True,
        )
        self.capture()

    @staticmethod
    def create_parser():
        return create_parser()

    def capture(self):
        self.graph = None
        if not self.use_cuda_graph or not self.wp.get_device().is_cuda:
            return
        if self.sim_substeps % 2:
            print("[BagDrop] CUDA graph disabled because --view-substeps is odd.", flush=True)
            return
        try:
            with self.wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print(f"[BagDrop] CUDA graph capture enabled for {self.sim_substeps} substeps per frame.", flush=True)
        except Exception as exc:
            print(f"[BagDrop] CUDA graph capture failed; falling back to Python stepping: {exc!r}", flush=True)

    def simulate(self):
        self.collision_pipeline.collide(
            self.state_0,
            self.contacts,
            soft_contact_margin=float(self.cloth_params.soft_contact_margin),
        )
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            self.wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Bag drop particle state is not finite.")
        if len(q) == 0:
            raise ValueError("Bag drop model has no particles.")

        min_pos = np.min(q, axis=0)
        max_abs_position = float(np.max(np.abs(q)))
        max_speed = float(np.max(np.linalg.norm(qd, axis=1)))
        if min_pos[2] < -0.25:
            raise ValueError(f"Bag penetrated too far below the ground: z_min={min_pos[2]:.4f} m")
        if max_abs_position > 20.0 or max_speed > 100.0:
            raise ValueError(
                f"Bag simulation appears unstable: max_abs_position={max_abs_position:.3f}, "
                f"max_speed={max_speed:.3f} m/s"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    newton.examples.add_broad_phase_arg(parser)
    parser.add_argument("asset", nargs="?", default=DEFAULT_BAG_ASSET, help="OBJ/USD bag asset to import")
    parser.add_argument("--warp-cache-dir", default="/tmp/warp_cache")
    parser.add_argument("--scale", type=float, default=None, help="Asset scale override; default uses probe estimate")
    parser.add_argument("--view-fps", type=float, default=60.0, help="Simulation frames advanced per viewer frame")
    parser.add_argument("--view-substeps", type=int, default=6, help="Style3D solver substeps per viewer frame")
    parser.add_argument("--start-height", type=float, default=0.5, help="Lowest bag vertex height above the ground [m]")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture")
    parser.add_argument("--show-particles", action="store_true", help="Render cloth particles")

    parser.add_argument("--solver-iterations", type=int, default=5, help="Style3D nonlinear iterations per substep")
    parser.add_argument("--linear-iterations", type=int, default=10, help="Style3D linear iterations per substep")
    parser.add_argument(
        "--style3d-translation-preconditioner",
        action="store_true",
        help="Enable the Style3D coarse translation PCG preconditioner",
    )
    parser.set_defaults(step_dt=1.0 / 120.0)

    parser.add_argument("--cloth-density", type=float, default=0.3, help="Cloth areal density")
    parser.add_argument("--particle-radius", type=float, default=0.0015, help="Cloth particle/contact radius [m]")
    parser.add_argument("--tri-ka", type=float, default=500.0, help="Triangle area preservation stiffness")
    parser.add_argument("--tri-kd", type=float, default=1.0e-4, help="Triangle damping")
    parser.add_argument(
        "--tri-aniso-ke",
        type=parse_vec3_arg,
        default=(1000.0, 1000.0, 500.0),
        help="Style3D anisotropic stretch/shear stiffness as weft,warp,shear",
    )
    parser.add_argument(
        "--edge-aniso-ke",
        type=parse_vec3_arg,
        default=(2.0e-4, 2.0e-4, 2.0e-4),
        help="Style3D anisotropic bending stiffness as weft,warp,shear",
    )
    parser.add_argument("--edge-kd", type=float, default=1.0e-4, help="Edge/bending damping")

    parser.add_argument("--soft-contact-margin", type=float, default=0.0015, help="Ground/body soft contact margin [m]")
    parser.add_argument("--soft-contact-ke", type=float, default=10.0, help="Ground/body soft contact stiffness")
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6, help="Ground/body soft contact damping")
    parser.add_argument("--soft-contact-mu", type=float, default=0.2, help="Ground/body soft contact friction")

    parser.add_argument("--no-style3d-self-collision", action="store_true", help="Disable Style3D cloth self-collision")
    parser.add_argument(
        "--style3d-collision-radius", type=float, default=0.0005, help="Style3D self-collision radius [m]"
    )
    parser.add_argument(
        "--style3d-collision-stiff-vf", type=float, default=0.01, help="Vertex-face self-collision stiffness"
    )
    parser.add_argument(
        "--style3d-collision-stiff-ee", type=float, default=0.005, help="Edge-edge self-collision stiffness"
    )
    parser.add_argument("--style3d-collision-stiff-ef", type=float, default=0.05, help="Edge-face untangling stiffness")

    parser.add_argument("--style3d-panel-axes", choices=["xy", "xz", "yz"], default="xy")
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
