# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two disconnected Style3D cloths in one world.

This example is intended to visualize the coarse translation preconditioner
with separate cloth components in one world.

Command:
    python -m newton.examples cloth_style3d_two_cloths --device cuda:0
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.args = args
        self.use_cuda_graph = not bool(args.no_cuda_graph)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        dim_x = int(args.dim_x)
        dim_y = int(args.dim_y)
        width_x = float(args.width_x)
        width_y = float(args.width_y)
        mass = float(args.cloth_density) * width_x * width_y / (dim_x * dim_y)

        cloth_args = {
            "rot": wp.quat_identity(),
            "dim_x": dim_x,
            "dim_y": dim_y,
            "cell_x": width_x / dim_x,
            "cell_y": width_y / dim_y,
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "mass": mass,
            "tri_aniso_ke": wp.vec3(*map(float, args.tri_aniso_ke)),
            "tri_ka": float(args.tri_ka),
            "tri_kd": float(args.tri_kd),
            "edge_aniso_ke": wp.vec3(*map(float, args.edge_aniso_ke)),
            "edge_kd": float(args.edge_kd),
            "particle_radius": float(args.particle_radius),
        }
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(-0.35, -0.15, float(args.low_cloth_height)),
            **cloth_args,
        )
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(0.35, -0.15, float(args.high_cloth_height)),
            **cloth_args,
        )
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.soft_contact_margin = float(args.soft_contact_margin)
        self.model.soft_contact_ke = float(args.soft_contact_ke)
        self.model.soft_contact_kd = float(args.soft_contact_kd)
        self.model.soft_contact_mu = float(args.soft_contact_mu)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverStyle3D(
            self.model,
            iterations=max(1, int(args.solver_iterations)),
            linear_iterations=max(1, int(args.linear_iterations)),
            enable_translation_preconditioner=bool(args.style3d_translation_preconditioner),
        )
        if self.solver.collision is not None:
            self.solver.collision.radius = float(args.style3d_collision_radius)
            if args.no_style3d_self_collision:
                self.solver.collision.stiff_vf = 0.0
                self.solver.collision.stiff_ee = 0.0
                self.solver.collision.stiff_ef = 0.0
            self.solver.rebuild_bvh(self.state_0)

        particles_per_cloth = (dim_x + 1) * (dim_y + 1)
        self.low_slice = slice(0, particles_per_cloth)
        self.high_slice = slice(particles_per_cloth, 2 * particles_per_cloth)
        self.last_stats = {}

        self.viewer.show_particles = bool(args.show_particles)
        self.viewer.show_triangles = True
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -1.5, 0.8), pitch=-25.0, yaw=90.0)
        self.update_stats()
        self.capture()

    @staticmethod
    def create_parser():
        return create_parser()

    def capture(self):
        self.graph = None
        if not self.use_cuda_graph or not wp.get_device().is_cuda:
            return
        if self.sim_substeps % 2:
            print("[TwoCloths] CUDA graph disabled because --view-substeps is odd.", flush=True)
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print(f"[TwoCloths] CUDA graph capture enabled for {self.sim_substeps} substeps per frame.", flush=True)
        except Exception as exc:
            print(f"[TwoCloths] CUDA graph capture failed; falling back to Python stepping: {exc!r}", flush=True)

    def simulate(self):
        self.model.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_stats(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        speed = np.linalg.norm(qd, axis=1)
        self.last_stats = {
            "low_z_min": float(np.min(q[self.low_slice, 2])),
            "low_z_max": float(np.max(q[self.low_slice, 2])),
            "high_z_min": float(np.min(q[self.high_slice, 2])),
            "high_z_max": float(np.max(q[self.high_slice, 2])),
            "max_speed": float(np.max(speed)),
        }

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
            self.update_stats()
            s = self.last_stats
            print(
                "[TwoCloths] "
                f"frame={self.frame} time={self.sim_time:.3f} "
                f"low_z=[{s['low_z_min']:.4f}, {s['low_z_max']:.4f}] "
                f"high_z=[{s['high_z_min']:.4f}, {s['high_z_max']:.4f}] "
                f"max_speed={s['max_speed']:.4f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Two-cloth particle state is not finite.")


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--view-fps", type=float, default=60.0)
    parser.add_argument("--view-substeps", type=int, default=6)
    parser.add_argument("--dim-x", type=int, default=32)
    parser.add_argument("--dim-y", type=int, default=24)
    parser.add_argument("--width-x", type=float, default=0.45)
    parser.add_argument("--width-y", type=float, default=0.32)
    parser.add_argument("--low-cloth-height", type=float, default=0.012)
    parser.add_argument("--high-cloth-height", type=float, default=0.42)
    parser.add_argument("--show-particles", action="store_true")
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--solver-iterations", type=int, default=5)
    parser.add_argument("--linear-iterations", type=int, default=10)
    parser.add_argument("--style3d-translation-preconditioner", action="store_true")
    parser.add_argument("--cloth-density", type=float, default=0.3)
    parser.add_argument("--particle-radius", type=float, default=0.003)
    parser.add_argument("--tri-ka", type=float, default=200.0)
    parser.add_argument("--tri-kd", type=float, default=1.0e-5)
    parser.add_argument("--tri-aniso-ke", type=float, nargs=3, default=(200.0, 200.0, 50.0))
    parser.add_argument("--edge-aniso-ke", type=float, nargs=3, default=(5.0e-6, 5.0e-6, 5.0e-6))
    parser.add_argument("--edge-kd", type=float, default=1.0e-4)
    parser.add_argument("--soft-contact-margin", type=float, default=0.003)
    parser.add_argument("--soft-contact-ke", type=float, default=10.0)
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6)
    parser.add_argument("--soft-contact-mu", type=float, default=0.2)
    parser.add_argument("--no-style3d-self-collision", action="store_true")
    parser.add_argument("--style3d-collision-radius", type=float, default=0.002)
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
