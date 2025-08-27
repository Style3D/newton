# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example IK H1 (positions + rotations)
#
# Inverse kinematics on H1 with four interactive end-effector
# targets (left/right hands + left/right feet) controlled via ViewerGL.log_gizmo().
#
# - Uses both IKPositionObjective and IKRotationObjective per end-effector
# - Re-solves IK every frame from the latest gizmo transforms
#
# Command: python -m newton.examples ik_h1
###########################################################################

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.utils


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.num_substeps = 6
        self.iterations = 8
        self.dt = self.frame_dt / self.num_substeps

        self.viewer = viewer

        # ------------------------------------------------------------------
        # Build a single H1 (fixed base for stability) + ground
        # ------------------------------------------------------------------
        h1 = newton.Style3DModelBuilder()
        h1.add_mjcf(
            newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
            floating=False,
            mesh_maxhullvert=256,
        )
        h1.add_ground_plane()

        # add cloth builder
        cloth_builder = newton.Style3DModelBuilder()
        # asset_path = newton.utils.download_asset("style3d_description")
        # garment_usd_name = "Women_Sweatshirt"
        # usd_stage = Usd.Stage.Open(str(asset_path / "garments" / (garment_usd_name + ".usd")))
        # usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath(str("/Root/" + garment_usd_name + "/Root_Garment")))

        usd_stage = Usd.Stage.Open(os.path.join(newton.examples.get_asset_directory(), "CakeSkirt_H1.usd"))
        usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/CakeSkirt_H1/Root_Garment"))
        garment_prim = UsdGeom.PrimvarsAPI(usd_geom_garment.GetPrim()).GetPrimvar("st")
        garment_mesh_indices = np.array(usd_geom_garment.GetFaceVertexIndicesAttr().Get())
        garment_mesh_points = np.array(usd_geom_garment.GetPointsAttr().Get())
        garment_mesh_uv_indices = np.array(garment_prim.GetIndices())
        garment_mesh_uv = np.array(garment_prim.Get()) * 1e-3

        cloth_builder.add_aniso_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(0, 0, 1), angle=wp.half_pi),
            vel=wp.vec3(0.0, 0.0, 0.0),
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2),
            edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 2.0,
            panel_verts=garment_mesh_uv.tolist(),
            panel_indices=garment_mesh_uv_indices.tolist(),
            vertices=garment_mesh_points.tolist(),
            indices=garment_mesh_indices.tolist(),
            density=0.3,
            scale=1.0,
            particle_radius=1.0e-2,
        )
        h1.add_builder(cloth_builder)

        self.graph = None
        self.model = h1.finalize()

        self.model.soft_contact_ke = 1.0e2
        # self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.0

        self.viewer.set_model(self.model)

        # states
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # End effectors
        # ------------------------------------------------------------------
        self.ee = [
            ("left_hand", 16),
            ("right_hand", 33),
            ("left_foot", 5),
            ("right_foot", 10),
        ]
        num_ees = len(self.ee)

        # ------------------------------------------------------------------
        # Persistent gizmo transforms (pass-by-ref objects mutated by viewer)
        # ------------------------------------------------------------------
        body_q_np = self.state.body_q.numpy()
        self.ee_tfs = [wp.transform(*body_q_np[link_idx]) for _, link_idx in self.ee]

        # ------------------------------------------------------------------
        # IK setup (single problem)
        # ------------------------------------------------------------------
        total_residuals = num_ees * 3 * 2 + self.model.joint_coord_count  # positions + rotations + joint limits

        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # Position & rotation objectives
        self.pos_objs = []
        self.rot_objs = []
        for ee_i, (_, link_idx) in enumerate(self.ee):
            tf = self.ee_tfs[ee_i]

            self.pos_objs.append(
                ik.IKPositionObjective(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                    n_problems=1,
                    total_residuals=total_residuals,
                    residual_offset=ee_i * 3,  # 0,3,6,9 for 4 EEs
                )
            )

            self.rot_objs.append(
                ik.IKRotationObjective(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_q2v4(wp.transform_get_rotation(tf))], dtype=wp.vec4),
                    n_problems=1,
                    total_residuals=total_residuals,
                    residual_offset=num_ees * 3 + ee_i * 3,  # 12,15,18,21 for 4 EEs
                )
            )

        # Joint limit objective
        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            n_problems=1,
            total_residuals=total_residuals,
            residual_offset=num_ees * 6,  # 24 when 4 EEs
            weight=10.0,
        )

        # Variables the solver will update
        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.ik_iters = 24
        self.solver = ik.IKSolver(
            model=self.model,
            joint_q=self.joint_q,
            objectives=[*self.pos_objs, *self.rot_objs, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        self.cloth_solver = newton.solvers.SolverStyle3D(
            self.model,
            self.iterations,
        )
        self.cloth_solver.precompute(
            h1,
        )
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state)

        self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def integrate_frame_substeps(self):
        for _ in range(self.num_substeps):
            self.contacts = self.model.collide(self.state, soft_contact_margin=0.1)
            self.cloth_solver.step(self.state, self.state1, self.control, self.contacts, self.dt)
            (self.state, self.state1) = (self.state1, self.state)

    def simulate(self):
        self.solver.solve(iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.integrate_frame_substeps()

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transforms and push into IK objectives."""
        for i, tf in enumerate(self.ee_tfs):
            self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self._push_targets_from_gizmos()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmos (the viewer will draw & mutate transforms in-place)
        for (name, _), tf in zip(self.ee, self.ee_tfs, strict=False):
            self.viewer.log_gizmo(f"target_{name}", tf)

        # Visualize the current articulated state
        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.log_state(self.state)

        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer)
    newton.examples.run(example)
