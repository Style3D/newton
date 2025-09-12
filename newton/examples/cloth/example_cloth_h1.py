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

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.utils
import style3d


class Example:
    def __init__(self, viewer):
        # frame timing
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        # must be an even number when using CUDA Graph
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10

        self.viewer = viewer
        self.viewer._paused = True
        self.frame_index = 0

        # ------------------------------------------------------------------
        # Build a single H1 (fixed base for stability) + ground
        # ------------------------------------------------------------------
        h1 = newton.Style3DModelBuilder()
        h1.add_mjcf(
            newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
            floating=False,
        )
        h1.add_ground_plane()

        # ------------------------------------------------------------------
        # Build a cloth
        # ------------------------------------------------------------------
        cloth_builder = newton.Style3DModelBuilder()
        # asset_path = newton.utils.download_asset("style3d_description")
        # garment_usd_name = "Women_Sweatshirt"
        # usd_stage = Usd.Stage.Open(str(asset_path / "garments" / (garment_usd_name + ".usd")))
        # usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath(str("/Root/" + garment_usd_name + "/Root_Garment")))

        usd_stage = Usd.Stage.Open(os.path.join(newton.examples.get_asset_directory(), "piyi.usd"))
        usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/piyi/Root_Garment"))
        garment_prim = UsdGeom.PrimvarsAPI(usd_geom_garment.GetPrim()).GetPrimvar("st")
        garment_mesh_indices = np.array(usd_geom_garment.GetFaceVertexIndicesAttr().Get())
        garment_mesh_points = np.array(usd_geom_garment.GetPointsAttr().Get())
        garment_mesh_uv_indices = np.array(garment_prim.GetIndices())
        garment_mesh_uv = np.array(garment_prim.Get()) * 1.45

        cloth_builder.add_aniso_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(0, 0, 1), angle=-wp.half_pi),
            vel=wp.vec3(0.0, 0.0, 0.0),
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2) * 10.0,
            edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 50.0,
            panel_verts=garment_mesh_uv.tolist(),
            panel_indices=garment_mesh_uv_indices.tolist(),
            vertices=garment_mesh_points.tolist(),
            indices=garment_mesh_indices.tolist(),
            density=0.5,
            scale=1.0,
            particle_radius=3.0e-3,
        )
        h1.add_builder(cloth_builder)

        self.graph = None
        self.model = h1.finalize()
        self.model.soft_contact_ke = 1e4
        self.model.shape_material_mu.fill_(0.0)
        self.viewer.set_model(self.model)

        # states
        self.state = self.model.state()
        self.state1 = self.model.state()
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
        self.ik_solver = ik.IKSolver(
            model=self.model,
            joint_q=self.joint_q,
            objectives=[*self.pos_objs, *self.rot_objs, self.obj_joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        # ------------------------------------------------------------------
        # Cloth solver
        # ------------------------------------------------------------------
        self.cloth_solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
            # collision_handler=newton._src.solvers.style3d.CollisionHandler(self.model),
            collision_handler=style3d.Collision(self.model),
        )
        self.cloth_solver.precompute(h1)
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state)
        self.shape_flags = self.model.shape_flags.numpy()

        # self.render()
        # self.capture()

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph

    def simulate(self):
        self.ik_solver.solve(iterations=self.ik_iters)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.state1.body_qd.assign(self.state.body_qd)
        self.state1.body_q.assign(self.state.body_q)
        for _ in range(self.sim_substeps):
            self.contacts = self.model.collide(self.state, soft_contact_margin=0.3)
            self.cloth_solver.step(self.state, self.state1, self.control, self.contacts, self.sim_dt)
            (self.state, self.state1) = (self.state1, self.state)

    def _push_targets_from_gizmos(self):
        """Read gizmo-updated transforms and push into IK objectives."""
        for i, tf in enumerate(self.ee_tfs):
            self.pos_objs[i].set_target_position(0, wp.transform_get_translation(tf))
            q = wp.transform_get_rotation(tf)
            self.rot_objs[i].set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def _force_update_targets(self):
        """Rise hands and then wave."""
        target_pos = [
            wp.vec3(0.16589675843715668, 0.6483738422393799, 1.7127180099487305),
            wp.vec3(0.279998242855072, -0.49536585807800293, 1.1899967193603516),
        ]
        target_rot = [
            wp.quat(0.5709694027900696, -0.3505951762199402, 0.2904931306838989, 0.6831477284431458),
            wp.quat(0.00, 0.00, 0.00, 0.00),
        ]

        transition_time = 2.0  # seconds
        if self.sim_time < transition_time:
            lerp_ratio = 0.1 * self.sim_time / transition_time
            for i in range(len(target_pos)):
                tf = self.ee_tfs[i]
                wp.transform_set_rotation(
                    tf,
                    wp.quat_slerp(wp.transform_get_rotation(tf), target_rot[i], wp.clamp(lerp_ratio * 3.0, 0.0, 1.0)),
                )
                wp.transform_set_translation(
                    tf, wp.lerp(wp.transform_get_translation(tf), target_pos[i], wp.clamp(lerp_ratio * 1.0, 0.0, 1.0))
                )
        else:
            rot = (
                wp.quat_from_axis_angle(
                    axis=wp.vec3(1, 0, 0), angle=wp.sin((self.sim_time - transition_time) * 5.0) * 0.3
                )
                * target_rot[0],
            )
            pos0 = target_pos[0] + wp.vec3(wp.sin((self.sim_time - transition_time) * 5.0) * 0.1, 0.0, 0.0)
            pos1 = target_pos[1] + wp.vec3(0.0, wp.sin((self.sim_time - transition_time) * 2.0) * 0.05, 0.0)
            wp.transform_set_rotation(self.ee_tfs[0], wp.quat(rot))
            wp.transform_set_translation(self.ee_tfs[0], pos0)
            wp.transform_set_translation(self.ee_tfs[1], pos1)

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        if self.frame_index > 0:
            self._force_update_targets()
            self._push_targets_from_gizmos()

            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.capture()
            self.sim_time += self.frame_dt

        self.frame_index += 1

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Register gizmos (the viewer will draw & mutate transforms in-place)
        # for (name, _), tf in zip(self.ee, self.ee_tfs, strict=False):
        #    self.viewer.log_gizmo(f"target_{name}", tf)

        # Visualize the current articulated state
        # newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        if isinstance(self.viewer, newton._src.viewer.ViewerUSD):
            self.viewer.log_mesh(
                "/model/triangles", self.state.particle_q, self.model.tri_indices.flatten(), backface_culling=False
            )

            # Register body entity
            for i in range(self.model.body_count):
                shape_indices = self.model.body_shapes[i]
                for shape_idx in shape_indices:
                    if isinstance(self.model.shape_source[shape_idx], newton.Mesh):
                        if self.shape_flags[shape_idx] & 1 == 0:
                            continue

                        @wp.kernel
                        def transform_vertices_kernel(
                            shape_index: wp.int32,
                            scale: float,
                            shape_body: wp.array(dtype=int),
                            vertices_in: wp.array(dtype=wp.vec3),
                            scaling3d: wp.array(dtype=wp.vec3),
                            transforms0: wp.array(dtype=wp.transform),
                            transforms1: wp.array(dtype=wp.transform),
                            vertices_out: wp.array(dtype=wp.vec3),
                        ):
                            tid = wp.tid()
                            scaling = scaling3d[shape_index] * scale
                            new_pos = wp.transform_point(transforms0[shape_index], vertices_in[tid])
                            new_pos = wp.transform_point(transforms1[shape_body[shape_index]], new_pos)
                            new_pos[0] *= scaling[0]
                            new_pos[1] *= scaling[1]
                            new_pos[2] *= scaling[2]
                            vertices_out[tid] = new_pos

                        shape_vertices = wp.array(self.model.shape_source[shape_idx].vertices, dtype=wp.vec3)
                        shape_indices = wp.array(self.model.shape_source[shape_idx].indices.flatten(), dtype=int)

                        wp.launch(
                            transform_vertices_kernel,
                            dim=len(shape_vertices),
                            inputs=[
                                shape_idx,
                                1.0,
                                self.model.shape_body,
                                shape_vertices,
                                self.model.shape_scale,
                                self.model.shape_transform,
                                self.state.body_q,
                            ],
                            outputs=[shape_vertices],
                        )
                        self.viewer.log_mesh(
                            f"/model/triangles_{self.model.shape_key[shape_idx]}", shape_vertices, shape_indices
                        )
            print(f"[{self.viewer._frame_index}/300]")
        else:
            self.viewer.log_state(self.state)

        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    # parser.set_defaults(num_frames=301, viewer="usd", output_path="D:/desktop/example_cloth_h1_uv.usd")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example)
