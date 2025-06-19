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
import newton.utils
from newton.geometry import PARTICLE_FLAG_ACTIVE, Mesh
from newton.utils.renderer.polyscope_renderer import PolyscopeRenderer


class Example:
    def __init__(self, stage_path="example_cloth_style3d.usd", num_frames=600):
        fps = 60
        self.frame_dt = 1.0 / fps
        # must be an even number when using CUDA Graph
        self.num_substeps = 2
        self.iterations = 10
        self.dt = self.frame_dt / self.num_substeps
        self.num_frames = num_frames
        self.use_cuda_graph = wp.get_device().is_cuda

        builder = newton.sim.Style3DModelBuilder(up_axis=newton.Axis.Y)

        use_cloth_mesh = True
        if use_cloth_mesh:
            usd_stage = Usd.Stage.Open(os.path.join(newton.examples.get_asset_directory(), "women_skirt.usda"))

            # Grament
            usd_geom_garment = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/women_skirt/Root_Garment"))
            garment_prim = UsdGeom.PrimvarsAPI(usd_geom_garment.GetPrim()).GetPrimvar("st")
            garment_mesh_indices = np.array(usd_geom_garment.GetFaceVertexIndicesAttr().Get())
            garment_mesh_points = np.array(usd_geom_garment.GetPointsAttr().Get())
            garment_mesh_uv_indices = np.array(garment_prim.GetIndices())
            garment_mesh_uv = np.array(garment_prim.Get()) * 1e-3

            # Avatar
            usd_geom_avatar = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/Root/women_skirt/Root_SkinnedMesh_Avatar_0_Sub_0"))
            avatar_mesh_indices = np.array(usd_geom_avatar.GetFaceVertexIndicesAttr().Get())
            avatar_mesh_points = np.array(usd_geom_avatar.GetPointsAttr().Get())

            builder.add_aniso_cloth_mesh(
                pos=wp.vec3(0, 0, 0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
                panel_verts=garment_mesh_uv.tolist(),
                panel_indices=garment_mesh_uv_indices.tolist(),
                vertices=garment_mesh_points.tolist(),
                indices=garment_mesh_indices.tolist(),
                density=0.3,
                scale=1.0,
                particle_radius=5.0e-3,
            )
            # add avatar
            builder.add_shape_mesh(
                body=builder.add_body(),
                mesh=Mesh(avatar_mesh_points, avatar_mesh_indices),
            )
            # set fixed points
            # fixed_points = [0, 100, 2000, 10000, 20000, 30000]
            fixed_points = []
        else:
            grid_dim = 100
            grid_width = 1.0
            cloth_density = 0.3
            builder.add_aniso_cloth_grid(
                pos=wp.vec3(-0.5, 1.5, 0.0),
                rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
                dim_x=grid_dim,
                dim_y=grid_dim,
                cell_x=grid_width / grid_dim,
                cell_y=grid_width / grid_dim,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=cloth_density * (grid_width * grid_width) / (grid_dim * grid_dim),
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                tri_ka=1.0e2,
                tri_kd=2.0e-6,
                edge_aniso_ke=wp.vec3(1.0e-5, 1.0e-5, 1.0e-5),
            )
            fixed_points = [0, grid_dim]

        self.model = builder.finalize()
        self.model.ground = False
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2

        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_points:
            flags[fixed_vertex_id] = wp.uint32(int(flags[fixed_vertex_id]) & ~int(PARTICLE_FLAG_ACTIVE))
        self.model.particle_flags = wp.array(flags)

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2e-2
        self.model.soft_contact_margin = 0.35e-2
        self.model.soft_contact_ke = 1.0e1
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2

        self.solver = newton.solvers.Style3DSolver(
            model=self.model,
            iterations=self.iterations,
            enable_mouse_dragging=True,
        )
        self.solver.precompute(builder)
        self.state0 = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state0)

        self.renderer = PolyscopeRenderer(self.model)
        self.renderer.set_user_update(self.update)
        self.cuda_graph = None

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.integrate_frame_substeps()
            self.cuda_graph = capture.graph

    def integrate_frame_substeps(self):
        self.contacts = self.model.collide(self.state0)
        for _ in range(self.num_substeps):
            self.solver.step(self.model, self.state0, self.state1, self.control, self.contacts, self.dt)
            (self.state0, self.state1) = (self.state1, self.state0)

    def advance_frame(self):
        if self.use_cuda_graph:
            wp.capture_launch(self.cuda_graph)
        else:
            self.integrate_frame_substeps()
        self.renderer.sim_time += self.dt
        self.renderer.sim_frames += 1

    def update(self):
        if self.renderer.drag_info_chg:
            self.solver.update_drag_info(
                self.renderer.drag_index,
                self.renderer.drag_position,
                self.renderer.drag_bary_coord,
            )
            self.renderer.drag_info_chg = False
        self.advance_frame()
        self.advance_frame()
        self.renderer.update_state(self.state0)

    def run(self):
        self.renderer.run()


if __name__ == "__main__":
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        example = Example()
        example.run()
