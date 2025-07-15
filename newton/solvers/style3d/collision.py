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

import warp as wp

from newton.sim import EdgeBvh, TriBvh

from .kernels import (
    handle_edge_edge_contacts_kernel,
    handle_vertex_triangle_contacts_kernel,
    solve_untangling_kernel,
)

########################################################################################################################
###################################################    Collision    ####################################################
########################################################################################################################


class Collision:
    def __init__(
        self,
        device,
        particle_count: int,
        edge_count: int,
        tri_count: int,
    ):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            device: The target Warp device (e.g. "cpu" or "cuda").
            tri_count: Number of triangles for the triangle BVH.
            edge_count: Number of edges for the edge BVH.
            particle_count: Number of particles (used for VF broad-phase buffer).
        """
        self.device = device
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.1  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.tri_bvh = TriBvh(tri_count, self.device)
        self.edge_bvh = EdgeBvh(edge_count, self.device)
        self.broad_phase_ee = wp.array(shape=(128, edge_count), dtype=int)
        self.broad_phase_ef = wp.array(shape=(128, edge_count), dtype=int)
        self.broad_phase_vf = wp.array(shape=(128, particle_count), dtype=int)

    def rebuild_bvh(
        self,
        pos: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=int, ndim=2),
        edge_indices: wp.array(dtype=int, ndim=2),
    ):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
            tri_indices: Array of triangle vertex indices.
            edge_indices: Array of edge vertex indices.
        """
        self.tri_bvh.build(pos, tri_indices, self.radius)
        self.edge_bvh.build(pos, edge_indices, self.radius)

    def refit_bvh(
        self,
        pos: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=int, ndim=2),
        edge_indices: wp.array(dtype=int, ndim=2),
    ):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
            tri_indices: Array of triangle vertex indices.
            edge_indices: Array of edge vertex indices.
        """
        self.tri_bvh.refit(pos, tri_indices, self.radius)
        self.edge_bvh.refit(pos, edge_indices, self.radius)

    def broad_phase(
        self,
        pos: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=int, ndim=2),
        edge_indices: wp.array(dtype=int, ndim=2),
    ):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            pos: Array of vertex positions.
            tri_indices: Triangle connectivity.
            edge_indices: Edge connectivity.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius * 2.0

        # Vertex-face collision candidates
        self.tri_bvh.triangle_vs_point(
            pos,
            pos,
            tri_indices,
            self.broad_phase_vf,
            True,
            max_dist,
            query_radius,
        )

        # Edge-edge collision candidates
        self.edge_bvh.edge_vs_edge(
            pos,
            edge_indices,
            pos,
            edge_indices,
            self.broad_phase_ee,
            True,
            max_dist,
            query_radius,
        )

        self.tri_bvh.aabb_vs_aabb(
            self.edge_bvh.lower_bounds,
            self.edge_bvh.upper_bounds,
            self.broad_phase_ef,
            query_radius,
            False,
        )

    def narrow_phase(
        self,
        pos: wp.array(dtype=wp.vec3),
        tri_indices: wp.array(dtype=int, ndim=2),
        edge_indices: wp.array(dtype=int, ndim=2),
        vert_stiff: wp.array(dtype=float),
        # outputs
        forces: wp.array(dtype=wp.vec3),
        hessian_diags: wp.array(dtype=wp.mat33),
    ):
        """
        Perform narrow-phase collision handling by applying collision response forces.

        Args:
            pos: Array of current vertex positions.
            tri_indices: Triangle indices for vertex-face collision detection.
            edge_indices: Edge indices for edge-edge collision detection.
            vert_stiff: Per-vertex stiffness multipliers.
            forces: Output force array (accumulated).
            hessian_diags: Output diagonal blocks of the collision Hessian.
        """
        thickness = 2.0 * self.radius

        if self.stiff_vf > 0:
            wp.launch(
                handle_vertex_triangle_contacts_kernel,
                dim=len(pos),
                inputs=[
                    thickness,
                    self.stiff_vf,
                    pos,
                    tri_indices,
                    self.broad_phase_vf,
                    vert_stiff,
                ],
                outputs=[forces, hessian_diags],
                device=self.device,
            )

        if self.stiff_ee > 0:
            wp.launch(
                handle_edge_edge_contacts_kernel,
                dim=edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ee,
                    pos,
                    edge_indices,
                    self.broad_phase_ee,
                    vert_stiff,
                ],
                outputs=[forces, hessian_diags],
                device=self.device,
            )

        if self.stiff_ef > 0:
            wp.launch(
                solve_untangling_kernel,
                dim=edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ef,
                    pos,
                    tri_indices,
                    edge_indices,
                    self.broad_phase_ef,
                    vert_stiff,
                ],
                outputs=[forces, hessian_diags],
                device=self.device,
            )

    def multiply(self, x: wp.array(dtype=wp.vec3), out_Hx: wp.array(dtype=wp.vec3)):
        """
        Evaluate matrix-free Hessian-vector product for collision constraints.

        Args:
            x: Input vector.
            out_Hx: Output vector representing H*x.
        """
        # TODO
        pass
