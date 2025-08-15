########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Collision handling class                                                                           #
#   Author:         Wenchao Huang (physhuangwenchao@gmail.com)                                                         #
#   Date:           2025/07/04                                                                                         #
########################################################################################################################

import warp as wp

from newton import Contacts, Model, State
from style3d.bvh import EdgeBvh, TriBvh
from style3d.collision.kernels import (
    handle_edge_edge_contacts_kernel,
    handle_vertex_triangle_contacts_kernel,
    solve_untangling_kernel,
)

########################################################################################################################
###################################################    Collision    ####################################################
########################################################################################################################


class Collision:
    def __init__(self, model: Model):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            model:
            device: The target Warp device (e.g. "cpu" or "cuda:0").
        """
        self.model = model
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.5  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.tri_bvh = TriBvh(model.tri_count, self.model.device)
        self.edge_bvh = EdgeBvh(model.edge_count, self.model.device)
        self.broad_phase_ee = wp.array(shape=(64, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_ef = wp.array(shape=(16, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_vf = wp.array(shape=(64, model.particle_count), dtype=int, device=self.model.device)

        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.contact_hessian_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.model.device)
        self.rebuild_bvh(model.particle_q)

    def rebuild_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.build(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.build(pos, self.model.edge_indices, self.radius)

    def refit_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
            tri_indices: Array of triangle vertex indices.
            edge_indices: Array of edge vertex indices.
        """
        self.tri_bvh.refit(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.refit(pos, self.model.edge_indices, self.radius)

    def frame_begin(self, particle_q: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            pos: Array of vertex positions.
            tri_indices: Triangle connectivity.
            edge_indices: Edge connectivity.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius * 2.0

        self.refit_bvh(particle_q)

        # Vertex-face collision candidates
        if self.stiff_vf > 0.0:
            self.tri_bvh.triangle_vs_point(
                particle_q,
                particle_q,
                self.model.tri_indices,
                self.broad_phase_vf,
                True,
                max_dist,
                query_radius,
            )

        # Edge-edge collision candidates
        if self.stiff_ee > 0.0:
            self.edge_bvh.edge_vs_edge(
                particle_q,
                self.model.edge_indices,
                particle_q,
                self.model.edge_indices,
                self.broad_phase_ee,
                True,
                max_dist,
                query_radius,
            )

        # Face-edge collision candidates
        if self.stiff_ef > 0.0:
            self.tri_bvh.aabb_vs_aabb(
                self.edge_bvh.lower_bounds,
                self.edge_bvh.upper_bounds,
                self.broad_phase_ef,
                query_radius,
                False,
            )

    def accumulate_contact_force(
        self,
        dt: float,
        _iter: int,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        particle_forces: wp.array(dtype=wp.vec3),
        particle_q_prev: wp.array(dtype=wp.vec3) = None,
        particle_stiff: wp.array(dtype=wp.vec3) = None,
    ):
        """
        Evaluates contact forces and the diagonal of the Hessian for implicit time integration.

        Steps:
            1. Refits BVH based on current positions.
            2. Detects edge-edge and vertex-triangle collisions.
            3. Launches kernel to accumulate forces and Hessians for all particles.

        Args:
            dt (float): Time step.
            state_in (State): Current simulation state (input).
            state_out (State): Next simulation state (output).
            contacts (Contacts): Contact data structure containing contact information.
            particle_forces (wp.array): Output array for computed contact forces.
            particle_q_prev (wp.array): Previous positions (optional, for velocity-based damping).
            particle_stiff (wp.array): Optional stiffness array for particles.
        """
        thickness = 2.0 * self.radius
        self.contact_hessian_diags.zero_()

        if self.stiff_vf > 0:
            wp.launch(
                handle_vertex_triangle_contacts_kernel,
                dim=len(state_in.particle_q),
                inputs=[
                    thickness,
                    self.stiff_vf,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.broad_phase_vf,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ee > 0:
            wp.launch(
                handle_edge_edge_contacts_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ee,
                    state_in.particle_q,
                    self.model.edge_indices,
                    self.broad_phase_ee,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ef > 0:
            wp.launch(
                solve_untangling_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ef,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.broad_phase_ef,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

    def contact_hessian_diagonal(self):
        """Return diagonal of contact Hessian for preconditioning.
        Note:
            Should be called after `eval_contact_force_hessian()`.
        """
        return self.contact_hessian_diags

    def hessian_multiply(self, x: wp.array(dtype=wp.vec3)):
        """Computes the Hessian-vector product for implicit integration."""

        @wp.kernel
        def hessian_multiply_kernel(
            hessian_diags: wp.array(dtype=wp.mat33),
            x: wp.array(dtype=wp.vec3),
            # outputs
            Hx: wp.array(dtype=wp.vec3),
        ):
            tid = wp.tid()
            Hx[tid] = hessian_diags[tid] * x[tid]

        wp.launch(
            hessian_multiply_kernel,
            dim=self.model.particle_count,
            inputs=[self.contact_hessian_diags, x],
            outputs=[self.Hx],
            device=self.model.device,
        )
        return self.Hx

    def linear_iteration_end(self, dx: wp.array(dtype=wp.vec3)):
        """Displacement constraints"""
        pass

    def frame_end(self, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """Apply post-processing"""
        pass
