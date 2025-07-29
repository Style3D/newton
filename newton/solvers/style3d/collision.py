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

from newton.sim import Contacts, Model, State

from ..vbd.solver_vbd import NUM_THREADS_PER_COLLISION_PRIMITIVE, accumulate_contact_force_and_hessian
from ..vbd.tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo

########################################################################################################################
################################################    Collision Handler   ################################################
########################################################################################################################


@wp.kernel
def particle_conservative_bounds_kernel(
    collision_query_radius: float,
    conservative_bound_relaxation: float,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_dx: wp.array(dtype=wp.vec3),
):
    """
    Ensures particle displacements remain within a conservative bound to prevent penetration.

    Args:
        collision_query_radius (float): Maximum allowed distance to check for collisions.
        conservative_bound_relaxation (float): Relaxation factor for conservative displacement bound.
        collision_info (TriMeshCollisionInfo): Collision information for all particles.
        particle_dx (wp.array): Particle displacement vectors to be adjusted if necessary.
    """
    particle_index = wp.tid()

    # Compute the minimum distance between the query radius and the nearest collision triangle distance
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # Calculate the conservative bound based on relaxation factor and minimum distance
    conservative_bound = conservative_bound_relaxation * min_dist

    # Current displacement of the particle
    dx = particle_dx[particle_index]

    # If displacement exceeds conservative bound, clamp it to avoid excessive movement
    if wp.length(dx) > conservative_bound:
        particle_dx[particle_index] = wp.normalize(dx) * conservative_bound
        # wp.printf("conservative_bound = %f\n", conservative_bound)


########################################################################################################################
################################################    Collision Handler   ################################################
########################################################################################################################


class CollisionHandler:
    """Handles collision detection and response for cloth simulation."""

    def __init__(
        self,
        model: Model,
        friction_epsilon: float = 1e-2,
        self_contact_radius: float = 3e-3,
        self_contact_margin: float = 5e-3,
        edge_edge_parallel_epsilon: float = 1e-5,
        edge_collision_buffer_pre_alloc: int = 64,
        vertex_collision_buffer_pre_alloc: int = 32,
        integrate_with_external_rigid_solver: bool = True,
        penetration_free_conservative_bound_relaxation: float = 0.42,
    ):
        """
        Initializes the collision handler.

        Args:
            model (Model): The simulation model containing particle and geometry data.
            friction_epsilon (float): Small epsilon used to prevent division by zero in friction calculations.
            self_contact_radius (float): Radius for self-collision detection (used for cloth self-contact).
            self_contact_margin (float): Extra margin added to self-contact detection radius to improve robustness.
            edge_edge_parallel_epsilon (float): Tolerance for detecting nearly parallel edges in edge-edge collision.
            edge_collision_buffer_pre_alloc (int): Pre-allocated size for edge collision buffer.
            vertex_collision_buffer_pre_alloc (int): Pre-allocated size for vertex collision buffer.
            integrate_with_external_rigid_solver (bool): Whether to integrate with an external rigid body solver.
            penetration_free_conservative_bound_relaxation (float): Relaxation factor for penetration-free displacement bound.
        """
        self.model = model
        self.friction_epsilon = friction_epsilon
        self.self_contact_margin = self_contact_margin
        self.self_contact_radius = self_contact_radius
        self.integrate_with_external_rigid_solver = integrate_with_external_rigid_solver
        self.penetration_free_conservative_bound_relaxation = penetration_free_conservative_bound_relaxation

        self.particle_colors = wp.zeros(model.particle_count, dtype=int)
        self.trimesh_collision_detector = TriMeshCollisionDetector(
            model,
            vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
            edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
            edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
        )
        self.trimesh_collision_info = wp.array(
            [self.trimesh_collision_detector.collision_info], dtype=TriMeshCollisionInfo, device=self.model.device
        )
        self.collision_evaluation_kernel_launch_size = max(
            self.model.particle_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            self.model.edge_count * NUM_THREADS_PER_COLLISION_PRIMITIVE,
            model.shape_count * model.particle_count,
        )

    def rebuild_bvh(self, pos: wp.array(dtype=wp.vec3)):
        """Rebuilds the BVH for collision detection using updated particle positions."""
        self.trimesh_collision_detector.rebuild(pos)

    def frame_begin(self, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """Initializes collision detection for a new simulation frame."""
        pass

    def eval_contact_force_hessian_diag(
        self,
        dt: float,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        particle_forces: wp.array(dtype=wp.vec3),
        particle_hessians: wp.array(dtype=wp.mat33),
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
            particle_hessians (wp.array): Output array for Hessian diagonals.
            particle_q_prev (wp.array): Previous positions (optional, for velocity-based damping).
            particle_stiff (wp.array): Optional stiffness array for particles.
        """
        self.trimesh_collision_detector.refit(state_in.particle_q)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.self_contact_margin)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.self_contact_margin)

        curr_color = 0
        wp.launch(
            kernel=accumulate_contact_force_and_hessian,
            dim=self.collision_evaluation_kernel_launch_size,
            inputs=[
                dt,
                curr_color,
                particle_q_prev,
                state_in.particle_q,
                self.particle_colors,
                self.model.tri_indices,
                self.model.edge_indices,
                # self-contact
                self.trimesh_collision_info,
                self.self_contact_radius,
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                # body-particle contact
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.model.shape_material_mu,
                self.model.shape_body,
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                state_in.body_q if self.integrate_with_external_rigid_solver else None,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
            ],
            outputs=[particle_forces, particle_hessians],
            device=self.model.device,
            max_blocks=self.model.device.sm_count,
        )

    def hessian_multiply(self, x: wp.array(dtype=wp.vec3)):
        """Computes the Hessian-vector product for implicit integration."""
        return None

    def linear_iteration_end(self, dx: wp.array(dtype=wp.vec3)):
        """
        Enforces conservative displacement bounds after each solver iteration to maintain stability
        and prevent excessive motion leading to penetration.

        Args:
            dx (wp.array): Current displacement for each particle, which may be modified.
        """
        wp.launch(
            particle_conservative_bounds_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.self_contact_margin,
                self.penetration_free_conservative_bound_relaxation,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[dx],
            device=self.model.device,
        )

    def frame_end(self, pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
        """Applies final collision response and velocity damping."""
        pass
