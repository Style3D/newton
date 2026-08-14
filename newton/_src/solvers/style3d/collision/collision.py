# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from newton import Contacts, Model, State
from newton._src.solvers.style3d.collision.bvh import BvhEdge, BvhTri
from newton._src.solvers.style3d.collision.kernels import (
    accumulate_rigid_edge_cloth_edge_reaction_kernel,
    accumulate_rigid_vertex_cloth_face_reaction_kernel,
    accumulate_body_reaction_kernel,
    accumulate_projection_impulse_kernel,
    apply_contact_projection_kernel,
    count_particle_contacts_kernel,
    clamp_body_wrench_kernel,
    eval_rigid_edge_cloth_edge_contacts_kernel,
    eval_rigid_vertex_cloth_face_contacts_kernel,
    eval_body_contact_kernel,
    finalize_contact_projection_kernel,
    hessian_multiply_kernel,
    handle_edge_edge_contacts_kernel,
    handle_vertex_triangle_contacts_kernel,
    project_body_particle_contacts_kernel,
    project_rigid_edge_cloth_edge_kernel,
    project_rigid_vertex_cloth_face_kernel,
    solve_untangling_kernel,
    summarize_feature_query_kernel,
    transform_rigid_feature_vertices_kernel,
)

########################################################################################################################
###################################################    Collision    ####################################################
########################################################################################################################


class Collision:
    """
    Collision handler for cloth simulation.
    """

    def __init__(self, model: Model):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            model: The simulation model containing particle and geometry data.
        """
        self.model = model
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.5  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.friction_epsilon = 1e-2
        self.integrate_with_external_rigid_solver = True
        self.tri_bvh = BvhTri(model.tri_count, self.model.device)
        self.edge_bvh = BvhEdge(model.edge_count, self.model.device)
        self.body_contact_max = model.shape_count * model.particle_count
        self.broad_phase_ee = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_ef = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_vf = wp.array(shape=(32, model.particle_count), dtype=int, device=self.model.device)

        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.contact_hessian_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.model.device)

        # S1 static-friction anchors. Allocated always (fixed size, CUDA-graph
        # safe); ``anchor_kt_ratio <= 0`` keeps the stock viscous friction law.
        self.anchor_kt_ratio = 0.0
        self.anchor_local = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.anchor_shape = wp.full(model.particle_count, -1, dtype=wp.int32, device=self.model.device)

        # S3 AVBD tangential multipliers (dual ascent, cone-clamped per iteration).
        self.avbd_dual_k = 0.0
        self.lambda_t = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.lambda_shape = wp.full(model.particle_count, -1, dtype=wp.int32, device=self.model.device)
        self.shape_contact_ke = wp.full(
            model.shape_count,
            float(model.soft_contact_ke),
            dtype=float,
            device=self.model.device,
        )
        self.feature_vertex_shape = None
        self.feature_edge_shape = None
        self.projection_iterations = 0
        self.projection_interleaved = False

        self.edge_bvh.build(model.particle_q, self.model.edge_indices, self.radius)
        self.tri_bvh.build(model.particle_q, self.model.tri_indices, self.radius)

    def enable_friction_anchors(self, kt_ratio: float = 0.1) -> None:
        """Enable S1 static-friction anchors on particle-rigid contacts.

        The stock law regularises friction from this substep's slip only, so a
        pinched grip creeps out at a steady rate. Anchors make the tangential
        force a spring to a persistent contact point with elastoplastic return
        mapping, which is what actually sticks.

        Args:
            kt_ratio: Tangential stiffness as a fraction of the contact's normal
                stiffness. 0 disables (stock viscous friction). Calibrate upward
                from 0.1; 8 is known to lock first grasps and is not a free knob.
        """
        self.anchor_kt_ratio = max(float(kt_ratio), 0.0)
        print(f"[collision] friction anchors: kt_ratio={self.anchor_kt_ratio:g}", flush=True)

    def enable_avbd_friction(self, dual_k: float = 0.1) -> None:
        """Enable S3 AVBD tangential multipliers on particle-rigid contacts.

        Args:
            dual_k: Dual-ascent step as a fraction of the contact's normal
                stiffness. Only sets convergence speed, not the final force --
                the multiplier saturates on the Coulomb cone either way.
        """
        self.avbd_dual_k = max(float(dual_k), 0.0)
        print(f"[collision] AVBD friction duals: dual_k={self.avbd_dual_k:g}", flush=True)

    def set_shape_contact_stiffness(self, shape_ids, stiffness: float):
        """Override particle-contact stiffness for arbitrary rigid shapes."""
        values = self.shape_contact_ke.numpy()
        values[np.asarray(shape_ids, dtype=np.int64)] = max(float(stiffness), 0.0)
        self.shape_contact_ke.assign(values)

    def enable_rigid_feature_contacts(
        self,
        shape_ids,
        contact_radius: float,
        search_margin: float,
        crease_angle_deg: float = 10.0,
        candidate_capacity: int = 128,
    ) -> None:
        """Enable symmetric mesh-feature contact for arbitrary rigid shapes.

        Uses the rigid meshes' existing vertices and sharp/boundary edges
        against the cloth's existing triangles and edges.  It does not sample
        cloth faces, add particles, or assume gripper names or pairs.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids:
            return
        local_vertices = []
        vertex_shapes = []
        vertex_weights = []
        rigid_edges = []
        edge_shapes = []
        edge_weights = []
        shape_scale = self.model.shape_scale.numpy()
        vertex_offset = 0
        crease_cos = float(np.cos(np.deg2rad(float(crease_angle_deg))))

        for shape in shape_ids:
            source = self.model.shape_source[shape]
            vertices = np.asarray(source.vertices, dtype=np.float32)
            faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            if not len(vertices) or not len(faces):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)

            face_normal = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            )
            face_area2 = np.linalg.norm(face_normal, axis=1)
            face_normal /= np.maximum(face_area2[:, None], 1.0e-20)
            area_weight = np.zeros(len(vertices), dtype=np.float32)
            for corner in range(3):
                np.add.at(area_weight, faces[:, corner], face_area2 / 6.0)
            nonzero = area_weight[area_weight > 0.0]
            if len(nonzero):
                area_weight /= float(np.mean(nonzero))
            else:
                area_weight.fill(1.0)

            adjacency: dict[tuple[int, int], list[int]] = {}
            for face_id, (a, b, c) in enumerate(faces):
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (int(min(u, v)), int(max(u, v)))
                    adjacency.setdefault(key, []).append(face_id)
            feature_edges = []
            for (u, v), adjacent in adjacency.items():
                keep = len(adjacent) == 1
                if not keep:
                    base = face_normal[adjacent[0]]
                    keep = any(
                        float(np.dot(base, face_normal[other])) < crease_cos
                        for other in adjacent[1:]
                    )
                if keep:
                    feature_edges.append((u, v))

            local_vertices.append(vertices)
            vertex_shapes.append(np.full(len(vertices), shape, dtype=np.int32))
            vertex_weights.append(area_weight)
            if feature_edges:
                feature_edges_np = np.asarray(feature_edges, dtype=np.int32)
                packed = np.full((len(feature_edges_np), 4), -1, dtype=np.int32)
                packed[:, 2:] = feature_edges_np + vertex_offset
                rigid_edges.append(packed)
                edge_shapes.append(np.full(len(packed), shape, dtype=np.int32))
                lengths = np.linalg.norm(
                    vertices[feature_edges_np[:, 1]] - vertices[feature_edges_np[:, 0]],
                    axis=1,
                ).astype(np.float32)
                positive = lengths[lengths > 0.0]
                if len(positive):
                    lengths /= float(np.mean(positive))
                else:
                    lengths.fill(1.0)
                edge_weights.append(lengths)
            vertex_offset += len(vertices)

        if not local_vertices or not rigid_edges:
            return

        device = self.model.device
        self.feature_local_pos = wp.array(
            np.concatenate(local_vertices), dtype=wp.vec3, device=device)
        self.feature_vertex_shape = wp.array(
            np.concatenate(vertex_shapes), dtype=int, device=device)
        self.feature_vertex_weight = wp.array(
            np.concatenate(vertex_weights), dtype=float, device=device)
        self.feature_edge_indices = wp.array(
            np.concatenate(rigid_edges), dtype=int, device=device)
        self.feature_edge_shape = wp.array(
            np.concatenate(edge_shapes), dtype=int, device=device)
        self.feature_edge_weight = wp.array(
            np.concatenate(edge_weights), dtype=float, device=device)
        self.feature_pos_prev = wp.zeros(
            len(self.feature_local_pos), dtype=wp.vec3, device=device)
        self.feature_pos = wp.zeros(
            len(self.feature_local_pos), dtype=wp.vec3, device=device)

        capacity = max(8, int(candidate_capacity))
        self.feature_broad_phase_vf = wp.zeros(
            (capacity + 1, len(self.feature_local_pos)), dtype=int, device=device)
        self.feature_broad_phase_ee = wp.zeros(
            (capacity + 1, len(self.feature_edge_indices)), dtype=int, device=device)
        self.feature_stats = wp.zeros(4, dtype=int, device=device)
        self.feature_contact_radius = max(float(contact_radius), 0.0)
        self.feature_search_margin = max(float(search_margin), 0.0)
        self.feature_candidate_capacity = capacity
        print(
            "[collision] rigid feature contacts: "
            f"{len(shape_ids)} shapes, {len(self.feature_local_pos)} vertices, "
            f"{len(self.feature_edge_indices)} sharp edges, "
            f"radius={self.feature_contact_radius:g} m, "
            f"search_margin={self.feature_search_margin:g} m",
            flush=True,
        )

    def _update_rigid_feature_queries(
        self,
        cloth_pos: wp.array[wp.vec3],
        body_q_prev: wp.array[wp.transform],
        body_q: wp.array[wp.transform],
    ) -> None:
        if self.feature_vertex_shape is None:
            return
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_local_pos,
                self.feature_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q_prev,
            ],
            outputs=[self.feature_pos_prev],
            device=self.model.device,
        )
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_local_pos,
                self.feature_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
            ],
            outputs=[self.feature_pos],
            device=self.model.device,
        )
        self.tri_bvh.refit(cloth_pos, self.model.tri_indices, self.radius)
        max_dist = self.feature_contact_radius + self.feature_search_margin
        self.tri_bvh.triangle_vs_point(
            self.feature_pos,
            cloth_pos,
            self.model.tri_indices,
            self.feature_broad_phase_vf,
            False,
            max_dist,
            self.feature_search_margin,
        )
        wp.launch(
            summarize_feature_query_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_broad_phase_vf,
                self.feature_candidate_capacity,
                0,
            ],
            outputs=[self.feature_stats],
            device=self.model.device,
        )
        self._update_rigid_feature_edge_query(cloth_pos)

    def _update_rigid_feature_edge_query(
        self,
        cloth_pos: wp.array[wp.vec3],
    ) -> None:
        """Refresh the edge-edge candidate set."""
        self.edge_bvh.refit(cloth_pos, self.model.edge_indices, self.radius)
        max_dist = self.feature_contact_radius + self.feature_search_margin
        self.edge_bvh.edge_vs_edge(
            self.feature_pos,
            self.feature_edge_indices,
            cloth_pos,
            self.model.edge_indices,
            self.feature_broad_phase_ee,
            False,
            max_dist,
            self.feature_search_margin,
        )
        wp.launch(
            summarize_feature_query_kernel,
            dim=len(self.feature_edge_indices),
            inputs=[
                self.feature_broad_phase_ee,
                self.feature_candidate_capacity,
                2,
            ],
            outputs=[self.feature_stats],
            device=self.model.device,
        )

    def rebuild_bvh(self, pos: wp.array[wp.vec3]):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.rebuild(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.rebuild(pos, self.model.edge_indices, self.radius)

    def refit_bvh(self, pos: wp.array[wp.vec3]):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.refit(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.refit(pos, self.model.edge_indices, self.radius)

    def frame_begin(self, particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3], dt: float):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            particle_q: Array of vertex positions.
            particle_qd: Array of vertex velocities.
            dt: simulation time step.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius

        self.refit_bvh(particle_q)
        if self.feature_vertex_shape is not None:
            self.feature_stats.zero_()

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
        particle_forces: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        particle_stiff: wp.array[wp.vec3] = None,
    ):
        """
        Evaluates contact forces and the diagonal of the Hessian for implicit time integration.

        This method launches kernels to compute contact forces and Hessian contributions
        based on broad-phase collision candidates computed in frame_begin().

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

        wp.launch(
            kernel=eval_body_contact_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                state_in.particle_q,
                # body-particle contact
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
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
                self.anchor_kt_ratio,
                # The solver calls this once per iteration; advancing the anchor
                # every time would drag it 20x per substep and destroy the very
                # persistence it exists for.
                1 if _iter == 0 else 0,
                self.anchor_local,
                self.anchor_shape,
                self.avbd_dual_k,
                self.lambda_t,
                self.lambda_shape,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

        if self.feature_vertex_shape is not None:
            body_q = (
                state_out.body_q
                if self.integrate_with_external_rigid_solver
                else state_in.body_q
            )
            body_q_prev = state_in.body_q
            if _iter == 0:
                self._update_rigid_feature_queries(
                    state_in.particle_q, body_q_prev, body_q)
            else:
                self._update_rigid_feature_edge_query(state_in.particle_q)
            wp.launch(
                eval_rigid_vertex_cloth_face_contacts_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.feature_vertex_shape,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )
            wp.launch(
                eval_rigid_edge_cloth_edge_contacts_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.feature_edge_shape,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    state_in.particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

    def accumulate_body_reaction(
        self,
        dt: float,
        particle_q_prev: wp.array[wp.vec3],
        particle_q: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
        max_force: float = 0.0,
    ):
        """Accumulate cloth contact reaction wrenches onto rigid bodies.

        Evaluates the particle-body contact force once at the given (typically
        converged, post-step) particle positions and adds the equal-and-opposite
        wrench per body into ``body_f`` (world frame, about the body COM — the
        :attr:`newton.State.body_f` convention). Feed the result to an external
        rigid solver for two-way cloth-rigid coupling.

        Args:
            dt: Time step used for the contact damping/friction terms.
            particle_q_prev: Particle positions at the start of the step.
            particle_q: Particle positions after the cloth solve.
            contacts: Contacts filled by the collision pipeline for this step.
            body_q: Current body transforms.
            body_q_prev: Body transforms at the start of the step.
            body_enabled: Per-body int mask; only bodies with 1 receive forces.
            body_f: Output wrench accumulator (not zeroed here).
            max_force: If > 0, per-body wrenches are uniformly scaled so the
                linear force magnitude stays below this bound.
        """
        wp.launch(
            kernel=accumulate_body_reaction_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                particle_q,
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.model.shape_material_mu,
                self.model.shape_body,
                body_q,
                body_q_prev,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                self.anchor_kt_ratio,
                self.anchor_local,
                self.anchor_shape,
                self.avbd_dual_k,
                self.lambda_t,
                self.lambda_shape,
                body_enabled,
            ],
            outputs=[body_f],
            device=self.model.device,
        )
        if self.feature_vertex_shape is not None:
            self._update_rigid_feature_queries(particle_q, body_q_prev, body_q)
            wp.launch(
                accumulate_rigid_vertex_cloth_face_reaction_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_shape,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                    self.model.shape_body,
                    body_q,
                    self.model.body_com,
                    body_enabled,
                ],
                outputs=[body_f],
                device=self.model.device,
            )
            wp.launch(
                accumulate_rigid_edge_cloth_edge_reaction_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_shape,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                    self.model.shape_body,
                    body_q,
                    self.model.body_com,
                    body_enabled,
                ],
                outputs=[body_f],
                device=self.model.device,
            )
        if max_force > 0.0:
            wp.launch(
                kernel=clamp_body_wrench_kernel,
                dim=body_f.shape[0],
                inputs=[max_force],
                outputs=[body_f],
                device=self.model.device,
            )

    def enable_contact_projection(
        self,
        slack: float = 2.0e-3,
        iterations: int = 3,
        relaxation: float = 1.0,
        friction_scale: float = 0.0,
        interleaved: bool = False,
    ) -> None:
        """Enable the post-solve position-level non-penetration pass.

        After the implicit solve, cloth positions are projected so the
        penetration depth of particle-shape and rigid-feature contacts is
        bounded by ``slack``; the applied displacement is folded into the
        velocities. Penalty forces stay in charge of the reaction channel —
        evaluate reactions AFTER calling :meth:`project_contacts` so the
        residual (= slack) keeps feeding force back to two-way coupled bodies.

        Args:
            slack: Residual depth [m] left for the penalty force channel.
            iterations: Jacobi projection iterations per substep.
            relaxation: Under-relaxation factor for the averaged corrections.
            friction_scale: Multiplies ``shape_material_mu`` for the position-level
                Coulomb branch. 0 keeps the pass normal-only (previous behaviour).
            interleaved: Run one sweep inside each solver iteration
                (:meth:`project_contacts_iteration`) instead of a post-solve pass.
                This is the PBD arrangement: elasticity and contact alternate to a
                self-consistent state, rather than contact patching a converged
                elastic solve. The post-solve entry point becomes a no-op.
        """
        device = self.model.device
        self.projection_slack = max(float(slack), 0.0)
        self.projection_iterations = max(int(iterations), 0)
        self.projection_relaxation = float(relaxation)
        self.projection_friction_scale = max(float(friction_scale), 0.0)
        self.projection_interleaved = bool(interleaved)
        self.proj_delta = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_weight = wp.zeros(self.model.particle_count, dtype=float, device=device)
        self.proj_accum = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_accum_kept = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_contact_share = wp.zeros(self.model.particle_count, dtype=float, device=device)
        print(
            "[collision] contact projection: "
            f"slack={self.projection_slack:g} m, iterations={self.projection_iterations}, "
            f"friction_scale={self.projection_friction_scale:g}, "
            f"mode={'interleaved' if self.projection_interleaved else 'post'}",
            flush=True,
        )

    def accumulate_projection_impulse(
        self,
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
        dt: float,
        max_force: float = 0.0,
    ):
        """Feed the last projection pass back to the rigid side as an impulse.

        Call AFTER the reaction accumulation (which zeroes ``body_f``). With this
        the rigid side feels the position solve directly, so the projection no
        longer has to leave residual penetration behind just to keep a penalty
        force alive -- see :meth:`enable_contact_projection`.
        """
        if self.projection_iterations <= 0:
            return
        self.proj_contact_share.zero_()
        wp.launch(
            count_particle_contacts_kernel,
            dim=self.body_contact_max,
            inputs=[
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
            ],
            outputs=[self.proj_contact_share],
            device=self.model.device,
        )
        wp.launch(
            accumulate_projection_impulse_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                self.proj_accum_kept,
                self.model.particle_mass,
                self.proj_contact_share,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.model.shape_body,
                body_q,
                body_com,
                body_enabled,
            ],
            outputs=[body_f],
            device=self.model.device,
        )
        # Re-clamp: the reaction pass already clamped its own contribution, and
        # a projection impulse spike must not slip past that guard.
        if max_force > 0.0:
            wp.launch(
                clamp_body_wrench_kernel,
                dim=len(body_f),
                inputs=[max_force],
                outputs=[body_f],
                device=self.model.device,
            )

    def project_contacts_iteration(
        self,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        dt: float,
        body_q_prev: wp.array[wp.transform] | None = None,
    ):
        """One projection sweep, to be called INSIDE the solver's iteration loop.

        Same three kernels as :meth:`project_contacts`, run once instead of
        ``projection_iterations`` times: the solver's own loop supplies the
        iteration count, so elasticity and contact alternate the way a PBD
        solver does. Velocities are NOT touched here -- the solver derives the
        whole step's velocity from ``(x_out - x_prev)/dt`` after the loop, which
        already contains the projected displacement.

        No-op unless :meth:`enable_contact_projection` was called with
        ``interleaved=True``.
        """
        if not self.projection_interleaved or self.projection_iterations <= 0:
            return
        self._project_sweep(particle_q, particle_q_prev, contacts, body_q, body_q_prev)

    def _project_sweep(
        self,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform] | None,
    ):
        """One Jacobi sweep: accumulate corrections from all contact types, apply."""
        wp.launch(
            project_body_particle_contacts_kernel,
            dim=self.body_contact_max,
            inputs=[
                self.projection_slack,
                self.projection_friction_scale,
                particle_q,
                particle_q_prev,
                self.proj_accum,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.model.shape_body,
                self.model.shape_material_mu,
                body_q,
                body_q if body_q_prev is None else body_q_prev,
            ],
            outputs=[self.proj_delta, self.proj_weight],
            device=self.model.device,
        )
        if self.feature_vertex_shape is not None:
            wp.launch(
                project_rigid_vertex_cloth_face_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    self.projection_slack,
                    self.feature_contact_radius,
                    particle_q_prev,
                    particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
            wp.launch(
                project_rigid_edge_cloth_edge_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    self.projection_slack,
                    self.feature_contact_radius,
                    particle_q_prev,
                    particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
        wp.launch(
            apply_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.projection_relaxation,
                self.model.particle_flags,
            ],
            outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
            device=self.model.device,
        )

    def project_contacts(
        self,
        particle_q: wp.array[wp.vec3],
        particle_qd: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        dt: float,
        body_q_prev: wp.array[wp.transform] | None = None,
    ):
        """Run the position-level projection pass (see :meth:`enable_contact_projection`).

        Call after the cloth solve and before reaction accumulation. Reuses
        the rigid-feature broad-phase candidates from the solve iterations
        (positions drift sub-millimetre per substep, covered by the search
        margin).

        Args:
            particle_q: Converged particle positions (modified in place).
            particle_qd: Particle velocities (corrected in place).
            particle_q_prev: Particle positions at the start of the substep.
            contacts: Contacts filled by the collision pipeline for this step.
            body_q: Current body transforms (the pose the cloth solved against).
            dt: Substep time step.
        """
        if self.projection_iterations <= 0 or self.projection_interleaved:
            return
        for _ in range(self.projection_iterations):
            wp.launch(
                project_body_particle_contacts_kernel,
                dim=self.body_contact_max,
                inputs=[
                    self.projection_slack,
                    self.projection_friction_scale,
                    particle_q,
                    particle_q_prev,
                    self.proj_accum,
                    self.model.particle_radius,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    self.model.shape_body,
                    self.model.shape_material_mu,
                    body_q,
                    body_q if body_q_prev is None else body_q_prev,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
            if self.feature_vertex_shape is not None:
                wp.launch(
                    project_rigid_vertex_cloth_face_kernel,
                    dim=len(self.feature_local_pos),
                    inputs=[
                        self.projection_slack,
                        self.feature_contact_radius,
                        particle_q_prev,
                        particle_q,
                        self.model.tri_indices,
                        self.feature_pos_prev,
                        self.feature_pos,
                        self.feature_vertex_weight,
                        self.feature_broad_phase_vf,
                    ],
                    outputs=[self.proj_delta, self.proj_weight],
                    device=self.model.device,
                )
                wp.launch(
                    project_rigid_edge_cloth_edge_kernel,
                    dim=len(self.feature_edge_indices),
                    inputs=[
                        self.projection_slack,
                        self.feature_contact_radius,
                        particle_q_prev,
                        particle_q,
                        self.model.edge_indices,
                        self.feature_pos_prev,
                        self.feature_pos,
                        self.feature_edge_indices,
                        self.feature_edge_weight,
                        self.feature_broad_phase_ee,
                    ],
                    outputs=[self.proj_delta, self.proj_weight],
                    device=self.model.device,
                )
            wp.launch(
                apply_contact_projection_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self.projection_relaxation,
                    self.model.particle_flags,
                ],
                outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
                device=self.model.device,
            )
        wp.launch(
            finalize_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[dt],
            outputs=[self.proj_accum, particle_qd, self.proj_accum_kept],
            device=self.model.device,
        )

    def contact_hessian_diagonal(self):
        """Return diagonal of contact Hessian for preconditioning.
        Note:
            Should be called after `accumulate_contact_force()`.
        """
        return self.contact_hessian_diags

    def hessian_multiply(self, x: wp.array[wp.vec3]):
        """Computes the Hessian-vector product for implicit integration."""
        wp.launch(
            hessian_multiply_kernel,
            dim=self.model.particle_count,
            inputs=[self.contact_hessian_diags, x],
            outputs=[self.Hx],
            device=self.model.device,
        )
        return self.Hx

    def linear_iteration_end(self, dx: wp.array[wp.vec3]):
        """No post-solve contact projection; contacts live in the Newton solve."""
        pass

    def frame_end(self, pos: wp.array[wp.vec3], vel: wp.array[wp.vec3], dt: float):
        """Apply post-processing"""
