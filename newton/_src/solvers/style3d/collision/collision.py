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
    bake_shape_sdf_kernel,
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
    project_tri_sdf_kernel,
    eval_tri_sdf_contact_kernel,
    accumulate_tri_sdf_reaction_kernel,
    gjs_accumulate_kernel,
    gjs_setup_kernel,
    gjs_solve_kernel,
    gjs_writeback_kernel,
    solve_untangling_kernel,
    solve_rigid_untangling_kernel,
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
        # E4 per-shape penalty band (PhysX ``contact_offset``). Allocated always
        # (fixed size, CUDA-graph safe); all zeros = every shape uses
        # ``particle_radius`` as its band, i.e. the stock law, bit-identical.
        self.shape_contact_offset = wp.zeros(
            model.shape_count, dtype=float, device=self.model.device
        )
        # Material-derived contact stiffness (default off: material_ke <= 0 keeps
        # the per-shape constant, and the kernel takes the same branch it always
        # did, so the default path is bit-identical).
        self.material_ke = 0.0
        self.particle_contact_ke = wp.zeros(model.particle_count, dtype=float, device=self.model.device)
        self.feature_vertex_shape = None
        self.feature_edge_shape = None
        # R2-3 rigid untangling (ICM against rigid triangles). ``None`` =
        # disabled; nothing is allocated and no kernel is launched, so the
        # default path is unchanged.
        self.icm_tri_indices = None
        # R5-B gripper joint DOFs inside the cloth solve. ``None`` = disabled;
        # nothing is allocated and no kernel is launched, so the default path
        # is unchanged.
        self.gjs_joint_coord = None
        self.icm_stiff_factor = 0.0
        self.icm_thickness = 0.0
        self.icm_query_radius = 0.0
        self.projection_iterations = 0
        self.projection_interleaved = False
        # E3 triangle-level SDF contact. ``None`` = disabled; nothing is
        # allocated and no kernel is launched, so the default path is unchanged.
        self.tri_sdf_slot_shape = None
        self.tri_sdf_compliant = False
        self.proj_delta = None
        self.projection_relaxation = None

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

    def set_shape_contact_offset(self, shape_ids, offset: float):
        """E4: decouple the penalty's range band from ``particle_radius``.

        The stock law ``f = ke * (r - d)`` uses ``particle_radius`` both as the
        geometric probe radius and as the band the force acts over, so taking the
        radius down to the cloth's physical half-thickness collapses the normal
        load and, with it, the Coulomb cone mu*N (measured 15.4 / 8.5 / 0.8 N at
        r = 8 / 4 / 0.5 mm). Setting an offset on the gripper shapes gives those
        shapes their own band -- normal load AND friction act over
        ``h < d < offset`` -- while the geometric stop stays wherever the geometry
        channel puts it.

        Args:
            shape_ids: shapes that get the band (typically the gripper fingers).
            offset: band [m]. 0 restores the stock ``particle_radius`` behaviour.
                It must be LARGER than the geometric stop of whatever holds the
                cloth out (``particle_radius`` for the vertex projection, the
                triangle constraint's half-thickness for the SDF one); an offset
                at or below that stop leaves zero overlap, hence N = 0 and
                mu*N = 0 -- a frictionless wall.

        Note: the contact-generation margin must also cover the band, otherwise
        no contact is even generated at ``d`` inside it (see
        ``CollisionPipeline(soft_contact_margin=...)``: a pair is emitted while
        ``d < margin + particle_radius``).
        """
        values = self.shape_contact_offset.numpy()
        values[np.asarray(shape_ids, dtype=np.int64)] = max(float(offset), 0.0)
        self.shape_contact_offset.assign(values)
        print(f"[collision] contact offset band: shapes={list(np.asarray(shape_ids).ravel())} "
              f"offset={float(offset) * 1000:.2f}mm", flush=True)

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

    def enable_rigid_untangling(
        self,
        shape_ids,
        stiff_factor: float = 1.0,
        thickness: float = -1.0,
        query_radius: float = -1.0,
        candidate_capacity: int = 32,
    ) -> None:
        """R2-3: enable Intersection Contour Minimisation against rigid shapes.

        The stock cloth-cloth ICM (``stiff_ef``) is the only channel in this
        solver that can recover from an ALREADY-tangled state; every other
        channel is a proximity law that pushes a particle to the nearest
        surface. For a blade thinner than the penetration depth "nearest" is the
        wrong side, and an edge can thread a triangle with both its endpoints
        outside the shell -- neither the penalty nor the projection sees it.
        This gives the same contour-length gradient a rigid face side.

        Args:
            shape_ids: rigid shapes whose triangles take part (typically the
                gripper fingers). An empty list leaves the channel disabled.
            stiff_factor: multiplies the cloth's own PD diagonal, exactly like
                ``stiff_ef`` does for cloth-cloth. 0 disables.
            thickness: displacement scale [m]; the per-iteration correction is
                ``2 * thickness``. Negative takes the cloth-cloth default
                (``2 * self.radius``), which is the same number the stock ICM
                uses, so "same law, rigid face" is the literal default.
            query_radius: broad-phase padding [m]. Negative takes ``self.radius``.
            candidate_capacity: max rigid triangles examined per cloth edge.

        Default OFF: unless this is called, ``icm_tri_indices`` stays ``None``,
        nothing is allocated and the kernel is never launched.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids or float(stiff_factor) <= 0.0:
            return
        local_vertices = []
        vertex_shapes = []
        rigid_tris = []
        shape_scale = self.model.shape_scale.numpy()
        vertex_offset = 0
        for shape in shape_ids:
            source = self.model.shape_source[shape]
            if source is None:
                continue
            vertices = np.asarray(source.vertices, dtype=np.float32)
            faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            if not len(vertices) or not len(faces):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)
            local_vertices.append(vertices)
            vertex_shapes.append(np.full(len(vertices), shape, dtype=np.int32))
            rigid_tris.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        if not rigid_tris:
            return

        device = self.model.device
        self.icm_local_pos = wp.array(
            np.concatenate(local_vertices), dtype=wp.vec3, device=device)
        self.icm_vertex_shape = wp.array(
            np.concatenate(vertex_shapes), dtype=int, device=device)
        self.icm_tri_indices = wp.array(
            np.concatenate(rigid_tris), dtype=int, device=device)
        self.icm_pos = wp.zeros(len(self.icm_local_pos), dtype=wp.vec3, device=device)
        self.icm_stiff_factor = float(stiff_factor)
        self.icm_thickness = (
            2.0 * self.radius if float(thickness) < 0.0 else float(thickness)
        )
        self.icm_query_radius = (
            self.radius if float(query_radius) < 0.0 else float(query_radius)
        )
        capacity = max(4, int(candidate_capacity))
        self.icm_capacity = capacity
        self.icm_broad_phase = wp.zeros(
            (capacity + 1, self.model.edge_count), dtype=int, device=device)
        # [0] = cloth-edge x rigid-triangle intersections found this substep,
        # [1] = broad-phase candidates handed to the kernel. Zeroed every substep.
        self.icm_stats = wp.zeros(2, dtype=int, device=device)
        # Never zeroed: episode totals, so "did the cloth EVER thread a finger"
        # can be answered without a readback inside the substep loop.
        self.icm_stats_total = wp.zeros(4, dtype=int, device=device)
        self.icm_bvh = BvhTri(self.icm_tri_indices.shape[0], device)
        # Seed the hierarchy from the rest pose; every substep only refits it.
        # ``build`` allocates, so it must stay outside any graph capture.
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.icm_local_pos),
            inputs=[
                self.icm_local_pos,
                self.icm_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                self.model.body_q,
            ],
            outputs=[self.icm_pos],
            device=device,
        )
        self.icm_bvh.build(self.icm_pos, self.icm_tri_indices, self.icm_query_radius)
        import atexit

        def _report_icm_totals(stats=self.icm_stats_total):
            # One line at process exit: crossings summed over the run and the
            # worst single substep. Zero here means the channel had nothing to
            # recover from, which is a result, not a silence.
            try:
                t = stats.numpy()
                print(
                    f"[collision] rigid untangling (ICM) totals: crossings={int(t[0])} "
                    f"max_per_substep={int(t[1])} max_candidates={int(t[2])} "
                    f"capacity_hits={int(t[3])}",
                    flush=True,
                )
            except Exception:
                pass

        atexit.register(_report_icm_totals)
        print(
            "[collision] rigid untangling (ICM): "
            f"{len(shape_ids)} shapes, {len(self.icm_local_pos)} vertices, "
            f"{self.icm_tri_indices.shape[0]} triangles, "
            f"stiff_factor={self.icm_stiff_factor:g}, "
            f"thickness={self.icm_thickness * 1000.0:.2f}mm, "
            f"query_radius={self.icm_query_radius * 1000.0:.2f}mm",
            flush=True,
        )

    def enable_gripper_joint_solve(
        self,
        joint_coord,
        joint_dof,
        child_bodies,
        finger_shapes,
        axis_child,
        kp,
        kd,
        effort,
        mass,
        limit_lo,
        limit_hi,
    ) -> None:
        """R5-B: make prismatic finger joint coordinates unknowns of the cloth solve.

        Each nonlinear iteration solves, per finger, the closed-form 1D
        minimisation of finger inertia + PD drive (effort-clamped) + the
        finger's own contact penalty energy, and translates a solver-side copy
        of that finger's body pose by the resulting ``dq`` — so the NEXT
        iteration's contact forces and friction see the blade move within the
        substep instead of one substep late (doc GRIPPER-CONTACT R5-B).

        Args:
            joint_coord: per-finger ``joint_q`` coordinate index.
            joint_dof: per-finger ``joint_qd``/DOF index (PD target array index).
            child_bodies: per-finger child body index (the moving blade).
            finger_shapes: list of per-finger collision shape id lists.
            axis_child: per-finger prismatic axis in the CHILD body frame
                (``rot(X_cj) @ joint_axis``), so the world axis is just the
                child rotation applied to it.
            kp / kd / effort / mass: per-finger PD gains [N/m, N·s/m], actuator
                effort rating [N] (<=0 = unclamped) and blade mass [kg].
            limit_lo / limit_hi: per-finger joint limits [m].

        Default OFF: unless this is called, ``gjs_joint_coord`` stays ``None``,
        nothing is allocated and no kernel is launched.
        """
        device = self.model.device
        n = len(joint_coord)
        if n == 0:
            return
        slot_map = np.full(self.model.shape_count, -1, dtype=np.int32)
        for slot, shapes in enumerate(finger_shapes):
            for s in shapes:
                slot_map[int(s)] = slot
        self.gjs_joint_coord = wp.array(np.asarray(joint_coord, dtype=np.int32), dtype=int, device=device)
        self.gjs_joint_dof = wp.array(np.asarray(joint_dof, dtype=np.int32), dtype=int, device=device)
        self.gjs_body = wp.array(np.asarray(child_bodies, dtype=np.int32), dtype=int, device=device)
        self.gjs_shape_slot = wp.array(slot_map, dtype=int, device=device)
        self.gjs_axis_child = wp.array(np.asarray(axis_child, dtype=np.float32), dtype=wp.vec3, device=device)
        self.gjs_kp = wp.array(np.asarray(kp, dtype=np.float32), dtype=float, device=device)
        self.gjs_kd = wp.array(np.asarray(kd, dtype=np.float32), dtype=float, device=device)
        self.gjs_effort = wp.array(np.asarray(effort, dtype=np.float32), dtype=float, device=device)
        self.gjs_mass = wp.array(np.asarray(mass, dtype=np.float32), dtype=float, device=device)
        self.gjs_limit_lo = wp.array(np.asarray(limit_lo, dtype=np.float32), dtype=float, device=device)
        self.gjs_limit_hi = wp.array(np.asarray(limit_hi, dtype=np.float32), dtype=float, device=device)
        self.gjs_q = wp.zeros(n, dtype=float, device=device)
        self.gjs_q_prev = wp.zeros(n, dtype=float, device=device)
        self.gjs_q_inertia = wp.zeros(n, dtype=float, device=device)
        self.gjs_axis_w = wp.zeros(n, dtype=wp.vec3, device=device)
        self.gjs_s1 = wp.zeros(n, dtype=float, device=device)
        self.gjs_s2 = wp.zeros(n, dtype=float, device=device)
        self.gjs_body_q = wp.zeros(self.model.body_count, dtype=wp.transform, device=device)
        print(
            "[collision] gripper joint DOFs in cloth solve (R5-B): "
            f"{n} fingers, shapes/slot={[len(s) for s in finger_shapes]}, "
            f"kp={list(np.asarray(kp, dtype=float))}, kd={list(np.asarray(kd, dtype=float))}, "
            f"effort={list(np.asarray(effort, dtype=float))}, mass={list(np.asarray(mass, dtype=float))}",
            flush=True,
        )

    def gripper_joint_begin(self, state_in: State, state_out: State) -> None:
        """Per-substep init: seed jaw pose + DOF state from the rigid solver's step.

        ``state_out.joint_q`` is the external rigid solver's integration of the
        actuator-disabled finger joint (its actuator on these DOFs must be
        disabled by the caller), i.e. the inertial prediction the 1D energy
        needs. All launches are fixed-dimension, so this captures.
        """
        wp.copy(self.gjs_body_q, state_out.body_q)
        wp.launch(
            gjs_setup_kernel,
            dim=len(self.gjs_joint_coord),
            inputs=[
                state_in.joint_q,
                state_out.joint_q,
                self.gjs_body_q,
                self.gjs_joint_coord,
                self.gjs_body,
                self.gjs_axis_child,
            ],
            outputs=[self.gjs_q_prev, self.gjs_q_inertia, self.gjs_q, self.gjs_axis_w],
            device=self.model.device,
        )

    def gripper_joint_iteration(
        self,
        dt: float,
        state_out: State,
        contacts: Contacts,
        control,
    ) -> None:
        """One per-iteration finger DOF update (call after ``nonlinear_step``)."""
        self.gjs_s1.zero_()
        self.gjs_s2.zero_()
        wp.launch(
            gjs_accumulate_kernel,
            dim=self.body_contact_max,
            inputs=[
                state_out.particle_q,
                self.model.particle_radius,
                self.shape_contact_offset,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.material_ke,
                self.particle_contact_ke,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.gjs_shape_slot,
                self.gjs_body,
                self.gjs_axis_w,
                self.gjs_body_q,
            ],
            outputs=[self.gjs_s1, self.gjs_s2],
            device=self.model.device,
        )
        wp.launch(
            gjs_solve_kernel,
            dim=len(self.gjs_joint_coord),
            inputs=[
                dt,
                self.gjs_q_prev,
                self.gjs_q_inertia,
                self.gjs_axis_w,
                self.gjs_s1,
                self.gjs_s2,
                self.gjs_kp,
                self.gjs_kd,
                self.gjs_effort,
                self.gjs_mass,
                self.gjs_limit_lo,
                self.gjs_limit_hi,
                self.gjs_joint_dof,
                self.gjs_body,
                control.joint_target_q,
                control.joint_target_qd,
            ],
            outputs=[self.gjs_q, self.gjs_body_q],
            device=self.model.device,
        )

    def gripper_joint_finish(self, state_out: State, dt: float) -> None:
        """Publish converged q / qd / blade pose into ``state_out`` (see kernel doc)."""
        wp.launch(
            gjs_writeback_kernel,
            dim=len(self.gjs_joint_coord),
            inputs=[
                dt,
                self.gjs_q,
                self.gjs_q_prev,
                self.gjs_joint_coord,
                self.gjs_joint_dof,
                self.gjs_body,
                self.gjs_body_q,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd, state_out.body_q],
            device=self.model.device,
        )

    def _update_rigid_untangling_query(
        self,
        cloth_pos: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
    ) -> None:
        """Refresh the per-cloth-edge candidate list of rigid triangles.

        Same shape as the cloth-cloth EF broad phase (``frame_begin``): the
        query boxes are the cloth EDGE bounds, the BVH holds the rigid
        triangles. All in-place, so it is safe inside a graph capture.
        """
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.icm_local_pos),
            inputs=[
                self.icm_local_pos,
                self.icm_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
            ],
            outputs=[self.icm_pos],
            device=self.model.device,
        )
        self.icm_bvh.refit(self.icm_pos, self.icm_tri_indices, self.icm_query_radius)
        self.edge_bvh.refit(cloth_pos, self.model.edge_indices, self.radius)
        self.icm_bvh.aabb_vs_aabb(
            self.edge_bvh.lower_bounds,
            self.edge_bvh.upper_bounds,
            self.icm_broad_phase,
            self.icm_query_radius,
            False,
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

        if self.icm_tri_indices is not None:
            # R2-3 rigid untangling. Guarded on a Python attribute that is
            # ``None`` unless ``enable_rigid_untangling`` ran, so on the default
            # path nothing below is even queued -- and the branch is resolved at
            # capture time, so the captured graph stays fixed.
            icm_body_q = (
                state_out.body_q
                if self.integrate_with_external_rigid_solver
                else state_in.body_q
            )
            if _iter == 0:
                # The gripper pose is constant within a substep, so the
                # candidate list only has to be rebuilt once per substep; the
                # query box padding covers the cloth's motion across iterations.
                self.icm_stats.zero_()
                self._update_rigid_untangling_query(state_in.particle_q, icm_body_q)
            wp.launch(
                solve_rigid_untangling_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    self.icm_thickness,
                    self.icm_stiff_factor,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.icm_pos,
                    self.icm_tri_indices,
                    self.icm_broad_phase,
                    particle_stiff,
                    self.icm_capacity,
                    1 if _iter == 0 else 0,
                ],
                outputs=[
                    particle_forces,
                    self.contact_hessian_diags,
                    self.icm_stats,
                    self.icm_stats_total,
                ],
                device=self.model.device,
            )

        # R5-B: when the finger DOF solve is on, the current body pose the
        # contact law sees is the per-iteration jaw pose buffer (only the
        # finger entries ever differ from state_out.body_q). Python-level
        # branch, resolved at capture time; with the feature off the value is
        # exactly the expression that was passed inline before.
        body_q_cur = state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q
        if self.gjs_joint_coord is not None:
            body_q_cur = self.gjs_body_q
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
                self.shape_contact_offset,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.material_ke,
                self.particle_contact_ke,
                self.model.shape_material_mu,
                self.model.shape_body,
                body_q_cur,
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

        if self.tri_sdf_slot_shape is not None and self.tri_sdf_compliant:
            wp.launch(
                eval_tri_sdf_contact_kernel,
                dim=self.tri_sdf_slots * self.model.tri_count,
                inputs=[
                    state_out.particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    self.tri_sdf_stiffness,
                    self.tri_sdf_data,
                    self.tri_sdf_base,
                    self.tri_sdf_nx,
                    self.tri_sdf_ny,
                    self.tri_sdf_nz,
                    self.tri_sdf_origin,
                    self.tri_sdf_voxel,
                    self.tri_sdf_bg,
                    self.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                    self.tri_sdf_h,
                    self.tri_sdf_max_correction,
                    self.tri_sdf_refine,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

    def accumulate_tri_sdf_reaction(
        self,
        particle_q: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
    ):
        """Equal-and-opposite wrench of the compliant triangle contact.

        Call AFTER :meth:`accumulate_body_reaction` (which zeroes ``body_f``) so
        both channels land in the same buffer.
        """
        if self.tri_sdf_slot_shape is None or not self.tri_sdf_compliant:
            return
        wp.launch(
            accumulate_tri_sdf_reaction_kernel,
            dim=self.tri_sdf_slots * self.model.tri_count,
            inputs=[
                particle_q,
                self.model.tri_indices,
                int(self.model.tri_count),
                self.tri_sdf_stiffness,
                self.tri_sdf_data,
                self.tri_sdf_base,
                self.tri_sdf_nx,
                self.tri_sdf_ny,
                self.tri_sdf_nz,
                self.tri_sdf_origin,
                self.tri_sdf_voxel,
                self.tri_sdf_bg,
                self.tri_sdf_slot_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
                body_com,
                body_enabled,
                self.tri_sdf_h,
                self.tri_sdf_max_correction,
                self.tri_sdf_refine,
            ],
            outputs=[body_f],
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
                self.shape_contact_offset,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.material_ke,
                self.particle_contact_ke,
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
        # Positions as the solve left them, before the last sweep moved them.
        # With slack = 0 the projected state carries no penetration and so no
        # penalty reaction; this snapshot is where the reaction has to be read
        # instead (ke x the depth the solve settled at = the load the jaw is
        # actually pressing with).
        self.proj_pre_q = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        # Which shapes the projection acts on. It exists to fix gripper-cloth
        # penetration; applying it to the table as well makes the cloth resist
        # sliding as it spreads (measured: 0.22 of flatten coverage). Default
        # keeps every shape so behaviour is unchanged unless a caller narrows it.
        if getattr(self, "projection_shape_enabled", None) is None:
            self.projection_shape_enabled = wp.ones(
                self.model.shape_count, dtype=int, device=device
            )
        print(
            "[collision] contact projection: "
            f"slack={self.projection_slack:g} m, iterations={self.projection_iterations}, "
            f"friction_scale={self.projection_friction_scale:g}, "
            f"mode={'interleaved' if self.projection_interleaved else 'post'}",
            flush=True,
        )

    def enable_material_contact_stiffness(self, modulus: float, thickness: float) -> None:
        """Derive the contact stiffness from the cloth material instead of a constant.

        ``k_i = modulus * A_i / thickness`` with ``A_i`` the particle's tributary
        area on the rest mesh (each incident triangle contributes a third of its
        area). A per-shape constant ``ke`` is mesh-coupled: the penalty is
        per-particle, so refining the cloth 9k -> 40k multiplies the particle
        count by ~4 and the total normal load with it, and the constant has to be
        retuned per mesh. ``sum_i A_i`` is the cloth's area whatever the
        tessellation, so this form holds the load invariant under remeshing and
        leaves ONE parameter with physical units (a transverse compression
        modulus [Pa]) instead of a per-configuration knob.

        Args:
            modulus: E_t [Pa]. <= 0 disables and restores the per-shape constant.
            thickness: cloth thickness t0 [m] (asset meta ``thickness``).
        """
        modulus = float(modulus)
        if modulus <= 0.0 or thickness <= 0.0:
            self.material_ke = 0.0
            return
        pos = self.model.particle_q.numpy().astype(np.float64)
        tris = self.model.tri_indices.numpy().astype(np.int64)
        e1 = pos[tris[:, 1]] - pos[tris[:, 0]]
        e2 = pos[tris[:, 2]] - pos[tris[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        tributary = np.zeros(self.model.particle_count, dtype=np.float64)
        for corner in range(3):
            np.add.at(tributary, tris[:, corner], area / 3.0)
        ke = modulus * tributary / float(thickness)
        self.particle_contact_ke.assign(ke.astype(np.float32))
        self.material_ke = modulus
        nz = tributary[tributary > 0.0]
        print(
            "[collision] material contact stiffness: "
            f"E_t={modulus:g} Pa, t0={thickness * 1000:g} mm, "
            f"cloth area={tributary.sum():.5f} m^2, "
            f"A_i mean={nz.mean() * 1e6:.3f} mm^2 -> k_i mean={ke[tributary > 0.0].mean():.1f} N/m "
            f"(range {ke[tributary > 0.0].min():.1f}..{ke.max():.1f})",
            flush=True,
        )

    def enable_triangle_sdf_contacts(
        self,
        shape_ids,
        half_thickness: float,
        voxel: float = 1.0e-3,
        pad: float = 6.0e-3,
        refine_steps: int = 3,
        max_correction: float = 5.0e-3,
        compliant: bool = False,
        modulus: float = 0.0,
    ) -> None:
        """Enable E3 triangle-level contact against a baked per-shape SDF.

        Vertex-sphere detection is blind to a blade crossing the INTERIOR of a
        cloth triangle, which is why ``particle_radius`` had to be inflated to
        the mesh aperture (8 mm for a 7.5 mm p99 aperture) -- a contact model
        coupled to the tessellation. This constrains the whole triangle instead:
        ``min_{x in tri} SDF_shape(x) >= half_thickness``. Nothing in the kernel
        reads ``particle_radius`` or an edge length, so the constraint is mesh
        independent.

        The SDF is baked once per shape (rigid bodies, so the local-frame field
        never changes) on a dense ``voxel`` grid padded by ``pad``; a 45x71x22 mm
        finger at 1 mm is ~0.2 M voxels, i.e. under a megabyte.

        Args:
            shape_ids: Shapes to constrain against (typically the gripper fingers).
            half_thickness: h [m] -- the cloth half-thickness the surface is held
                off the solid by. This is the ONLY length scale in the model.
            voxel: SDF grid spacing [m].
            pad: Grid padding around the shape bounds [m]; triangles further than
                this from the shape are rejected by one compare.
            refine_steps: Projected steepest-descent steps in barycentric space
                after the 3-vertex + centroid seeding.
            max_correction: Per-sweep displacement cap [m]. Bounds the response
                to a triangle that is already deep inside (e.g. right after a
                teleporting reset) instead of launching it.
            compliant: Run the constraint as a PENALTY FORCE into the implicit
                solve (and its equal-and-opposite onto the body) instead of as a
                position projection. The projection form removes exactly the
                overlap the penalty channel needs, so the two compete for the
                same quantity and the reaction collapses once the collision
                radius is retired; the compliant form has ONE mechanism carrying
                both the geometry and the force.
            modulus: E_t [Pa] for the compliant form. ``k_tri = E_t * A_tri / t0``
                with A_tri the REST triangle area, so sum_tri A_tri is the cloth's
                area regardless of tessellation and the load is mesh independent.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids:
            return
        device = self.model.device
        shape_scale = self.model.shape_scale.numpy()

        blocks, bases, dims, origins, slots = [], [], [], [], []
        total = 0
        # ``bg`` is the out-of-grid sentinel only. The bake itself queries out to
        # ``bake_max_dist`` so the SOLID INTERIOR carries a true negative
        # distance: querying only to the pad would leave anything deeper than the
        # pad reading +pad, i.e. a deeply penetrated triangle would look free and
        # the constraint would release it.
        bg = 1.0e3
        bake_max_dist = 0.05
        for shape in shape_ids:
            source = self.model.shape_source[shape]
            vertices = np.asarray(source.vertices, dtype=np.float32)
            indices = np.asarray(source.indices, dtype=np.int32).reshape(-1)
            if not len(vertices) or not len(indices):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)
            mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=device),
                indices=wp.array(indices, dtype=wp.int32, device=device),
            )
            lo = vertices.min(axis=0) - pad
            hi = vertices.max(axis=0) + pad
            n = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int32) + 1, 2)
            count = int(n[0]) * int(n[1]) * int(n[2])
            grid = wp.zeros(count, dtype=float, device=device)
            wp.launch(
                bake_shape_sdf_kernel,
                dim=(int(n[0]), int(n[1]), int(n[2])),
                inputs=[
                    mesh.id,
                    wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
                    float(voxel),
                    int(n[0]),
                    int(n[1]),
                    int(n[2]),
                    0,
                    bake_max_dist,
                ],
                outputs=[grid],
                device=device,
            )
            blocks.append(grid.numpy())
            bases.append(total)
            dims.append(n)
            origins.append(lo)
            slots.append(shape)
            total += count

        if not blocks:
            return
        dims = np.asarray(dims, dtype=np.int32)
        self.tri_sdf_data = wp.array(np.concatenate(blocks).astype(np.float32), dtype=float, device=device)
        self.tri_sdf_base = wp.array(np.asarray(bases, dtype=np.int32), dtype=int, device=device)
        self.tri_sdf_nx = wp.array(dims[:, 0].copy(), dtype=int, device=device)
        self.tri_sdf_ny = wp.array(dims[:, 1].copy(), dtype=int, device=device)
        self.tri_sdf_nz = wp.array(dims[:, 2].copy(), dtype=int, device=device)
        self.tri_sdf_origin = wp.array(np.asarray(origins, dtype=np.float32), dtype=wp.vec3, device=device)
        self.tri_sdf_slot_shape = wp.array(np.asarray(slots, dtype=np.int32), dtype=int, device=device)
        self.tri_sdf_voxel = float(voxel)
        self.tri_sdf_bg = bg
        self.tri_sdf_h = float(half_thickness)
        self.tri_sdf_refine = max(int(refine_steps), 0)
        self.tri_sdf_max_correction = float(max_correction)
        self.tri_sdf_slots = len(slots)
        self.tri_sdf_compliant = bool(compliant)
        pos_rest = self.model.particle_q.numpy().astype(np.float64)
        tris = self.model.tri_indices.numpy().astype(np.int64)
        e1 = pos_rest[tris[:, 1]] - pos_rest[tris[:, 0]]
        e2 = pos_rest[tris[:, 2]] - pos_rest[tris[:, 0]]
        tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        t0 = max(float(half_thickness) * 2.0, 1.0e-9)
        k_tri = (float(modulus) * tri_area / t0) if modulus > 0.0 else np.zeros_like(tri_area)
        self.tri_sdf_stiffness = wp.array(k_tri.astype(np.float32), dtype=float, device=device)
        if self.tri_sdf_compliant:
            print(
                "[collision] triangle SDF contact is COMPLIANT: "
                f"E_t={modulus:g} Pa, t0={t0 * 1000:g} mm, "
                f"cloth area={tri_area.sum():.5f} m^2, "
                f"k_tri mean={k_tri.mean():.1f} N/m (range {k_tri.min():.1f}..{k_tri.max():.1f})",
                flush=True,
            )
        # Jacobi accumulators. The position-projection pass owns the same three
        # buffers; allocate only if it did not (the two are independent switches).
        if getattr(self, "proj_delta", None) is None:
            self.proj_delta = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
            self.proj_weight = wp.zeros(self.model.particle_count, dtype=float, device=device)
            self.proj_accum = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        if getattr(self, "projection_relaxation", None) is None:
            self.projection_relaxation = 1.0
        print(
            "[collision] triangle SDF contacts: "
            f"{len(slots)} shapes, {total} voxels @ {voxel * 1000:g} mm, "
            f"h={half_thickness * 1000:g} mm, refine={self.tri_sdf_refine}, "
            f"max_corr={max_correction * 1000:g} mm",
            flush=True,
        )

    def _tri_sdf_sweep(self, particle_q: wp.array[wp.vec3], body_q: wp.array[wp.transform]):
        """One Jacobi sweep of the triangle-level SDF constraint."""
        wp.launch(
            project_tri_sdf_kernel,
            dim=self.tri_sdf_slots * self.model.tri_count,
            inputs=[
                particle_q,
                self.model.tri_indices,
                int(self.model.tri_count),
                self.tri_sdf_data,
                self.tri_sdf_base,
                self.tri_sdf_nx,
                self.tri_sdf_ny,
                self.tri_sdf_nz,
                self.tri_sdf_origin,
                self.tri_sdf_voxel,
                self.tri_sdf_bg,
                self.tri_sdf_slot_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
                self.tri_sdf_h,
                self.tri_sdf_max_correction,
                self.tri_sdf_refine,
            ],
            outputs=[self.proj_delta, self.proj_weight],
            device=self.model.device,
        )
        wp.launch(
            apply_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[self.projection_relaxation, self.model.particle_flags],
            outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
            device=self.model.device,
        )

    def set_projection_shapes(self, shape_indices) -> None:
        """Restrict the position projection to the given shapes.

        Args:
            shape_indices: Shapes the projection may move particles out of.
                Everything else keeps only its penalty contact.
        """
        import numpy as np

        mask = np.zeros(self.model.shape_count, dtype=np.int32)
        idx = np.asarray(list(shape_indices), dtype=np.int32)
        if len(idx):
            mask[idx] = 1
        self.projection_shape_enabled = wp.array(mask, dtype=int, device=self.model.device)
        print(f"[collision] contact projection limited to {int(mask.sum())} shapes", flush=True)

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
        if self.projection_interleaved and self.projection_iterations > 0:
            self._project_sweep(particle_q, particle_q_prev, contacts, body_q, body_q_prev)
        # E3 triangle-level SDF constraint: an independent switch, so it can run
        # on the simplified stack (position projection off, E0c) without
        # dragging the particle-radius projection back in.
        if self.tri_sdf_slot_shape is not None and not self.tri_sdf_compliant:
            # compliant mode carries the constraint as a force instead (see
            # accumulate_contact_force), so there is no position sweep here.
            self._tri_sdf_sweep(particle_q, body_q)

    def _project_sweep(
        self,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform] | None,
    ):
        """One Jacobi sweep: accumulate corrections from all contact types, apply."""
        self.proj_pre_q.assign(particle_q)
        wp.launch(
            project_body_particle_contacts_kernel,
            dim=self.body_contact_max,
            inputs=[
                self.projection_slack,
                self.projection_friction_scale,
                self.model.soft_contact_mu,
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
                self.projection_shape_enabled,
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
                    self.model.soft_contact_mu,
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
                    self.projection_shape_enabled,
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
