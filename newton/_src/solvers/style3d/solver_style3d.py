# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import ParticleFlags
from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from .builder import PDMatrixBuilder
from .collision import Collision
from .kernels import (
    accumulate_dragging_pd_diag_kernel,
    eval_bend_kernel,
    eval_drag_force_kernel,
    eval_stretch_kernel,
    init_rhs_kernel,
    init_step_kernel,
    nonlinear_step_kernel,
    prepare_jacobi_preconditioner_kernel,
    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
    update_velocity,
)
from .linear_solver import PcgSolver, SparseMatrixELL

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


@wp.kernel
def _accumulate_translation_preconditioner_kernel(
    dt: float,
    residual: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_masses: wp.array[float],
    particle_flags: wp.array[wp.int32],
    contact_hessian_diags: wp.array[wp.mat33],
    # outputs
    coarse_rhs: wp.array[wp.vec3],
    coarse_diag: wp.array[wp.vec3],
):
    tid = wp.tid()
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        comp_idx = wp.min(wp.max(particle_component[tid], 0), coarse_diag.shape[0] - 1)
        mass_diag = particle_masses[tid] / (dt * dt)
        contact_hess = contact_hessian_diags[tid]
        wp.atomic_add(coarse_rhs, comp_idx, residual[tid])
        wp.atomic_add(
            coarse_diag,
            comp_idx,
            wp.vec3(
                mass_diag + contact_hess[0, 0],
                mass_diag + contact_hess[1, 1],
                mass_diag + contact_hess[2, 2],
            ),
        )


@wp.kernel
def _apply_translation_preconditioner_kernel(
    coarse_rhs: wp.array[wp.vec3],
    coarse_diag: wp.array[wp.vec3],
    particle_component: wp.array[wp.int32],
    particle_flags: wp.array[wp.int32],
    # outputs
    z: wp.array[wp.vec3],
):
    tid = wp.tid()
    comp_idx = wp.min(wp.max(particle_component[tid], 0), coarse_diag.shape[0] - 1)
    denom = coarse_diag[comp_idx]
    rhs = coarse_rhs[comp_idx]
    if particle_flags[tid] & ParticleFlags.ACTIVE:
        correction = wp.vec3(0.0)
        if denom[0] > 0.0:
            correction[0] = rhs[0] / denom[0]
        if denom[1] > 0.0:
            correction[1] = rhs[1] / denom[1]
        if denom[2] > 0.0:
            correction[2] = rhs[2] / denom[2]
        z[tid] = z[tid] + correction


########################################################################################################################
#################################################    Style3D Solver    #################################################
########################################################################################################################


class SolverStyle3D(SolverBase):
    r"""Projective dynamics based cloth solver.

    References:
        1. Baraff, D. & Witkin, A. "Large Steps in Cloth Simulation."
        2. Liu, T. et al. "Fast Simulation of Mass-Spring Systems."

    Implicit-Euler method solves the following non-linear equation:

    .. math::

        (M / dt^2 + H(x)) \cdot dx &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt - x) + f_{ext}(x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{prev} + v_{prev} \cdot dt + (dt^2 / M) \cdot f_{ext}(x) - x) + f_{int}(x) \\
                                   &= (M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)

    Notations:
        - :math:`M`: mass matrix
        - :math:`x`: unsolved particle position
        - :math:`H`: Hessian matrix (function of x)
        - :math:`P`: PD-approximated Hessian matrix (constant)
        - :math:`A`: :math:`M / dt^2 + H(x)` or :math:`M / dt^2 + P`
        - :math:`rhs`: Right hand side of the equation: :math:`(M / dt^2) \cdot (x_{inertia} - x) + f_{int}(x)`
        - :math:`res`: Residual: :math:`rhs - A \cdot dx_{init}`, or rhs if :math:`dx_{init} = 0`

    See Also:
        :doc:`newton.solvers.style3d </api/newton_solvers_style3d>` exposes
        helper functions that populate Style3D cloth data on a
        :class:`~newton.ModelBuilder`.

    Example:
        Build a mesh-based cloth with
        :func:`newton.solvers.style3d.add_cloth_mesh`::

            from newton.solvers import style3d

            builder = newton.ModelBuilder()
            SolverStyle3D.register_custom_attributes(builder)
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=mesh.vertices.tolist(),
                indices=mesh.indices.tolist(),
                density=0.3,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-5, 1.0e-5, 5.0e-6),
            )

        Or build a grid with :func:`newton.solvers.style3d.add_cloth_grid`::

            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(-0.5, 0.0, 2.0),
                rot=wp.quat_identity(),
                dim_x=64,
                dim_y=32,
                cell_x=0.1,
                cell_y=0.1,
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=0.1,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
                edge_aniso_ke=wp.vec3(2.0e-4, 1.0e-4, 5.0e-5),
            )

    """

    def __init__(
        self,
        model: Model,
        iterations: int = 10,
        linear_iterations: int = 10,
        drag_spring_stiff: float = 1e2,
        enable_mouse_dragging: bool = False,
        enable_translation_preconditioner: bool = False,
    ):
        """
        Args:
            model: The :class:`~newton.Model` containing Style3D attributes to integrate.
            iterations: Number of non-linear iterations per step.
            linear_iterations: Number of linear iterations (currently PCG iter) per non-linear iteration.
            drag_spring_stiff: The stiffness of spring connecting barycentric-weighted drag-point and target-point.
            enable_mouse_dragging: Enable/disable dragging kernel.
            enable_translation_preconditioner: Enable a coarse per-component translation preconditioner for PCG.
        """

        super().__init__(model)
        if not hasattr(model, "style3d"):
            raise AttributeError(
                "Style3D custom attributes are missing from the model. "
                "Call SolverStyle3D.register_custom_attributes() before building the model."
            )
        self.style3d = model.style3d
        self.collision: Collision | None = Collision(model)  # set None to disable
        self.linear_iterations = linear_iterations
        self.nonlinear_iterations = iterations
        self.drag_spring_stiff = drag_spring_stiff
        self.enable_mouse_dragging = enable_mouse_dragging
        self.enable_translation_preconditioner = enable_translation_preconditioner
        self.linear_solver = PcgSolver(model.particle_count, self.device)

        # Fixed PD matrix
        self.pd_non_diags = SparseMatrixELL()
        self.pd_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self._precompute(model)

        # Non-linear equation variables
        self.dx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.rhs = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_prev = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)
        self.x_inertia = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device)

        # Static part of A_diag, full A_diag, and inverse of A_diag
        self.static_A_diags = wp.zeros(model.particle_count, dtype=float, device=self.device)
        self.inv_A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self.A_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)

        # Coarse translation preconditioner buffers.
        self._translation_preconditioner_dt = 0.0
        self._translation_component_count, particle_component = self._build_particle_components(model)
        self._translation_particle_component = wp.array(particle_component, dtype=wp.int32, device=self.device)
        self._translation_coarse_rhs = wp.zeros(self._translation_component_count, dtype=wp.vec3, device=self.device)
        self._translation_coarse_diag = wp.zeros(self._translation_component_count, dtype=wp.vec3, device=self.device)
        self._translation_zero_contact_hessian = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.device)
        self._translation_contact_hessian_diags = self._translation_zero_contact_hessian

        # Drag info
        self.drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.drag_index = wp.array([-1], dtype=int, device=self.device)
        self.drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    @override
    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float) -> None:
        """Advance the Style3D solver by one time step.

        The solver performs non-linear projective dynamics iterations with
        optional collision handling. During the solve, positions in
        ``state_in`` are updated in-place to the current iterate; the final
        positions and velocities are written to ``state_out``.

        Args:
            state_in: Input :class:`newton.State` (positions updated in-place).
            state_out: Output :class:`newton.State` with the final state.
            control: :class:`newton.Control` input (currently unused).
            contacts: :class:`newton.Contacts` used for collision response.
            dt: Time step in seconds.
        """
        if self.collision is not None:
            self.collision.frame_begin(state_in.particle_q, state_in.particle_qd, dt)
        self._translation_preconditioner_dt = dt
        self._translation_contact_hessian_diags = self._translation_zero_contact_hessian

        wp.launch(
            kernel=init_step_kernel,
            dim=self.model.particle_count,
            inputs=[
                dt,
                self.model.gravity,
                self.model.particle_world,
                state_in.particle_f,
                state_in.particle_qd,
                state_in.particle_q,
                self.x_prev,
                self.pd_diags,
                self.model.particle_mass,
                self.model.particle_flags,
            ],
            outputs=[
                self.x_inertia,
                self.static_A_diags,
                self.dx,
            ],
            device=self.device,
        )

        if self.enable_mouse_dragging:
            wp.launch(
                accumulate_dragging_pd_diag_kernel,
                dim=1,
                inputs=[
                    self.drag_spring_stiff,
                    self.drag_index,
                    self.drag_bary_coord,
                    self.model.tri_indices,
                    self.model.particle_flags,
                ],
                outputs=[self.static_A_diags],
                device=self.device,
            )

        for _iter in range(self.nonlinear_iterations):
            wp.launch(
                init_rhs_kernel,
                dim=self.model.particle_count,
                inputs=[
                    dt,
                    state_in.particle_q,
                    self.x_inertia,
                    self.model.particle_mass,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_stretch_kernel,
                dim=len(self.model.tri_areas),
                inputs=[
                    state_in.particle_q,
                    self.model.tri_areas,
                    self.model.tri_poses,
                    self.model.tri_indices,
                    self.style3d.tri_aniso_ke,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            wp.launch(
                eval_bend_kernel,
                dim=len(self.style3d.edge_rest_area),
                inputs=[
                    state_in.particle_q,
                    self.style3d.edge_rest_area,
                    self.style3d.edge_bending_cot,
                    self.model.edge_indices,
                    self.model.edge_bending_properties,
                ],
                outputs=[self.rhs],
                device=self.device,
            )

            if self.enable_mouse_dragging:
                wp.launch(
                    eval_drag_force_kernel,
                    dim=1,
                    inputs=[
                        self.drag_spring_stiff,
                        self.drag_index,
                        self.drag_pos,
                        self.drag_bary_coord,
                        self.model.tri_indices,
                        state_in.particle_q,
                    ],
                    outputs=[self.rhs],
                    device=self.device,
                )

            if self.collision is not None:
                self.collision.accumulate_contact_force(
                    dt,
                    _iter,
                    state_in,
                    state_out,
                    contacts,
                    self.rhs,
                    self.x_prev,
                    self.static_A_diags,
                )
                self._translation_contact_hessian_diags = self.collision.contact_hessian_diagonal()
                wp.launch(
                    prepare_jacobi_preconditioner_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.static_A_diags,
                        self.collision.contact_hessian_diagonal(),
                        self.model.particle_flags,
                    ],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )
            else:
                wp.launch(
                    prepare_jacobi_preconditioner_no_contact_hessian_kernel,
                    dim=self.model.particle_count,
                    inputs=[self.static_A_diags],
                    outputs=[self.inv_A_diags],
                    device=self.device,
                )

            hessian_multiply = None if self.collision is None else self.collision.hessian_multiply
            if self.enable_translation_preconditioner:
                self.linear_solver.solve(
                    self.pd_non_diags,
                    self.static_A_diags,
                    self.dx if _iter == 0 else None,
                    self.rhs,
                    self.inv_A_diags,
                    self.dx,
                    self.linear_iterations,
                    hessian_multiply,
                    self._apply_translation_preconditioner,
                )
            else:
                self.linear_solver.solve(
                    self.pd_non_diags,
                    self.static_A_diags,
                    self.dx if _iter == 0 else None,
                    self.rhs,
                    self.inv_A_diags,
                    self.dx,
                    wp.min(_iter + 1, 10),
                    hessian_multiply,
                )

            if self.collision is not None:
                self.collision.linear_iteration_end(self.dx)

            wp.launch(
                nonlinear_step_kernel,
                dim=self.model.particle_count,
                inputs=[state_in.particle_q],
                outputs=[state_out.particle_q, self.dx],
                device=self.device,
            )

            if self.collision is not None:
                # PBD-style alternation: project contacts BEFORE the assign, so
                # the next iteration's nonlinear_step_kernel (which reads
                # state_in) starts from the projected positions. Projecting
                # after the assign would silently discard the correction every
                # iteration. No-op unless projection is in interleaved mode.
                self.collision.project_contacts_iteration(
                    particle_q=state_out.particle_q,
                    particle_q_prev=self.x_prev,
                    contacts=contacts,
                    body_q=state_out.body_q if self.collision.integrate_with_external_rigid_solver else state_in.body_q,
                    dt=dt,
                    body_q_prev=state_in.body_q if self.collision.integrate_with_external_rigid_solver else None,
                )

            state_in.particle_q.assign(state_out.particle_q)

        wp.launch(
            kernel=update_velocity,
            dim=self.model.particle_count,
            inputs=[dt, self.x_prev, state_out.particle_q],
            outputs=[state_out.particle_qd],
            device=self.device,
        )

        if self.collision is not None:
            self.collision.frame_end(state_out.particle_q, state_out.particle_qd, dt)

    def rebuild_bvh(self, state: State):
        if self.collision is not None:
            self.collision.rebuild_bvh(state.particle_q)

    @staticmethod
    def _build_particle_components(model: Model) -> tuple[int, np.ndarray]:
        parent = list(range(model.particle_count))
        rank = [0] * model.particle_count
        particle_world = model.particle_world.numpy() if model.particle_world is not None else None

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            if a < 0 or b < 0 or a >= model.particle_count or b >= model.particle_count:
                return
            if particle_world is not None and particle_world[a] != particle_world[b]:
                return
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        if model.tri_indices is not None:
            for tri in model.tri_indices.numpy():
                a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
                union(a, b)
                union(b, c)
                union(c, a)

        component_map: dict[int, int] = {}
        particle_component = np.empty(model.particle_count, dtype=np.int32)
        for particle in range(model.particle_count):
            root = find(particle)
            component = component_map.setdefault(root, len(component_map))
            particle_component[particle] = component
        return max(1, len(component_map)), particle_component

    def _apply_translation_preconditioner(self, residual: wp.array[wp.vec3], z: wp.array[wp.vec3]) -> None:
        self._translation_coarse_rhs.zero_()
        self._translation_coarse_diag.zero_()
        wp.launch(
            _accumulate_translation_preconditioner_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_preconditioner_dt,
                residual,
                self._translation_particle_component,
                self.model.particle_mass,
                self.model.particle_flags,
                self._translation_contact_hessian_diags,
            ],
            outputs=[
                self._translation_coarse_rhs,
                self._translation_coarse_diag,
            ],
            device=self.device,
        )
        wp.launch(
            _apply_translation_preconditioner_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._translation_coarse_rhs,
                self._translation_coarse_diag,
                self._translation_particle_component,
                self.model.particle_flags,
            ],
            outputs=[z],
            device=self.device,
        )

    @override
    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Declare Style3D custom attributes under the ``style3d`` namespace.

        See Also:
            :ref:`custom_attributes` for the custom attribute system overview.
        """
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="tri_aniso_ke",
                frequency=AttributeFrequency.TRIANGLE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_rest_area",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.float32,
                default=0.0,
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="edge_bending_cot",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec4,
                default=wp.vec4(0.0, 0.0, 0.0, 0.0),
                namespace="style3d",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="aniso_ke",
                frequency=AttributeFrequency.EDGE,
                assignment=AttributeAssignment.MODEL,
                dtype=wp.vec3,
                default=wp.vec3(0.0),
                namespace="style3d",
            )
        )

    def _precompute(self, model: Model):
        with wp.ScopedTimer("SolverStyle3D::precompute()"):
            if (
                not hasattr(model, "style3d")
                or not hasattr(model.style3d, "tri_aniso_ke")
                or not hasattr(model.style3d, "edge_rest_area")
                or not hasattr(model.style3d, "edge_bending_cot")
            ):
                raise AttributeError(
                    "Style3D custom attributes are missing from the model. "
                    "Call SolverStyle3D.register_custom_attributes() before building the model."
                )

            pd_matrix_builder = PDMatrixBuilder(model.particle_count)
            tri_indices = model.tri_indices.numpy().tolist()
            tri_poses = model.tri_poses.numpy().tolist()
            tri_areas = model.tri_areas.numpy().tolist()
            edge_indices = model.edge_indices.numpy().tolist()
            edge_bending_properties = model.edge_bending_properties.numpy().tolist()
            tri_aniso_ke = model.style3d.tri_aniso_ke.numpy().tolist()
            edge_rest_area = model.style3d.edge_rest_area.numpy().tolist()
            edge_bending_cot = model.style3d.edge_bending_cot.numpy().tolist()

            pd_matrix_builder.add_stretch_constraints(tri_indices, tri_poses, tri_aniso_ke, tri_areas)
            pd_matrix_builder.add_bend_constraints(
                edge_indices,
                edge_bending_properties,
                edge_rest_area,
                edge_bending_cot,
            )
            self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = pd_matrix_builder.finalize(self.device)

    def _update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self.drag_bary_coord.fill_(bary_coord)
        self.drag_index.fill_(index)
        self.drag_pos.fill_(pos)
