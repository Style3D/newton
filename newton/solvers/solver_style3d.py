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
import numpy as np
from .solver import SolverBase
from newton.core import PARTICLE_FLAG_ACTIVE, Contact, Control, Model, State


wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    def to(self, device):
        if device.is_cpu:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]





@wp.func
def triangle_deformation_gradient(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, inv_dm: wp.mat22):
    x01 = x1 - x0
    x02 = x2 - x0
    Fu = x01 * inv_dm[0, 0] + x02 * inv_dm[1, 0]
    Fv = x01 * inv_dm[0, 1] + x02 * inv_dm[1, 1]
    return Fu, Fv




@wp.func
def evalate_stretch_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    inv_dm: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
):
    """ Ref. Large Steps in Cloth Simulation, Baraff & Witkin in 1998.
    """
    Fu, Fv = triangle_deformation_gradient(pos[tri_indices[face, 0]],
                                           pos[tri_indices[face, 1]],
                                           pos[tri_indices[face, 2]],
                                           inv_dm)
    
    Fs1 = (Fu + Fv) / wp.sqrt(2.0)
    Fs2 = (Fu - Fv) / wp.sqrt(2.0)

    len_Fu = wp.length(Fu)
    len_Fv = wp.length(Fv)

    len_Fs1 = wp.length(Fs1)
    len_Fs2 = wp.length(Fs2)

    normalized_Fu = wp.normalize(Fu) if (len_Fu > 1e-6) else wp.vec3(0.0)
    normalized_Fv = wp.normalize(Fv) if (len_Fv > 1e-6) else wp.vec3(0.0)
    normalized_Fs1 = wp.normalize(Fs1) if (len_Fs1 > 1e-6) else wp.vec3(0.0)
    normalized_Fs2 = wp.normalize(Fs2) if (len_Fs2 > 1e-6) else wp.vec3(0.0)

    if v_order == 0:
        dFu_dx = -inv_dm[0, 0] - inv_dm[1, 0]
        dFv_dx = -inv_dm[0, 1] - inv_dm[1, 1]
    elif v_order == 1:
        dFu_dx = inv_dm[0, 0]
        dFv_dx = inv_dm[0, 1]
    else:
        dFu_dx = inv_dm[1, 0]
        dFv_dx = inv_dm[1, 1]

    dFs1_dx = (dFu_dx + dFv_dx) / wp.sqrt(2.0)
    dFs2_dx = (dFu_dx - dFv_dx) / wp.sqrt(2.0)

    ku = 5e4
    kv = 5e4
    ks = 1e4

    f = -area * (ku * (len_Fu - 1.0) * dFu_dx * normalized_Fu +
                 kv * (len_Fv - 1.0) * dFv_dx * normalized_Fv +
                 ks * (len_Fs1 - 1.0) * dFs1_dx * normalized_Fs1 +
                 ks * (len_Fs2 - 1.0) * dFs2_dx * normalized_Fs2)

    h = area * (ku / len_Fu * dFu_dx * dFu_dx * wp.outer(normalized_Fu, normalized_Fu) +
                kv / len_Fv * dFv_dx * dFv_dx * wp.outer(normalized_Fv, normalized_Fv) +
                ks / len_Fs1 * dFs1_dx * dFs1_dx * wp.outer(normalized_Fs1, normalized_Fs1) +
                ks / len_Fs2 * dFs2_dx * dFs2_dx * wp.outer(normalized_Fs2, normalized_Fs2))

    if len_Fu > 1.0:
        h += area * ku * dFu_dx * dFu_dx * (1.0 - 1.0 / len_Fu) * wp.identity(n = 3, dtype = float)

    if len_Fv > 1.0:
        h += area * kv * dFv_dx * dFv_dx * (1.0 - 1.0 / len_Fv) * wp.identity(n = 3, dtype = float)



    return f, wp.identity(n = 3, dtype = float) * (area * (ku * dFu_dx * dFu_dx + kv * dFv_dx * dFv_dx + ks * dFs1_dx * dFs1_dx + ks * dFs2_dx * dFs2_dx))
    #return f, h


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.vec3,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    prev_pos[particle] = pos[particle]
    if not particle_flags[particle] & PARTICLE_FLAG_ACTIVE:
        inertia[particle] = prev_pos[particle]
        return
    vel_new = vel[particle] + (gravity + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]


@wp.kernel
def VBD_solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    # ground-particle contact
    has_ground: bool,
    ground: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    # output
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    particle_index = tid
    particle_pos = pos[particle_index]

    if not particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE:
        pos_new[particle_index] = particle_pos
        return

    particle_prev_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_id, 0],
                tri_indices[tri_id, 1],
                tri_indices[tri_id, 2],
            )
        # fmt: on

        temp = wp.vec3i(tri_indices[tri_id, 0], tri_indices[tri_id, 1], tri_indices[tri_id, 2])

        f_tri, h_tri = evalate_stretch_force_hessian(
            tri_id,
            particle_order,
            pos,
            tri_indices,
            tri_poses[tri_id],
            tri_areas[tri_id],
            tri_materials[tri_id, 0],
            tri_materials[tri_id, 1],
            tri_materials[tri_id, 2],
        )
        # compute damping
        k_d = tri_materials[tri_id, 2]
        h_d = h_tri * (k_d / dt)

        f_d = h_d * (prev_pos[particle_index] - pos[particle_index])

        f = f + f_tri + f_d * 0.1
        h = h + h_tri + h_d * 0.1

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_adj_tri,
                particle_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
        # fmt: on


    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


@wp.kernel
def VBD_copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = tid
    pos[particle] = pos_new[particle]


@wp.kernel
def update_velocity(dt: float, prev_pos: wp.array(dtype = wp.vec3), pos: wp.array(dtype = wp.vec3), vel: wp.array(dtype = wp.vec3)):
    particle = wp.tid()

    vel[particle] = (pos[particle] - prev_pos[particle]) / dt


class Style3DSolver(SolverBase):
    """
    """

    def __init__(
        self,
        model: Model,
        iterations=10,
        handle_self_contact=False,
        penetration_free_conservative_bound_relaxation=0.42,
        friction_epsilon=1e-2,
        body_particle_contact_buffer_pre_alloc=4,
        vertex_collision_buffer_pre_alloc=32,
        edge_collision_buffer_pre_alloc=64,
        triangle_collision_buffer_pre_alloc=32,
        edge_edge_parallel_epsilon=1e-5,
    ):
        super().__init__(model)
        self.iterations = iterations

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact

        self.collision_evaluation_kernel_launch_size = self.model.soft_contact_max

        # spaces for particle force and hessian
        self.particle_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.particle_hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.device)

        self.friction_epsilon = friction_epsilon

        if len(self.model.particle_color_groups) == 0:
            raise ValueError(
                "model.particle_color_groups is empty! When using the VBDIntegrator you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )


    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")

        if edges_array.size:
            # build vertex-edge adjacency data
            num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")

            wp.launch(
                kernel=self.count_num_adjacent_edges,
                inputs=[edges_array, num_vertex_adjacent_edges],
                dim=1,
                device="cpu",
            )

            num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
            vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
            vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
            vertex_adjacent_edges_offsets[0] = 0
            adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32, device="cpu")

            # temporal variables to record how much adjacent edges has been filled to each vertex
            vertex_adjacent_edges_fill_count = wp.zeros(
                shape=(self.model.particle_count,), dtype=wp.int32, device="cpu"
            )

            edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
            # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
            adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32, device="cpu")

            wp.launch(
                kernel=self.fill_adjacent_edges,
                inputs=[
                    edges_array,
                    adjacency.v_adj_edges_offsets,
                    vertex_adjacent_edges_fill_count,
                    adjacency.v_adj_edges,
                ],
                dim=1,
                device="cpu",
            )
        else:
            adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32, device="cpu")
            adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32, device="cpu")

        # compute adjacent triangles

        # count number of adjacent faces for each vertex
        face_indices = model.tri_indices.to("cpu")
        num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")
        wp.launch(
            kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1, device="cpu"
        )

        # preallocate memory based on counting results
        num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
        vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
        vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
        vertex_adjacent_faces_offsets[0] = 0
        adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32, device="cpu")

        vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")

        face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
        # (face, vertex_order) * num_adj_faces * num_particles
        # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
        adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32, device="cpu")

        wp.launch(
            kernel=self.fill_adjacent_faces,
            inputs=[
                face_indices,
                adjacency.v_adj_faces_offsets,
                vertex_adjacent_faces_fill_count,
                adjacency.v_adj_faces,
            ],
            dim=1,
            device="cpu",
        )

        return adjacency

    def step(self, model: Model, state_in: State, state_out: State, control: Control, contacts: Contact, dt: float):
        if model is not self.model:
            raise ValueError("model must be the one used to initialize VBDSolver")

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
                self.inertia,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        for _iter in range(self.iterations):
            for color in range(1):
                wp.launch(
                    kernel=VBD_solve_trimesh_no_self_contact,
                    inputs=[
                        dt,
                        self.model.particle_color_groups[color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        state_in.particle_qd,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.adjacency,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        #   ground-particle contact
                        self.model.ground,
                        self.model.ground_plane,
                        self.model.particle_radius,
                    ],
                    outputs=[
                        state_out.particle_q,
                    ],
                    dim=self.model.particle_count,
                    device=self.device,
                )

                wp.launch(
                    kernel=VBD_copy_particle_positions_back,
                    inputs=[self.model.particle_color_groups[color], state_in.particle_q, state_out.particle_q],
                    dim=self.model.particle_count,
                    device=self.device,
                )

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def rebuild_bvh(self, state: State):
        pass


    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1


    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1


    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1
