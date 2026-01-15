########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Style3D Solver Plus                                                                                #
#   Author:         Wenchao Huang                                                                                      #
#   Date:           2025/10/27                                                                                         #
########################################################################################################################

import numpy as np
import style3dsim as sim
import warp as wp

import newton
import newton as nt
from newton import Contacts, Control, State, Style3DModelBuilder


def sim_log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
    if level == sim.LogLevel.INFO:
        print("[info]: ", message)
    elif level == sim.LogLevel.ERROR:
        print("[error]: ", message)
    elif level == sim.LogLevel.WARNING:
        print("[warning]: ", message)
    elif level == sim.LogLevel.DEBUG:
        print("[debug]: ", message)


class SolverStyle3DPro(nt.solvers.SolverBase):
    def __init__(
        self,
        model: nt.Style3DModel,
        iterations=10,
        linear_iterations=10,
        drag_spring_stiff: float = 1e2,
        enable_mouse_dragging: bool = False,
    ):
        super().__init__(model)
        self.enable_mouse_dragging = enable_mouse_dragging

        if sim.is_login():
            # Create world
            self.world = sim.World()

            # Create Cloth
            verts_np = model.particle_q.numpy()
            faces_np = model.tri_indices.numpy()
            flags_np = model.particle_flags.numpy()
            self.is_fixed = [False] * len(verts_np)
            self.fixed_indices = list(range(len(verts_np)))

            for i in range(len(verts_np)):
                self.is_fixed[i] = not (flags_np[i] & newton.ParticleFlags.ACTIVE)

            self.faces = faces_np
            self.cloth = sim.Cloth(faces_np, verts_np, [], False)
            self.cloth.set_pin(self.is_fixed, self.fixed_indices)
            self.cloth.attach(self.world)

            # Create rigid bodies
            self.body_entities = {}
            self.shape_flags = model.shape_flags.numpy()
            body_trans_np = model.body_q.numpy()

            # # Register body entity
            for i in range(model.body_count):
                shape_indices = model.body_shapes[i]
                for shape_idx in shape_indices:
                    if isinstance(model.shape_source[shape_idx], newton.Mesh):
                        if self.shape_flags[shape_idx] & 1 == 0:
                            continue

                        @wp.kernel
                        def transform_vertices_kernel(
                            index: wp.int32,
                            vertices_in: wp.array(dtype=wp.vec3),
                            scaling3d: wp.array(dtype=wp.vec3),
                            transforms: wp.array(dtype=wp.transform),
                            vertices_out: wp.array(dtype=wp.vec3),
                        ):
                            tid = wp.tid()
                            new_pos = wp.transform_point(transforms[index], vertices_in[tid])
                            new_pos[0] *= scaling3d[index][0]
                            new_pos[1] *= scaling3d[index][1]
                            new_pos[2] *= scaling3d[index][2]
                            vertices_out[tid] = new_pos

                        shape_vertices = wp.array(model.shape_source[shape_idx].vertices, dtype=wp.vec3)

                        wp.launch(
                            transform_vertices_kernel,
                            dim=len(shape_vertices),
                            inputs=[
                                shape_idx,
                                shape_vertices,
                                model.shape_scale,
                                model.shape_transform,
                            ],
                            outputs=[shape_vertices],
                        )

                        trans = body_trans_np[i]
                        translation = sim.Vec3f(trans[0], trans[1], trans[2])
                        rotation = sim.Quat(trans[3], trans[4], trans[5], trans[6])
                        scaling = sim.Vec3f(1.0, 1.0, 1.0)
                        transform = sim.Transform(translation, rotation, scaling)
                        static_mesh = sim.Mesh(model.shape_source[shape_idx].indices.flatten(), shape_vertices.numpy())
                        rigid_body = sim.RigidBody(static_mesh, transform)
                        rigid_body.attach(self.world)
                        rigid_body.set_pin(True)
                        attrib = sim.RigidBodyAttrib()
                        attrib.mass = 10.0
                        rigid_body.set_attrib(attrib)
                        self.body_entities[model.shape_key[shape_idx]] = rigid_body

        # Drag info
        self.drag_pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.drag_index = wp.array([-1], dtype=int, device=self.device)
        self.drag_bary_coord = wp.zeros(1, dtype=wp.vec3, device=self.device)

    def step(self, state_in: State, state_out: State, control: Control, contacts: Contacts, dt: float):
        if not sim.is_login():
            return

        if state_in.body_q is not None:
            trans_in = state_in.body_q.numpy()
            trans_out = state_out.body_q.numpy()
            for i in range(self.model.body_count):
                trans_0 = trans_in[i]
                trans_1 = trans_out[i]
                shape_indices = self.model.body_shapes[i]
                for shape_idx in shape_indices:
                    if isinstance(self.model.shape_source[shape_idx], newton.Mesh):
                        if self.shape_flags[shape_idx] & 1:
                            translation_0 = sim.Vec3f(trans_0[0], trans_0[1], trans_0[2])
                            translation_1 = sim.Vec3f(trans_1[0], trans_1[1], trans_1[2])
                            rotation_0 = sim.Quat(trans_0[3], trans_0[4], trans_0[5], trans_0[6])
                            rotation_1 = sim.Quat(trans_1[3], trans_1[4], trans_1[5], trans_1[6])
                            begin_trans = sim.Transform(translation_0, rotation_0, sim.Vec3f(1.0, 1.0, 1.0))
                            end_trans = sim.Transform(translation_1, rotation_1, sim.Vec3f(1.0, 1.0, 1.0))
                            self.body_entities[self.model.shape_key[shape_idx]].move(begin_trans, end_trans)

        self.world.step_sim()
        self.world.fetch_sim(0)
        verts = self.cloth.get_positions()
        state_out.particle_q.assign(verts)

    def rebuild_bvh(self, state: State):
        pass

    def precompute(self, builder: Style3DModelBuilder):
        pass

    def update_drag_info(self, index: int, pos: wp.vec3, bary_coord: wp.vec3):
        """Should be invoked when state changed."""
        # print([index, pos, bary_coord])
        self.drag_bary_coord.fill_(bary_coord)
        self.drag_index.fill_(index)
        self.drag_pos.fill_(pos)

        if self.enable_mouse_dragging:
            if index != -1:
                face = self.faces[index]
                vIdx = face[0]
                coord = bary_coord[0]
                if bary_coord[1] > coord:
                    vIdx = face[1]
                    coord = bary_coord[1]
                if bary_coord[2] > coord:
                    vIdx = face[2]

                is_fixed = self.is_fixed.copy()
                is_fixed[vIdx] = True
                pos_np = np.array([[pos.x, pos.y, pos.z]], dtype=np.float32)
                index_np = np.array([vIdx], dtype=np.int32)
                self.cloth.set_positions(pos_np, index_np)
                self.cloth.set_pin(is_fixed, self.fixed_indices)
            else:
                self.cloth.set_pin(self.is_fixed, self.fixed_indices)
