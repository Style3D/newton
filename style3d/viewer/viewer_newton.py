########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Style3D Viewer                                                                                     #
#   Author:         Wenchao Huang (physhuangwenchao@gmail.com)                                                         #
#   Date:           2025/06/19                                                                                         #
########################################################################################################################

import numpy as np
import polyscope as ps
import polyscope.imgui
import warp as wp
from typing_extensions import override

from newton import Axis, AxisType, Mesh, State
from newton._src.viewer.viewer import ViewerBase
from style3d.viewer.viewer import Viewer

########################################################################################################################
#####################################################    Viewer    #####################################################
########################################################################################################################


class ViewerNewton(Viewer, ViewerBase):
    def __init__(
        self,
        up_axis: AxisType = Axis.Y,
        window_size: tuple[int, int] = (1920, 1080),
        scale: float = 1.0,
        vsync=False,
    ):
        """Initialize a 3D renderer with customizable window properties.
        Args:
            window_size (Tuple[int, int]): Window dimensions (width, height)
            vsync (bool): Enable vertical synchronization (default: False)
        """
        super().__init__(
            title="Newton Viewer",
            window_size=window_size,
            vsync=vsync,
        )
        self.scale = scale
        self.up_axis = up_axis
        self.tri_indices = None

        # Cache variables
        self.shape_flags = None
        self._body_transform_mat4x4 = None

        # Render entities
        self.tri_entity = None
        self.particle_entity = None
        self.body_entities = {}

        # Drag info
        self.drag_index = -1
        self.drag_info_chg = False
        self.drag_position = wp.vec3(0, 0, 0)
        self.drag_bary_coord = wp.vec3(0, 0, 0)

        self.set_on_pick(self.handle_pick)
        self.set_on_drag(self.handle_drag)
        self.set_on_release_drag(self.handle_release_drag)

    def handle_pick(self, pick_result: ps.PickResult):
        if pick_result is not None:
            if pick_result.is_hit:
                if pick_result.structure_name == self.tri_entity.get_name():
                    self.drag_index = pick_result.structure_data["index"]
                    self.drag_bary_coord = pick_result.structure_data["bary_coords"]
                    self.drag_position = wp.vec3(pick_result.position)
                    self.drag_info_chg = True

    def handle_drag(self, drag_pos: tuple[float, float, float]):
        self.drag_position = wp.vec3(drag_pos[0], drag_pos[1], drag_pos[2])
        self.drag_info_chg = True

    def handle_release_drag(self):
        self.drag_info_chg = self.drag_index != -1
        self.drag_index = -1

    def _array_to_y_up(self, np_array):
        if self.up_axis == Axis.Z:
            return np_array[:, [1, 2, 0]]
        elif self.up_axis == Axis.X:
            return np_array[:, [2, 0, 1]]
        else:
            return np_array

    def _transform_to_y_up(self, np_array):
        if self.up_axis == Axis.Z:
            return np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ np_array
        elif self.up_axis == Axis.X:
            return np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ np_array
        else:
            return np_array

    @override
    def set_model(self, model):
        super().set_model(model)
        # Add meshes
        if model is not None:
            # Cache
            self.shape_flags = model.shape_flags.numpy()
            self._body_transform_mat4x4 = wp.zeros(model.body_count, dtype=wp.mat44)
            particle_q = self._array_to_y_up(model.particle_q.numpy().reshape(model.particle_count, 3)) * self.scale

            # Register particle entity
            if model.particle_count > 0:
                self.particle_entity = ps.register_point_cloud(
                    name="Particles",
                    enabled=False,
                    points=particle_q,
                    radius=model.particle_radius.numpy()[0] * self.scale,
                )

            # Register triangle entity
            if model.tri_count > 0:
                self.tri_indices = model.tri_indices.numpy()
                self.tri_entity = ps.register_surface_mesh(
                    name="Triangles",
                    vertices=particle_q,
                    faces=model.tri_indices.numpy().reshape(model.tri_count, 3),
                    color=(184 / 255.0, 67 / 255.0, 1),
                    back_face_policy="custom",
                    edge_color=(0, 0, 0),
                    smooth_shade=False,
                    edge_width=0.3,
                )
                self.tri_entity.set_selection_mode("faces_only")

            # Register body entity
            for i in range(model.body_count):
                shape_indices = model.body_shapes[i]
                for shape_idx in shape_indices:
                    if isinstance(model.shape_source[shape_idx], Mesh):
                        if self.shape_flags[shape_idx] & 1 == 0:
                            continue

                        @wp.kernel
                        def transform_vertices_kernel(
                            index: wp.int32,
                            scale: float,
                            vertices_in: wp.array(dtype=wp.vec3),
                            scaling3d: wp.array(dtype=wp.vec3),
                            transforms: wp.array(dtype=wp.transform),
                            vertices_out: wp.array(dtype=wp.vec3),
                        ):
                            tid = wp.tid()
                            scaling = scaling3d[index] * scale
                            new_pos = wp.transform_point(transforms[index], vertices_in[tid])
                            new_pos[0] *= scaling[0]
                            new_pos[1] *= scaling[1]
                            new_pos[2] *= scaling[2]
                            vertices_out[tid] = new_pos

                        shape_vertices = wp.array(model.shape_source[shape_idx].vertices, dtype=wp.vec3)

                        wp.launch(
                            transform_vertices_kernel,
                            dim=len(shape_vertices),
                            inputs=[
                                shape_idx,
                                self.scale,
                                shape_vertices,
                                model.shape_scale,
                                model.shape_transform,
                            ],
                            outputs=[shape_vertices],
                        )

                        self.body_entities[model.shape_key[shape_idx]] = ps.register_surface_mesh(
                            name=model.shape_key[shape_idx],
                            vertices=shape_vertices.numpy(),
                            faces=model.shape_source[shape_idx].indices.reshape(-1, 3),
                            back_face_policy="cull",
                            edge_color=(1, 1, 1),
                            smooth_shade=True,
                            edge_width=0.0,
                            color=(0, 0, 0),
                            material="wax",
                        )

            self.particle_count = model.particle_count
            self.body_count = model.body_count
            self.tri_count = model.tri_count
            self.tet_count = model.tet_count
        else:
            self.particle_count = 0
            self.body_count = 0
            self.tri_count = 0
            self.tet_count = 0

    @override
    def log_state(self, state: State):
        # Download to host.
        if state.particle_q is not None:
            particle_q = self._array_to_y_up(state.particle_q.numpy().reshape(state.particle_count, 3)) * self.scale

        if self.particle_entity is not None:
            if self.particle_entity.is_enabled():
                self.particle_entity.update_point_positions(particle_q)

        if self.tri_entity is not None:
            self.tri_entity.update_vertex_positions(particle_q)
            if self.pick_result is not None:
                if self.pick_result.structure_name == self.tri_entity.get_name():
                    index = self.pick_result.structure_data["index"]
                    face = self.tri_indices[index, 0:3]
                    x0 = wp.vec3(particle_q[face[0], 0:3])
                    x1 = wp.vec3(particle_q[face[1], 0:3])
                    x2 = wp.vec3(particle_q[face[2], 0:3])
                    bary_coord = wp.vec3(self.pick_result.structure_data["bary_coords"])
                    self._dragged_point.set_position(x0 * bary_coord[0] + x1 * bary_coord[1] + x2 * bary_coord[2])

        if len(self.body_entities) > 0:

            @wp.kernel
            def transform_to_mat4x4_kernel(
                scale: float,
                transform_in: wp.array(dtype=wp.transform),
                transform_out: wp.array(dtype=wp.mat44),
            ):
                tid = wp.tid()
                transform = transform_in[tid]
                translation = wp.transform_get_translation(transform) * scale
                wp.transform_set_translation(transform, translation)
                transform_out[tid] = wp.transform_to_matrix(transform)

            wp.launch(
                transform_to_mat4x4_kernel,
                dim=self.model.body_count,
                inputs=[self.scale, state.body_q],
                outputs=[self._body_transform_mat4x4],
            )

            body_q = self._body_transform_mat4x4.numpy()

            for i in range(self.model.body_count):
                shape_indices = self.model.body_shapes[i]
                for shape_idx in shape_indices:
                    if isinstance(self.model.shape_source[shape_idx], Mesh):
                        if self.shape_flags[shape_idx] & 1:
                            self.body_entities[self.model.shape_key[shape_idx]].set_transform(
                                self._transform_to_y_up(body_q[i])
                            )

        ps.request_redraw()

    @override
    def _process_key_inputs(self):
        super()._process_key_inputs()
        if ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_X):  # Show/hide edges
            for _, entity in self.body_entities.items():
                entity.set_edge_width(0 if entity.get_edge_width() != 0 else 0.3)
            if self.tri_entity is not None:
                self.tri_entity.set_edge_width(0 if self.tri_entity.get_edge_width() != 0 else 0.3)

    @override
    def is_running(self) -> bool:
        return not self.requests_close()

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def begin_frame(self, time):
        super().begin_frame(time)
        self.sim_time = time

    @override
    def end_frame(self):
        super().end_frame()
        self.frame_tick()


if __name__ == "__main__":
    viewer = ViewerNewton()
    viewer.run()
