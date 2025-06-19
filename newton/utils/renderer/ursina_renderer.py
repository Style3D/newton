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

from typing import Optional

import ursina as ua
import warp as wp
from panda3d.core import AntialiasAttrib, loadPrcFileData
from ursina.shaders import unlit_shader

from newton.sim import Model, State
from newton.utils.renderer.shaders.cloth_shader import cloth_shader
from newton.utils.renderer.shaders.floor_shader import create_floor_entity

########################################################################################################################
################################################    Ursina Renderer    #################################################
########################################################################################################################


@wp.func
def to_ursina_coord(x: wp.vec3):
    return wp.vec3(x[0], x[1], -x[2])


@wp.kernel
def compute_face_normals_kernel(
    faces: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    # outputs
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    face = wp.vec3i(faces[3 * tid + 0], faces[3 * tid + 1], faces[3 * tid + 2])
    v0 = positions[face[0]]
    v1 = positions[face[1]]
    v2 = positions[face[2]]

    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = wp.normalize(wp.cross(edge1, edge2))

    wp.atomic_add(normals, face[0], normal)
    wp.atomic_add(normals, face[1], normal)
    wp.atomic_add(normals, face[2], normal)


@wp.kernel
def pack_postion_normal_kernel(
    positions: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    # outputs
    vertex_buffer: wp.array(dtype=float),
):
    tid = wp.tid()
    pos = to_ursina_coord(positions[tid])
    normal = to_ursina_coord(wp.normalize(normals[tid]))
    vertex_buffer[6 * tid + 0] = pos[0]
    vertex_buffer[6 * tid + 1] = pos[1]
    vertex_buffer[6 * tid + 2] = pos[2]
    vertex_buffer[6 * tid + 3] = normal[0]
    vertex_buffer[6 * tid + 4] = normal[1]
    vertex_buffer[6 * tid + 5] = normal[2]


class UrsinaRenderer:
    def __init__(
        self,
        model: Model = None,
        title: str = "Newton",
        window_pos: Optional[tuple[int, int]] = None,
        window_size: Optional[tuple[int, int]] = None,
        full_screen: bool = False,
        multi_sample: bool = True,
        borderless: bool = False,
        vsync: bool = False,
        **ursina_kwargs,
    ):
        """
        Initialize a 3D renderer with customizable window properties.

        Args:
            model (newton.Model): The Newton physics model to render.
            title (str): Window title (default: "Newton")
            window_pos (Optional[Tuple[int, int]]): Window position (x,y) coordinates.
                                                   None for system default.
            window_size (Optional[Tuple[int, int]]): Window dimensions (width, height).
                                                   None for system default.
            full_screen (bool): Enable fullscreen mode (default: False)
            multi_sample (bool): Enable multisample anti-aliasing (default: True)
            borderless (bool): Use borderless window (default: False)
            vsync (bool): Enable vertical synchronization (default: False)
            **ursina_kwargs: Additional arguments passed to the underlying Ursina Renderer.
        """
        self.model = model

        # Graphics pipeline configuration
        loadPrcFileData("", f"sync-video {vsync}")  # Vertical sync
        loadPrcFileData("", f"framebuffer-multisample {multi_sample}")  # Anti-aliasing

        # Additional recommended settings
        loadPrcFileData("", "gl-debug false")  # Disable debug for production
        loadPrcFileData("", "textures-power-2 down")  # Optimize texture handling

        ua.scene.setAntialias(AntialiasAttrib.MMultisample)
        self.app = ua.Ursina(
            title=title,
            position=window_pos,
            size=window_size,
            borderless=borderless,
            full_screen=full_screen,
            **ursina_kwargs,
        )

        # configure camera
        ua.EditorCamera(rotation_speed=200, zoom_speed=-2)
        ua.camera.position = ua.Vec3(0, 1.5, 0)
        ua.camera.clip_plane_near = 2e-2
        ua.camera.clip_plane_far = 1e5
        ua.camera.orthographic = False
        ua.camera.collider = None
        ua.camera.fov = 25

        self._setup_coord_axes()
        self.floor = create_floor_entity(scale=4, texture_scale=(40, 40))

        # Hit point
        self.hit_point = ua.Entity(model="sphere", color=ua.color.yellow, scale=1e-2)
        self.hit_point.visible = False

        # Camera rotation center
        self.rot_point = ua.Entity(model="sphere", color=ua.color.orange, scale=1e-2)
        self.rot_point.visible = False

        if model is not None:
            self.render_normals = wp.zeros(self.model.particle_count, dtype=wp.vec3)
            self.vertex_buffer = wp.zeros(self.model.particle_count * 6, dtype=float)
            self.render_triangles = wp.array(self.model.tri_indices, dtype=int).reshape(len(self.model.tri_indices) * 3)
            self._update_pos_normal_buffer(self.model.particle_q)

            mesh = ua.Mesh(
                vertex_buffer=self.vertex_buffer.numpy().tobytes(),
                triangles=self.render_triangles.numpy(),
                vertex_buffer_format="p3f,n3f",
                vertex_buffer_length=self.model.particle_count,
            )
            self.cloth = ua.Entity(model=mesh, shader=cloth_shader, dynamic=True)
            self.cloth.double_sided = True

            for geo_mesh in self.model.shape_geo_src:
                ua.Entity(
                    model=ua.Mesh(
                        vertices=geo_mesh.vertices,
                        triangles=geo_mesh.indices,
                    ),
                    shader=cloth_shader,
                )

    def _setup_coord_axes(self, resolution: int = 16, radius: float = 0.002, scale: float = 0.2):
        x_axis = ua.Entity(
            model=ua.Cylinder(resolution=resolution, radius=radius, direction=ua.Vec3(scale, 0, 0)),
            color=ua.color.red,
            shader=unlit_shader,
        )
        y_axis = ua.Entity(
            model=ua.Cylinder(resolution=resolution, radius=radius, direction=ua.Vec3(0, scale, 0)),
            color=ua.color.green,
            shader=unlit_shader,
        )
        z_axis = ua.Entity(
            model=ua.Cylinder(resolution=resolution, radius=radius, direction=ua.Vec3(0, 0, scale)),
            color=ua.color.blue,
            shader=unlit_shader,
        )
        x_arrow = ua.Entity(
            model=ua.Cone(resolution=resolution, radius=0.01, height=0.05),
            color=ua.color.red,
            position=(scale, 0, 0),
            rotation=(0, 0, 90),
            shader=unlit_shader,
        )
        y_arrow = ua.Entity(
            model=ua.Cone(resolution=resolution, radius=0.01, height=0.05),
            color=ua.color.green,
            position=(0, scale, 0),
            rotation=(0, 0, 0),
            shader=unlit_shader,
        )
        z_arrow = ua.Entity(
            model=ua.Cone(resolution=resolution, radius=0.01, height=0.05),
            color=ua.color.blue,
            position=(0, 0, scale),
            rotation=(90, 0, 0),
            shader=unlit_shader,
        )
        self.coord_axes = [x_axis, y_axis, z_axis, x_arrow, y_arrow, z_arrow]

    def _update_pos_normal_buffer(self, pos: wp.array(dtype=wp.vec3)):
        self.render_normals.fill_(0.0)
        wp.launch(
            compute_face_normals_kernel,
            inputs=[self.render_triangles, pos],
            outputs=[self.render_normals],
            dim=self.model.tri_count,
        )
        wp.launch(
            pack_postion_normal_kernel,
            dim=self.model.particle_count,
            inputs=[pos, self.render_normals],
            outputs=[self.vertex_buffer],
        )

    def update(self):
        pass

    def render(self, state: State):
        if self.model is not None:
            self._update_pos_normal_buffer(state.particle_q)
            self.cloth.model.vertex_buffer = self.vertex_buffer.numpy().tobytes()
            self.cloth.model.generate()

    def default_input(self, key) -> bool:
        if key == "escape":
            ua.application.quit()
        elif key == "x":
            if self.model is not None:
                self.cloth.wireframe = not self.cloth.wireframe  # Toggle wireframe rendering
        elif key == "b":
            if self.model is not None:
                self.cloth.double_sided = not self.cloth.double_sided  # Toggle backface culling
        elif key == "g":
            self.floor.visible = not self.floor.visible  # Show/hide floor
        elif key == "c":
            for entity in self.coord_axes:
                entity.visible = not entity.visible  # Show/hide coordinate system axes
        elif key == "left mouse down" or key == "left mouse up":
            self.hit_point.visible = key == "left mouse down"
        else:
            return False
        return True

    def run(self):
        self.app.run()


if __name__ == "__main__":
    renderer = UrsinaRenderer()

    def update():
        renderer.update()

    def input(key):
        renderer.default_input(key)

    renderer.run()
