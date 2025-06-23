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

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import warp as wp

from newton.sim import Model, State

########################################################################################################################
###############################################    Polyscope Renderer    ###############################################
########################################################################################################################


def rotate_around_point(point, origin, deg):
    # Convert degrees to radians
    angle_rad = -deg / 180.0 * wp.pi

    cos_angle = wp.cos(angle_rad)
    sin_angle = wp.sin(angle_rad)
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    # Return rotated point
    return (origin[0] + (dx * cos_angle - dy * sin_angle), origin[1] + (dx * sin_angle + dy * cos_angle))


def gen_cone_mesh(resolution: int = 16, radius: float = 0.5, height: float = 1, add_bottom: bool = True):
    # Initialize base vertex and origin
    v = wp.vec3(radius, 0, 0)
    origin = wp.vec3(0, 0, 0)

    # Calculate rotation angle per segment
    degrees_to_rotate = 360.0 / resolution

    verts = []
    faces = []

    # Apex point
    v0 = wp.vec3(0, height, 0)
    verts.append(v0)

    # Generate side vertices and triangles
    for i in range(resolution):
        # Current base point
        v1 = wp.vec3(v[0], 0, v[1])
        # Rotate to get next base point
        v = rotate_around_point(v, origin, -degrees_to_rotate)
        v2 = wp.vec3(v[0], 0, v[1])  # Next base point

        verts.append(v1)
        verts.append(v2)

        # Side triangle (clockwise)
        faces.append([0, i * 2 + 1, i * 2 + 2])

    # Generate bottom vertices and triangles if required
    if add_bottom:
        bottom_start_idx = len(verts)
        center_idx = bottom_start_idx
        # Add bottom center point
        verts.append(wp.vec3(0, 0, 0))

        # Reset v to initial position
        v = wp.vec3(radius, 0, 0)

        # Bottom triangles (clockwise, connecting center and circle points)
        for i in range(resolution):
            # Add base circle points
            verts.append(wp.vec3(v[0], 0, v[1]))
            v = rotate_around_point(v, origin, -degrees_to_rotate)

            v0_idx = center_idx  # Center point
            v1_idx = bottom_start_idx + 1 + i  # Current circle point
            v2_idx = bottom_start_idx + 1 + ((i + 1) % resolution)  # Next circle point
            faces.append([v0_idx, v1_idx, v2_idx])

    # Return vertices and faces as numpy arrays
    return np.array(verts).reshape(-1, 3), np.array(faces)


def gen_cylinder_mesh(
    resolution: int = 16, radius: float = 0.5, start: float = 0, height: float = 1, direction=(0, 1, 0)
):
    # Normalize direction vector
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # Find two vectors perpendicular to direction to form a basis
    if np.abs(direction[1]) < 0.99:  # Avoid parallel to y-axis
        v1 = np.cross(direction, [0, 1, 0])
    else:
        v1 = np.cross(direction, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(direction, v1)
    v2 = v2 / np.linalg.norm(v2)

    # Initialize lists for vertices and faces
    verts = []
    faces = []

    # Calculate rotation angle per segment
    degrees_to_rotate = 360.0 / resolution

    # Generate bottom circle vertices (at y=start)
    bottom_center = np.array([0, start, 0])
    v = wp.vec3(radius, 0, 0)
    origin = wp.vec3(0, 0, 0)
    bottom_verts_start = len(verts)
    verts.append(bottom_center)  # Bottom center point
    for i in range(resolution):
        # Transform to cylinder's coordinate system
        point_2d = rotate_around_point(v, origin, i * -degrees_to_rotate)
        point = point_2d[0] * v1 + point_2d[1] * v2 + bottom_center
        verts.append(point)

    # Generate top circle vertices (at y=start+height)
    top_center = bottom_center + height * direction
    top_verts_start = len(verts)
    verts.append(top_center)  # Top center point
    for i in range(resolution):
        point_2d = rotate_around_point(v, origin, i * -degrees_to_rotate)
        point = point_2d[0] * v1 + point_2d[1] * v2 + top_center
        verts.append(point)

    # Generate side faces (quads split into two triangles)
    for i in range(resolution):
        bottom_v0 = bottom_verts_start + 1 + i
        bottom_v1 = bottom_verts_start + 1 + ((i + 1) % resolution)
        top_v0 = top_verts_start + 1 + i
        top_v1 = top_verts_start + 1 + ((i + 1) % resolution)
        # First triangle
        faces.append([bottom_v0, bottom_v1, top_v1])
        # Second triangle
        faces.append([bottom_v0, top_v1, top_v0])

    # Generate bottom cap faces (triangle fan)
    for i in range(resolution):
        v0 = bottom_verts_start  # Bottom center
        v1 = bottom_verts_start + 1 + i
        v2 = bottom_verts_start + 1 + ((i + 1) % resolution)
        faces.append([v0, v2, v1])  # Reversed for correct normal

    # Generate top cap faces (triangle fan)
    for i in range(resolution):
        v0 = top_verts_start  # Top center
        v1 = top_verts_start + 1 + i
        v2 = top_verts_start + 1 + ((i + 1) % resolution)
        faces.append([v0, v1, v2])  # Normal outward

    # Convert to numpy arrays
    return np.array(verts).reshape(-1, 3), np.array(faces)


class PolyscopeRenderer:
    def __init__(
        self,
        model: Model = None,
        window_size: list[int] = (1920, 1080),
        vsync=False,
    ):
        self.vsync = vsync
        self.paused = True
        self._tri_mesh = None
        self.user_update = None
        self.coord_axes_mesh = []
        self.ground_plane_mode = "tile_reflection"

        # Setup camera
        self._last_mouse_pos = (0, 0)
        self._camera_origin = [0, 1, 0]
        self._camera_radius = 3.0
        self._camera_theta = 90.0
        self._camera_phi = 0.0

        # Setup polyscope scene parameters
        ps.init()
        ps.set_SSAA_factor(4)
        ps.set_enable_vsync(vsync)
        ps.set_ground_plane_height(0)
        ps.set_ground_plane_mode(self.ground_plane_mode)
        ps.set_window_size(window_size[0], window_size[1])
        ps.set_background_color((0.02, 0.02, 0.02))
        ps.set_do_default_mouse_interaction(False)
        ps.set_user_callback(self._update)
        ps.set_max_fps(200)
        ps.set_program_name("Style3D-Newton")

        # Add coordinate axes
        self._set_up_coord_axes()

        # Add look-at point
        self._look_at_point = ps.register_point_cloud(
            name="Look-At",
            points=np.array([0.0, 0.0, 0.0]).reshape(-1,3),
            color=(1, 0.1, 0.1),
            enabled=False,
        )
        self._look_at_point.set_position(self._camera_origin)
        self._look_at_point.set_material("candy")
        self._update_camera()

        # Add meshes
        if model is not None:
            self._tri_mesh = ps.register_surface_mesh(
                name="Garment",
                vertices=model.particle_q.numpy().reshape(model.particle_count, 3),
                faces=model.tri_indices.numpy().reshape(model.tri_count, 3),
                back_face_policy="custom",
                smooth_shade=False,
            )


    def set_user_update(self, callback):
        self.user_update = callback


    def update_state(self, state: State):
        if self._tri_mesh is not None:
            self._tri_mesh.update_vertex_positions(state.particle_q.numpy().reshape(state.particle_count, 3))
            ps.request_redraw()


    def _set_up_coord_axes(self):
        verts, faces = gen_cylinder_mesh(radius=0.005, height=0.2, direction=(1, 0, 0))
        self.coord_axes_mesh.append(
            ps.register_surface_mesh(
                name="Axis-X",
                vertices=verts,
                faces=faces,
                color=(1, 0, 0),
                smooth_shade=True,
            )
        )

        verts, faces = gen_cylinder_mesh(radius=0.005, height=0.2, direction=(0, 1, 0))
        self.coord_axes_mesh.append(
            ps.register_surface_mesh(
                name="Axis-Y",
                vertices=verts,
                faces=faces,
                color=(0, 1, 0),
                smooth_shade=True,
            )
        )

        verts, faces = gen_cylinder_mesh(radius=0.005, height=0.2, direction=(0, 0, 1))
        self.coord_axes_mesh.append(
            ps.register_surface_mesh(
                name="Axis-Z",
                vertices=verts,
                faces=faces,
                color=(0, 0, 1),
                smooth_shade=True,
            )
        )


    def _process_key_inputs(self):
        if psim.IsKeyPressed(psim.ImGuiKey_Space):
            self.paused = not self.paused  # Run/pause
            print(f"Paused = {self.paused}")
        elif psim.IsKeyPressed(psim.ImGuiKey_Escape):
            ps.unshow()  # Exit
        elif psim.IsKeyPressed(psim.ImGuiKey_X):
            if self._tri_mesh is not None:
                # Show/hide edges
                self._tri_mesh.set_edge_width(0 if self._tri_mesh.get_edge_width() != 0 else 0.3)
        elif psim.IsKeyPressed(psim.ImGuiKey_C):
            for mesh in self.coord_axes_mesh:
                # Show/hide coordinate axes
                mesh.set_enabled(not mesh.is_enabled())
        elif psim.IsKeyPressed(psim.ImGuiKey_V):
            self.vsync = not self.vsync
            ps.set_enable_vsync(self.vsync)
            print(f"Vsync = {self.vsync}")
        elif psim.IsKeyPressed(psim.ImGuiKey_G):
            # Rolling ground plane mode
            if self.ground_plane_mode == "none":
                self.ground_plane_mode = "tile"
            elif self.ground_plane_mode == "tile":
                self.ground_plane_mode = "tile_reflection"
            elif self.ground_plane_mode == "tile_reflection":
                self.ground_plane_mode = "none"
            ps.set_ground_plane_mode(self.ground_plane_mode)
            print(f"Ground plane mode: {self.ground_plane_mode}")


    def _process_mouse_inputs(self):
        # Mouse delta Pos
        mouse_pos = psim.GetMousePos()
        dx = mouse_pos[0] - self._last_mouse_pos[0]
        dy = mouse_pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = mouse_pos

        # Button key
        LeftButton, RightButton, MiddleButton = 0, 1, 2

        # Click evnet
        if psim.IsMouseClicked(LeftButton):
            hit_info = ps.pick(screen_coords = mouse_pos)
        if psim.IsMouseClicked(RightButton):
            pass
        if psim.IsMouseClicked(MiddleButton):
            pass

        # Show/hide look-at point
        self._look_at_point.set_enabled(psim.IsMouseDown(MiddleButton) or psim.IsMouseDown(RightButton))
        if self._tri_mesh is not None:
            if psim.IsMouseDown(MiddleButton) or psim.IsMouseDown(RightButton):
                self._tri_mesh.set_transparency(0.9)
            else:
                self._tri_mesh.set_transparency(1.0)

        # Dragging
        if psim.IsMouseDown(LeftButton):
            pass
        else:
            pass

        should_update_camera = False

        # Rotate camera
        if  psim.IsMouseDown(RightButton) and ((dx != 0) or (dy != 0)):
            self._camera_phi -= dx / 2.0
            self._camera_theta -= dy / 4.0
            self._camera_theta = wp.clamp(self._camera_theta, 1.0, 179.0)
            should_update_camera = True

        # Translate camera
        if psim.IsMouseDown(MiddleButton) and ((dx != 0) or (dy != 0)):
            camera_params = ps.get_view_camera_parameters()
            fov = camera_params.get_fov_vertical_deg()
            window_height = ps.get_window_size()[1]

            up_dir = camera_params.get_up_dir()
            right_dir = camera_params.get_right_dir()
            delta = up_dir * dy * 2.0 / window_height
            delta -= right_dir * dx * 2.0 / window_height
            delta *= self._camera_radius
            delta *= wp.tan(wp.radians(fov) / 2.0)

            delta[0] += self._camera_origin[0]
            delta[1] += self._camera_origin[1]
            delta[2] += self._camera_origin[2]

            self._camera_origin = (delta[0], delta[1], delta[2])
            self._look_at_point.set_position(self._camera_origin)
            should_update_camera = True

        # Zoom camera
        io = psim.GetIO()
        if io.MouseWheel != 0.0:
            ratio = 0.9 if (io.MouseWheel < 0.0) else (1.0 / 0.9)
            self._camera_radius = wp.clamp(self._camera_radius * ratio, 5e-2, 5e1)
            should_update_camera = True

        if should_update_camera:
            self._update_camera()


    def _update_camera(self):
        r = wp.sin(wp.radians(self._camera_theta))
        x = r * wp.sin(wp.radians(self._camera_phi))
        z = r * wp.cos(wp.radians(self._camera_phi))
        y = wp.cos(wp.radians(self._camera_theta))
        x = x * self._camera_radius + self._camera_origin[0]
        y = y * self._camera_radius + self._camera_origin[1]
        z = z * self._camera_radius + self._camera_origin[2]
        ps.look_at_dir(camera_location=(x, y, z), target=self._camera_origin, up_dir=(0, 1, 0))
        self._look_at_point.set_radius(self._camera_radius * 5e-3, False)


    def _update(self):
        self._process_key_inputs()
        self._process_mouse_inputs()
        if self.user_update is not None:
            if not self.paused:
                self.user_update()


    def run(self):
        ps.show()


if __name__ == "__main__":
    renderer = PolyscopeRenderer()
    renderer.run()
