########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Viewer class                                                                                       #
#   Author:         Wenchao Huang (physhuangwenchao@gmail.com)                                                         #
#   Date:           2025/11/27                                                                                         #
########################################################################################################################

import time

import numpy as np
import polyscope as ps
import polyscope.imgui
import warp as wp

########################################################################################################################
#####################################################    Viewer    #####################################################
########################################################################################################################


class Viewer:
    def __init__(
        self,
        title: str = "Style3D Viewer",
        window_size: tuple[int, int] = (1920, 1080),
        vsync: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize a 3D renderer with customizable window properties.
        Args:
            title (str): Title of the window (default: Style3D Viewer)
            window_size (Tuple[int, int]): Window dimensions (width, height)
            vsync (bool): Enable vertical synchronization (default: False)
        """
        super().__init__(*args, **kwargs)
        self._ground_plane_mode = "tile_reflection"
        self._request_close = False
        self._paused = True
        self._vsync = vsync

        # Simulation state
        self.sim_time = 0.0
        self.sim_frames = 0

        # Simulation statistics
        self.particle_count = 0
        self.spring_count = 0
        self.body_count = 0
        self.tri_count = 0
        self.tet_count = 0

        # FPS counting
        self._rendering_fps = 1.0
        self._rendering_fps_counter = 0
        self._rendering_fps_last_time = 0.0

        # Drag info
        self._pick_dist = 0.0
        self.pick_result = None

        # User callbacks
        self._user_update = None  # _user_update()
        self._on_release_drag = None  # _on_release_drag()
        self._on_drag = None  # _on_drag(target_pos: tuple[float, float, float])
        self._on_pick = None  # _on_pick(pick_res: ps.PickResult)

        # Setup polyscope scene parameters
        ps.set_program_name(title)  # should be called before init()
        ps.init()
        ps.set_SSAA_factor(4)
        ps.set_enable_vsync(vsync)
        ps.set_ground_plane_height(0)
        ps.set_give_focus_on_show(True)
        ps.set_ground_plane_mode(self._ground_plane_mode)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_window_size(window_size[0], window_size[1])
        ps.set_background_color([0.015, 0.015, 0.015])
        ps.set_do_default_mouse_interaction(False)
        ps.set_user_callback(self._update)
        ps.set_max_fps(200)

        # Add look-at point
        self._look_at_point = ps.register_point_cloud(
            name="Look-At",
            points=np.array([0, 0, 0]).reshape(-1, 3),
            color=(1, 0.1, 0.1),
            material="candy",
            enabled=False,
        )

        # Add dragged-point
        self._dragged_point = ps.register_point_cloud(
            name="Dragged-Point",
            points=np.array([0, 0, 0]).reshape(-1, 3),
            color=(72 / 255, 167 / 255, 1),
            material="flat",
            enabled=False,
        )

        # Add coordinate axes
        edges = np.array([[0, 1], [2, 3], [4, 5]])
        colors = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        nodes = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0, 0], [0, 0.2, 0], [0, 0, 0], [0, 0, 0.2]])
        self._coord_axes = ps.register_curve_network(name="Axes", nodes=nodes, edges=edges, enabled=True, radius=3e-3)
        self._coord_axes.add_color_quantity(name="Axes color", values=colors, defined_on="nodes", enabled=True)

        # Inner group (hide children)
        self._inner_group = ps.create_group("Inner")
        self._inner_group.set_show_child_details(False)
        self._inner_group.set_hide_descendants_from_structure_lists(True)
        self._look_at_point.add_to_group(self._inner_group)
        self._dragged_point.add_to_group(self._inner_group)
        self._coord_axes.add_to_group(self._inner_group)

        # Setup camera
        self._last_mouse_pos = (0, 0)
        self._camera_origin = [0, 1, 0]
        self._camera_radius = 2.3
        self._camera_theta = 80.0
        self._camera_phi = 0.0
        self._update_camera()

    def set_user_update(self, user_func):
        self._user_update = user_func

    def set_on_pick(self, callback):
        self._on_pick = callback

    def set_on_drag(self, callback):
        self._on_drag = callback

    def set_on_release_drag(self, callback):
        self._on_release_drag = callback

    def _update_camera(self):
        r = wp.sin(wp.radians(self._camera_theta))
        x = r * wp.sin(wp.radians(self._camera_phi))
        z = r * wp.cos(wp.radians(self._camera_phi))
        y = wp.cos(wp.radians(self._camera_theta))
        x = x * self._camera_radius + self._camera_origin[0]
        y = y * self._camera_radius + self._camera_origin[1]
        z = z * self._camera_radius + self._camera_origin[2]
        ps.look_at_dir(camera_location=(x, y, z), target=self._camera_origin, up_dir=(0, 1, 0))
        self._dragged_point.set_radius(self._camera_radius * 2e-3, False)
        self._look_at_point.set_radius(self._camera_radius * 5e-3, False)
        self._look_at_point.set_position(self._camera_origin)

    def _process_key_inputs(self):
        if ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_Escape):
            self._request_close = True
            ps.unshow()  # Exit
        elif ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_Space):
            self._paused = not self._paused  # Run/pause
        elif ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_S):
            if self._user_update is not None:
                self._user_update()  # single step
        elif ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_C):
            # Show/hide coordinate axes
            self._coord_axes.set_enabled(not self._coord_axes.is_enabled())
        elif ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_V):
            self._vsync = not self._vsync
            ps.set_enable_vsync(self._vsync)
        elif ps.imgui.IsKeyPressed(ps.imgui.ImGuiKey_G):
            # Rolling ground plane mode
            if self._ground_plane_mode == "none":
                self._ground_plane_mode = "tile"
            elif self._ground_plane_mode == "tile":
                self._ground_plane_mode = "tile_reflection"
            elif self._ground_plane_mode == "tile_reflection":
                self._ground_plane_mode = "none"
            ps.set_ground_plane_mode(self._ground_plane_mode)

    def _update_gui(self):
        # Skip generate "imgui.ini" file
        ps.imgui.GetIO().IniFilename = None

        # update render fps
        curr_time = time.time()
        if (self._rendering_fps_counter > 0) and (curr_time - self._rendering_fps_last_time > 0.1):
            self._rendering_fps = self._rendering_fps_counter / (curr_time - self._rendering_fps_last_time)
            self._rendering_fps_last_time = curr_time
            self._rendering_fps_counter = 0
        self._rendering_fps_counter += 1

        ps.imgui.Text("State: ")
        ps.imgui.SameLine()
        if self._paused:
            ps.imgui.TextColored([1, 0, 0, 1], "Paused")
        else:
            ps.imgui.TextColored([0, 1, 0, 1], "Running")
        ps.imgui.Text(f"Sim Time: {self.sim_time:.1f} s")
        ps.imgui.Text(f"Frame Count: {self.sim_frames}")
        ps.imgui.Text(f"FPS: {self._rendering_fps:.1f} / {1e3 / self._rendering_fps:.1f}ms")

        if self.body_count or self.particle_count or self.spring_count or self.tri_count or self.tet_count:
            ps.imgui.Separator()
            ps.imgui.Text("Statistics:")
            if self.body_count:
                ps.imgui.Text(f" - Body Count: {self.body_count}")
            if self.particle_count:
                ps.imgui.Text(f" - Particle Count: {self.particle_count}")
            if self.spring_count:
                ps.imgui.Text(f" - Spring Count: {self.spring_count}")
            if self.tri_count:
                ps.imgui.Text(f" - Triangle Count: {self.tri_count}")
            if self.tet_count:
                ps.imgui.Text(f" - Tetrahedral Count: {self.tet_count}")

        if self.pick_result is not None:
            ps.imgui.Separator()
            ps.imgui.Text("Pick Result:")
            ps.imgui.Text(f" - Structure Name: {self.pick_result.structure_name}")
            ps.imgui.Text(f" - Structure Type: {self.pick_result.structure_type_name}")
            ps.imgui.Text(f" - Screen Coordinate: {self.pick_result.screen_coords}")
            ps.imgui.Text(f" - World Position: {self.pick_result.position}")
            if self.pick_result.structure_type_name != "Point Cloud":
                if self.pick_result.structure_data["element_type"] == "face":
                    ps.imgui.Text(f" - Bary Coordinate: {self.pick_result.structure_data['bary_coords']}")
                ps.imgui.Text(f" - Element Type: {self.pick_result.structure_data['element_type']}")
                ps.imgui.Text(f" - Index: {self.pick_result.structure_data['index']}")

    def _process_mouse_inputs(self):
        # Mouse delta Pos
        mouse_pos = ps.imgui.GetMousePos()
        dx = mouse_pos[0] - self._last_mouse_pos[0]
        dy = mouse_pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = mouse_pos

        # Click event
        if ps.imgui.IsMouseClicked(ps.imgui.ImGuiMouseButton_Left):
            pick_result = ps.pick(screen_coords=mouse_pos)
            if pick_result.is_hit:
                self.pick_result = pick_result
                self._dragged_point.set_enabled(True)
                self._dragged_point.set_position(pick_result.position)
                eye_pos = ps.get_view_camera_parameters().get_position()
                self._pick_dist = wp.length(wp.vec3(pick_result.position - eye_pos))
            else:
                self._dragged_point.set_enabled(False)  #
                self.pick_result = None
            if self._on_pick is not None:
                self._on_pick(pick_result)
            self.drag_info_chg = True
        if ps.imgui.IsMouseClicked(ps.imgui.ImGuiMouseButton_Middle):
            pass
        if ps.imgui.IsMouseClicked(ps.imgui.ImGuiMouseButton_Right):
            pass

        # Show/hide look-at point
        self._look_at_point.set_enabled(
            ps.imgui.IsMouseDown(ps.imgui.ImGuiMouseButton_Middle)
            or ps.imgui.IsMouseDown(ps.imgui.ImGuiMouseButton_Right)
        )

        # Dragging
        if (self.pick_result is not None) and ps.imgui.IsMouseDragging(ps.imgui.ImGuiMouseButton_Left):
            if self._on_drag is not None:
                camera_params = ps.get_view_camera_parameters()
                aspect = camera_params.get_aspect()
                view_mat = camera_params.get_view_mat()
                fov = wp.radians(camera_params.get_fov_vertical_deg())
                axis_x = wp.vec3(view_mat[0, 0], view_mat[0, 1], view_mat[0, 2])
                axis_y = wp.vec3(view_mat[1, 0], view_mat[1, 1], view_mat[1, 2])
                axis_z = wp.vec3(view_mat[2, 0], view_mat[2, 1], view_mat[2, 2])
                origin = wp.vec3(camera_params.get_position())
                (width, height) = ps.get_window_size()
                nx = 2.0 * (mouse_pos[0] + 0.5) / width - 1.0
                ny = 2.0 * (mouse_pos[1] + 0.5) / height - 1.0
                u = nx * wp.tan(fov / 2.0) * aspect
                v = ny * wp.tan(fov / 2.0)
                ray_dir = wp.normalize(axis_x * u - axis_y * v - axis_z)
                self._on_drag(origin + ray_dir * self._pick_dist)
        elif ps.imgui.IsMouseReleased(ps.imgui.ImGuiMouseButton_Left):
            if self._on_release_drag is not None:
                self._on_release_drag()

        should_update_camera = False

        # Rotate camera
        if ps.imgui.IsMouseDown(ps.imgui.ImGuiMouseButton_Right) and ((dx != 0) or (dy != 0)):
            self._camera_phi -= dx / 2.0
            self._camera_theta -= dy / 4.0
            self._camera_theta = wp.clamp(self._camera_theta, low=1.0, high=179.0)
            should_update_camera = True

        # Translate camera
        if ps.imgui.IsMouseDown(ps.imgui.ImGuiMouseButton_Middle) and ((dx != 0) or (dy != 0)):
            camera_params = ps.get_view_camera_parameters()
            fov = camera_params.get_fov_vertical_deg()
            window_height = ps.get_window_size()[1]
            up_dir = camera_params.get_up_dir()
            right_dir = camera_params.get_right_dir()
            delta = up_dir * dy * 2.0 / window_height
            delta -= right_dir * dx * 2.0 / window_height
            delta *= wp.tan(wp.radians(fov) / 2.0)
            delta *= self._camera_radius
            delta[0] += self._camera_origin[0]
            delta[1] += self._camera_origin[1]
            delta[2] += self._camera_origin[2]
            self._camera_origin = (delta[0], delta[1], delta[2])
            should_update_camera = True

        # Zoom camera
        if ps.imgui.GetIO().MouseWheel != 0.0:
            ratio = 0.9 if (ps.imgui.GetIO().MouseWheel < 0) else (1 / 0.9)
            self._camera_radius = wp.clamp(self._camera_radius * ratio, low=2e-2, high=1e2)
            should_update_camera = True

        if should_update_camera:
            self._update_camera()

    def _update(self):
        self._update_gui()
        self._process_key_inputs()
        self._process_mouse_inputs()
        if self._user_update is not None:
            if not self._paused:
                self._user_update()

    def requests_close(self):
        return self._request_close or ps.window_requests_close()

    def shut_dwon(self):
        ps.shutdown(True)

    def frame_tick(self):
        ps.frame_tick()

    def exit(self):
        ps.unshow()

    def run(self):
        ps.show()


if __name__ == "__main__":

    def user_update():
        print("user_update")

    def on_pick(pick_result: ps.PickResult):
        print("on pick")

    def on_drag(drag_pos: tuple[float, float, float]):
        print(f"on drag: {drag_pos}")

    def on_release_drag():
        print("on release drag")

    viewer = Viewer()
    viewer.set_user_update(user_update)
    viewer.set_on_release_drag(on_release_drag)
    viewer.set_on_drag(on_drag)
    viewer.set_on_pick(on_pick)
    viewer.run()
