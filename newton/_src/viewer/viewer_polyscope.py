########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Style3D Viewer                                                                                     #
#   Author:         Wenchao Huang (physhuangwenchao@gmail.com)                                                         #
#   Date:           2025/06/19                                                                                         #
########################################################################################################################

import time

import numpy as np
import warp as wp
from typing_extensions import override

import newton
from newton import Axis, Mesh, State
from newton._src.solvers.style3d.collision.kernels import triangle_barycentric
from newton.utils import load_texture, normalize_texture
from .picking import Picking
from .viewer import ViewerBase

try:
    import polyscope as ps
except ImportError:
    ps = None

########################################################################################################################
####################################################    Kernels    #####################################################
########################################################################################################################

@wp.kernel
def transform_shape_vertices_kernel(
    index: wp.int32,
    scale_vertices: wp.int32,
    vertices_in: wp.array[wp.vec3],
    scaling3d: wp.array[wp.vec3],
    transforms: wp.array[wp.transform],
    vertices_out: wp.array[wp.vec3],
):
    tid = wp.tid()
    pos = vertices_in[tid]
    if scale_vertices:
        scaling = scaling3d[index]
        pos = wp.vec3(pos[0] * scaling[0], pos[1] * scaling[1], pos[2] * scaling[2])
    pos = wp.transform_point(transforms[index], pos)
    vertices_out[tid] = pos


@wp.kernel
def transform_to_mat4x4_kernel(
    transform_in: wp.array[wp.transform],
    transform_out: wp.array[wp.mat44],
):
    tid = wp.tid()
    transform = transform_in[tid]
    translation = wp.transform_get_translation(transform)
    wp.transform_set_translation(transform, translation)
    transform_out[tid] = wp.transform_to_matrix(transform)


@wp.kernel
def apply_particle_picking_force_kernel(
    particle_q: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    pick_particles: wp.array[int],
    pick_weights: wp.array[float],
    pick_target: wp.array[wp.vec3],
    pick_stiffness: float,
    pick_max_force: float,
):
    tid = wp.tid()
    particle_idx = pick_particles[tid]
    if particle_idx < 0:
        return
    if (particle_flags[particle_idx] & int(newton.ParticleFlags.ACTIVE)) == 0:
        return

    mass = particle_mass[particle_idx]
    if mass <= 0.0:
        return

    i0 = pick_particles[0]
    i1 = pick_particles[1]
    i2 = pick_particles[2]
    if i0 < 0 or i1 < 0 or i2 < 0:
        return

    w0 = pick_weights[0]
    w1 = pick_weights[1]
    w2 = pick_weights[2]
    picked_point = particle_q[i0] * w0 + particle_q[i1] * w1 + particle_q[i2] * w2
    force = pick_stiffness * (pick_target[0] - picked_point)

    force_mag = wp.length(force)
    if force_mag > pick_max_force:
        force = force * (pick_max_force / force_mag)

    wp.atomic_add(particle_f, particle_idx, force * pick_weights[tid])


########################################################################################################################
#####################################################    Viewer    #####################################################
########################################################################################################################

class ViewerPolyscope(ViewerBase):
    """Polyscope-backed viewer for Newton models.
    """

    # Keep to Polyscope materials that are stable across bundled versions.
    _POLYSCOPE_MATERIALS = {"wax", "candy", "flat", "mud", "clay", "ceramic"}
    _POLYSCOPE_GROUND_MODE = "tile_reflection"

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        vsync: bool = False,
        paused: bool = True,
    ) -> None:
        """Initialize a 3D renderer with customizable window properties.
        Args:
            window_size (Tuple[int, int]): Window dimensions (width, height)
        """
        if ps is None:
            raise ImportError("polyscope package is required for ViewerPolyscope. Install with: pip install polyscope")
    
        super().__init__()
        self._paused = paused
        self._request_close = False

        # Simulation state
        self.sim_time = 0.0
        self._step_requested = False

        # FPS counting
        self._rendering_fps = 1.0
        self._rendering_fps_counter = 0
        self._rendering_fps_last_time = 0.0

        ps.set_program_name("Newton Viewer")
        ps.init()
        ps.set_max_fps(360)
        ps.set_SSAA_factor(4)
        ps.set_enable_vsync(vsync)
        ps.set_give_focus_on_show(True)
        ps.set_window_size(width, height)
        ps.set_user_callback(self._update)
        ps.set_navigation_style("first_person")
        ps.set_do_default_mouse_interaction(True)
        ps.set_open_imgui_window_for_user_callback(True)
        ps.set_automatically_compute_scene_extents(False)
        ps.set_background_color([0.015, 0.015, 0.015])

        self.picking = None
        self.tri_indices = None
        self._last_state = None
        self._picking_line_entity = None
        self._particle_pick_target = None
        self._particle_pick_target_wp = None
        self._particle_pick_indices_wp = None
        self._particle_pick_weights_wp = None
        self._particle_pick_weights = np.zeros(3, dtype=np.float32)
        self._particle_pick_indices = np.full(3, -1, dtype=np.int32)
        self._particle_pick_world_offset = np.zeros(3, dtype=np.float32)
        self._particle_pick_ray_distance = 0.0
        self._particle_pick_stiffness = 1000.0
        self._particle_pick_max_force = 1000.0
        self._particle_pick_index = -1

        # Cache variables
        self.shape_body = None
        self.shape_world = None
        self.shape_flags = None
        self.shape_colors = None
        self.particle_world = None
        self._scene_extents_dirty = False
        self._body_transform_mat4x4 = None

        # Render entities
        self.tri_entity = None
        self.body_entities = {}
        self.particle_entity = None
        self._point_entities = {}
        self._point_enabled_state = {}
        self._point_quantity_state = {}
        self._shape_groups = {}
        self._shape_entities = {}
        self._shape_entity_colors = {}
        self._shape_entity_materials = {}
        self._shape_group_parent_name = "Worlds"

    @staticmethod
    def _polyscope_up_dir(up_axis) -> str:
        """Convert Newton's up-axis enum to Polyscope's up-dir string."""
        up_axis = Axis.from_any(up_axis)
        if up_axis == Axis.Z:
            return "z_up"
        elif up_axis == Axis.X:
            return "x_up"
        else:
            return "y_up"

    @staticmethod
    def _normalize(value, fallback=None) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if norm > 1.0e-8:
            return value / norm
        if fallback is None:
            return value
        return np.asarray(fallback, dtype=np.float32)

    @staticmethod
    def _mouse_pos_tuple(mouse_pos) -> tuple[float, float]:
        if hasattr(mouse_pos, "x") and hasattr(mouse_pos, "y"):
            return float(mouse_pos.x), float(mouse_pos.y)
        return float(mouse_pos[0]), float(mouse_pos[1])

    def _polyscope_window_size(self) -> tuple[int, int]:
        width, height = ps.get_window_size()
        if width <= 0 or height <= 0:
            width, height = ps.get_buffer_size()
        return max(int(width), 1), max(int(height), 1)

    def _polyscope_buffer_size(self) -> tuple[int, int]:
        width, height = ps.get_buffer_size()
        if width <= 0 or height <= 0:
            width, height = self._polyscope_window_size()
        return max(int(width), 1), max(int(height), 1)

    def _mouse_pos_to_buffer_coords(self, mouse_pos) -> tuple[float, float]:
        win_w, win_h = self._polyscope_window_size()
        buf_w, buf_h = self._polyscope_buffer_size()
        x, y = self._mouse_pos_tuple(mouse_pos)
        return x * (buf_w / win_w), y * (buf_h / win_h)

    def _mouse_ray(self, mouse_pos) -> tuple[wp.vec3, wp.vec3] | None:
        """Build a world-space ray from a Polyscope mouse position."""
        width, height = self._polyscope_buffer_size()
        x, y = self._mouse_pos_to_buffer_coords(mouse_pos)
        x = max(0.0, min(float(width - 1), float(x)))
        y = max(0.0, min(float(height - 1), float(y)))

        try:
            params = ps.get_view_camera_parameters()
            ray_start_np = np.asarray(params.get_position(), dtype=np.float32)
            look_dir = np.asarray(params.get_look_dir(), dtype=np.float32)
            right_dir = np.asarray(params.get_right_dir(), dtype=np.float32)
            up_dir = np.asarray(params.get_up_dir(), dtype=np.float32)
            fov_y = float(params.get_fov_vertical_deg())
            aspect = float(params.get_aspect())
        except Exception:
            return None

        look_norm = float(np.linalg.norm(look_dir))
        right_norm = float(np.linalg.norm(right_dir))
        up_norm = float(np.linalg.norm(up_dir))
        if look_norm <= 1.0e-8 or right_norm <= 1.0e-8 or up_norm <= 1.0e-8:
            return None
        look_dir /= look_norm
        right_dir /= right_norm
        up_dir /= up_norm

        tan_y = float(np.tan(np.radians(fov_y) * 0.5))
        screen_x = (2.0 * x / float(width) - 1.0) * aspect * tan_y
        screen_y = (1.0 - 2.0 * y / float(height)) * tan_y
        ray_dir_np = look_dir + right_dir * screen_x + up_dir * screen_y

        ray_norm = float(np.linalg.norm(ray_dir_np))
        if ray_norm <= 1.0e-8:
            return None
        ray_dir_np /= ray_norm
        ray_start = wp.vec3(float(ray_start_np[0]), float(ray_start_np[1]), float(ray_start_np[2]))
        ray_dir = wp.vec3(float(ray_dir_np[0]), float(ray_dir_np[1]), float(ray_dir_np[2]))
        return ray_start, ray_dir

    @staticmethod
    def _imgui_key(name: str):
        return getattr(ps.imgui, f"ImGuiKey_{name}", None)

    def _is_key_pressed_name(self, *names: str) -> bool:
        return any((key := self._imgui_key(name)) is not None and ps.imgui.IsKeyPressed(key) for name in names)

    def _release_particle_picking(self) -> None:
        self._particle_pick_index = -1
        self._particle_pick_indices[:] = -1
        self._particle_pick_weights[:] = 0.0
        self._particle_pick_target = None
        self._particle_pick_ray_distance = 0.0
        self._particle_pick_world_offset[:] = 0.0
        if self._particle_pick_indices_wp is not None:
            self._particle_pick_indices_wp.fill_(-1)
        if self._particle_pick_weights_wp is not None:
            self._particle_pick_weights_wp.zero_()

    def _is_particle_picking(self) -> bool:
        return self._particle_pick_index >= 0

    def _particle_world_offset(
        self,
        particle_idx: int,
        particle_world: np.ndarray | None = None,
        world_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        if (
            particle_world is None
            or world_offsets is None
            or self.world_offsets is None
            or self.world_offsets.shape[0] <= 0
        ):
            return np.zeros(3, dtype=np.float32)

        world_idx = int(particle_world[particle_idx])
        if world_idx < 0 or world_idx >= len(world_offsets):
            return np.zeros(3, dtype=np.float32)
        return np.asarray(world_offsets[world_idx], dtype=np.float32)

    @staticmethod
    def _normalized_pick_weights(weights) -> np.ndarray | None:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (3,) or not np.all(np.isfinite(weights)):
            return None
        weights = np.clip(weights, 0.0, 1.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1.0e-8:
            return None
        return weights / weight_sum

    @staticmethod
    def _is_tri_pick_result(pick_result) -> bool:
        if not pick_result.is_hit or pick_result.structure_type_name != "Surface Mesh":
            return False
        structure_name = str(pick_result.structure_name)
        return structure_name == "Triangles" or structure_name.endswith("/Triangles")

    def _pick_cloth_with_polyscope(self, mouse_pos, state: State, ray_start, ray_dir) -> bool:
        """Pick cloth through Polyscope so mesh occlusion matches what the user sees."""
        if (
            self.model is None
            or state.particle_q is None
            or self.model.particle_count <= 0
            or self.model.particle_mass is None
            or self.model.particle_flags is None
            or self.tri_entity is None
            or self.tri_indices is None
        ):
            return False

        try:
            pick_result = ps.pick(screen_coords=self._mouse_pos_tuple(mouse_pos))
        except Exception:
            return False

        if not self._is_tri_pick_result(pick_result):
            return False

        pick_data = pick_result.structure_data
        element_type = str(pick_data.get("element_type", "")).lower()
        if element_type and element_type != "face":
            return False

        face_idx = int(pick_data.get("index", pick_result.local_index))
        if face_idx < 0 or face_idx >= len(self.tri_indices):
            return False

        particle_q = state.particle_q.numpy().reshape(self.model.particle_count, 3)
        particle_mass = self.model.particle_mass.numpy()
        particle_flags = self.model.particle_flags.numpy()
        particle_world = self.model.particle_world.numpy() if self.model.particle_world is not None else None
        world_offsets = self.world_offsets.numpy() if self.world_offsets is not None else None

        positions = np.array(particle_q, copy=True)
        if particle_world is not None and world_offsets is not None and len(world_offsets) > 0:
            valid_world = (particle_world >= 0) & (particle_world < len(world_offsets))
            positions[valid_world] += world_offsets[particle_world[valid_world]]

        particle_valid = (particle_mass > 0.0) & ((particle_flags & int(newton.ParticleFlags.ACTIVE)) != 0)
        if particle_world is not None:
            particle_valid &= np.asarray([self._should_render_world(int(world_idx)) for world_idx in particle_world])

        ray_start_np = np.asarray([float(ray_start[0]), float(ray_start[1]), float(ray_start[2])], dtype=np.float32)
        ray_dir_np = self._normalize(
            np.asarray([float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2])], dtype=np.float32),
            fallback=(0.0, 0.0, -1.0),
        )

        tri = np.asarray(self.tri_indices[face_idx], dtype=np.int32)
        if not np.any(particle_valid[tri]):
            return False

        hit_point = np.asarray(pick_result.position, dtype=np.float32)
        weights = self._normalized_pick_weights(pick_data.get("bary_coords"))
        if weights is None:
            bary = triangle_barycentric(
                wp.vec3(*positions[tri[0]]),
                wp.vec3(*positions[tri[1]]),
                wp.vec3(*positions[tri[2]]),
                wp.vec3(*hit_point),
            )
            weights = self._normalized_pick_weights([bary[0], bary[1], bary[2]])
            if weights is None:
                return False
        offset = self._particle_world_offset(int(tri[int(np.argmax(weights))]), particle_world, world_offsets)
        target = hit_point - offset

        self._particle_pick_index = int(tri[int(np.argmax(weights))])
        self._particle_pick_indices = tri
        self._particle_pick_weights = weights
        self._particle_pick_world_offset = offset
        self._particle_pick_target = target.astype(np.float32, copy=True)
        self._particle_pick_ray_distance = max(float(np.dot(hit_point - ray_start_np, ray_dir_np)), 0.0)

        if self._particle_pick_indices_wp is not None:
            self._particle_pick_indices_wp.assign(self._particle_pick_indices)
        if self._particle_pick_weights_wp is not None:
            self._particle_pick_weights_wp.assign(self._particle_pick_weights)
        if self._particle_pick_target_wp is not None:
            self._particle_pick_target_wp.assign(np.asarray([self._particle_pick_target], dtype=np.float32))

        return True

    def _update_particle_pick_target(self, ray_start, ray_dir) -> None:
        if not self._is_particle_picking():
            return

        ray_start_np = np.asarray([float(ray_start[0]), float(ray_start[1]), float(ray_start[2])], dtype=np.float32)
        ray_dir_np = self._normalize(
            np.asarray([float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2])], dtype=np.float32),
            fallback=(0.0, 0.0, -1.0),
        )
        target_offset = ray_start_np + ray_dir_np * float(self._particle_pick_ray_distance)
        target = target_offset - self._particle_pick_world_offset
        self._particle_pick_target = target.astype(np.float32, copy=True)
        if self._particle_pick_target_wp is not None:
            self._particle_pick_target_wp.assign(np.asarray([self._particle_pick_target], dtype=np.float32))

    def _process_mouse_picking_drag(self) -> None:
        if not self.picking_enabled or self.picking is None:
            return

        right_button = ps.imgui.ImGuiMouseButton_Middle
        right_clicked = ps.imgui.IsMouseClicked(right_button)
        right_down = ps.imgui.IsMouseDown(right_button)
        right_released = ps.imgui.IsMouseReleased(right_button)

        if right_released:
            self.picking.release()
            self._release_particle_picking()
            return

        if not right_clicked and not right_down:
            return

        if ps.imgui.GetIO().WantCaptureMouse and not self.picking.is_picking() and not self._is_particle_picking():
            return

        mouse_pos = ps.imgui.GetMousePos()
        ray = self._mouse_ray(mouse_pos)
        if ray is None:
            return

        ray_start, ray_dir = ray
        if right_clicked and self._last_state is not None:
            self._release_particle_picking()
            self.picking.release()
            if not self._pick_cloth_with_polyscope(mouse_pos, self._last_state, ray_start, ray_dir):
                self.picking.pick(self._last_state, ray_start, ray_dir)

        if right_down and self.picking.is_picking():
            self.picking.update(ray_start, ray_dir)
        elif right_down and self._is_particle_picking():
            self._update_particle_pick_target(ray_start, ray_dir)

    def _render_picking_line(self) -> None:
        if not self.picking_enabled or self.picking is None:
            if self._picking_line_entity is not None:
                self._picking_line_entity.set_enabled(False)
            return

        if self.picking.is_picking():
            pick_body_idx = int(self.picking.pick_body.numpy()[0])
            if pick_body_idx < 0:
                if self._picking_line_entity is not None:
                    self._picking_line_entity.set_enabled(False)
                return

            pick_state = self.picking.pick_state.numpy()
            picked_point = np.asarray(pick_state[0]["picked_point_world"], dtype=np.float32).copy()
            pick_target = np.asarray(pick_state[0]["picking_target_world"], dtype=np.float32).copy()

            if self.world_offsets is not None and self.world_offsets.shape[0] > 0 and self.model.body_world is not None:
                body_world_idx = int(self.model.body_world.numpy()[pick_body_idx])
                if 0 <= body_world_idx < self.world_offsets.shape[0]:
                    world_offset = self.world_offsets.numpy()[body_world_idx]
                    pick_target += world_offset
                    picked_point += world_offset
        elif self._is_particle_picking() and self._last_state is not None and self._particle_pick_target is not None:
            particle_q = self._last_state.particle_q.numpy().reshape(self.model.particle_count, 3)
            picked_point = (
                particle_q[self._particle_pick_indices[0]] * self._particle_pick_weights[0]
                + particle_q[self._particle_pick_indices[1]] * self._particle_pick_weights[1]
                + particle_q[self._particle_pick_indices[2]] * self._particle_pick_weights[2]
            ).astype(np.float32, copy=True)
            pick_target = np.asarray(self._particle_pick_target, dtype=np.float32).copy()
            picked_point += self._particle_pick_world_offset
            pick_target += self._particle_pick_world_offset
        else:
            if self._picking_line_entity is not None:
                self._picking_line_entity.set_enabled(False)
            return

        nodes = np.asarray([picked_point, pick_target], dtype=np.float32)
        if self._picking_line_entity is None:
            self._picking_line_entity = ps.register_curve_network(
                nodes=nodes,
                name="picking_line",
                edges=np.asarray([[0, 1]], dtype=np.int32),
                color=(0.0, 1.0, 1.0),
                material="flat",
            )
            self._picking_line_entity.set_radius(0.003, relative=False)
        else:
            self._picking_line_entity.update_node_positions(nodes)
            self._picking_line_entity.set_enabled(True)

    def _process_key_controls(self) -> None:
        """Handle viewer lifecycle keys that are outside Polyscope navigation."""
        if self._is_key_pressed_name("Escape"):
            self.exit()
            return

        if ps.imgui.GetIO().WantCaptureKeyboard:
            return

        # Keep simulation playback control local while Polyscope owns navigation.
        if self._is_key_pressed_name("Space"):
            self._paused = not self._paused
        if self._is_key_pressed_name("Period") and self._paused:
            self._step_requested = True

    def _update_gui(self) -> None:
        """Draw the compact runtime status panel in Polyscope's UI."""
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
        ps.imgui.Text(f"FPS: {self._rendering_fps:.1f} / {1e3 / self._rendering_fps:.1f}ms")

        model = self.model
        if model is not None and (
            model.body_count or model.particle_count or model.spring_count or model.tri_count or model.tet_count
        ):
            ps.imgui.Separator()
            ps.imgui.Text("Statistics:")
            if model.world_count:
                ps.imgui.Text(f" - Worlds: {model.world_count}")
            if model.body_count:
                ps.imgui.Text(f" - Bodies: {model.body_count}")
            if model.shape_count:
                ps.imgui.Text(f" - Shapes: {model.shape_count}")
            if model.joint_count:
                ps.imgui.Text(f" - Joints: {model.joint_count}")
            if model.particle_count:
                ps.imgui.Text(f" - Particles: {model.particle_count}")
            if model.spring_count:
                ps.imgui.Text(f" - Springs: {model.spring_count}")
            if model.tri_count:
                ps.imgui.Text(f" - Triangles: {model.tri_count}")
            if model.edge_count:
                ps.imgui.Text(f" - Edges: {model.edge_count}")
            if model.tet_count:
                ps.imgui.Text(f" - Tetrahedrals: {model.tet_count}")

    def _update(self) -> None:
        """Per-frame Polyscope callback for lightweight controls and UI."""
        self._process_key_controls()
        self._process_mouse_picking_drag()
        self._update_gui()

    def _shape_material(self, shape_idx: int) -> str:
        """Map Newton mesh metadata to a stable Polyscope material name."""
        source = self.model.shape_source[shape_idx]
        material = getattr(source, "material", None)
        if isinstance(material, str) and material in self._POLYSCOPE_MATERIALS:
            return material

        metallic = getattr(source, "metallic", None)
        if metallic is not None and metallic >= 0.5:
            return "ceramic"

        roughness = getattr(source, "roughness", None)
        if roughness is None:
            return "wax"
        if roughness <= 0.25:
            return "candy"
        if roughness >= 0.75:
            return "clay"
        return "wax"

    @staticmethod
    def _shape_name(model: newton.Model, shape_idx: int) -> str:
        label = model.shape_label[shape_idx]
        if label:
            return f"{label}/shape_{shape_idx}"
        return f"shape_{shape_idx}"

    @staticmethod
    def _is_ground_shape(model: newton.Model, shape_idx: int) -> bool:
        """Return whether a shape is Newton's global ground plane."""
        if int(model.shape_type.numpy()[shape_idx]) != newton.GeoType.PLANE:
            return False

        label = model.shape_label[shape_idx]
        if label == "ground" or label == "ground_plane" or label.startswith("ground_"):
            return True

        scale = model.shape_scale.numpy()[shape_idx]
        shape_body = int(model.shape_body.numpy()[shape_idx])
        return shape_body == -1 and float(scale[0]) <= 0.0 and float(scale[1]) <= 0.0

    def _sync_ground_plane_from_model(self, model: newton.Model) -> None:
        """Use Polyscope's ground plane when the Newton model has ground."""
        shape_transform = model.shape_transform.numpy()
        for shape_idx in range(model.shape_count):
            if not self._is_ground_shape(model, shape_idx):
                continue

            transform = wp.transform(*shape_transform[shape_idx])
            translation = wp.transform_get_translation(transform)
            height = float(translation[int(Axis.from_any(model.up_axis))])
            ps.set_ground_plane_height_mode(ps.GroundPlaneHeightMode.manual)
            ps.set_ground_plane_mode(self._POLYSCOPE_GROUND_MODE)
            ps.set_ground_plane_height(height)
            return

        ps.set_ground_plane_mode("none")

    @staticmethod
    def _shape_mesh(model: newton.Model, shape_idx: int) -> tuple[Mesh | None, bool]:
        """Return the display mesh and whether model scale must be applied."""
        geo_type = int(model.shape_type.numpy()[shape_idx])
        geo_scale = [float(v) for v in model.shape_scale.numpy()[shape_idx]]
        source = model.shape_source[shape_idx]

        if geo_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            return source, True
        if geo_type == newton.GeoType.HFIELD:
            actual_heights = source.min_z + source.data * (source.max_z - source.min_z)
            return newton.Mesh.create_heightfield(
                heightfield=actual_heights.T,
                extent_x=source.hx * 2.0,
                extent_y=source.hy * 2.0,
                ground_z=source.min_z,
                compute_inertia=False,
            ), False
        if geo_type == newton.GeoType.PLANE:
            width = geo_scale[0] if geo_scale and geo_scale[0] > 0.0 else 1000.0
            length = geo_scale[1] if len(geo_scale) > 1 and geo_scale[1] > 0.0 else 1000.0
            return newton.Mesh.create_plane(width, length, compute_inertia=False), False
        if geo_type == newton.GeoType.SPHERE:
            return newton.Mesh.create_sphere(geo_scale[0], compute_inertia=False), False
        if geo_type == newton.GeoType.CAPSULE:
            return newton.Mesh.create_capsule(
                geo_scale[0], geo_scale[1], up_axis=newton.Axis.Z, compute_inertia=False
            ), False
        if geo_type == newton.GeoType.CYLINDER:
            return newton.Mesh.create_cylinder(
                geo_scale[0], geo_scale[1], up_axis=newton.Axis.Z, compute_inertia=False
            ), False
        if geo_type == newton.GeoType.CONE:
            return newton.Mesh.create_cone(geo_scale[0], geo_scale[1], up_axis=newton.Axis.Z, compute_inertia=False), False
        if geo_type == newton.GeoType.BOX:
            if len(geo_scale) == 1:
                hx = hy = hz = geo_scale[0]
            else:
                hx, hy, hz = geo_scale[:3]
            return newton.Mesh.create_box(hx, hy, hz, duplicate_vertices=True, compute_inertia=False), False
        if geo_type == newton.GeoType.ELLIPSOID:
            rx = geo_scale[0] if len(geo_scale) > 0 else 1.0
            ry = geo_scale[1] if len(geo_scale) > 1 else rx
            rz = geo_scale[2] if len(geo_scale) > 2 else rx
            return newton.Mesh.create_ellipsoid(rx, ry, rz, compute_inertia=False), False

        return None, False

    def _shape_color(self, shape_idx: int) -> tuple[float, float, float]:
        if self.shape_colors is None:
            return (0.0, 0.0, 0.0)

        color = self.shape_colors[shape_idx]
        # Newton GL treats shape colors as sRGB and converts to linear before
        # lighting; Polyscope expects linear-like colors here.
        color = np.power(np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0), 2.2)
        return (float(color[0]), float(color[1]), float(color[2]))

    def _sync_shape_colors_from_model(self) -> None:
        if self.model is None or self.model.shape_color is None or len(self._shape_entities) == 0:
            return

        self.shape_colors = self.model.shape_color.numpy()
        for shape_idx, entity in self._shape_entities.items():
            color = self._shape_color(shape_idx)
            if self._shape_entity_colors.get(shape_idx) != color:
                entity.set_color(color)
                self._shape_entity_colors[shape_idx] = color

    def _sync_shape_materials_from_model(self) -> None:
        if self.model is None or len(self._shape_entities) == 0:
            return

        for shape_idx, entity in self._shape_entities.items():
            material = self._shape_material(shape_idx)
            if self._shape_entity_materials.get(shape_idx) != material:
                entity.set_material(material)
                self._shape_entity_materials[shape_idx] = material

    def _shape_texture_data(self, shape_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return texture pixels and UVs for a rendered shape mesh."""
        source = self.model.shape_source[shape_idx]
        if isinstance(source, Mesh) and source.texture is not None and source.uvs is not None:
            texture = normalize_texture(load_texture(source.texture), require_channels=True)
            if texture is None:
                return None

            texture = np.asarray(texture[:, :, :3], dtype=np.float32) / 255.0
            # Match the sRGB-to-linear conversion used for flat shape colors.
            texture = np.power(np.clip(texture, 0.0, 1.0), 2.2)
            return texture, np.asarray(source.uvs, dtype=np.float32)

        return None

    def _add_shape_texture_quantity(self, entity, shape_idx: int, shape_mesh: Mesh) -> None:
        """Attach a texture quantity when the rendered mesh has compatible UVs."""
        texture_data = self._shape_texture_data(shape_idx)
        if texture_data is None:
            return

        texture, uvs = texture_data
        vertices = np.asarray(shape_mesh.vertices)
        if len(uvs) != len(vertices):
            return

        param_name = "UV"
        entity.add_parameterization_quantity(param_name, uvs, defined_on="vertices", enabled=False)
        entity.add_color_quantity(
            "Texture",
            texture,
            defined_on="texture",
            param_name=param_name,
            enabled=True,
        )

    def _world_offsets_numpy(self) -> np.ndarray | None:
        if self.world_offsets is None or self.world_offsets.shape[0] <= 0:
            return None
        return self.world_offsets.numpy()

    @staticmethod
    def _world_offset(world_offsets: np.ndarray | None, world_idx: int) -> np.ndarray | None:
        if world_offsets is None or world_idx < 0 or world_idx >= len(world_offsets):
            return None
        return np.asarray(world_offsets[world_idx], dtype=np.float32)

    def _apply_particle_world_offsets(self, particle_q: np.ndarray, world_offsets: np.ndarray | None) -> np.ndarray:
        """Apply per-world display offsets to particle positions."""
        if self.particle_world is None or world_offsets is None or len(world_offsets) == 0:
            return particle_q

        particle_q = np.array(particle_q, dtype=np.float32, copy=True)
        valid_world = (self.particle_world >= 0) & (self.particle_world < len(world_offsets))
        particle_q[valid_world] += world_offsets[self.particle_world[valid_world]]
        return particle_q

    def _array_numpy(self, value) -> np.ndarray:
        if isinstance(value, wp.array):
            return value.numpy()
        return np.asarray(value)

    def _points_numpy(self, points) -> np.ndarray:
        points = self._array_numpy(points).reshape(-1, 3).astype(np.float32, copy=False)
        world_offsets = self._world_offsets_numpy()
        if self.model is not None and len(points) == self.model.particle_count:
            points = self._apply_particle_world_offsets(points, world_offsets)
        return points

    def _set_point_enabled_from_log(self, entity_name: str, entity, enabled: bool) -> None:
        """Only write enabled state when the logging-side intent changes."""
        if self._point_enabled_state.get(entity_name) == enabled:
            return
        entity.set_enabled(enabled)
        self._point_enabled_state[entity_name] = enabled

    def _clear_shape_groups(self) -> None:
        for group_name in list(self._shape_groups):
            if group_name == self._shape_group_parent_name:
                continue
            ps.remove_group(group_name, error_if_absent=False)
        if self._shape_group_parent_name in self._shape_groups:
            ps.remove_group(self._shape_group_parent_name, error_if_absent=False)
        self._shape_groups.clear()

    def _get_or_create_shape_group(self, group_name: str):
        """Create Polyscope hierarchy groups lazily for multi-world models."""
        group = self._shape_groups.get(group_name)
        if group is not None:
            return group

        try:
            group = ps.create_group(group_name)
        except Exception:
            group = ps.get_group(group_name)
        group.set_show_child_details(True)
        group.set_hide_descendants_from_structure_lists(True)
        self._shape_groups[group_name] = group

        if group_name != self._shape_group_parent_name:
            parent = self._get_or_create_shape_group(self._shape_group_parent_name)
            parent.add_child_group(group)

        return group

    def _shape_group_name(self, shape_idx: int) -> str | None:
        if self.model is None or self.model.world_count <= 1:
            return None

        world_idx = int(self.shape_world[shape_idx]) if self.shape_world is not None else -1
        if world_idx < 0:
            return "Global"
        return f"World {world_idx}"

    @override
    def set_model(self, model: newton.Model | None) -> None:
        """Register Newton model geometry with Polyscope."""
        super().set_model(model)
        self.shape_body = None
        self.shape_world = None
        self.shape_colors = None
        self.particle_world = None
        self.body_entities.clear()
        self._point_entities.clear()
        self._point_enabled_state.clear()
        self._point_quantity_state.clear()
        self._shape_entity_materials.clear()
        self._shape_entity_colors.clear()
        self._shape_entities.clear()
        self._clear_shape_groups()
        self._scene_extents_dirty = False
        self.picking = None
        self._last_state = None
        self._release_particle_picking()
        self._particle_pick_target_wp = None
        self._particle_pick_indices_wp = None
        self._particle_pick_weights_wp = None

        # Add meshes
        if model is not None:
            self._sync_ground_plane_from_model(model)
            ps.set_up_dir(self._polyscope_up_dir(model.up_axis))
            self.picking = Picking(model, world_offsets=self.world_offsets)
            self.picking.visible_worlds_mask = self._visible_worlds_mask
            self._particle_pick_weights_wp = wp.zeros(3, dtype=float, device=model.device)
            self._particle_pick_target_wp = wp.zeros(1, dtype=wp.vec3, device=model.device)
            self._particle_pick_indices_wp = wp.array([-1, -1, -1], dtype=int, device=model.device)

            # Cache
            self.shape_body = model.shape_body.numpy()
            self.shape_flags = model.shape_flags.numpy()
            self.shape_world = model.shape_world.numpy() if model.shape_world is not None else None
            self.shape_colors = model.shape_color.numpy() if model.shape_color is not None else None
            self.particle_world = model.particle_world.numpy() if model.particle_world is not None else None
            self._body_transform_mat4x4 = wp.zeros(model.body_count, dtype=wp.mat44)
            particle_q = model.particle_q.numpy().reshape(model.particle_count, 3)

            # Register particle entity
            if model.particle_count > 0:
                self.particle_entity = ps.register_point_cloud(
                    name="Particles",
                    enabled=False,
                    points=particle_q,
                    point_render_mode="sphere",
                    material="wax",
                )
                self._point_entities["Particles"] = self.particle_entity
                self._point_entities["/model/particles"] = self.particle_entity
                self._point_quantity_state["Particles"] = set()
                self._point_enabled_state["Particles"] = False

            # Register triangle entity
            if model.tri_count > 0:
                self.tri_indices = model.tri_indices.numpy().reshape(model.tri_count, 3)
                self.tri_entity = ps.register_surface_mesh(
                    name="Triangles",
                    vertices=particle_q,
                    faces=self.tri_indices,
                    color=(184 / 255.0, 67 / 255.0, 1),
                    back_face_policy="custom",
                    edge_color=(0, 0, 0),
                    smooth_shade=False,
                    edge_width=0.3,
                )
                self.tri_entity.set_selection_mode("faces_only")

            # Walk model shapes directly: body=-1 static shapes are not covered
            # by body_shapes but still need to be rendered.
            for shape_idx in range(model.shape_count):
                if self._is_ground_shape(model, shape_idx):
                    continue

                if self.shape_flags[shape_idx] & int(newton.ShapeFlags.VISIBLE) == 0:
                    continue

                shape_mesh, scale_vertices = self._shape_mesh(model, shape_idx)
                if shape_mesh is None:
                    continue

                # Bake the shape-local transform into the mesh. Dynamic body
                # transforms are applied per-frame via the Polyscope entity transform.
                shape_vertices = wp.array(shape_mesh.vertices, dtype=wp.vec3)
                shape_material = self._shape_material(shape_idx)
                shape_color = self._shape_color(shape_idx)

                wp.launch(
                    transform_shape_vertices_kernel,
                    dim=len(shape_vertices),
                    inputs=[
                        shape_idx,
                        int(scale_vertices),
                        shape_vertices,
                        model.shape_scale,
                        model.shape_transform,
                    ],
                    outputs=[shape_vertices],
                )

                shape_name = self._shape_name(model, shape_idx)
                shape_entity = ps.register_surface_mesh(
                    name=shape_name,
                    vertices=shape_vertices.numpy(),
                    faces=shape_mesh.indices.reshape(-1, 3),
                    back_face_policy="cull",
                    edge_color=(0, 0, 0),
                    smooth_shade=False,
                    color=shape_color,
                    material=shape_material,
                )
                self._add_shape_texture_quantity(shape_entity, shape_idx, shape_mesh)

                shape_group_name = self._shape_group_name(shape_idx)
                if shape_group_name is not None:
                    shape_entity.add_to_group(self._get_or_create_shape_group(shape_group_name))

                self.body_entities[shape_name] = shape_entity
                self._shape_entities[shape_idx] = shape_entity
                self._shape_entity_colors[shape_idx] = shape_color
                self._shape_entity_materials[shape_idx] = shape_material

            ps.set_automatically_compute_scene_extents(True)
            ps.update_scene_extents()
            ps.set_automatically_compute_scene_extents(False)
            self._scene_extents_dirty = self.world_offsets is not None and self.world_offsets.shape[0] > 0
        else:
            ps.set_ground_plane_mode("none")

    @override
    def log_state(self, state: State) -> None:
        """Update registered Polyscope structures from the current Newton state."""
        self._last_state = state
        world_offsets = self._world_offsets_numpy()

        # Download to host.
        if state.particle_q is not None:
            particle_q = state.particle_q.numpy().reshape(state.particle_count, 3)
            particle_q = self._apply_particle_world_offsets(particle_q, world_offsets)

        if self.particle_entity is not None:
            self._set_point_enabled_from_log(
                "Particles", self.particle_entity, self.show_particles and not self._layer_force_hidden()
            )
            if self.particle_entity.is_enabled():
                self.particle_entity.update_point_positions(particle_q)

        if self.tri_entity is not None:
            self.tri_entity.update_vertex_positions(particle_q)

        if len(self.body_entities) > 0:
            self._sync_shape_colors_from_model()
            self._sync_shape_materials_from_model()

            wp.launch(
                transform_to_mat4x4_kernel,
                dim=self.model.body_count,
                inputs=[state.body_q],
                outputs=[self._body_transform_mat4x4],
            )

            body_q = self._body_transform_mat4x4.numpy()
            for shape_idx, entity in self._shape_entities.items():
                body_idx = int(self.shape_body[shape_idx])
                world_idx = int(self.shape_world[shape_idx]) if self.shape_world is not None else -1
                world_offset = self._world_offset(world_offsets, world_idx)
                if body_idx >= 0:
                    transform = np.array(body_q[body_idx], dtype=np.float32, copy=True)
                    if world_offset is not None:
                        transform[:3, 3] += world_offset
                    entity.set_transform(transform)
                elif world_offset is not None:
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, 3] = world_offset
                    entity.set_transform(transform)

        if self._scene_extents_dirty:
            ps.set_automatically_compute_scene_extents(True)
            ps.update_scene_extents()
            ps.set_automatically_compute_scene_extents(False)
            self._scene_extents_dirty = False

        self._render_picking_line()
        ps.request_redraw()

    @override
    def is_key_down(self, key: str | int) -> bool:
        if isinstance(key, str):
            if len(key) == 1 and key.isdigit():
                key_name = f"ImGuiKey_{key}"
            elif len(key) == 1 and key.isalpha():
                key_name = f"ImGuiKey_{key.upper()}"
            else:
                key_name = f"ImGuiKey_{key}"
            key = getattr(ps.imgui, key_name, None)
            if key is None:
                return False
        return ps.imgui.IsKeyDown(key)

    @override
    def is_running(self) -> bool:
        return not self.requests_close()

    @override
    def is_paused(self) -> bool:
        return self._paused

    @override
    def should_step(self) -> bool:
        if not self._paused:
            self._step_requested = False
            return True
        if self._step_requested:
            self._step_requested = False
            return True
        return False

    @override
    def begin_frame(self, time: float) -> None:
        super().begin_frame(time)
        self.sim_time = time

    @override
    def end_frame(self) -> None:
        self.frame_tick()

    def requests_close(self) -> bool:
        return self._request_close or ps.window_requests_close()

    def shut_dwon(self) -> None:
        ps.shutdown(True)

    def frame_tick(self) -> None:
        ps.frame_tick()

    def exit(self) -> None:
        self._request_close = True
        ps.unshow()

    def run(self) -> None:
        ps.show()

    @override
    def apply_forces(self, state: newton.State) -> None:
        if self.picking_enabled and self.picking is not None:
            self.picking._apply_picking_force(state)
        if (
            self.picking_enabled
            and self.model is not None
            and state.particle_q is not None
            and state.particle_qd is not None
            and state.particle_f is not None
            and self.model.particle_flags is not None
            and self.model.particle_mass is not None
            and self._particle_pick_indices_wp is not None
            and self._particle_pick_weights_wp is not None
            and self._particle_pick_target_wp is not None
        ):
            wp.launch(
                kernel=apply_particle_picking_force_kernel,
                dim=3,
                inputs=[
                    state.particle_q,
                    state.particle_f,
                    self.model.particle_flags,
                    self.model.particle_mass,
                    self._particle_pick_indices_wp,
                    self._particle_pick_weights_wp,
                    self._particle_pick_target_wp,
                    self._particle_pick_stiffness,
                    self._particle_pick_max_force,
                ],
                device=self.model.device,
            )

    @override
    def log_array(self, name: str, array) -> None:
        pass

    @override
    def log_instances(
        self,
        name: str,
        mesh: Mesh,
        xforms,
        scales,
        colors,
        materials,
        hidden: bool = False,
    ) -> None:
        pass

    @override
    def log_lines(
        self,
        name: str,
        starts,
        ends,
        colors,
        width: float = 0.01,
        hidden: bool = False,
    ) -> None:
        pass

    @override
    def log_mesh(
        self,
        name: str,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden: bool = False,
        backface_culling: bool = True,
    ) -> None:
        pass

    @override
    def log_points(
        self,
        name: str,
        points,
        radii=None,
        colors=None,
        hidden: bool = False,
    ) -> None:
        """Log or update a Polyscope point cloud without clobbering UI-edited display settings."""
        name = self._qualify(name)
        if name == "/model/particles" and self.particle_entity is not None:
            entity_name = "Particles"
        else:
            entity_name = name

        if points is None:
            entity = self._point_entities.get(entity_name)
            if entity is not None:
                self._set_point_enabled_from_log(entity_name, entity, False)
            return

        points_np = self._points_numpy(points)
        entity = self._point_entities.get(entity_name)
        if entity is not None and entity.n_points() != len(points_np):
            ps.remove_point_cloud(entity_name, error_if_absent=False)
            entity = None
            self._point_quantity_state.pop(entity_name, None)
            self._point_enabled_state.pop(entity_name, None)

        if entity is None:
            radius = 0.01
            if radii is not None:
                if isinstance(radii, (int, float, np.integer, np.floating)):
                    radius = float(radii)
                else:
                    radii_np = np.asarray(self._array_numpy(radii), dtype=np.float32).reshape(-1)
                    if len(radii_np) > 0:
                        radius = float(np.mean(radii_np))

            entity = ps.register_point_cloud(
                name=entity_name,
                points=points_np,
                enabled=not hidden,
                radius=radius,
                point_render_mode="sphere",
                material="wax",
            )
            self._point_entities[entity_name] = entity
            if name == "/model/particles":
                self.particle_entity = entity
                self._point_entities["/model/particles"] = entity
            self._point_quantity_state[entity_name] = set()
            self._point_enabled_state[entity_name] = not hidden
        else:
            entity.update_point_positions(points_np)
            self._set_point_enabled_from_log(entity_name, entity, not hidden)

        quantity_state = self._point_quantity_state.setdefault(entity_name, set())

        if radii is not None:
            if isinstance(radii, (int, float, np.integer, np.floating)):
                if "radius_quantity" not in quantity_state and "scalar_radius" not in quantity_state:
                    entity.clear_point_radius_quantity()
                    entity.set_radius(float(radii), relative=False)
                    quantity_state.add("scalar_radius")
            else:
                radii_np = np.asarray(self._array_numpy(radii), dtype=np.float32).reshape(-1)
                if len(radii_np) == len(points_np):
                    if "radius_quantity" in quantity_state:
                        entity.get_quantity_buffer("radius", "values").update_data_from_host(radii_np)
                    else:
                        entity.add_scalar_quantity("radius", radii_np, enabled=False)
                        entity.set_point_radius_quantity("radius", autoscale=False)
                        quantity_state.discard("scalar_radius")
                        quantity_state.add("radius_quantity")

        if colors is not None:
            colors_np = np.asarray(self._array_numpy(colors), dtype=np.float32)
            if colors_np.ndim == 1 and colors_np.shape[0] == 3:
                if "color_quantity" not in quantity_state and "solid_color" not in quantity_state:
                    entity.set_color(colors_np)
                    quantity_state.add("solid_color")
            else:
                colors_np = colors_np.reshape(-1, 3)
                if len(colors_np) == len(points_np):
                    if "color_quantity" in quantity_state:
                        entity.get_quantity_buffer("colors", "colors").update_data_from_host(colors_np)
                    else:
                        entity.add_color_quantity("colors", colors_np, enabled=True)
                        quantity_state.discard("solid_color")
                        quantity_state.add("color_quantity")

    @override
    def log_scalar(self, name: str, value, *, clear: bool = False, smoothing: int = 1) -> None:
        pass

    @override
    def close(self) -> None:
        pass
