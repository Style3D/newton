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

import newton
from newton import Axis, AxisType, Mesh, State
from newton._src.utils.texture import load_texture, normalize_texture
from newton._src.viewer.viewer import ViewerBase
from style3d.viewer.viewer import Viewer

########################################################################################################################
####################################################    Kernels    #####################################################
########################################################################################################################

@wp.kernel
def transform_shape_vertices_kernel(
    index: wp.int32,
    scale: float,
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
    pos = wp.transform_point(transforms[index], pos) * scale
    vertices_out[tid] = pos


@wp.kernel
def transform_to_mat4x4_kernel(
    scale: float,
    transform_in: wp.array[wp.transform],
    transform_out: wp.array[wp.mat44],
):
    tid = wp.tid()
    transform = transform_in[tid]
    translation = wp.transform_get_translation(transform) * scale
    wp.transform_set_translation(transform, translation)
    transform_out[tid] = wp.transform_to_matrix(transform)

########################################################################################################################
#####################################################    Viewer    #####################################################
########################################################################################################################

class ViewerNewton(Viewer, ViewerBase):
    # Keep to Polyscope materials that are stable across bundled versions.
    _POLYSCOPE_MATERIALS = {"wax", "candy", "flat", "mud", "clay", "ceramic"}

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
        self.shape_colors = None
        self.shape_body = None
        self._body_transform_mat4x4 = None

        # Render entities
        self.tri_entity = None
        self.particle_entity = None
        self.body_entities = {}
        self._shape_entities = {}
        self._shape_entity_colors = {}
        self._shape_entity_materials = {}

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
            if pick_result.is_hit and self.tri_entity is not None:
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

    def _shape_material(self, shape_idx: int):
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
    def _shape_name(model, shape_idx: int):
        label = model.shape_label[shape_idx]
        return label if label else f"shape_{shape_idx}"

    @staticmethod
    def _is_ground_shape(model, shape_idx: int):
        geo_type = int(model.shape_type.numpy()[shape_idx])
        if geo_type != newton.GeoType.PLANE:
            return False

        # Polyscope already owns the visual ground plane; skip Newton's infinite
        # ground shape to avoid z-fighting and duplicate floor visuals.
        label = model.shape_label[shape_idx]
        if label == "ground_plane" or label.startswith("ground_plane_"):
            return True

        shape_body = int(model.shape_body.numpy()[shape_idx])
        scale = model.shape_scale.numpy()[shape_idx]
        return shape_body == -1 and float(scale[0]) <= 0.0 and float(scale[1]) <= 0.0

    # Newton's UV primitives duplicate pole vertices for texture seams. With
    # Polyscope smooth shading this can produce black pole artifacts, so the
    # display meshes below use a single pole vertex and no degenerate triangles.
    @staticmethod
    def _create_sphere_mesh(radius: float, num_latitudes: int = 32, num_longitudes: int = 32):
        vertices = [[0.0, radius, 0.0]]
        normals = [[0.0, 1.0, 0.0]]
        uvs = [[0.5, 0.0]]

        for i in range(1, num_latitudes):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for j in range(num_longitudes):
                phi = j * 2.0 * np.pi / num_longitudes
                x = np.cos(phi) * sin_theta
                y = cos_theta
                z = np.sin(phi) * sin_theta
                vertices.append([x * radius, y * radius, z * radius])
                normals.append([x, y, z])
                uvs.append([float(j) / num_longitudes, float(i) / num_latitudes])

        bottom = len(vertices)
        vertices.append([0.0, -radius, 0.0])
        normals.append([0.0, -1.0, 0.0])
        uvs.append([0.5, 1.0])

        indices = []
        first_ring = 1
        for j in range(num_longitudes):
            j_next = (j + 1) % num_longitudes
            indices.extend([0, first_ring + j_next, first_ring + j])

        for i in range(num_latitudes - 2):
            upper = 1 + i * num_longitudes
            lower = upper + num_longitudes
            for j in range(num_longitudes):
                j_next = (j + 1) % num_longitudes
                indices.extend([upper + j, lower + j, upper + j_next])
                indices.extend([upper + j_next, lower + j, lower + j_next])

        last_ring = 1 + (num_latitudes - 2) * num_longitudes
        for j in range(num_longitudes):
            j_next = (j + 1) % num_longitudes
            indices.extend([bottom, last_ring + j, last_ring + j_next])

        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        triangles = vertices[indices]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        # Keep culling and lighting stable even if a construction branch flips winding.
        inward = np.einsum("ij,ij->i", face_normals, centers) < 0.0
        flipped = indices[inward].copy()
        flipped[:, [1, 2]] = flipped[:, [2, 1]]
        indices[inward] = flipped

        return newton.Mesh(
            vertices=vertices,
            indices=indices.reshape(-1),
            normals=np.asarray(normals, dtype=np.float32),
            uvs=np.asarray(uvs, dtype=np.float32),
            compute_inertia=False,
        )

    @staticmethod
    def _create_capsule_mesh(radius: float, half_height: float, segments: int = 32):
        hemi_segments = max(4, segments // 2)
        vertices = [[0.0, 0.0, half_height + radius]]
        normals = [[0.0, 0.0, 1.0]]
        uvs = [[0.5, 0.0]]
        rings = []

        def add_ring(z: float, ring_radius: float, normal_z: float, v: float):
            start = len(vertices)
            rings.append(start)
            for j in range(segments):
                phi = j * 2.0 * np.pi / segments
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                vertices.append([ring_radius * cos_phi, ring_radius * sin_phi, z])
                radial_normal = np.sqrt(max(0.0, 1.0 - normal_z * normal_z))
                normals.append([cos_phi * radial_normal, sin_phi * radial_normal, normal_z])
                uvs.append([float(j) / segments, v])

        for i in range(1, hemi_segments + 1):
            alpha = i * 0.5 * np.pi / hemi_segments
            add_ring(
                half_height + radius * np.cos(alpha),
                radius * np.sin(alpha),
                np.cos(alpha),
                0.5 * float(i) / hemi_segments,
            )

        add_ring(-half_height, radius, 0.0, 0.5)

        for i in range(1, hemi_segments):
            alpha = 0.5 * np.pi + i * 0.5 * np.pi / hemi_segments
            add_ring(
                -half_height + radius * np.cos(alpha),
                radius * np.sin(alpha),
                np.cos(alpha),
                0.5 + 0.5 * float(i) / hemi_segments,
            )

        bottom = len(vertices)
        vertices.append([0.0, 0.0, -half_height - radius])
        normals.append([0.0, 0.0, -1.0])
        uvs.append([0.5, 1.0])

        indices = []
        first_ring = rings[0]
        for j in range(segments):
            j_next = (j + 1) % segments
            indices.extend([0, first_ring + j, first_ring + j_next])

        for upper, lower in zip(rings[:-1], rings[1:]):
            for j in range(segments):
                j_next = (j + 1) % segments
                indices.extend([upper + j, lower + j, upper + j_next])
                indices.extend([upper + j_next, lower + j, lower + j_next])

        last_ring = rings[-1]
        for j in range(segments):
            j_next = (j + 1) % segments
            indices.extend([bottom, last_ring + j_next, last_ring + j])

        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        triangles = vertices[indices]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        # Capsule is centered at the origin, so center dot face normal is enough
        # to detect inward-facing triangles.
        inward = np.einsum("ij,ij->i", face_normals, centers) < 0.0
        flipped = indices[inward].copy()
        flipped[:, [1, 2]] = flipped[:, [2, 1]]
        indices[inward] = flipped

        normals = np.asarray(normals, dtype=np.float32)
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normal_lengths, 1.0e-8)

        return newton.Mesh(
            vertices=vertices,
            indices=indices.reshape(-1),
            normals=normals,
            uvs=np.asarray(uvs, dtype=np.float32),
            compute_inertia=False,
        )

    @staticmethod
    def _create_ellipsoid_mesh(rx: float, ry: float, rz: float, num_latitudes: int = 32, num_longitudes: int = 32):
        vertices = [[0.0, ry, 0.0]]
        normals = [[0.0, 1.0, 0.0]]
        uvs = [[0.5, 0.0]]

        def ellipsoid_normal(x: float, y: float, z: float):
            # Ellipsoid normals come from the implicit surface gradient, not
            # from the unscaled sphere direction.
            normal = np.array(
                [
                    x / max(rx * rx, 1.0e-12),
                    y / max(ry * ry, 1.0e-12),
                    z / max(rz * rz, 1.0e-12),
                ],
                dtype=np.float32,
            )
            return (normal / max(float(np.linalg.norm(normal)), 1.0e-8)).tolist()

        for i in range(1, num_latitudes):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            for j in range(num_longitudes):
                phi = j * 2.0 * np.pi / num_longitudes
                ux = np.cos(phi) * sin_theta
                uy = cos_theta
                uz = np.sin(phi) * sin_theta
                x = ux * rx
                y = uy * ry
                z = uz * rz
                vertices.append([x, y, z])
                normals.append(ellipsoid_normal(x, y, z))
                uvs.append([float(j) / num_longitudes, float(i) / num_latitudes])

        bottom = len(vertices)
        vertices.append([0.0, -ry, 0.0])
        normals.append([0.0, -1.0, 0.0])
        uvs.append([0.5, 1.0])

        indices = []
        first_ring = 1
        for j in range(num_longitudes):
            j_next = (j + 1) % num_longitudes
            indices.extend([0, first_ring + j_next, first_ring + j])

        for i in range(num_latitudes - 2):
            upper = 1 + i * num_longitudes
            lower = upper + num_longitudes
            for j in range(num_longitudes):
                j_next = (j + 1) % num_longitudes
                indices.extend([upper + j, lower + j, upper + j_next])
                indices.extend([upper + j_next, lower + j, lower + j_next])

        last_ring = 1 + (num_latitudes - 2) * num_longitudes
        for j in range(num_longitudes):
            j_next = (j + 1) % num_longitudes
            indices.extend([bottom, last_ring + j, last_ring + j_next])

        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        triangles = vertices[indices]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        gradient_centers = np.column_stack(
            (
                centers[:, 0] / max(rx * rx, 1.0e-12),
                centers[:, 1] / max(ry * ry, 1.0e-12),
                centers[:, 2] / max(rz * rz, 1.0e-12),
            )
        )
        # For an ellipsoid, compare face normals against the implicit gradient
        # at the face center to detect inward winding.
        inward = np.einsum("ij,ij->i", face_normals, gradient_centers) < 0.0
        flipped = indices[inward].copy()
        flipped[:, [1, 2]] = flipped[:, [2, 1]]
        indices[inward] = flipped

        return newton.Mesh(
            vertices=vertices,
            indices=indices.reshape(-1),
            normals=np.asarray(normals, dtype=np.float32),
            uvs=np.asarray(uvs, dtype=np.float32),
            compute_inertia=False,
        )

    @staticmethod
    def _shape_mesh(model, shape_idx: int):
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
            return ViewerNewton._create_sphere_mesh(geo_scale[0]), False
        if geo_type == newton.GeoType.CAPSULE:
            return ViewerNewton._create_capsule_mesh(geo_scale[0], geo_scale[1]), False
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
            return ViewerNewton._create_ellipsoid_mesh(rx, ry, rz), False

        return None, False

    def _shape_color(self, shape_idx: int):
        if self.shape_colors is None:
            return (0.0, 0.0, 0.0)

        color = self.shape_colors[shape_idx]
        # Newton GL treats shape colors as sRGB and converts to linear before
        # lighting; Polyscope expects linear-like colors here.
        color = np.power(np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0), 2.2)
        return (float(color[0]), float(color[1]), float(color[2]))

    def _sync_shape_colors_from_model(self):
        if self.model is None or self.model.shape_color is None or len(self._shape_entities) == 0:
            return

        self.shape_colors = self.model.shape_color.numpy()
        for shape_idx, entity in self._shape_entities.items():
            color = self._shape_color(shape_idx)
            if self._shape_entity_colors.get(shape_idx) != color:
                entity.set_color(color)
                self._shape_entity_colors[shape_idx] = color

    def _sync_shape_materials_from_model(self):
        if self.model is None or len(self._shape_entities) == 0:
            return

        for shape_idx, entity in self._shape_entities.items():
            material = self._shape_material(shape_idx)
            if self._shape_entity_materials.get(shape_idx) != material:
                entity.set_material(material)
                self._shape_entity_materials[shape_idx] = material

    def _add_shape_texture_quantity(self, entity, shape_idx: int):
        source = self.model.shape_source[shape_idx]
        if not isinstance(source, Mesh) or source.texture is None or source.uvs is None:
            return

        texture = normalize_texture(load_texture(source.texture), require_channels=True)
        if texture is None:
            return

        uvs = np.asarray(source.uvs, dtype=np.float32)
        vertices = np.asarray(source.vertices)
        if len(uvs) != len(vertices):
            return

        texture = np.asarray(texture[:, :, :3], dtype=np.float32) / 255.0
        # Match the sRGB-to-linear conversion used for flat shape colors.
        texture = np.power(np.clip(texture, 0.0, 1.0), 2.2)
        param_name = "UV"
        entity.add_parameterization_quantity(param_name, uvs, defined_on="vertices", enabled=False)
        entity.add_color_quantity(
            "Texture",
            texture,
            defined_on="texture",
            param_name=param_name,
            enabled=True,
        )

    @override
    def set_model(self, model):
        super().set_model(model)
        self.shape_colors = None
        self.shape_body = None
        self.body_entities.clear()
        self._shape_entities.clear()
        self._shape_entity_colors.clear()
        self._shape_entity_materials.clear()
        # Add meshes
        if model is not None:
            # Cache
            self.shape_flags = model.shape_flags.numpy()
            self.shape_colors = model.shape_color.numpy() if model.shape_color is not None else None
            self.shape_body = model.shape_body.numpy()
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
                shape_color = self._shape_color(shape_idx)
                shape_material = self._shape_material(shape_idx)

                wp.launch(
                    transform_shape_vertices_kernel,
                    dim=len(shape_vertices),
                    inputs=[
                        shape_idx,
                        self.scale,
                        int(scale_vertices),
                        shape_vertices,
                        model.shape_scale,
                        model.shape_transform,
                    ],
                    outputs=[shape_vertices],
                )

                vertices = shape_vertices.numpy()
                if self.shape_body[shape_idx] == -1:
                    vertices = self._array_to_y_up(vertices)

                shape_name = self._shape_name(model, shape_idx)
                shape_entity = ps.register_surface_mesh(
                    name=shape_name,
                    vertices=vertices,
                    faces=shape_mesh.indices.reshape(-1, 3),
                    back_face_policy="cull",
                    edge_color=(1, 1, 1),
                    smooth_shade=True,
                    edge_width=0.0,
                    color=shape_color,
                    material=shape_material,
                )
                self._add_shape_texture_quantity(shape_entity, shape_idx)
                self.body_entities[shape_name] = shape_entity
                self._shape_entities[shape_idx] = shape_entity
                self._shape_entity_colors[shape_idx] = shape_color
                self._shape_entity_materials[shape_idx] = shape_material

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
            self._sync_shape_colors_from_model()
            self._sync_shape_materials_from_model()

            wp.launch(
                transform_to_mat4x4_kernel,
                dim=self.model.body_count,
                inputs=[self.scale, state.body_q],
                outputs=[self._body_transform_mat4x4],
            )

            body_q = self._body_transform_mat4x4.numpy()

            for shape_idx, entity in self._shape_entities.items():
                body_idx = int(self.shape_body[shape_idx])
                if body_idx >= 0:
                    entity.set_transform(self._transform_to_y_up(body_q[body_idx]))

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

    @override
    def apply_forces(self, state: newton.State):
        pass

    @override
    def log_array(self, name: str, array):
        pass

    @override
    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False):
        pass

    @override
    def log_lines(self, name, starts, ends, colors, width=0.01, hidden=False):
        pass

    @override
    def log_mesh(
        self,
        name,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden=False,
        backface_culling=True,
    ):
        pass

    @override
    def log_points(self, name, points, radii=None, colors=None, hidden=False):
        pass

    @override
    def log_scalar(self, name, value, *, clear=False, smoothing=1):
        pass

    @override
    def close(self):
        pass


########################################################################################################################
####################################################    __main__    ####################################################
########################################################################################################################

if __name__ == "__main__":
    viewer = ViewerNewton()
    viewer.run()
