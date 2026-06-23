# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import newton


def _quat_xyzw_to_matrix(q):
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _transform_points(points, xform):
    rot = _quat_xyzw_to_matrix(xform[3:7])
    return points @ rot.T + xform[:3]


def _shape_mesh(model, shape_idx):
    shape_type = model.shape_type.numpy()[shape_idx]
    shape_source = model.shape_source[shape_idx]
    shape_scale = np.asarray(model.shape_scale.numpy()[shape_idx], dtype=np.float32)

    if shape_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
        vertices = np.asarray(shape_source.vertices, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(shape_source.indices, dtype=np.int32).reshape(-1, 3)
        return vertices * shape_scale.reshape(1, 3), faces

    if shape_type == newton.GeoType.BOX:
        mesh = newton.Mesh.create_box(float(shape_scale[0]), float(shape_scale[1]), float(shape_scale[2]))
    elif shape_type == newton.GeoType.SPHERE:
        mesh = newton.Mesh.create_sphere(float(shape_scale[0]), num_latitudes=16, num_longitudes=16)
    elif shape_type == newton.GeoType.CAPSULE:
        mesh = newton.Mesh.create_capsule(float(shape_scale[0]), float(shape_scale[1]), up_axis=newton.Axis.X)
    elif shape_type == newton.GeoType.CYLINDER:
        mesh = newton.Mesh.create_cylinder(float(shape_scale[0]), float(shape_scale[1]), up_axis=newton.Axis.X)
    elif shape_type == newton.GeoType.CONE:
        mesh = newton.Mesh.create_cone(float(shape_scale[0]), float(shape_scale[1]), up_axis=newton.Axis.X)
    else:
        return None, None

    return np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3), np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)


def _write_obj_mesh(file, name, vertices, faces, vertex_offset):
    file.write(f"o {name}\n")
    for v in vertices:
        file.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for face in faces:
        a, b, c = face + vertex_offset + 1
        file.write(f"f {a} {b} {c}\n")
    return vertex_offset + len(vertices)


def _as_numpy(array):
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)


def export_scene_obj(model, state, path, viz_scale, add_cloth=True):
    path.parent.mkdir(parents=True, exist_ok=True)

    body_q = _as_numpy(state.body_q) if model.body_count > 0 else np.empty((0, 7), dtype=np.float32)
    shape_body = model.shape_body.numpy()
    shape_transform = model.shape_transform.numpy()
    shape_flags = model.shape_flags.numpy()

    vertex_offset = 0
    with open(path, "w", encoding="utf-8") as file:
        file.write("# Exported from style3d/export_obj.py\n")
        file.write("# Units: meters\n")

        if add_cloth and model.tri_count > 0:
            cloth_vertices = _as_numpy(state.particle_q).reshape(model.particle_count, 3) * viz_scale
            cloth_faces = model.tri_indices.numpy().reshape(model.tri_count, 3)
            vertex_offset = _write_obj_mesh(file, "cloth", cloth_vertices, cloth_faces, vertex_offset)

        for shape_idx in range(model.shape_count):
            if not (shape_flags[shape_idx] & int(newton.ShapeFlags.VISIBLE)):
                continue
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue

            vertices, faces = _shape_mesh(model, shape_idx)
            if vertices is None or len(vertices) == 0 or len(faces) == 0:
                continue

            vertices = _transform_points(vertices, shape_transform[shape_idx])
            vertices = _transform_points(vertices, body_q[body_idx]) * viz_scale
            label = model.shape_label[shape_idx].replace("/", "_").replace(" ", "_")
            vertex_offset = _write_obj_mesh(file, f"franka_{label}", vertices, faces, vertex_offset)

    print(f"Exported Franka and cloth OBJ: {path}")
