# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D cloth construction helpers for examples."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from style3d.examples.tools.mesh_asset_utils import MeshData


@dataclass(frozen=True)
class NewtonClothParams:
    density: float = 0.3
    particle_radius: float = 5.0e-3
    tri_ka: float = 100.0
    tri_kd: float = 1.5e-6
    tri_aniso_ke: tuple[float, float, float] = (100.0, 100.0, 10.0)
    edge_aniso_ke: tuple[float, float, float] = (2.0e-5, 1.0e-5, 5.0e-6)
    edge_kd: float = 1.0e-3
    soft_contact_margin: float = 3.5e-3
    soft_contact_ke: float = 10.0
    soft_contact_kd: float = 1.0e-6
    soft_contact_mu: float = 0.2
    solver_iterations: int = 4
    linear_iterations: int = 10
    step_dt: float = 1.0 / 120.0
    style3d_self_collision: bool = True
    style3d_collision_radius: float = 3.0e-3
    style3d_collision_stiff_vf: float = 0.5
    style3d_collision_stiff_ee: float = 0.1
    style3d_collision_stiff_ef: float = 1.0


def parse_vec3_arg(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers, e.g. 100,100,10")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def cloth_params_from_args(args: argparse.Namespace) -> NewtonClothParams:
    return NewtonClothParams(
        density=float(args.cloth_density),
        particle_radius=float(args.particle_radius),
        tri_ka=float(args.tri_ka),
        tri_kd=float(args.tri_kd),
        tri_aniso_ke=tuple(map(float, args.tri_aniso_ke)),
        edge_aniso_ke=tuple(map(float, args.edge_aniso_ke)),
        edge_kd=float(args.edge_kd),
        soft_contact_margin=float(args.soft_contact_margin),
        soft_contact_ke=float(args.soft_contact_ke),
        soft_contact_kd=float(args.soft_contact_kd),
        soft_contact_mu=float(args.soft_contact_mu),
        solver_iterations=max(1, int(args.solver_iterations)),
        linear_iterations=max(1, int(args.linear_iterations)),
        step_dt=float(args.step_dt),
        style3d_self_collision=not bool(args.no_style3d_self_collision),
        style3d_collision_radius=float(args.style3d_collision_radius),
        style3d_collision_stiff_vf=float(args.style3d_collision_stiff_vf),
        style3d_collision_stiff_ee=float(args.style3d_collision_stiff_ee),
        style3d_collision_stiff_ef=float(args.style3d_collision_stiff_ef),
    )


def _import_newton(warp_cache_dir: str | None = None, device: str | None = None):
    import warp as wp  # noqa: PLC0415

    if warp_cache_dir:
        Path(warp_cache_dir).mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(Path(warp_cache_dir).resolve())
    if device:
        wp.set_device(device)

    print(
        "[Newton] Importing current checkout. First import can take a minute while Warp registers kernels.",
        flush=True,
    )
    import newton  # noqa: PLC0415

    return newton, wp


def collide_model(model, state, contacts=None, *, soft_contact_margin: float | None = None):
    if soft_contact_margin is None:
        return model.collide(state, contacts)

    if getattr(model, "_collision_pipeline", None) is None:
        model._init_collision_pipeline()
    contacts = contacts if contacts is not None else model._collision_pipeline.contacts()
    model._collision_pipeline.collide(state, contacts, soft_contact_margin=soft_contact_margin)
    return contacts


def configure_style3d_solver_collision(solver, state, cloth_params: NewtonClothParams) -> None:
    collision = getattr(solver, "collision", None)
    if collision is None:
        return
    collision.radius = float(cloth_params.style3d_collision_radius)
    if cloth_params.style3d_self_collision:
        collision.stiff_vf = float(cloth_params.style3d_collision_stiff_vf)
        collision.stiff_ee = float(cloth_params.style3d_collision_stiff_ee)
        collision.stiff_ef = float(cloth_params.style3d_collision_stiff_ef)
    else:
        collision.stiff_vf = 0.0
        collision.stiff_ee = 0.0
        collision.stiff_ef = 0.0
    if hasattr(solver, "rebuild_bvh"):
        solver.rebuild_bvh(state)


def build_style3d_cloth_model(
    mesh: MeshData,
    *,
    scale: float,
    device: str | None,
    warp_cache_dir: str | None,
    style3d_panel_axes: str = "xy",
    start_height: float = 0.05,
    style3d_fix_panel_winding: bool = True,
    style3d_clean_nonmanifold: bool = False,
    style3d_min_triangle_area_ratio: float = 0.05,
    style3d_sew_distance: float = 0.0,
    style3d_sew_ke: float = 100.0,
    style3d_sew_kd: float = 1.0e-3,
    cloth_params: NewtonClothParams = NewtonClothParams(),
):
    newton, wp = _import_newton(warp_cache_dir=warp_cache_dir, device=device)
    build_vertices_np = mesh.vertices
    build_faces_np = mesh.faces
    panel_vertices_np = mesh.panel_vertices
    panel_faces_np = mesh.panel_faces

    print(
        f"[Newton] Building Style3D cloth model: "
        f"verts={len(build_vertices_np)}, tris={len(build_faces_np)}, device={wp.get_device()}",
        flush=True,
    )

    if hasattr(newton, "Style3DModelBuilder"):
        builder = newton.Style3DModelBuilder()
        has_style3d_builder = True
    else:
        builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        has_style3d_builder = False

    if panel_vertices_np is not None and panel_faces_np is not None:
        build_faces_np, panel_faces_np = repair_panel_winding(
            build_faces_np,
            panel_faces_np,
            panel_vertices_np,
            enabled=style3d_fix_panel_winding,
        )
        build_vertices_np, build_faces_np, panel_faces_np = clean_nonmanifold_style3d_faces(
            build_vertices_np,
            build_faces_np,
            panel_faces_np,
            enabled=style3d_clean_nonmanifold,
            min_area_ratio=float(style3d_min_triangle_area_ratio),
        )

    vertices = [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in build_vertices_np]
    flat_indices = build_faces_np.astype(np.int32).reshape(-1)
    common = {
        "vertices": vertices,
        "indices": flat_indices.tolist(),
        "rot": wp.quat_identity(),
        "pos": wp.vec3(0.0, 0.0, float(start_height) - float(np.min(build_vertices_np[:, 2])) * float(scale)),
        "vel": wp.vec3(0.0, 0.0, 0.0),
        "density": float(cloth_params.density),
        "scale": float(scale),
        "tri_ka": float(cloth_params.tri_ka),
        "tri_kd": float(cloth_params.tri_kd),
        "particle_radius": float(cloth_params.particle_radius),
    }

    if panel_vertices_np is not None and panel_faces_np is not None:
        print(
            f"[Newton] Using panel vertices from {mesh.panel_coordinate_source}: "
            f"{len(panel_vertices_np)} verts, {len(panel_faces_np)} tris",
            flush=True,
        )
        panel_verts = [wp.vec2(float(v[0]), float(v[1])) for v in panel_vertices_np]
        panel_indices = panel_faces_np.astype(np.int32).reshape(-1).tolist()
    else:
        print(
            f"[Newton] No OBJ panel coordinates found; projecting vertices with axes={style3d_panel_axes}",
            flush=True,
        )
        panel_verts = project_panel_vertices(wp, build_vertices_np, style3d_panel_axes)
        panel_indices = flat_indices.tolist()

    if has_style3d_builder:
        builder.add_aniso_cloth_mesh(
            **common,
            panel_verts=panel_verts,
            panel_indices=panel_indices,
            tri_aniso_ke=wp.vec3(*map(float, cloth_params.tri_aniso_ke)),
            edge_aniso_ke=wp.vec3(*map(float, cloth_params.edge_aniso_ke)),
            edge_kd=float(cloth_params.edge_kd),
        )
    else:
        import numpy as _np  # noqa: PLC0415

        if not hasattr(_np, "atan2"):
            _np.atan2 = _np.arctan2
        if not hasattr(_np, "pow"):
            _np.pow = _np.power
        from newton.solvers import style3d  # noqa: PLC0415

        style3d.add_cloth_mesh(
            builder,
            **common,
            panel_verts=panel_verts,
            panel_indices=panel_indices,
            tri_aniso_ke=wp.vec3(*map(float, cloth_params.tri_aniso_ke)),
            edge_aniso_ke=wp.vec3(*map(float, cloth_params.edge_aniso_ke)),
            edge_kd=float(cloth_params.edge_kd),
        )

    sew_count = add_close_vertex_springs(
        builder,
        build_vertices_np,
        faces=build_faces_np,
        distance=float(style3d_sew_distance),
        ke=float(style3d_sew_ke),
        kd=float(style3d_sew_kd),
    )
    if sew_count:
        print(
            f"[Newton] Added {sew_count} seam springs for close Style3D vertices "
            f"(distance<={style3d_sew_distance:g}, ke={style3d_sew_ke:g}, kd={style3d_sew_kd:g})",
            flush=True,
        )

    builder.add_ground_plane()
    try:
        model = builder.finalize(requires_grad=False)
    except TypeError:
        model = builder.finalize()
    _apply_contact_params(model, cloth_params)
    state = model.state()
    return newton, wp, model, state, builder


def _apply_contact_params(model, cloth_params: NewtonClothParams) -> None:
    for name, value in (
        ("soft_contact_ke", cloth_params.soft_contact_ke),
        ("soft_contact_kd", cloth_params.soft_contact_kd),
        ("soft_contact_mu", cloth_params.soft_contact_mu),
    ):
        if hasattr(model, name):
            setattr(model, name, float(value))


def repair_panel_winding(
    faces: np.ndarray,
    panel_faces: np.ndarray,
    panel_vertices: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled or len(faces) == 0:
        return faces, panel_faces
    tri = panel_vertices[panel_faces]
    # NumPy 2.x dropped np.cross support for 2D vectors.
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    area2 = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    neg = area2 < -1.0e-14
    zero = np.abs(area2) <= 1.0e-14
    if not np.any(neg):
        if np.any(zero):
            print(f"[Newton] Warning: {int(np.count_nonzero(zero))} zero-area panel triangles remain.", flush=True)
        return faces, panel_faces

    repaired_faces = faces.copy()
    repaired_panel_faces = panel_faces.copy()
    repaired_faces[neg] = repaired_faces[neg][:, [0, 2, 1]]
    repaired_panel_faces[neg] = repaired_panel_faces[neg][:, [0, 2, 1]]
    tri_repaired = panel_vertices[repaired_panel_faces]
    repaired_edge_1 = tri_repaired[:, 1] - tri_repaired[:, 0]
    repaired_edge_2 = tri_repaired[:, 2] - tri_repaired[:, 0]
    area2_repaired = repaired_edge_1[:, 0] * repaired_edge_2[:, 1] - repaired_edge_1[:, 1] * repaired_edge_2[:, 0]
    valid = area2_repaired > 1.0e-14
    print(
        f"[Newton] Repaired Style3D panel winding: flipped={int(np.count_nonzero(neg))}, "
        f"valid_after={int(np.count_nonzero(valid))}/{len(faces)}, "
        f"zero_after={int(np.count_nonzero(np.abs(area2_repaired) <= 1.0e-14))}",
        flush=True,
    )
    return repaired_faces, repaired_panel_faces


def clean_nonmanifold_style3d_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    panel_faces: np.ndarray,
    *,
    enabled: bool,
    min_area_ratio: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled or len(faces) == 0:
        return vertices, faces, panel_faces

    keep = np.ones(len(faces), dtype=bool)
    seen_faces: set[tuple[int, int, int]] = set()
    duplicate_count = 0
    for idx, face in enumerate(faces):
        key = tuple(sorted(map(int, face)))
        if key in seen_faces:
            keep[idx] = False
            duplicate_count += 1
        else:
            seen_faces.add(key)

    tri = vertices[faces]
    face_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    positive_area = face_area[face_area > 0.0]
    area_threshold = 0.0
    if len(positive_area) and min_area_ratio > 0.0:
        area_threshold = float(np.median(positive_area) * min_area_ratio)
        keep &= face_area >= area_threshold
    dropped_small_area = int(np.count_nonzero(face_area < area_threshold)) if area_threshold > 0.0 else 0

    cleaned_vertices = vertices
    cleaned_faces = faces[keep]
    cleaned_panel_faces = panel_faces[keep]
    dropped_vertices = 0
    if len(cleaned_faces):
        used_vertices = np.unique(cleaned_faces.reshape(-1))
        if len(used_vertices) != len(vertices):
            vertex_remap = np.full(len(vertices), -1, dtype=np.int64)
            vertex_remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)
            cleaned_vertices = vertices[used_vertices]
            cleaned_faces = vertex_remap[cleaned_faces]
            dropped_vertices = len(vertices) - len(cleaned_vertices)
    print(
        f"[Newton] Cleaned Style3D topology: dropped_duplicate_tris={duplicate_count}, "
        f"dropped_small_area_tris={dropped_small_area}, min_area={area_threshold:g}, "
        f"tris={len(faces)}->{len(cleaned_faces)}, dropped_orphan_vertices={dropped_vertices}",
        flush=True,
    )
    return cleaned_vertices, cleaned_faces, cleaned_panel_faces


def add_close_vertex_springs(
    builder,
    vertices: np.ndarray,
    *,
    faces: np.ndarray,
    distance: float,
    ke: float,
    kd: float,
) -> int:
    if distance <= 0.0 or len(vertices) == 0:
        return 0

    structural_edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if i > j:
                i, j = j, i
            structural_edges.add((i, j))

    quantized = np.round(vertices / distance).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(map(tuple, quantized.tolist())):
        buckets[key].append(idx)

    added = 0
    dist2 = distance * distance
    for group in buckets.values():
        if len(group) < 2:
            continue
        for offset, i in enumerate(group[:-1]):
            for j in group[offset + 1 :]:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in structural_edges:
                    continue
                delta = vertices[i] - vertices[j]
                if float(np.dot(delta, delta)) <= dist2:
                    builder.add_spring(int(i), int(j), ke, kd, control=0.0)
                    added += 1
    return added


def project_panel_vertices(wp, vertices: np.ndarray, axes: str):
    axes = axes.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if len(axes) != 2 or axes[0] not in axis_map or axes[1] not in axis_map or axes[0] == axes[1]:
        raise ValueError(f"Unsupported style3d panel axes: {axes!r}; expected one of xy, xz, yz")
    a0, a1 = axis_map[axes[0]], axis_map[axes[1]]
    return [wp.vec2(float(v[a0]), float(v[a1])) for v in vertices]


__all__ = [
    "NewtonClothParams",
    "add_close_vertex_springs",
    "build_style3d_cloth_model",
    "clean_nonmanifold_style3d_faces",
    "cloth_params_from_args",
    "collide_model",
    "configure_style3d_solver_collision",
    "parse_vec3_arg",
    "project_panel_vertices",
    "repair_panel_winding",
]
