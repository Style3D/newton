#!/usr/bin/env python3
"""
Probe whether local cloth/bag/garment assets are ready for Newton import.

This is intentionally lightweight:
  1. Parse OBJ/USD mesh geometry.
  2. Report topology risks that matter for Newton cloth solvers.
  3. Optionally write a minimal USDA mesh for OBJ assets when pxr is available.
  4. Optionally try to build a Newton cloth model and run a few solver steps.

Examples:
    python scripts/newton_asset_probe.py --preset bags_clothes --write-usd
    python scripts/newton_asset_probe.py newton/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt.obj --try-newton --steps 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ASSET_ROOT = PROJECT_ROOT / "newton" / "examples" / "assets" / "style3d_probe"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "newton_asset_probe"


DEFAULT_PRESET_ASSETS = [
    "newton/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt.obj",
    "newton/examples/assets/style3d_probe/cloth/green_tshirt/green_tshirt_40k.obj",
    "newton/examples/assets/style3d_probe/cloth/blue_tshirt_14k/blue_tshirt_14k.obj",
    "newton/examples/assets/style3d_probe/bag/nonwoven_small_6/nonwoven_small_6.obj",
    "newton/examples/assets/style3d_probe/bag/nonwoven_small_6/nonwoven_small_6.usda",
    "newton/examples/assets/style3d_probe/bag/nonwoven_711_small_2/nonwoven_711_small_2.obj",
]


@dataclass
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    source_format: str
    mesh_prim_path: str | None = None
    face_counts: list[int] = field(default_factory=list)
    panel_vertices: np.ndarray | None = None
    panel_faces: np.ndarray | None = None
    style3d_vertices: np.ndarray | None = None
    style3d_faces: np.ndarray | None = None
    style3d_panel_vertices: np.ndarray | None = None
    panel_coordinate_source: str | None = None


@dataclass
class MeshReport:
    path: str
    status: str
    source_format: str | None = None
    vertex_count: int = 0
    face_count: int = 0
    triangle_count: int = 0
    non_triangle_face_count: int = 0
    component_count: int = 0
    component_sizes: list[int] = field(default_factory=list)
    boundary_edge_count: int = 0
    nonmanifold_edge_count: int = 0
    duplicate_vertex_pairs_1e_6: int = 0
    degenerate_face_count: int = 0
    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    bbox_size: list[float] = field(default_factory=list)
    recommended_scale: float | None = None
    newton_readiness: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    generated_usd: str | None = None
    usd_prim_path: str | None = None
    panel_vertex_count: int = 0
    panel_triangle_count: int = 0
    panel_coordinate_source: str | None = None
    style3d_vertex_count: int = 0
    style3d_triangle_count: int = 0
    newton_build: dict[str, object] | None = None


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


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def parse_obj(path: Path) -> MeshData:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_texcoords: list[list[int | None]] = []
    face_counts: list[int] = []

    def parse_index(text: str, count: int) -> int:
        idx = int(text)
        if idx < 0:
            return count + idx
        return idx - 1

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                raw = line.split()[1:]
                indices: list[int] = []
                tex_indices: list[int | None] = []
                for token in raw:
                    if not token:
                        continue
                    token_parts = token.split("/")
                    idx_text = token_parts[0]
                    if not idx_text:
                        continue
                    indices.append(parse_index(idx_text, len(vertices)))
                    if len(token_parts) >= 2 and token_parts[1]:
                        tex_indices.append(parse_index(token_parts[1], len(texcoords)))
                    else:
                        tex_indices.append(None)
                if len(indices) >= 3:
                    face_counts.append(len(indices))
                    for i in range(1, len(indices) - 1):
                        faces.append([indices[0], indices[i], indices[i + 1]])
                        face_texcoords.append([tex_indices[0], tex_indices[i], tex_indices[i + 1]])

    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    panel_vertices: np.ndarray | None = None
    panel_faces: np.ndarray | None = None
    style3d_vertices: np.ndarray | None = None
    style3d_faces: np.ndarray | None = None
    style3d_panel_vertices: np.ndarray | None = None
    panel_source: str | None = None

    has_complete_uv_faces = (
        len(texcoords) > 0
        and len(face_texcoords) == len(faces)
        and all(all(t is not None and 0 <= int(t) < len(texcoords) for t in tri) for tri in face_texcoords)
    )
    if has_complete_uv_faces:
        texcoords_np = np.asarray(texcoords, dtype=np.float64)
        panel_vertices = texcoords_np
        panel_faces = np.asarray(face_texcoords, dtype=np.int64).reshape((-1, 3))
        panel_source = "obj_vt_indexed"
        uv_by_vertex: dict[int, int] = {}
        consistent = True
        for tri, tri_uv in zip(faces_np, face_texcoords, strict=True):
            for vertex_idx, uv_idx in zip(tri, tri_uv, strict=True):
                vi = int(vertex_idx)
                ui = int(uv_idx)
                previous = uv_by_vertex.get(vi)
                if previous is None:
                    uv_by_vertex[vi] = ui
                elif previous != ui:
                    consistent = False
                    break
            if not consistent:
                break

        if consistent and len(uv_by_vertex) == len(vertices_np):
            style3d_panel_vertices = texcoords_np[[uv_by_vertex[i] for i in range(len(vertices_np))]]
            panel_source = "obj_vt"
        else:
            pair_to_index: dict[tuple[int, int], int] = {}
            expanded_vertices: list[np.ndarray] = []
            expanded_panel_vertices: list[np.ndarray] = []
            expanded_faces: list[list[int]] = []
            for tri, tri_uv in zip(faces_np, face_texcoords, strict=True):
                expanded_tri: list[int] = []
                for vertex_idx, uv_idx in zip(tri, tri_uv, strict=True):
                    key = (int(vertex_idx), int(uv_idx))
                    mapped = pair_to_index.get(key)
                    if mapped is None:
                        mapped = len(expanded_vertices)
                        pair_to_index[key] = mapped
                        expanded_vertices.append(vertices_np[key[0]])
                        expanded_panel_vertices.append(texcoords_np[key[1]])
                    expanded_tri.append(mapped)
                expanded_faces.append(expanded_tri)
            style3d_vertices = np.asarray(expanded_vertices, dtype=np.float64)
            style3d_faces = np.asarray(expanded_faces, dtype=np.int64).reshape((-1, 3))
            style3d_panel_vertices = np.asarray(expanded_panel_vertices, dtype=np.float64)
            panel_source = "obj_vt_expanded"
    return MeshData(
        vertices=vertices_np,
        faces=faces_np,
        source_format="obj",
        face_counts=face_counts,
        panel_vertices=panel_vertices,
        panel_faces=panel_faces,
        style3d_vertices=style3d_vertices,
        style3d_faces=style3d_faces,
        style3d_panel_vertices=style3d_panel_vertices,
        panel_coordinate_source=panel_source,
    )


def _find_mesh_prims(stage, Usd, UsdGeom, prim_path: str | None):
    if prim_path:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"USD prim path is invalid: {prim_path}")
        if prim.IsA(UsdGeom.Mesh):
            return [prim]
        return [p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)]
    return [p for p in Usd.PrimRange(stage.GetPseudoRoot()) if p.IsA(UsdGeom.Mesh)]


def parse_usd(path: Path, prim_path: str | None = None) -> MeshData:
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(f"pxr is required to read USD files: {exc}") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(f"Failed to open USD: {path}")
    mesh_prims = _find_mesh_prims(stage, Usd, UsdGeom, prim_path)
    if len(mesh_prims) != 1:
        paths = [str(p.GetPath()) for p in mesh_prims[:8]]
        raise ValueError(f"Expected exactly one Mesh prim, found {len(mesh_prims)}: {paths}")

    prim = mesh_prims[0]
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if points is None or indices is None or counts is None:
        raise ValueError(f"Mesh prim is missing points/indices/counts: {prim.GetPath()}")

    vertices = np.asarray(points, dtype=np.float64)
    flat_indices = np.asarray(indices, dtype=np.int64)
    counts_np = np.asarray(counts, dtype=np.int64)
    faces: list[list[int]] = []
    cursor = 0
    for count in counts_np:
        poly = flat_indices[cursor: cursor + int(count)].tolist()
        cursor += int(count)
        if len(poly) >= 3:
            for i in range(1, len(poly) - 1):
                faces.append([poly[0], poly[i], poly[i + 1]])
    return MeshData(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        source_format=path.suffix.lower().lstrip("."),
        mesh_prim_path=str(prim.GetPath()),
        face_counts=counts_np.astype(int).tolist(),
    )


def load_mesh(path: Path, prim_path: str | None = None) -> MeshData:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return parse_obj(path)
    if suffix in {".usd", ".usda", ".usdc"}:
        return parse_usd(path, prim_path=prim_path)
    raise ValueError(f"Unsupported mesh format: {suffix}")


def connected_components(vertex_count: int, faces: np.ndarray) -> list[int]:
    adj: list[list[int]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces:
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].extend([bi, ci])
        adj[bi].extend([ai, ci])
        adj[ci].extend([ai, bi])

    seen = np.zeros(vertex_count, dtype=bool)
    sizes: list[int] = []
    for start in range(vertex_count):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def edge_stats(faces: np.ndarray) -> tuple[int, int]:
    counts: Counter[tuple[int, int]] = Counter()
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            counts[(u, v)] += 1
    boundary = sum(1 for value in counts.values() if value == 1)
    nonmanifold = sum(1 for value in counts.values() if value > 2)
    return boundary, nonmanifold


def degenerate_faces(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-14) -> int:
    if len(faces) == 0:
        return 0
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    return int(np.count_nonzero((area2 <= eps) | repeated))


def duplicate_pairs(vertices: np.ndarray, tol: float = 1e-6) -> int:
    if len(vertices) == 0:
        return 0
    quant = np.round(vertices / tol).astype(np.int64)
    counts = Counter(map(tuple, quant.tolist()))
    return int(sum(n * (n - 1) // 2 for n in counts.values() if n > 1))


def estimate_recommended_scale(bbox_size: np.ndarray) -> float | None:
    max_dim = float(np.max(bbox_size)) if bbox_size.size else 0.0
    if max_dim <= 0.0:
        return None
    # Style3D-exported OBJ assets in this repo are often millimeters; Newton examples use CLOTH_SCALE=0.001.
    if max_dim > 10.0:
        return 0.001
    return 1.0


def analyze_mesh(path: Path, mesh: MeshData) -> MeshReport:
    report = MeshReport(path=str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path), status="ok")
    report.source_format = mesh.source_format
    report.vertex_count = int(mesh.vertices.shape[0])
    report.face_count = len(mesh.face_counts)
    report.triangle_count = int(mesh.faces.shape[0])
    report.non_triangle_face_count = int(sum(1 for c in mesh.face_counts if c != 3))
    report.usd_prim_path = mesh.mesh_prim_path
    report.panel_vertex_count = int(mesh.panel_vertices.shape[0]) if mesh.panel_vertices is not None else 0
    report.panel_triangle_count = int(mesh.panel_faces.shape[0]) if mesh.panel_faces is not None else 0
    report.panel_coordinate_source = mesh.panel_coordinate_source
    report.style3d_vertex_count = int(mesh.style3d_vertices.shape[0]) if mesh.style3d_vertices is not None else 0
    report.style3d_triangle_count = int(mesh.style3d_faces.shape[0]) if mesh.style3d_faces is not None else 0

    if report.vertex_count == 0 or report.triangle_count == 0:
        report.status = "failed"
        report.newton_readiness = "blocked"
        report.warnings.append("No usable vertices or triangles were found.")
        return report

    invalid_index = bool(np.any(mesh.faces < 0) or np.any(mesh.faces >= report.vertex_count))
    if invalid_index:
        report.status = "failed"
        report.newton_readiness = "blocked"
        report.warnings.append("Face indices reference vertices outside the vertex array.")
        return report

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    report.bbox_min = [float(x) for x in bbox_min]
    report.bbox_max = [float(x) for x in bbox_max]
    report.bbox_size = [float(x) for x in bbox_size]
    report.recommended_scale = estimate_recommended_scale(bbox_size)
    report.component_sizes = connected_components(report.vertex_count, mesh.faces)
    report.component_count = len(report.component_sizes)
    report.boundary_edge_count, report.nonmanifold_edge_count = edge_stats(mesh.faces)
    report.degenerate_face_count = degenerate_faces(mesh.vertices, mesh.faces)
    report.duplicate_vertex_pairs_1e_6 = duplicate_pairs(mesh.vertices, tol=1e-6)

    if report.non_triangle_face_count:
        report.warnings.append("Newton cloth import expects triangle meshes; this asset needs triangulation.")
    if report.component_count > 1:
        report.warnings.append("Mesh has multiple connected components; bags may need seam/stitch constraints between components.")
    if report.boundary_edge_count:
        report.warnings.append("Mesh has boundary edges; this is normal for panels, but not a closed two-manifold cloth.")
    if report.nonmanifold_edge_count:
        report.warnings.append("Mesh has non-manifold edges; Newton add_cloth_mesh notes that the mesh should be two-manifold.")
    if report.degenerate_face_count:
        report.warnings.append("Mesh has degenerate triangles; clean or weld before Newton import.")
    if report.duplicate_vertex_pairs_1e_6:
        report.warnings.append("Mesh has near-duplicate vertices; this often indicates Style3D seam points that need weld or stitch handling.")
    if mesh.panel_coordinate_source == "obj_vt_expanded":
        report.warnings.append("OBJ uses face-varying UVs; panel coordinates are imported with separate panel indices.")
    if report.recommended_scale == 0.001:
        report.warnings.append("Asset dimensions look like millimeters; the probe will import it with scale=0.001.")

    if report.status == "ok":
        severe = report.nonmanifold_edge_count > 0 or report.degenerate_face_count > 0
        if severe:
            report.newton_readiness = "needs_cleanup"
        elif report.component_count > 1 or report.duplicate_vertex_pairs_1e_6 > 0:
            report.newton_readiness = "needs_stitch_or_weld"
        elif report.non_triangle_face_count > 0:
            report.newton_readiness = "needs_triangulation"
        else:
            report.newton_readiness = "direct_candidate"
    return report


def make_valid_prim_name(name: str) -> str:
    clean = []
    for ch in name:
        clean.append(ch if ch.isalnum() or ch == "_" else "_")
    text = "".join(clean).strip("_")
    if not text or text[0].isdigit():
        text = f"cloth_{text}"
    return text


def write_usda(mesh: MeshData, source_path: Path, output_dir: Path) -> tuple[str, str]:
    try:
        from pxr import Sdf, Usd, UsdGeom, Vt
    except Exception as exc:
        raise RuntimeError(f"pxr is required to write USDA files: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = make_valid_prim_name(source_path.stem)
    output_path = output_dir / f"{stem}.usda"
    prim_path = f"/World/{stem}/{stem}_mesh"

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, f"/World/{stem}")
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = Vt.Vec3fArray([tuple(map(float, v)) for v in mesh.vertices])
    flat_indices = Vt.IntArray([int(x) for x in mesh.faces.reshape(-1).tolist()])
    counts = Vt.IntArray([3] * int(mesh.faces.shape[0]))
    usd_mesh.GetPointsAttr().Set(points)
    usd_mesh.GetFaceVertexIndicesAttr().Set(flat_indices)
    usd_mesh.GetFaceVertexCountsAttr().Set(counts)
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateOrientationAttr().Set(UsdGeom.Tokens.rightHanded)
    stage.GetRootLayer().Save()
    return str(output_path), prim_path


def import_newton(warp_cache_dir: str | None = None, device: str | None = None):
    import warp as wp  # noqa: PLC0415
    if warp_cache_dir:
        Path(warp_cache_dir).mkdir(parents=True, exist_ok=True)
        wp.config.kernel_cache_dir = str(Path(warp_cache_dir).resolve())
    if device:
        wp.set_device(device)

    print(
        "[Newton] Importing current checkout. First import can take a minute while Warp "
        "registers kernels.",
        flush=True,
    )
    import newton  # noqa: PLC0415
    return newton, wp


def try_newton_build(
    mesh: MeshData,
    *,
    backend: str,
    scale: float,
    steps: int,
    device: str | None,
    warp_cache_dir: str | None,
    style3d_panel_axes: str,
    start_height: float,
    style3d_fix_panel_winding: bool,
    style3d_clean_nonmanifold: bool,
    style3d_min_triangle_area_ratio: float,
    style3d_sew_distance: float,
    style3d_sew_ke: float,
    style3d_sew_kd: float,
    cloth_params: NewtonClothParams,
) -> dict[str, object]:
    started = time.perf_counter()
    newton, wp, model, state_0, _builder = build_newton_cloth_model(
        mesh,
        backend=backend,
        scale=scale,
        device=device,
        warp_cache_dir=warp_cache_dir,
        style3d_panel_axes=style3d_panel_axes,
        start_height=start_height,
        style3d_fix_panel_winding=style3d_fix_panel_winding,
        style3d_clean_nonmanifold=style3d_clean_nonmanifold,
        style3d_min_triangle_area_ratio=style3d_min_triangle_area_ratio,
        style3d_sew_distance=style3d_sew_distance,
        style3d_sew_ke=style3d_sew_ke,
        style3d_sew_kd=style3d_sew_kd,
        cloth_params=cloth_params,
    )
    build_payload: dict[str, object] = {
        "status": "built",
        "backend": backend,
        "newton_file": str(getattr(newton, "__file__", "")),
        "device": str(wp.get_device()),
        "particle_count": int(model.particle_count),
        "triangle_count": int(model.tri_count),
        "edge_count": int(model.edge_count),
        "cloth_params": asdict(cloth_params),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }

    if steps <= 0:
        return build_payload

    final_state, _contacts = step_newton_model(
        newton,
        wp,
        model,
        state_0,
        backend=backend,
        steps=steps,
        dt=float(cloth_params.step_dt),
        iterations=int(cloth_params.solver_iterations),
        linear_iterations=int(cloth_params.linear_iterations),
        cloth_params=cloth_params,
    )
    wp.synchronize()
    build_payload["status"] = "stepped"
    build_payload["steps"] = int(steps)
    build_payload["state_stats"] = particle_state_stats(final_state)
    build_payload["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    return build_payload


def collide_model(model, state, contacts=None, *, soft_contact_margin: float | None = None):
    if soft_contact_margin is None:
        return model.collide(state, contacts)

    if getattr(model, "_collision_pipeline", None) is None:
        model._init_collision_pipeline()
    contacts = contacts if contacts is not None else model._collision_pipeline.contacts()
    model._collision_pipeline.collide(state, contacts, soft_contact_margin=soft_contact_margin)
    return contacts


def step_newton_model(
    newton,
    wp,
    model,
    state_0,
    *,
    backend: str,
    steps: int,
    dt: float,
    iterations: int,
    linear_iterations: int,
    cloth_params: NewtonClothParams,
):
    state_1 = model.state()
    control = model.control()
    contacts = collide_model(model, state_0, soft_contact_margin=cloth_params.soft_contact_margin)
    if backend == "style3d":
        solver = newton.solvers.SolverStyle3D(
            model,
            iterations=max(1, int(iterations)),
            linear_iterations=max(1, int(linear_iterations)),
        )
        configure_style3d_solver_collision(solver, state_0, cloth_params)
    else:
        solver = newton.solvers.SolverVBD(model, iterations=max(1, int(iterations)))
    for _ in range(int(steps)):
        state_0.clear_forces()
        contacts = collide_model(model, state_0, contacts, soft_contact_margin=cloth_params.soft_contact_margin)
        solver.step(state_0, state_1, control, contacts, dt=dt)
        state_0, state_1 = state_1, state_0
    return state_0, contacts


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


def particle_state_stats(state) -> dict[str, object]:
    q = state.particle_q.numpy()
    finite = np.isfinite(q)
    all_finite = bool(np.all(finite))
    if len(q) == 0:
        return {"all_finite": all_finite, "particle_count": 0}
    return {
        "all_finite": all_finite,
        "particle_count": int(len(q)),
        "bbox_min": np.nanmin(q, axis=0).astype(float).tolist(),
        "bbox_max": np.nanmax(q, axis=0).astype(float).tolist(),
        "bbox_size": np.ptp(q, axis=0).astype(float).tolist(),
        "max_abs_position": float(np.nanmax(np.abs(q))),
    }


def build_newton_cloth_model(
    mesh: MeshData,
    *,
    backend: str,
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
    newton, wp = import_newton(warp_cache_dir=warp_cache_dir, device=device)
    build_vertices_np = mesh.vertices
    build_faces_np = mesh.faces
    panel_vertices_np = None
    panel_faces_np = None
    if backend == "style3d" and mesh.panel_vertices is not None and mesh.panel_faces is not None:
        panel_vertices_np = mesh.panel_vertices
        panel_faces_np = mesh.panel_faces
    print(
        f"[Newton] Building cloth model: backend={backend}, "
        f"verts={len(build_vertices_np)}, tris={len(build_faces_np)}, device={wp.get_device()}",
        flush=True,
    )
    has_style3d_builder = hasattr(newton, "Style3DModelBuilder")
    if backend == "style3d" and has_style3d_builder:
        builder = newton.Style3DModelBuilder()
    else:
        builder = newton.ModelBuilder()
        if backend == "style3d":
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    if backend == "style3d" and panel_vertices_np is not None and panel_faces_np is not None:
        build_faces_np, panel_faces_np = repair_panel_winding(
            build_faces_np,
            panel_faces_np,
            panel_vertices_np,
            enabled=style3d_fix_panel_winding,
        )
        build_faces_np, panel_faces_np = clean_nonmanifold_style3d_faces(
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
    if backend == "style3d":
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
            panel_verts = project_panel_vertices(wp, mesh.vertices, style3d_panel_axes)
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
    else:
        builder.add_cloth_mesh(
            **common,
            tri_ke=float(cloth_params.tri_aniso_ke[0]),
            edge_ke=float(cloth_params.edge_aniso_ke[0]),
            edge_kd=float(cloth_params.edge_kd),
        )
        builder.color()

    builder.add_ground_plane()
    try:
        model = builder.finalize(requires_grad=False)
    except TypeError:
        model = builder.finalize()
    apply_contact_params(model, cloth_params)
    state = model.state()
    return newton, wp, model, state, builder


def apply_contact_params(model, cloth_params: NewtonClothParams) -> None:
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
    area2 = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
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
    area2_repaired = np.cross(tri_repaired[:, 1] - tri_repaired[:, 0], tri_repaired[:, 2] - tri_repaired[:, 0])
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
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled or len(faces) == 0:
        return faces, panel_faces

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

    cleaned_faces = faces[keep]
    cleaned_panel_faces = panel_faces[keep]
    print(
        f"[Newton] Cleaned Style3D topology: dropped_duplicate_tris={duplicate_count}, "
        f"dropped_small_area_tris={dropped_small_area}, min_area={area_threshold:g}, "
        f"tris={len(faces)}->{len(cleaned_faces)}",
        flush=True,
    )
    return cleaned_faces, cleaned_panel_faces


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
            for j in group[offset + 1:]:
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


def view_newton_asset(
    mesh: MeshData,
    *,
    backend: str,
    scale: float,
    device: str | None,
    warp_cache_dir: str | None,
    simulate: bool,
    fps: float,
    substeps: int,
    use_cuda_graph: bool,
    style3d_panel_axes: str,
    start_height: float,
    style3d_fix_panel_winding: bool,
    style3d_clean_nonmanifold: bool,
    style3d_min_triangle_area_ratio: float,
    style3d_sew_distance: float,
    style3d_sew_ke: float,
    style3d_sew_kd: float,
    cloth_params: NewtonClothParams,
) -> None:
    newton, wp, model, state_0, _builder = build_newton_cloth_model(
        mesh,
        backend=backend,
        scale=scale,
        device=device,
        warp_cache_dir=warp_cache_dir,
        style3d_panel_axes=style3d_panel_axes,
        start_height=start_height,
        style3d_fix_panel_winding=style3d_fix_panel_winding,
        style3d_clean_nonmanifold=style3d_clean_nonmanifold,
        style3d_min_triangle_area_ratio=style3d_min_triangle_area_ratio,
        style3d_sew_distance=style3d_sew_distance,
        style3d_sew_ke=style3d_sew_ke,
        style3d_sew_kd=style3d_sew_kd,
        cloth_params=cloth_params,
    )
    import newton.viewer  # noqa: PLC0415

    viewer = newton.viewer.ViewerGL()
    viewer.show_particles = False
    viewer.show_triangles = True
    viewer.set_model(model)
    from pyglet.math import Vec3 as PyVec3  # noqa: PLC0415

    viewer.set_camera(PyVec3(0.0, -3.0, 1.6), pitch=-25.0, yaw=90.0)

    state_1 = model.state()
    control = model.control()
    contacts = collide_model(model, state_0, soft_contact_margin=cloth_params.soft_contact_margin)
    solver = None
    if simulate:
        solver_cls = newton.solvers.SolverStyle3D if backend == "style3d" else newton.solvers.SolverVBD
        if backend == "style3d":
            solver = solver_cls(
                model,
                iterations=max(1, int(cloth_params.solver_iterations)),
                linear_iterations=max(1, int(cloth_params.linear_iterations)),
            )
            configure_style3d_solver_collision(solver, state_0, cloth_params)
        else:
            solver = solver_cls(model, iterations=max(1, int(cloth_params.solver_iterations)))

    frame_dt = 1.0 / float(fps)
    sim_substeps = max(1, int(substeps))
    sim_dt = frame_dt / sim_substeps
    sim_time = 0.0
    graph = None

    def simulate_frame():
        nonlocal contacts, state_0, state_1
        contacts = collide_model(model, state_0, contacts, soft_contact_margin=cloth_params.soft_contact_margin)
        for _ in range(sim_substeps):
            state_0.clear_forces()
            viewer.apply_forces(state_0)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    if simulate and use_cuda_graph and getattr(wp.get_device(), "is_cuda", False):
        if sim_substeps % 2:
            print("[Viewer] CUDA graph disabled because --view-substeps is odd; use an even value to enable replay.", flush=True)
        else:
            try:
                with wp.ScopedCapture() as capture:
                    simulate_frame()
                graph = capture.graph
                print(f"[Viewer] CUDA graph capture enabled for {sim_substeps} substeps per frame.", flush=True)
            except Exception as exc:
                graph = None
                print(f"[Viewer] CUDA graph capture failed; falling back to Python stepping: {exc!r}", flush=True)

    print("[Viewer] Newton viewer opened. Close the window or press ESC to exit.", flush=True)
    try:
        while viewer.is_running():
            if simulate and not viewer.is_paused():
                if graph is not None:
                    wp.capture_launch(graph)
                else:
                    simulate_frame()
                sim_time += frame_dt

            viewer.begin_frame(sim_time)
            viewer.log_state(state_0)
            viewer.log_contacts(contacts, state_0)
            viewer.end_frame()
    finally:
        viewer.close()


def collect_asset_paths(args) -> list[Path]:
    raw_paths: list[str] = []
    if args.preset in {"bags_clothes", "all"}:
        raw_paths.extend(DEFAULT_PRESET_ASSETS)
    raw_paths.extend(args.assets)

    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        path = resolve_path(raw)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def report_to_row(report: MeshReport) -> dict[str, object]:
    data = asdict(report)
    for key in ("warnings", "component_sizes", "bbox_min", "bbox_max", "bbox_size", "newton_build"):
        data[key] = json.dumps(data[key], ensure_ascii=False)
    return data


def write_outputs(reports: list[MeshReport], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)

    csv_path = output.with_suffix(".csv")
    if reports:
        fieldnames = list(report_to_row(reports[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for report in reports:
                writer.writerow(report_to_row(report))


def print_summary(reports: list[MeshReport]) -> None:
    print("\n=== Newton Asset Probe Summary ===")
    for r in reports:
        print(
            f"{r.newton_readiness:18s} {r.status:7s} "
            f"V={r.vertex_count:7d} T={r.triangle_count:7d} C={r.component_count:3d} "
            f"B={r.boundary_edge_count:7d} NM={r.nonmanifold_edge_count:5d} "
            f"{r.path}"
        )
        if r.generated_usd:
            print(f"  usd: {r.generated_usd} prim={r.usd_prim_path}")
        if r.newton_build:
            print(f"  newton: {json.dumps(r.newton_build, ensure_ascii=False)}")
        for warning in r.warnings[:4]:
            print(f"  warn: {warning}")
        if len(r.warnings) > 4:
            print(f"  warn: ... {len(r.warnings) - 4} more")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", nargs="*", help="OBJ/USD assets to probe")
    parser.add_argument("--preset", choices=["none", "bags_clothes", "all"], default="bags_clothes")
    parser.add_argument("--usd-prim-path", default=None, help="USD mesh prim path for a single USD input")
    parser.add_argument("--write-usd", action="store_true", help="Write minimal USDA files for OBJ assets")
    parser.add_argument("--usd-dir", default=str(DEFAULT_OUTPUT_DIR / "usd"), help="Directory for generated USDA files")
    parser.add_argument("--try-newton", action="store_true", help="Try building the mesh with Newton locally")
    parser.add_argument("--view", action="store_true", help="Open Newton OpenGL viewer for the first asset")
    parser.add_argument("--view-simulate", action="store_true", help="Run a simple cloth simulation in the viewer")
    parser.add_argument("--simulate-only", action="store_true", help="Build Newton model and run --steps solver steps without opening a viewer")
    parser.add_argument("--view-fps", type=float, default=60.0, help="Viewer simulation FPS")
    parser.add_argument("--view-substeps", type=int, default=4, help="Physics substeps per viewer frame")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Disable CUDA graph capture in --view-simulate mode")
    parser.add_argument("--backend", choices=["vbd", "style3d"], default="vbd", help="Newton cloth backend to test")
    parser.add_argument("--style3d-panel-axes", choices=["xy", "xz", "yz"], default="xy", help="2D panel projection axes for --backend style3d")
    parser.add_argument("--start-height", type=float, default=0.05, help="Initial height of the asset's lowest vertex above the ground [m]")
    parser.add_argument("--no-style3d-fix-panel-winding", action="store_true", help="Disable automatic flipping of negative-area Style3D panel triangles")
    parser.add_argument(
        "--style3d-clean-nonmanifold",
        action="store_true",
        help="Drop duplicate triangles and tiny area triangles before Style3D build",
    )
    parser.add_argument(
        "--style3d-min-triangle-area-ratio",
        type=float,
        default=0.05,
        help="Drop triangles with 3D area below this fraction of the median positive triangle area",
    )
    parser.add_argument("--style3d-sew-distance", type=float, default=0.0, help="Add seam springs between non-edge vertices whose 3D positions are within this distance")
    parser.add_argument("--style3d-sew-ke", type=float, default=100.0, help="Stiffness for --style3d-sew-distance springs")
    parser.add_argument("--style3d-sew-kd", type=float, default=1.0e-3, help="Damping for --style3d-sew-distance springs")
    parser.add_argument("--cloth-density", type=float, default=0.3, help="Newton cloth areal density used by Style3D/VBD")
    parser.add_argument("--particle-radius", type=float, default=5.0e-3, help="Newton cloth particle/contact radius")
    parser.add_argument("--tri-ka", type=float, default=100.0, help="Newton triangle area preservation stiffness")
    parser.add_argument("--tri-kd", type=float, default=1.5e-6, help="Newton triangle damping")
    parser.add_argument("--tri-aniso-ke", type=parse_vec3_arg, default=(100.0, 100.0, 10.0), help="Style3D anisotropic stretch/shear stiffness as weft,warp,shear")
    parser.add_argument("--edge-aniso-ke", type=parse_vec3_arg, default=(2.0e-5, 1.0e-5, 5.0e-6), help="Style3D anisotropic bending stiffness as weft,warp,shear")
    parser.add_argument("--edge-kd", type=float, default=1.0e-3, help="Newton edge/bending damping")
    parser.add_argument("--soft-contact-margin", type=float, default=3.5e-3, help="Newton soft contact margin")
    parser.add_argument("--soft-contact-ke", type=float, default=10.0, help="Newton soft contact stiffness")
    parser.add_argument("--soft-contact-kd", type=float, default=1.0e-6, help="Newton soft contact damping")
    parser.add_argument("--soft-contact-mu", type=float, default=0.2, help="Newton soft contact friction coefficient")
    parser.add_argument("--solver-iterations", type=int, default=4, help="Newton nonlinear solver iterations per substep")
    parser.add_argument("--linear-iterations", type=int, default=10, help="Newton Style3D linear solver iterations per substep")
    parser.add_argument("--step-dt", type=float, default=1.0 / 120.0, help="Time step for --simulate-only/--steps")
    parser.add_argument("--no-style3d-self-collision", action="store_true", help="Disable Style3D cloth self-collision while keeping body/ground soft contact")
    parser.add_argument("--style3d-collision-radius", type=float, default=3.0e-3, help="Internal Style3D cloth collision BVH radius")
    parser.add_argument("--style3d-collision-stiff-vf", type=float, default=0.5, help="Style3D vertex-face self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ee", type=float, default=0.1, help="Style3D edge-edge self-collision stiffness")
    parser.add_argument("--style3d-collision-stiff-ef", type=float, default=1.0, help="Style3D edge-face untangling stiffness")
    parser.add_argument("--steps", type=int, default=0, help="Optional number of solver steps after build")
    parser.add_argument("--device", default=None, help="Warp device, e.g. cuda:0 or cpu")
    parser.add_argument("--warp-cache-dir", default="/tmp/warp_cache", help="Writable Warp kernel cache directory")
    parser.add_argument("--max-newton-verts", type=int, default=80000, help="Skip Newton build above this vertex count")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "report.json"), help="JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.view_simulate:
        args.view = True
    if args.simulate_only and args.steps <= 0:
        args.steps = 1
    if args.simulate_only:
        args.try_newton = True
    cloth_params = cloth_params_from_args(args)
    output = resolve_path(args.output)
    usd_dir = resolve_path(args.usd_dir)
    reports: list[MeshReport] = []
    viewed = False

    for path in collect_asset_paths(args):
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            mesh = load_mesh(path, prim_path=args.usd_prim_path)
            report = analyze_mesh(path, mesh)
            if args.write_usd and path.suffix.lower() == ".obj" and report.status == "ok":
                generated_usd, prim_path = write_usda(mesh, path, usd_dir)
                report.generated_usd = os.path.relpath(generated_usd, PROJECT_ROOT)
                report.usd_prim_path = prim_path
            if args.try_newton and report.status == "ok":
                if report.vertex_count > args.max_newton_verts:
                    report.newton_build = {
                        "status": "skipped",
                        "reason": f"vertex_count>{args.max_newton_verts}",
                    }
                else:
                    scale = float(report.recommended_scale or 1.0)
                    try:
                        report.newton_build = try_newton_build(
                            mesh,
                            backend=args.backend,
                            scale=scale,
                            steps=max(0, int(args.steps)),
                            device=args.device,
                            warp_cache_dir=args.warp_cache_dir,
                            style3d_panel_axes=args.style3d_panel_axes,
                            start_height=float(args.start_height),
                            style3d_fix_panel_winding=not args.no_style3d_fix_panel_winding,
                            style3d_clean_nonmanifold=bool(args.style3d_clean_nonmanifold),
                            style3d_min_triangle_area_ratio=float(args.style3d_min_triangle_area_ratio),
                            style3d_sew_distance=float(args.style3d_sew_distance),
                            style3d_sew_ke=float(args.style3d_sew_ke),
                            style3d_sew_kd=float(args.style3d_sew_kd),
                            cloth_params=cloth_params,
                        )
                    except Exception as exc:
                        report.newton_build = {
                            "status": "failed",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        report.warnings.append(f"Newton build failed: {exc!r}")
            if args.view and report.status == "ok" and not viewed:
                viewed = True
                scale = float(report.recommended_scale or 1.0)
                view_newton_asset(
                    mesh,
                    backend=args.backend,
                    scale=scale,
                    device=args.device,
                    warp_cache_dir=args.warp_cache_dir,
                    simulate=bool(args.view_simulate),
                    fps=float(args.view_fps),
                    substeps=int(args.view_substeps),
                    use_cuda_graph=not bool(args.no_cuda_graph),
                    style3d_panel_axes=args.style3d_panel_axes,
                    start_height=float(args.start_height),
                    style3d_fix_panel_winding=not args.no_style3d_fix_panel_winding,
                    style3d_clean_nonmanifold=bool(args.style3d_clean_nonmanifold),
                    style3d_min_triangle_area_ratio=float(args.style3d_min_triangle_area_ratio),
                    style3d_sew_distance=float(args.style3d_sew_distance),
                    style3d_sew_ke=float(args.style3d_sew_ke),
                    style3d_sew_kd=float(args.style3d_sew_kd),
                    cloth_params=cloth_params,
                )
            reports.append(report)
        except Exception as exc:
            reports.append(
                MeshReport(
                    path=str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
                    status="failed",
                    newton_readiness="blocked",
                    warnings=[repr(exc), traceback.format_exc()],
                )
            )

    write_outputs(reports, output)
    print_summary(reports)
    print(f"\nJSON report: {output}")
    print(f"CSV report:  {output.with_suffix('.csv')}")
    return 0 if all(r.status == "ok" for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
