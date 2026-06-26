# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate a plastic bag and drawstring as two VBD cloth meshes.

Command:
    python -m newton.examples cloth_vbd_plastic_bag_drawstring --device cuda:0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "style3d_probe" / "bag" / "plastic_bag"
DEFAULT_BAG_ASSET = ASSET_ROOT / "part1_m" / "part1_m.obj"
DEFAULT_DRAWSTRING_ASSET = ASSET_ROOT / "part2_m" / "part2_m.obj"
DEFAULT_DUSTBIN_ASSET = ASSET_ROOT / "dustbin" / "dustbin.obj"


@dataclass
class ObjMesh:
    vertices: np.ndarray
    indices: np.ndarray


def mesh_component_sizes(mesh: ObjMesh) -> list[int]:
    parent = list(range(len(mesh.vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b, c in mesh.indices:
        union(int(a), int(b))
        union(int(b), int(c))
        union(int(c), int(a))

    counts: dict[int, int] = {}
    for index in range(len(mesh.vertices)):
        root = find(index)
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values(), reverse=True)


def mesh_boundary_vertices(mesh: ObjMesh) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in mesh.indices:
        for i, j in ((a, b), (b, c), (c, a)):
            edge = (min(int(i), int(j)), max(int(i), int(j)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary = sorted({vertex for edge, count in edge_counts.items() if count == 1 for vertex in edge})
    return np.asarray(boundary, dtype=np.int32)


def select_top_boundary_vertices(mesh: ObjMesh, band: float) -> np.ndarray:
    boundary = mesh_boundary_vertices(mesh)
    if len(boundary) == 0:
        return np.empty(0, dtype=np.int32)

    boundary_z = mesh.vertices[boundary, 2]
    min_z = float(np.max(boundary_z) - band)
    return boundary[boundary_z >= min_z]


def parse_rgb(text: str) -> tuple[float, float, float]:
    values = [float(part.strip()) for part in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected an RGB triplet, e.g. 0.3,0.7,1.0")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 1]")
    return (values[0], values[1], values[2])


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer index")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("particle indices must be non-negative")
    return values


@wp.kernel
def record_particle_positions_kernel(
    indices: wp.array[wp.int32],
    q: wp.array[wp.vec3],
    anchors: wp.array[wp.vec3],
):
    tid = wp.tid()
    anchors[tid] = q[indices[tid]]


@wp.kernel
def drive_pull_particles_kernel(
    indices: wp.array[wp.int32],
    anchors: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[wp.float32],
    particle_inv_mass: wp.array[wp.float32],
    time: float,
    start_time: float,
    duration: float,
    height: float,
    outward_distance: float,
    arc_height: float,
    active_flag: int,
):
    tid = wp.tid()
    index = indices[tid]

    alpha = 1.0
    if duration > 0.0:
        alpha = (time - start_time) / duration
        alpha = wp.clamp(alpha, 0.0, 1.0)

    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    smooth_dot = 0.0
    if duration > 0.0 and alpha > 0.0 and alpha < 1.0:
        smooth_dot = 6.0 * alpha * (1.0 - alpha) / duration

    parabola = 4.0 * smooth * (1.0 - smooth)
    parabola_dot = 4.0 * (1.0 - 2.0 * smooth) * smooth_dot
    direction = directions[tid]
    offset = direction * (outward_distance * smooth) + wp.vec3(0.0, 0.0, height * smooth + arc_height * parabola)
    velocity = direction * (outward_distance * smooth_dot) + wp.vec3(
        0.0, 0.0, height * smooth_dot + arc_height * parabola_dot
    )

    q[index] = anchors[tid] + offset
    qd[index] = velocity
    particle_flags[index] = particle_flags[index] & ~active_flag
    particle_mass[index] = 0.0
    particle_inv_mass[index] = 0.0


def weld_vertices(mesh: ObjMesh, max_distance: float) -> tuple[ObjMesh, int]:
    if max_distance <= 0.0 or len(mesh.vertices) == 0:
        return mesh, 0

    parent = list(range(len(mesh.vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    max_distance2 = max_distance * max_distance
    for i in range(len(mesh.vertices)):
        delta = mesh.vertices[i + 1 :] - mesh.vertices[i]
        close = np.nonzero(np.einsum("ij,ij->i", delta, delta) <= max_distance2)[0]
        for offset in close:
            union(i, i + 1 + int(offset))

    root_to_index: dict[int, int] = {}
    new_vertices: list[np.ndarray] = []
    old_to_new = np.empty(len(mesh.vertices), dtype=np.int32)
    weld_count = 0
    for index in range(len(mesh.vertices)):
        root = find(index)
        mapped = root_to_index.get(root)
        if mapped is None:
            mapped = len(new_vertices)
            root_to_index[root] = mapped
            new_vertices.append(mesh.vertices[root].copy())
        else:
            weld_count += 1
        old_to_new[index] = mapped

    new_indices = old_to_new[mesh.indices]
    valid = (
        (new_indices[:, 0] != new_indices[:, 1])
        & (new_indices[:, 1] != new_indices[:, 2])
        & (new_indices[:, 2] != new_indices[:, 0])
    )
    return ObjMesh(vertices=np.asarray(new_vertices, dtype=np.float64), indices=new_indices[valid]), weld_count


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _parse_obj_index(text: str, count: int) -> int:
    index = int(text)
    return count + index if index < 0 else index - 1


def load_obj_mesh(path: Path) -> ObjMesh:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                face: list[int] = []
                for token in line.split()[1:]:
                    vertex_text = token.split("/")[0]
                    if vertex_text:
                        face.append(_parse_obj_index(vertex_text, len(vertices)))
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])

    if not vertices or not triangles:
        raise ValueError(f"OBJ mesh has no usable triangles: {path}")

    return ObjMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        indices=np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
    )


def infer_scale(meshes: list[ObjMesh]) -> float:
    vertices = np.concatenate([mesh.vertices for mesh in meshes], axis=0)
    extent = np.ptp(vertices, axis=0)
    return 0.001 if float(np.max(extent)) > 10.0 else 1.0


def add_cloth_obj(
    builder: newton.ModelBuilder,
    mesh: ObjMesh,
    *,
    pos: wp.vec3,
    scale: float,
    density: float,
    tri_ke: float,
    tri_ka: float,
    tri_kd: float,
    edge_ke: float,
    edge_kd: float,
    particle_radius: float,
    add_springs: bool,
    spring_ke: float,
    spring_kd: float,
) -> tuple[slice, slice]:
    start = len(builder.particle_q)
    tri_start = len(builder.tri_indices)
    builder.add_cloth_mesh(
        pos=pos,
        rot=wp.quat_identity(),
        scale=scale,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=mesh.vertices.tolist(),
        indices=mesh.indices.reshape(-1).tolist(),
        density=density,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
        add_springs=add_springs,
        spring_ke=spring_ke,
        spring_kd=spring_kd,
        particle_radius=particle_radius,
    )
    return slice(start, len(builder.particle_q)), slice(tri_start, len(builder.tri_indices))


def add_static_mesh_obj(
    builder: newton.ModelBuilder,
    mesh: ObjMesh,
    *,
    pos: wp.vec3,
    scale: float,
    color: tuple[float, float, float],
    contact_ke: float,
    contact_kd: float,
    contact_mu: float,
) -> tuple[int, int]:
    body = builder.add_body(
        xform=wp.transform(p=pos, q=wp.quat_identity()),
        label="dustbin",
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=contact_ke,
        kd=contact_kd,
        mu=contact_mu,
    )
    shape = builder.add_shape_mesh(
        body=body,
        mesh=newton.Mesh(
            mesh.vertices,
            mesh.indices.reshape(-1),
            compute_inertia=False,
            color=color,
        ),
        scale=wp.vec3(scale, scale, scale),
        cfg=shape_cfg,
        color=color,
        label="dustbin_mesh",
    )
    return body, shape


def fix_particles(builder: newton.ModelBuilder, particle_indices: np.ndarray) -> None:
    for particle in particle_indices:
        index = int(particle)
        builder.particle_flags[index] = builder.particle_flags[index] & ~ParticleFlags.ACTIVE
        builder.particle_mass[index] = 0.0


def add_nearby_stitch_springs(
    builder: newton.ModelBuilder,
    bag_slice: slice,
    drawstring_slice: slice,
    *,
    max_distance: float,
    max_springs: int,
    ke: float,
    kd: float,
) -> int:
    if max_distance <= 0.0 or max_springs <= 0:
        return 0

    particle_q = np.asarray(builder.particle_q, dtype=np.float64)
    bag_indices = np.arange(bag_slice.start, bag_slice.stop)
    drawstring_indices = np.arange(drawstring_slice.start, drawstring_slice.stop)
    bag_points = particle_q[bag_indices]
    drawstring_points = particle_q[drawstring_indices]

    pairs: list[tuple[float, int, int]] = []
    chunk_size = 256
    for start in range(0, len(drawstring_points), chunk_size):
        chunk = drawstring_points[start : start + chunk_size]
        delta = chunk[:, None, :] - bag_points[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)
        nearest = np.argmin(dist2, axis=1)
        nearest_dist2 = dist2[np.arange(len(chunk)), nearest]
        for local, d2 in enumerate(nearest_dist2):
            distance = float(np.sqrt(d2))
            if distance <= max_distance:
                pairs.append(
                    (
                        distance,
                        int(bag_indices[nearest[local]]),
                        int(drawstring_indices[start + local]),
                    )
                )

    pairs.sort(key=lambda item: item[0])
    used: set[tuple[int, int]] = set()
    count = 0
    for _distance, bag_index, drawstring_index in pairs:
        pair = (bag_index, drawstring_index)
        if pair in used:
            continue
        builder.add_spring(bag_index, drawstring_index, ke=ke, kd=kd, control=0.0)
        used.add(pair)
        count += 1
        if count >= max_springs:
            break
    return count


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / float(args.view_fps)
        self.sim_substeps = max(1, int(args.view_substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.use_cuda_graph = not bool(args.no_cuda_graph) and not bool(args.pull_drawstring)

        bag_path = _resolve_path(args.bag_asset)
        drawstring_path = _resolve_path(args.drawstring_asset)
        dustbin_path = _resolve_path(args.dustbin_asset)
        bag_mesh = load_obj_mesh(bag_path)
        drawstring_mesh = load_obj_mesh(drawstring_path)
        dustbin_mesh = None if bool(args.no_dustbin) else load_obj_mesh(dustbin_path)
        import_meshes = [bag_mesh, drawstring_mesh]
        if dustbin_mesh is not None:
            import_meshes.append(dustbin_mesh)
        scale = float(args.scale) if args.scale is not None else infer_scale(import_meshes)
        bag_components = mesh_component_sizes(bag_mesh)
        drawstring_components_before = mesh_component_sizes(drawstring_mesh)
        dustbin_components = [] if dustbin_mesh is None else mesh_component_sizes(dustbin_mesh)
        drawstring_weld_distance = float(args.drawstring_weld_distance) / scale
        drawstring_mesh, drawstring_weld_count = weld_vertices(drawstring_mesh, drawstring_weld_distance)
        drawstring_components_after = mesh_component_sizes(drawstring_mesh)
        mouth_band = float(args.mouth_band) / scale
        mouth_vertices_local = (
            select_top_boundary_vertices(bag_mesh, mouth_band) if bool(args.fix_mouth) else np.empty(0, dtype=np.int32)
        )

        all_vertices = np.concatenate([mesh.vertices for mesh in import_meshes], axis=0)
        bbox_min = all_vertices.min(axis=0)
        bbox_max = all_vertices.max(axis=0)
        center_xy = 0.5 * (bbox_min[:2] + bbox_max[:2])
        pos_np = np.array(
            [
                -float(center_xy[0] * scale),
                -float(center_xy[1] * scale),
                float(args.start_height) - float(bbox_min[2] * scale),
            ],
            dtype=np.float64,
        )
        pos = wp.vec3(
            float(pos_np[0]),
            float(pos_np[1]),
            float(pos_np[2]),
        )
        pull_center_xy = center_xy * scale + pos_np[:2]
        if dustbin_mesh is not None:
            pull_center_xy = 0.5 * (dustbin_mesh.vertices[:, :2].min(axis=0) + dustbin_mesh.vertices[:, :2].max(axis=0))
            pull_center_xy = pull_center_xy * scale + pos_np[:2]

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        self.dustbin_body = -1
        self.dustbin_shape = -1
        if dustbin_mesh is not None:
            self.dustbin_body, self.dustbin_shape = add_static_mesh_obj(
                builder,
                dustbin_mesh,
                pos=pos,
                scale=scale,
                color=tuple(map(float, args.dustbin_color)),
                contact_ke=float(args.dustbin_contact_ke),
                contact_kd=float(args.dustbin_contact_kd),
                contact_mu=float(args.dustbin_contact_mu),
            )

        self.bag_slice, self.bag_tri_slice = add_cloth_obj(
            builder,
            bag_mesh,
            pos=pos,
            scale=scale,
            density=float(args.bag_density),
            tri_ke=float(args.bag_tri_ke),
            tri_ka=float(args.bag_tri_ka),
            tri_kd=float(args.bag_tri_kd),
            edge_ke=float(args.bag_edge_ke),
            edge_kd=float(args.bag_edge_kd),
            particle_radius=float(args.bag_particle_radius),
            add_springs=bool(args.bag_springs),
            spring_ke=float(args.bag_spring_ke),
            spring_kd=float(args.bag_spring_kd),
        )
        self.mouth_particle_indices = np.asarray(
            [self.bag_slice.start + int(vertex) for vertex in mouth_vertices_local], dtype=np.int32
        )
        fix_particles(builder, self.mouth_particle_indices)

        self.drawstring_slice, self.drawstring_tri_slice = add_cloth_obj(
            builder,
            drawstring_mesh,
            pos=pos,
            scale=scale,
            density=float(args.drawstring_density),
            tri_ke=float(args.drawstring_tri_ke),
            tri_ka=float(args.drawstring_tri_ka),
            tri_kd=float(args.drawstring_tri_kd),
            edge_ke=float(args.drawstring_edge_ke),
            edge_kd=float(args.drawstring_edge_kd),
            particle_radius=float(args.drawstring_particle_radius),
            add_springs=not bool(args.no_drawstring_springs),
            spring_ke=float(args.drawstring_spring_ke),
            spring_kd=float(args.drawstring_spring_kd),
        )
        stitch_count = add_nearby_stitch_springs(
            builder,
            self.bag_slice,
            self.drawstring_slice,
            max_distance=float(args.stitch_distance),
            max_springs=max(0, int(args.stitch_max_springs)),
            ke=float(args.stitch_ke),
            kd=float(args.stitch_kd),
        )

        bag_tri_indices = np.asarray(builder.tri_indices[self.bag_tri_slice], dtype=np.int32).reshape(-1)
        drawstring_tri_indices = np.asarray(builder.tri_indices[self.drawstring_tri_slice], dtype=np.int32).reshape(-1)

        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = float(args.soft_contact_ke)
        self.model.soft_contact_kd = float(args.soft_contact_kd)
        self.model.soft_contact_mu = float(args.soft_contact_mu)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.bag_tri_indices = wp.array(bag_tri_indices, dtype=wp.int32, device=self.model.device)
        self.drawstring_tri_indices = wp.array(drawstring_tri_indices, dtype=wp.int32, device=self.model.device)
        drawstring_particle_count = self.drawstring_slice.stop - self.drawstring_slice.start
        pull_indices_local = tuple(args.pull_drawstring_indices) if bool(args.pull_drawstring) else ()
        invalid_pull_indices = [index for index in pull_indices_local if index >= drawstring_particle_count]
        if invalid_pull_indices:
            raise ValueError(
                f"Drawstring pull indices out of range for {drawstring_particle_count} particles: "
                f"{invalid_pull_indices}"
            )
        pull_indices_global = [self.drawstring_slice.start + index for index in pull_indices_local]
        self.pull_particle_indices_np = np.asarray(pull_indices_global, dtype=np.int32)
        self.pull_particle_indices = wp.array(self.pull_particle_indices_np, dtype=wp.int32, device=self.model.device)
        self.pull_anchor_positions = wp.zeros(
            len(self.pull_particle_indices_np), dtype=wp.vec3, device=self.model.device
        )
        pull_directions = np.zeros((len(self.pull_particle_indices_np), 3), dtype=np.float32)
        if len(self.pull_particle_indices_np):
            q_np = self.state_0.particle_q.numpy()
            direction_xy = q_np[self.pull_particle_indices_np, :2] - pull_center_xy[None, :]
            direction_norm = np.linalg.norm(direction_xy, axis=1)
            valid = direction_norm > 1.0e-8
            pull_directions[valid, :2] = direction_xy[valid] / direction_norm[valid, None]
            pull_directions[~valid, 0] = 1.0
        self.pull_directions = wp.array(pull_directions, dtype=wp.vec3, device=self.model.device)
        self.pull_initialized = False
        self.pull_start_time = float(args.pull_start_time)
        self.pull_duration = float(args.pull_duration)
        self.pull_height = float(args.pull_height)
        self.pull_outward_distance = float(args.pull_outward_distance)
        self.pull_arc_height = float(args.pull_arc_height)
        self.bag_color = tuple(map(float, args.bag_color))
        self.drawstring_color = tuple(map(float, args.drawstring_color))
        particle_colors = np.tile(np.asarray(self.bag_color, dtype=np.float32), (self.model.particle_count, 1))
        particle_colors[self.drawstring_slice] = np.asarray(self.drawstring_color, dtype=np.float32)
        if len(self.mouth_particle_indices):
            particle_colors[self.mouth_particle_indices] = np.asarray(args.mouth_color, dtype=np.float32)
        if len(self.pull_particle_indices_np):
            particle_colors[self.pull_particle_indices_np] = np.asarray(args.pull_color, dtype=np.float32)
        self.particle_render_colors = wp.array(particle_colors, dtype=wp.vec3, device=self.model.device)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=max(1, int(args.solver_iterations)),
            particle_enable_self_contact=not bool(args.no_self_contact),
            particle_self_contact_radius=float(args.self_contact_radius),
            particle_self_contact_margin=float(args.self_contact_margin),
            particle_topological_contact_filter_threshold=max(0, int(args.topological_contact_filter_threshold)),
            particle_enable_tile_solve=not bool(args.no_tile_solve),
            friction_epsilon=float(args.friction_epsilon),
            rigid_body_particle_contact_buffer_size=max(1, int(args.rigid_body_particle_contact_buffer_size)),
        )

        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.75, 0.34), pitch=-18.0, yaw=90.0)

        print(
            "[PlasticBagVBD] "
            f"bag={bag_path}, drawstring={drawstring_path}, dustbin={dustbin_path if dustbin_mesh is not None else None}, "
            f"scale={scale:g}, "
            f"bag_particles={self.bag_slice.stop - self.bag_slice.start}, "
            f"drawstring_particles={self.drawstring_slice.stop - self.drawstring_slice.start}, "
            f"triangles={self.model.tri_count}, shapes={self.model.shape_count}, springs={self.model.spring_count}, "
            f"stitch_springs={stitch_count}, drawstring_welded_vertices={drawstring_weld_count}, "
            f"fixed_mouth_particles={len(self.mouth_particle_indices)}, "
            f"pull_drawstring_indices={pull_indices_local}, pull_particle_indices={pull_indices_global}, "
            f"pull_center_xy={pull_center_xy.tolist()}",
            flush=True,
        )
        print(
            "[PlasticBagVBD] "
            f"bag_components={bag_components[:8]}, "
            f"drawstring_components_before={drawstring_components_before[:8]}, "
            f"drawstring_components_after={drawstring_components_after[:8]}, "
            f"dustbin_components={dustbin_components[:8]}",
            flush=True,
        )
        if bool(args.pull_drawstring) and bool(args.no_cuda_graph) is False:
            print(
                "[PlasticBagVBD] CUDA graph disabled because drawstring pull targets are updated each step.", flush=True
            )
        self.capture()

    @staticmethod
    def create_parser():
        return create_parser()

    def capture(self):
        self.graph = None
        if not self.use_cuda_graph or not wp.get_device().is_cuda:
            return
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            print(f"[PlasticBagVBD] CUDA graph capture enabled for {self.sim_substeps} substeps.", flush=True)
        except Exception as exc:
            print(f"[PlasticBagVBD] CUDA graph capture failed; falling back to Python stepping: {exc!r}", flush=True)

    def simulate(self):
        for substep in range(self.sim_substeps):
            time = self.sim_time + substep * self.sim_dt
            self._initialize_pull(time)
            self._drive_pull_particles(self.state_0, time)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self._drive_pull_particles(self.state_1, time + self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _initialize_pull(self, time: float) -> None:
        if self.pull_initialized or len(self.pull_particle_indices_np) == 0 or time < self.pull_start_time:
            return
        wp.launch(
            kernel=record_particle_positions_kernel,
            dim=len(self.pull_particle_indices_np),
            inputs=[self.pull_particle_indices, self.state_0.particle_q],
            outputs=[self.pull_anchor_positions],
            device=self.model.device,
        )
        self.pull_initialized = True
        print(
            "[PlasticBagVBD] "
            f"drawstring pull started at time={time:.3f}, "
            f"global_particles={self.pull_particle_indices_np.tolist()}, "
            f"height={self.pull_height:g}, outward_distance={self.pull_outward_distance:g}, "
            f"arc_height={self.pull_arc_height:g}, duration={self.pull_duration:g}",
            flush=True,
        )

    def _drive_pull_particles(self, state: newton.State, time: float) -> None:
        if not self.pull_initialized:
            return
        wp.launch(
            kernel=drive_pull_particles_kernel,
            dim=len(self.pull_particle_indices_np),
            inputs=[
                self.pull_particle_indices,
                self.pull_anchor_positions,
                self.pull_directions,
                state.particle_q,
                state.particle_qd,
                self.model.particle_flags,
                self.model.particle_mass,
                self.model.particle_inv_mass,
                float(time),
                self.pull_start_time,
                self.pull_duration,
                self.pull_height,
                self.pull_outward_distance,
                self.pull_arc_height,
                int(ParticleFlags.ACTIVE),
            ],
            device=self.model.device,
        )

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1

        if self.args.log_interval > 0 and self.frame % int(self.args.log_interval) == 0:
            q = self.state_0.particle_q.numpy()
            qd = self.state_0.particle_qd.numpy()
            drawstring_q = q[self.drawstring_slice]
            print(
                "[PlasticBagVBD] "
                f"frame={self.frame} time={self.sim_time:.3f} "
                f"z=[{float(np.min(q[:, 2])):.4f}, {float(np.max(q[:, 2])):.4f}] "
                f"drawstring_z=[{float(np.min(drawstring_q[:, 2])):.4f}, {float(np.max(drawstring_q[:, 2])):.4f}] "
                f"max_speed={float(np.max(np.linalg.norm(qd, axis=1))):.4f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/plastic_bag/bag",
            self.state_0.particle_q,
            self.bag_tri_indices,
            hidden=bool(self.args.no_colored_meshes),
            backface_culling=False,
            color=self.bag_color,
        )
        self.viewer.log_mesh(
            "/plastic_bag/drawstring",
            self.state_0.particle_q,
            self.drawstring_tri_indices,
            hidden=bool(self.args.no_colored_meshes),
            backface_culling=False,
            color=self.drawstring_color,
        )
        self.viewer.log_points(
            "/plastic_bag/colored_particles",
            self.state_0.particle_q,
            self.model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not bool(self.args.show_colored_particles),
        )
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        if len(q) == 0:
            raise ValueError("Plastic bag VBD example has no particles.")
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise ValueError("Plastic bag VBD particle state is not finite.")
        max_abs_position = float(np.max(np.abs(q)))
        max_speed = float(np.max(np.linalg.norm(qd, axis=1)))
        if max_abs_position > 10.0 or max_speed > 200.0:
            raise ValueError(
                f"Plastic bag VBD simulation appears unstable: "
                f"max_abs_position={max_abs_position:.3f}, max_speed={max_speed:.3f}"
            )


def create_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.description = __doc__
    parser.add_argument("--bag-asset", default=str(DEFAULT_BAG_ASSET), help="OBJ mesh for the plastic bag body")
    parser.add_argument(
        "--drawstring-asset",
        default=str(DEFAULT_DRAWSTRING_ASSET),
        help="OBJ mesh for the drawstring cloth strip",
    )
    parser.add_argument("--dustbin-asset", default=str(DEFAULT_DUSTBIN_ASSET), help="OBJ mesh for the rigid dustbin")
    parser.add_argument("--no-dustbin", action="store_true", help="Do not add the rigid dustbin collision mesh")
    parser.add_argument("--scale", type=float, default=None, help="Asset scale override")
    parser.add_argument("--start-height", type=float, default=0.015, help="Lowest imported vertex height [m]")
    parser.add_argument("--view-fps", type=float, default=60.0)
    parser.add_argument("--view-substeps", type=int, default=4)
    parser.add_argument("--solver-iterations", type=int, default=12)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--no-colored-meshes", action="store_true", help="Hide the separate bag/drawstring mesh layers")
    parser.add_argument(
        "--show-colored-particles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay colored particles so bag and drawstring remain distinguishable in all viewers",
    )
    parser.add_argument("--bag-color", type=parse_rgb, default=(0.35, 0.72, 1.0))
    parser.add_argument("--drawstring-color", type=parse_rgb, default=(1.0, 0.48, 0.12))
    parser.add_argument("--dustbin-color", type=parse_rgb, default=(0.62, 0.62, 0.58))
    parser.add_argument("--mouth-color", type=parse_rgb, default=(0.1, 0.95, 0.35))
    parser.add_argument("--pull-color", type=parse_rgb, default=(1.0, 0.0, 0.85))
    parser.add_argument(
        "--fix-mouth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fix the bag mouth boundary",
    )
    parser.add_argument(
        "--mouth-band",
        type=float,
        default=0.03,
        help="Fix boundary vertices within this distance below the highest boundary vertex [m]",
    )

    parser.add_argument("--bag-density", type=float, default=0.025)
    parser.add_argument("--bag-particle-radius", type=float, default=0.005)
    parser.add_argument("--bag-tri-ke", type=float, default=5.0e3)
    parser.add_argument("--bag-tri-ka", type=float, default=5.0e3)
    parser.add_argument("--bag-tri-kd", type=float, default=5.0e-2)
    parser.add_argument("--bag-edge-ke", type=float, default=2.0e-3)
    parser.add_argument("--bag-edge-kd", type=float, default=1.0e-4)
    parser.add_argument("--bag-springs", action="store_true", help="Add structural springs to the bag mesh")
    parser.add_argument("--bag-spring-ke", type=float, default=1.0e3)
    parser.add_argument("--bag-spring-kd", type=float, default=1.0e-2)

    parser.add_argument("--drawstring-density", type=float, default=0.05)
    parser.add_argument("--drawstring-particle-radius", type=float, default=0.005)
    parser.add_argument("--drawstring-tri-ke", type=float, default=2.0e4)
    parser.add_argument("--drawstring-tri-ka", type=float, default=2.0e4)
    parser.add_argument("--drawstring-tri-kd", type=float, default=5.0e-2)
    parser.add_argument("--drawstring-edge-ke", type=float, default=2.0e-2)
    parser.add_argument("--drawstring-edge-kd", type=float, default=1.0e-4)
    parser.add_argument(
        "--drawstring-weld-distance",
        type=float,
        default=5.0e-4,
        help="Merge drawstring vertices within this world-space distance before import [m]",
    )
    parser.add_argument("--no-drawstring-springs", action="store_true")
    parser.add_argument("--drawstring-spring-ke", type=float, default=2.0e4)
    parser.add_argument("--drawstring-spring-kd", type=float, default=5.0e-2)
    parser.add_argument(
        "--pull-drawstring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After settling, drive selected drawstring local particle indices outward and upward",
    )
    parser.add_argument(
        "--pull-drawstring-indices",
        type=parse_int_list,
        default=(603, 605, 269, 272, 268),
        help="Comma-separated local drawstring particle indices to pull",
    )
    parser.add_argument("--pull-start-time", type=float, default=0.5, help="Time to start pulling the drawstring [s]")
    parser.add_argument(
        "--pull-duration", type=float, default=1.5, help="Duration of the drawstring pull trajectory [s]"
    )
    parser.add_argument("--pull-height", type=float, default=0.45, help="Final upward drawstring pull distance [m]")
    parser.add_argument(
        "--pull-outward-distance", type=float, default=0.15, help="Final outward drawstring pull distance [m]"
    )
    parser.add_argument(
        "--pull-arc-height", type=float, default=0.06, help="Extra midpoint height for the parabolic pull arc [m]"
    )

    parser.add_argument(
        "--stitch-distance", type=float, default=0.0, help="Optional max bag/drawstring stitch distance [m]"
    )
    parser.add_argument("--stitch-max-springs", type=int, default=512)
    parser.add_argument("--stitch-ke", type=float, default=5.0e3)
    parser.add_argument("--stitch-kd", type=float, default=1.0e-2)

    parser.add_argument("--soft-contact-ke", type=float, default=5.0e4)
    parser.add_argument("--soft-contact-kd", type=float, default=1.0)
    parser.add_argument("--soft-contact-mu", type=float, default=0.5)
    parser.add_argument("--dustbin-contact-ke", type=float, default=5.0e5)
    parser.add_argument("--dustbin-contact-kd", type=float, default=50.0)
    parser.add_argument("--dustbin-contact-mu", type=float, default=0.7)
    parser.add_argument("--friction-epsilon", type=float, default=1.0e-2)
    parser.add_argument("--self-contact-radius", type=float, default=0.003)
    parser.add_argument("--self-contact-margin", type=float, default=0.006)
    parser.add_argument("--topological-contact-filter-threshold", type=int, default=2)
    parser.add_argument(
        "--rigid-body-particle-contact-buffer-size",
        type=int,
        default=16384,
        help="Per-body VBD contact-list capacity for rigid body / cloth particle contacts",
    )
    parser.add_argument("--no-self-contact", action="store_true")
    parser.add_argument("--no-tile-solve", action="store_true")
    return parser


def main() -> int:
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
