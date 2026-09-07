# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from newton import Contacts, Model, State
from newton._src.solvers.style3d.collision.bvh import BvhEdge, BvhTri
from newton._src.solvers.style3d.collision.kernels import (
    accumulate_rigid_edge_cloth_edge_reaction_kernel,
    accumulate_rigid_vertex_cloth_face_reaction_kernel,
    accumulate_body_reaction_kernel,
    accumulate_projection_impulse_kernel,
    apply_contact_projection_kernel,
    bake_shape_sdf_kernel,
    bake_shape_face_kernel,
    count_particle_contacts_kernel,
    clamp_body_wrench_kernel,
    eval_rigid_edge_cloth_edge_contacts_kernel,
    eval_rigid_vertex_cloth_face_contacts_kernel,
    eval_body_contact_kernel,
    finalize_contact_projection_kernel,
    hessian_multiply_kernel,
    handle_edge_edge_contacts_kernel,
    handle_vertex_triangle_contacts_kernel,
    project_body_particle_contacts_kernel,
    project_rigid_edge_cloth_edge_kernel,
    project_rigid_vertex_cloth_face_kernel,
    project_tri_sdf_kernel,
    eval_tri_sdf_contact_kernel,
    accumulate_tri_sdf_reaction_kernel,
    solve_untangling_kernel,
    solve_rigid_untangling_kernel,
    summarize_feature_query_kernel,
    transform_rigid_feature_vertices_kernel,
    tri_sdf_broadphase_kernel,
    tri_sdf_bp_selfcheck_kernel,
    add_tri_sdf_cache_kernel,
    tri_sdf_par_cull_kernel,
    tri_sdf_par_seed_kernel,
    SdfGridExact,
)
from newton._src.solvers.style3d.collision import _t14_prof


def _t14_bake_dir():
    import os

    d = os.environ.get(
        "T14_SDF_BAKE_CACHE",
        "/home/hwk/program/synreal-world/data/eval_out/gripper_penetration/T14/sdf_cache",
    )
    return d


def _t14_bake_verify():
    import os

    return os.environ.get("T14_SDF_BAKE_VERIFY", "0") not in ("", "0")


def _t14_bake_key(vertices, indices, voxel, pad, max_dist, lo, n):
    """Everything the baked field depends on, hashed."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
    h.update(np.asarray([voxel, pad, max_dist], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(lo, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(n, dtype=np.int32).tobytes())
    h.update(b"bake_shape_sdf_kernel/v1")
    return h.hexdigest()


def _t14_bake_load(key):
    import os

    fp = os.path.join(_t14_bake_dir(), key + ".npy")
    if not os.path.exists(fp):
        return None
    try:
        return np.load(fp)
    except Exception:  # noqa: BLE001  (a corrupt cache must never break a run)
        return None


# --- T15-A: nearest-triangle grid + pseudo-normal bake --------------------
# Same cache mechanism as the T14 SDF bake: the key covers everything the table
# depends on, so a hit is bitwise the table the bake would have produced.


# T15-A: fixed default spacing of the nearest-triangle table [m] (see
# ``_t15_build_gridexact``); overridable with T15_SDF_GX_VOXEL.
_T15_GX_DEFAULT_VOXEL = 2.5e-4


def _t15_gx_dir():
    import os

    return os.environ.get(
        "T15_SDF_GX_CACHE",
        "/home/hwk/program/synreal-world/data/eval_out/gripper_penetration/T15/gx_cache",
    )


def _t15_gx_key(vertices, indices, voxel, pad, max_dist, lo, n):
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
    h.update(np.asarray([voxel, pad, max_dist], dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(lo, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(n, dtype=np.int32).tobytes())
    h.update(b"bake_shape_face_kernel+pseudo_normals/v1")
    return h.hexdigest()


def _t15_gx_load(key):
    import os

    fp = os.path.join(_t15_gx_dir(), key + ".npz")
    if not os.path.exists(fp):
        return None, None, None
    try:
        z = np.load(fp)
        return z["face"], z["vn"], z["en"]
    except Exception:  # noqa: BLE001  (a corrupt cache must never break a run)
        return None, None, None


def _t15_gx_store(key, face, vn, en):
    import os

    d = _t15_gx_dir()
    try:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, key + ".tmp.npz")
        np.savez(tmp, face=face, vn=vn, en=en)
        os.replace(tmp, os.path.join(d, key + ".npz"))
    except Exception as e:  # noqa: BLE001
        print(f"[collision] T15 gx cache store failed ({e}); continuing", flush=True)


def _t15_ring_key(vertices, indices):
    """The vertex one-ring depends on the MESH only, not on the grid, so it gets
    its own cache key and the (large) face-grid caches stay valid."""
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(indices, dtype=np.int32).tobytes())
    h.update(b"vertex_one_ring_csr/v1")
    return h.hexdigest()


def _t15_vertex_one_ring(vertices, indices):
    """CSR of every face's VERTEX ONE-RING (faces sharing >=1 vertex, self excluded).

    Cached on disk under the geometry key: on the 21302-face blade the build is a
    few seconds of Python, and it is the same table for every grid spacing.
    """
    import os

    key = _t15_ring_key(vertices, indices)
    fp = os.path.join(_t15_gx_dir(), "ring_" + key + ".npz")
    if os.path.exists(fp):
        try:
            z = np.load(fp)
            return z["off"], z["idx"]
        except Exception:  # noqa: BLE001
            pass
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    nf = len(tris)
    nv = int(tris.max()) + 1
    vf_v = tris.reshape(-1)
    vf_f = np.repeat(np.arange(nf, dtype=np.int64), 3)
    order = np.argsort(vf_v, kind="stable")
    vf_v = vf_v[order]
    vf_f = vf_f[order]
    vstart = np.searchsorted(vf_v, np.arange(nv))
    vend = np.searchsorted(vf_v, np.arange(nv), side="right")
    off = np.zeros(nf + 1, dtype=np.int32)
    chunks = []
    total = 0
    for f in range(nf):
        acc = np.concatenate([vf_f[vstart[v]:vend[v]] for v in tris[f]])
        acc = np.unique(acc)
        acc = acc[acc != f]
        chunks.append(acc.astype(np.int32))
        total += len(acc)
        off[f + 1] = total
    idx = np.concatenate(chunks).astype(np.int32) if chunks else np.zeros(0, dtype=np.int32)
    try:
        os.makedirs(_t15_gx_dir(), exist_ok=True)
        tmp = fp + ".tmp.npz"
        np.savez(tmp, off=off, idx=idx)
        os.replace(tmp, fp)
    except Exception as e:  # noqa: BLE001
        print(f"[collision] T15 ring cache store failed ({e}); continuing", flush=True)
    return off, idx


def _t15_pseudo_normals(vertices, indices):
    """Angle-weighted vertex normals and two-face edge normals.

    Baerentzen & Aanaes 2005: with these the sign test ``dot(p - cp, N_feature)``
    is exact for a closed mesh whatever feature (face / edge / vertex) carries
    the closest point.  ``en`` is stored per (face, local edge) --
    ``en[3*f + 0]`` = edge (v0,v1) = AB, ``+1`` = (v0,v2) = AC, ``+2`` = (v1,v2)
    = BC -- so the query needs no edge table, just the winning face and feature
    code that ``triangle_closest_point`` already returns.
    """
    v = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    p0, p1, p2 = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1.0e-300)

    def _angle(o, x, y):
        e1 = x - o
        e2 = y - o
        e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1.0e-300)
        e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1.0e-300)
        return np.arccos(np.clip((e1 * e2).sum(axis=1), -1.0, 1.0))

    vn = np.zeros_like(v)
    for k, ang in enumerate((_angle(p0, p1, p2), _angle(p1, p2, p0), _angle(p2, p0, p1))):
        np.add.at(vn, tris[:, k], fn * ang[:, None])
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1.0e-300)

    # AB, AC, BC -- the order of TRI_CONTACT_FEATURE_EDGE_{AB,AC,BC}
    pairs = np.stack([tris[:, [0, 1]], tris[:, [0, 2]], tris[:, [1, 2]]], axis=1).reshape(-1, 2)
    keyed = np.sort(pairs, axis=1)
    _uk, inv = np.unique(keyed, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    acc = np.zeros((len(_uk), 3))
    np.add.at(acc, inv, np.repeat(fn, 3, axis=0))
    en = acc[inv]
    en /= np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1.0e-300)
    return vn.astype(np.float32), en.astype(np.float32)


def _t14_bake_store(key, arr):
    import os

    d = _t14_bake_dir()
    try:
        os.makedirs(d, exist_ok=True)
        # np.save appends ".npy" unless the name already ends in it, so the
        # staging name must too -- otherwise the rename below chases a file that
        # was never written.
        tmp = os.path.join(d, key + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, os.path.join(d, key + ".npy"))
    except Exception as e:  # noqa: BLE001
        print(f"[collision] tri-SDF bake cache store failed ({e}); continuing", flush=True)


########################################################################################################################
###################################################    Collision    ####################################################
########################################################################################################################


class Collision:
    """
    Collision handler for cloth simulation.
    """

    def __init__(self, model: Model):
        """
        Initialize the collision handler, including BVHs and buffers.

        Args:
            model: The simulation model containing particle and geometry data.
        """
        self.model = model
        self.radius = 3e-3  # Contact radius
        self.stiff_vf = 0.5  # Stiffness coefficient for vertex-face (VF) collision constraints
        self.stiff_ee = 0.1  # Stiffness coefficient for edge-edge (EE) collision constraints
        self.stiff_ef = 1.0  # Stiffness coefficient for edge-face (EF) collision constraints
        self.friction_epsilon = 1e-2
        self.integrate_with_external_rigid_solver = True
        self.tri_bvh = BvhTri(model.tri_count, self.model.device)
        self.edge_bvh = BvhEdge(model.edge_count, self.model.device)
        self.body_contact_max = model.shape_count * model.particle_count
        self.broad_phase_ee = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_ef = wp.array(shape=(32, model.edge_count), dtype=int, device=self.model.device)
        self.broad_phase_vf = wp.array(shape=(32, model.particle_count), dtype=int, device=self.model.device)

        self.Hx = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.contact_hessian_diags = wp.zeros(model.particle_count, dtype=wp.mat33, device=self.model.device)
        # T14: set by SolverStyle3D; how many Newton iterations a substep runs.
        self.nonlinear_iterations = 0

        # S1 static-friction anchors. Allocated always (fixed size, CUDA-graph
        # safe); ``anchor_kt_ratio <= 0`` keeps the stock viscous friction law.
        self.anchor_kt_ratio = 0.0
        self.anchor_local = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.anchor_shape = wp.full(model.particle_count, -1, dtype=wp.int32, device=self.model.device)

        # S3 AVBD tangential multipliers (dual ascent, cone-clamped per iteration).
        self.avbd_dual_k = 0.0
        self.lambda_t = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.lambda_shape = wp.full(model.particle_count, -1, dtype=wp.int32, device=self.model.device)
        self.shape_contact_ke = wp.full(
            model.shape_count,
            float(model.soft_contact_ke),
            dtype=float,
            device=self.model.device,
        )
        # E4 per-shape penalty band (PhysX ``contact_offset``). Allocated always
        # (fixed size, CUDA-graph safe); all zeros = every shape uses
        # ``particle_radius`` as its band, i.e. the stock law, bit-identical.
        self.shape_contact_offset = wp.zeros(
            model.shape_count, dtype=float, device=self.model.device
        )
        # R9 per-shape hardening eps_max for the compression law
        # ``f = ke*d/(1 - d/(eps_max*band))``. Allocated always (fixed size,
        # CUDA-graph safe); all zeros = every shape keeps the stock LINEAR
        # ``f = ke*d``, i.e. the default path is bit-identical.
        self.shape_hardening_eps = wp.zeros(
            model.shape_count, dtype=float, device=self.model.device
        )
        # Material-derived contact stiffness (default off: material_ke <= 0 keeps
        # the per-shape constant, and the kernel takes the same branch it always
        # did, so the default path is bit-identical).
        self.material_ke = 0.0
        self.particle_contact_ke = wp.zeros(model.particle_count, dtype=float, device=self.model.device)
        self.feature_vertex_shape = None
        self.feature_edge_shape = None
        # R2-3 rigid untangling (ICM against rigid triangles). ``None`` =
        # disabled; nothing is allocated and no kernel is launched, so the
        # default path is unchanged.
        self.icm_tri_indices = None
        self.icm_stiff_factor = 0.0
        self.icm_thickness = 0.0
        self.icm_query_radius = 0.0
        self.projection_iterations = 0
        self.projection_interleaved = False
        # E3 triangle-level SDF contact. ``None`` = disabled; nothing is
        # allocated and no kernel is launched, so the default path is unchanged.
        self.tri_sdf_slot_shape = None
        self.tri_sdf_compliant = False
        # R16-A2': the shape meshes the exact query backend evaluates against.
        self.tri_sdf_meshes = None
        self.tri_sdf_mesh_id = None
        self.tri_sdf_mesh_max_dist = 0.0
        # R13g: per-(slot, triangle) barycentric contact point, resolved once
        # per substep and held for the Newton iterations (see _R13G_SDF_FREEZE).
        self.tri_sdf_w = None
        # T8: allocated by enable_triangle_sdf_contacts; None until then.
        self.tri_sdf_anchor_p = None
        self.tri_sdf_anchor_valid = None
        self.tri_sdf_anchor_kt_ratio = 0.0
        self.tri_sdf_w_valid = False
        # R13g: how often inside a substep the barycentric contact point is
        # re-searched.  0 = once, at iteration 0 (full freeze).  n > 0 = every
        # n-th Newton iteration, a compromise between killing the chatter and
        # still tracking a fast-moving shape.  Only read when R13G_SDF_FREEZE.
        self.tri_sdf_resolve_every = int(
            __import__("os").environ.get("R13G_SDF_RESOLVE_EVERY", "0")
        )
        # R13f: inputs the compliant-SDF friction needs in the reaction pass,
        # stashed by accumulate_contact_force.  None => the reaction falls back
        # to the current state (zero relative displacement => zero friction).
        self._r13f_dt = 0.0
        self._r13f_particle_q_prev = None
        self._r13f_body_q_prev = None
        self.proj_delta = None
        self.projection_relaxation = None

        self.edge_bvh.build(model.particle_q, self.model.edge_indices, self.radius)
        self.tri_bvh.build(model.particle_q, self.model.tri_indices, self.radius)

    def enable_friction_anchors(self, kt_ratio: float = 0.1) -> None:
        """Enable S1 static-friction anchors on particle-rigid contacts.

        The stock law regularises friction from this substep's slip only, so a
        pinched grip creeps out at a steady rate. Anchors make the tangential
        force a spring to a persistent contact point with elastoplastic return
        mapping, which is what actually sticks.

        Args:
            kt_ratio: Tangential stiffness as a fraction of the contact's normal
                stiffness. 0 disables (stock viscous friction). Calibrate upward
                from 0.1; 8 is known to lock first grasps and is not a free knob.
        """
        self.anchor_kt_ratio = max(float(kt_ratio), 0.0)
        print(f"[collision] friction anchors: kt_ratio={self.anchor_kt_ratio:g}", flush=True)

    def enable_avbd_friction(self, dual_k: float = 0.1) -> None:
        """Enable S3 AVBD tangential multipliers on particle-rigid contacts.

        Args:
            dual_k: Dual-ascent step as a fraction of the contact's normal
                stiffness. Only sets convergence speed, not the final force --
                the multiplier saturates on the Coulomb cone either way.
        """
        self.avbd_dual_k = max(float(dual_k), 0.0)
        print(f"[collision] AVBD friction duals: dual_k={self.avbd_dual_k:g}", flush=True)

    def set_shape_contact_stiffness(self, shape_ids, stiffness: float):
        """Override particle-contact stiffness for arbitrary rigid shapes."""
        values = self.shape_contact_ke.numpy()
        values[np.asarray(shape_ids, dtype=np.int64)] = max(float(stiffness), 0.0)
        self.shape_contact_ke.assign(values)

    def set_shape_contact_offset(self, shape_ids, offset: float):
        """E4: decouple the penalty's range band from ``particle_radius``.

        The stock law ``f = ke * (r - d)`` uses ``particle_radius`` both as the
        geometric probe radius and as the band the force acts over, so taking the
        radius down to the cloth's physical half-thickness collapses the normal
        load and, with it, the Coulomb cone mu*N (measured 15.4 / 8.5 / 0.8 N at
        r = 8 / 4 / 0.5 mm). Setting an offset on the gripper shapes gives those
        shapes their own band -- normal load AND friction act over
        ``h < d < offset`` -- while the geometric stop stays wherever the geometry
        channel puts it.

        Args:
            shape_ids: shapes that get the band (typically the gripper fingers).
            offset: band [m]. 0 restores the stock ``particle_radius`` behaviour.
                It must be LARGER than the geometric stop of whatever holds the
                cloth out (``particle_radius`` for the vertex projection, the
                triangle constraint's half-thickness for the SDF one); an offset
                at or below that stop leaves zero overlap, hence N = 0 and
                mu*N = 0 -- a frictionless wall.

        Note: the contact-generation margin must also cover the band, otherwise
        no contact is even generated at ``d`` inside it (see
        ``CollisionPipeline(soft_contact_margin=...)``: a pair is emitted while
        ``d < margin + particle_radius``).
        """
        values = self.shape_contact_offset.numpy()
        values[np.asarray(shape_ids, dtype=np.int64)] = max(float(offset), 0.0)
        self.shape_contact_offset.assign(values)
        print(f"[collision] contact offset band: shapes={list(np.asarray(shape_ids).ravel())} "
              f"offset={float(offset) * 1000:.2f}mm", flush=True)

    def set_shape_contact_hardening(self, shape_ids, eps_max: float):
        """R9: hardening compression law on the given shapes' penalty band.

        The stock band law is LINEAR, ``f = ke*d``.  Its only handle on the
        normal load is the band width, and a wide band is exactly what rakes
        cloth into the jaw (doc GRIPPER-CONTACT R3(b): jaw stop = 1.60*offset,
        gathered material = 169*offset + 1844 mm^2).  A fabric squeezed
        transversely does not behave linearly -- it is soft, then hardens, then
        hits an incompressible core.  Doc 2.2 writes that as
        ``p(eps) = k0*eps/(1 - eps/eps_max)``.  Here the compressible layer is
        the band, so ``eps = d/band``, ``k0 = ke*band`` and

            f = ke * d / (1 - d/(eps_max*band))

        which is the same two-parameter law with the stiffness kept in the units
        the stack already uses.  Consequences: (i) penetration is bounded by
        ``eps_max*band`` BY CONSTRUCTION, not by tuning; (ii) the load can be
        raised without widening the band, which is the whole point -- narrow
        band for a tight stop, hardening for the newtons; (iii) friction rides
        on ``f_n`` downstream, so mu*N hardens with it.

        Per shape on purpose.  The table's band is ``particle_radius`` (0.5 mm);
        an asymptote there would make lying on the table a wall.

        Args:
            shape_ids: shapes that get the law (typically the gripper fingers).
            eps_max: strain at the incompressible core, in (0, 1) as a fraction
                of that shape's band.  <= 0 restores the stock linear law.
        """
        values = self.shape_hardening_eps.numpy()
        values[np.asarray(shape_ids, dtype=np.int64)] = max(float(eps_max), 0.0)
        self.shape_hardening_eps.assign(values)
        print(f"[collision] contact hardening: shapes={list(np.asarray(shape_ids).ravel())} "
              f"eps_max={float(eps_max):.3f} (d_max = eps_max x band)", flush=True)

    def enable_rigid_feature_contacts(
        self,
        shape_ids,
        contact_radius: float,
        search_margin: float,
        crease_angle_deg: float = 10.0,
        candidate_capacity: int = 128,
    ) -> None:
        """Enable symmetric mesh-feature contact for arbitrary rigid shapes.

        Uses the rigid meshes' existing vertices and sharp/boundary edges
        against the cloth's existing triangles and edges.  It does not sample
        cloth faces, add particles, or assume gripper names or pairs.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids:
            return
        local_vertices = []
        vertex_shapes = []
        vertex_weights = []
        rigid_edges = []
        edge_shapes = []
        edge_weights = []
        shape_scale = self.model.shape_scale.numpy()
        vertex_offset = 0
        crease_cos = float(np.cos(np.deg2rad(float(crease_angle_deg))))

        for shape in shape_ids:
            source = self.model.shape_source[shape]
            vertices = np.asarray(source.vertices, dtype=np.float32)
            faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            if not len(vertices) or not len(faces):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)

            face_normal = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            )
            face_area2 = np.linalg.norm(face_normal, axis=1)
            face_normal /= np.maximum(face_area2[:, None], 1.0e-20)
            area_weight = np.zeros(len(vertices), dtype=np.float32)
            for corner in range(3):
                np.add.at(area_weight, faces[:, corner], face_area2 / 6.0)
            nonzero = area_weight[area_weight > 0.0]
            if len(nonzero):
                area_weight /= float(np.mean(nonzero))
            else:
                area_weight.fill(1.0)

            adjacency: dict[tuple[int, int], list[int]] = {}
            for face_id, (a, b, c) in enumerate(faces):
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (int(min(u, v)), int(max(u, v)))
                    adjacency.setdefault(key, []).append(face_id)
            feature_edges = []
            for (u, v), adjacent in adjacency.items():
                keep = len(adjacent) == 1
                if not keep:
                    base = face_normal[adjacent[0]]
                    keep = any(
                        float(np.dot(base, face_normal[other])) < crease_cos
                        for other in adjacent[1:]
                    )
                if keep:
                    feature_edges.append((u, v))

            local_vertices.append(vertices)
            vertex_shapes.append(np.full(len(vertices), shape, dtype=np.int32))
            vertex_weights.append(area_weight)
            if feature_edges:
                feature_edges_np = np.asarray(feature_edges, dtype=np.int32)
                packed = np.full((len(feature_edges_np), 4), -1, dtype=np.int32)
                packed[:, 2:] = feature_edges_np + vertex_offset
                rigid_edges.append(packed)
                edge_shapes.append(np.full(len(packed), shape, dtype=np.int32))
                lengths = np.linalg.norm(
                    vertices[feature_edges_np[:, 1]] - vertices[feature_edges_np[:, 0]],
                    axis=1,
                ).astype(np.float32)
                positive = lengths[lengths > 0.0]
                if len(positive):
                    lengths /= float(np.mean(positive))
                else:
                    lengths.fill(1.0)
                edge_weights.append(lengths)
            vertex_offset += len(vertices)

        if not local_vertices or not rigid_edges:
            return

        device = self.model.device
        self.feature_local_pos = wp.array(
            np.concatenate(local_vertices), dtype=wp.vec3, device=device)
        self.feature_vertex_shape = wp.array(
            np.concatenate(vertex_shapes), dtype=int, device=device)
        self.feature_vertex_weight = wp.array(
            np.concatenate(vertex_weights), dtype=float, device=device)
        self.feature_edge_indices = wp.array(
            np.concatenate(rigid_edges), dtype=int, device=device)
        self.feature_edge_shape = wp.array(
            np.concatenate(edge_shapes), dtype=int, device=device)
        self.feature_edge_weight = wp.array(
            np.concatenate(edge_weights), dtype=float, device=device)
        self.feature_pos_prev = wp.zeros(
            len(self.feature_local_pos), dtype=wp.vec3, device=device)
        self.feature_pos = wp.zeros(
            len(self.feature_local_pos), dtype=wp.vec3, device=device)

        capacity = max(8, int(candidate_capacity))
        self.feature_broad_phase_vf = wp.zeros(
            (capacity + 1, len(self.feature_local_pos)), dtype=int, device=device)
        self.feature_broad_phase_ee = wp.zeros(
            (capacity + 1, len(self.feature_edge_indices)), dtype=int, device=device)
        self.feature_stats = wp.zeros(4, dtype=int, device=device)
        self.feature_contact_radius = max(float(contact_radius), 0.0)
        self.feature_search_margin = max(float(search_margin), 0.0)
        self.feature_candidate_capacity = capacity
        print(
            "[collision] rigid feature contacts: "
            f"{len(shape_ids)} shapes, {len(self.feature_local_pos)} vertices, "
            f"{len(self.feature_edge_indices)} sharp edges, "
            f"radius={self.feature_contact_radius:g} m, "
            f"search_margin={self.feature_search_margin:g} m",
            flush=True,
        )

    def _update_rigid_feature_queries(
        self,
        cloth_pos: wp.array[wp.vec3],
        body_q_prev: wp.array[wp.transform],
        body_q: wp.array[wp.transform],
    ) -> None:
        if self.feature_vertex_shape is None:
            return
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_local_pos,
                self.feature_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q_prev,
            ],
            outputs=[self.feature_pos_prev],
            device=self.model.device,
        )
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_local_pos,
                self.feature_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
            ],
            outputs=[self.feature_pos],
            device=self.model.device,
        )
        self.tri_bvh.refit(cloth_pos, self.model.tri_indices, self.radius)
        max_dist = self.feature_contact_radius + self.feature_search_margin
        self.tri_bvh.triangle_vs_point(
            self.feature_pos,
            cloth_pos,
            self.model.tri_indices,
            self.feature_broad_phase_vf,
            False,
            max_dist,
            self.feature_search_margin,
        )
        wp.launch(
            summarize_feature_query_kernel,
            dim=len(self.feature_local_pos),
            inputs=[
                self.feature_broad_phase_vf,
                self.feature_candidate_capacity,
                0,
            ],
            outputs=[self.feature_stats],
            device=self.model.device,
        )
        self._update_rigid_feature_edge_query(cloth_pos)

    def _update_rigid_feature_edge_query(
        self,
        cloth_pos: wp.array[wp.vec3],
    ) -> None:
        """Refresh the edge-edge candidate set."""
        self.edge_bvh.refit(cloth_pos, self.model.edge_indices, self.radius)
        max_dist = self.feature_contact_radius + self.feature_search_margin
        self.edge_bvh.edge_vs_edge(
            self.feature_pos,
            self.feature_edge_indices,
            cloth_pos,
            self.model.edge_indices,
            self.feature_broad_phase_ee,
            False,
            max_dist,
            self.feature_search_margin,
        )
        wp.launch(
            summarize_feature_query_kernel,
            dim=len(self.feature_edge_indices),
            inputs=[
                self.feature_broad_phase_ee,
                self.feature_candidate_capacity,
                2,
            ],
            outputs=[self.feature_stats],
            device=self.model.device,
        )

    def enable_rigid_untangling(
        self,
        shape_ids,
        stiff_factor: float = 1.0,
        thickness: float = -1.0,
        query_radius: float = -1.0,
        candidate_capacity: int = 32,
    ) -> None:
        """R2-3: enable Intersection Contour Minimisation against rigid shapes.

        The stock cloth-cloth ICM (``stiff_ef``) is the only channel in this
        solver that can recover from an ALREADY-tangled state; every other
        channel is a proximity law that pushes a particle to the nearest
        surface. For a blade thinner than the penetration depth "nearest" is the
        wrong side, and an edge can thread a triangle with both its endpoints
        outside the shell -- neither the penalty nor the projection sees it.
        This gives the same contour-length gradient a rigid face side.

        Args:
            shape_ids: rigid shapes whose triangles take part (typically the
                gripper fingers). An empty list leaves the channel disabled.
            stiff_factor: multiplies the cloth's own PD diagonal, exactly like
                ``stiff_ef`` does for cloth-cloth. 0 disables.
            thickness: displacement scale [m]; the per-iteration correction is
                ``2 * thickness``. Negative takes the cloth-cloth default
                (``2 * self.radius``), which is the same number the stock ICM
                uses, so "same law, rigid face" is the literal default.
            query_radius: broad-phase padding [m]. Negative takes ``self.radius``.
            candidate_capacity: max rigid triangles examined per cloth edge.

        Default OFF: unless this is called, ``icm_tri_indices`` stays ``None``,
        nothing is allocated and the kernel is never launched.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids or float(stiff_factor) <= 0.0:
            return
        local_vertices = []
        vertex_shapes = []
        rigid_tris = []
        shape_scale = self.model.shape_scale.numpy()
        vertex_offset = 0
        for shape in shape_ids:
            source = self.model.shape_source[shape]
            if source is None:
                continue
            vertices = np.asarray(source.vertices, dtype=np.float32)
            faces = np.asarray(source.indices, dtype=np.int32).reshape(-1, 3)
            if not len(vertices) or not len(faces):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)
            local_vertices.append(vertices)
            vertex_shapes.append(np.full(len(vertices), shape, dtype=np.int32))
            rigid_tris.append(faces + vertex_offset)
            vertex_offset += len(vertices)

        if not rigid_tris:
            return

        device = self.model.device
        self.icm_local_pos = wp.array(
            np.concatenate(local_vertices), dtype=wp.vec3, device=device)
        self.icm_vertex_shape = wp.array(
            np.concatenate(vertex_shapes), dtype=int, device=device)
        self.icm_tri_indices = wp.array(
            np.concatenate(rigid_tris), dtype=int, device=device)
        self.icm_pos = wp.zeros(len(self.icm_local_pos), dtype=wp.vec3, device=device)
        self.icm_stiff_factor = float(stiff_factor)
        self.icm_thickness = (
            2.0 * self.radius if float(thickness) < 0.0 else float(thickness)
        )
        self.icm_query_radius = (
            self.radius if float(query_radius) < 0.0 else float(query_radius)
        )
        capacity = max(4, int(candidate_capacity))
        self.icm_capacity = capacity
        self.icm_broad_phase = wp.zeros(
            (capacity + 1, self.model.edge_count), dtype=int, device=device)
        # [0] = cloth-edge x rigid-triangle intersections found this substep,
        # [1] = broad-phase candidates handed to the kernel. Zeroed every substep.
        self.icm_stats = wp.zeros(2, dtype=int, device=device)
        # Never zeroed: episode totals, so "did the cloth EVER thread a finger"
        # can be answered without a readback inside the substep loop.
        self.icm_stats_total = wp.zeros(4, dtype=int, device=device)
        self.icm_bvh = BvhTri(self.icm_tri_indices.shape[0], device)
        # Seed the hierarchy from the rest pose; every substep only refits it.
        # ``build`` allocates, so it must stay outside any graph capture.
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.icm_local_pos),
            inputs=[
                self.icm_local_pos,
                self.icm_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                self.model.body_q,
            ],
            outputs=[self.icm_pos],
            device=device,
        )
        self.icm_bvh.build(self.icm_pos, self.icm_tri_indices, self.icm_query_radius)
        import atexit

        def _report_icm_totals(stats=self.icm_stats_total):
            # One line at process exit: crossings summed over the run and the
            # worst single substep. Zero here means the channel had nothing to
            # recover from, which is a result, not a silence.
            try:
                t = stats.numpy()
                print(
                    f"[collision] rigid untangling (ICM) totals: crossings={int(t[0])} "
                    f"max_per_substep={int(t[1])} max_candidates={int(t[2])} "
                    f"capacity_hits={int(t[3])}",
                    flush=True,
                )
            except Exception:
                pass

        atexit.register(_report_icm_totals)
        print(
            "[collision] rigid untangling (ICM): "
            f"{len(shape_ids)} shapes, {len(self.icm_local_pos)} vertices, "
            f"{self.icm_tri_indices.shape[0]} triangles, "
            f"stiff_factor={self.icm_stiff_factor:g}, "
            f"thickness={self.icm_thickness * 1000.0:.2f}mm, "
            f"query_radius={self.icm_query_radius * 1000.0:.2f}mm",
            flush=True,
        )

    def _update_rigid_untangling_query(
        self,
        cloth_pos: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
    ) -> None:
        """Refresh the per-cloth-edge candidate list of rigid triangles.

        Same shape as the cloth-cloth EF broad phase (``frame_begin``): the
        query boxes are the cloth EDGE bounds, the BVH holds the rigid
        triangles. All in-place, so it is safe inside a graph capture.
        """
        wp.launch(
            transform_rigid_feature_vertices_kernel,
            dim=len(self.icm_local_pos),
            inputs=[
                self.icm_local_pos,
                self.icm_vertex_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
            ],
            outputs=[self.icm_pos],
            device=self.model.device,
        )
        self.icm_bvh.refit(self.icm_pos, self.icm_tri_indices, self.icm_query_radius)
        self.edge_bvh.refit(cloth_pos, self.model.edge_indices, self.radius)
        self.icm_bvh.aabb_vs_aabb(
            self.edge_bvh.lower_bounds,
            self.edge_bvh.upper_bounds,
            self.icm_broad_phase,
            self.icm_query_radius,
            False,
        )

    def rebuild_bvh(self, pos: wp.array[wp.vec3]):
        """
        Rebuild triangle and edge BVHs.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.rebuild(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.rebuild(pos, self.model.edge_indices, self.radius)

    def refit_bvh(self, pos: wp.array[wp.vec3]):
        """
        Refit (update) triangle and edge BVHs based on new positions without changing topology.

        Args:
            pos: Array of vertex positions.
        """
        self.tri_bvh.refit(pos, self.model.tri_indices, self.radius)
        self.edge_bvh.refit(pos, self.model.edge_indices, self.radius)

    def frame_begin(self, particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3], dt: float):
        """
        Perform broad-phase collision detection using BVHs.

        Args:
            particle_q: Array of vertex positions.
            particle_qd: Array of vertex velocities.
            dt: simulation time step.
        """
        max_dist = self.radius * 3.0
        query_radius = self.radius

        self.refit_bvh(particle_q)
        if self.feature_vertex_shape is not None:
            self.feature_stats.zero_()

        # Vertex-face collision candidates
        if self.stiff_vf > 0.0:
            self.tri_bvh.triangle_vs_point(
                particle_q,
                particle_q,
                self.model.tri_indices,
                self.broad_phase_vf,
                True,
                max_dist,
                query_radius,
            )

        # Edge-edge collision candidates
        if self.stiff_ee > 0.0:
            self.edge_bvh.edge_vs_edge(
                particle_q,
                self.model.edge_indices,
                particle_q,
                self.model.edge_indices,
                self.broad_phase_ee,
                True,
                max_dist,
                query_radius,
            )

        # Face-edge collision candidates
        if self.stiff_ef > 0.0:
            self.tri_bvh.aabb_vs_aabb(
                self.edge_bvh.lower_bounds,
                self.edge_bvh.upper_bounds,
                self.broad_phase_ef,
                query_radius,
                False,
            )

    # ------------------------------------------------------------------ T14
    def _t14_launch_plan(self, n_pairs: int):
        """(dim, bp_mode) pairs the tri-SDF kernels are launched over.

        OFF: exactly one full launch, ``bp_mode`` unread (dead code in the
        kernel), i.e. the launch this stack always issued.

        ON: two launches with FIXED dims -- the candidate pass over the list's
        capacity, and a full-scan pass that every thread exits on its first
        statement unless the gather overflowed.  Two fixed launches rather than
        one sized launch is what keeps this CUDA-graph capturable: the
        alternative needs the candidate count on the host.
        """
        if not self.tri_sdf_bp_enabled:
            return ((n_pairs, 0),)
        if not self.tri_sdf_bp_fallback:
            # measurement knob only -- drops the overflow safety net
            return ((self.tri_sdf_bp_capacity, 1),)
        return ((self.tri_sdf_bp_capacity, 1), (n_pairs, 0))

    def _t14_par_search(self, particle_q, body_q, n_pairs: int):
        """T14 (iii): run the two search passes that feed the per-seed scratch."""
        with _t14_prof.section("tri_sdf_par_cull"):
            wp.launch(
                tri_sdf_par_cull_kernel,
                dim=n_pairs,
                inputs=[
                    particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    self.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    body_q,
                    self.tri_sdf_mesh_id,
                    self.tri_sdf_bg,
                    self.tri_sdf_h,
                    self.tri_sdf_gx,
                ],
                outputs=[self.tri_sdf_par_g_best, self.tri_sdf_par_g_n],
                device=self.model.device,
            )
        with _t14_prof.section("tri_sdf_par_seed"):
            wp.launch(
                tri_sdf_par_seed_kernel,
                dim=4 * n_pairs,
                inputs=[
                    particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    self.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    body_q,
                    self.tri_sdf_mesh_id,
                    self.tri_sdf_bg,
                    self.tri_sdf_h,
                    self.tri_sdf_refine,
                    self.tri_sdf_par_g_best,
                    self.tri_sdf_par_g_n,
                    self.tri_sdf_gx,
                ],
                outputs=[self.tri_sdf_par_best, self.tri_sdf_par_w, self.tri_sdf_par_n],
                device=self.model.device,
            )

    def _t14_broadphase(self, particle_q, body_q, n_pairs: int):
        """Rebuild the tri-SDF candidate list for this substep."""
        self.tri_sdf_cand_count.zero_()
        self.tri_sdf_bp_overflow.zero_()
        with _t14_prof.section("tri_sdf_broadphase"):
            wp.launch(
                tri_sdf_broadphase_kernel,
                dim=n_pairs,
                inputs=[
                    particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    self.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    body_q,
                    self.tri_sdf_bp_aabb_lo,
                    self.tri_sdf_bp_aabb_hi,
                    self.tri_sdf_h,
                    self.tri_sdf_bp_slack,
                    self.tri_sdf_bp_capacity,
                    self.tri_sdf_anchor_valid,
                    self.tri_sdf_anchor_kt_ratio,
                ],
                outputs=[
                    self.tri_sdf_cand_idx,
                    self.tri_sdf_cand_count,
                    self.tri_sdf_bp_overflow,
                ],
                device=self.model.device,
            )
        if self.tri_sdf_bp_selfcheck:
            # Verification only: re-runs the REAL centroid cull on all pairs and
            # counts the ones the gather dropped.  Host readback -- never on a
            # timing run, never inside a graph.
            self.tri_sdf_bp_stats.zero_()
            wp.launch(
                tri_sdf_bp_selfcheck_kernel,
                dim=n_pairs,
                inputs=[
                    particle_q,
                    self.model.tri_indices,
                    int(self.model.tri_count),
                    self.tri_sdf_slot_shape,
                    self.model.shape_body,
                    self.model.shape_transform,
                    body_q,
                    self.tri_sdf_bp_aabb_lo,
                    self.tri_sdf_bp_aabb_hi,
                    self.tri_sdf_h,
                    self.tri_sdf_bp_slack,
                    self.tri_sdf_mesh_id,
                    self.tri_sdf_bg,
                ],
                outputs=[self.tri_sdf_bp_stats],
                device=self.model.device,
            )
            st = self.tri_sdf_bp_stats.numpy()
            ov = int(self.tri_sdf_bp_overflow.numpy()[0])
            cnt = int(self.tri_sdf_cand_count.numpy()[0])
            self.tri_sdf_bp_stats_hist.append(
                (cnt, int(st[0]), int(st[1]), int(st[2]), int(st[3]), ov)
            )

    def accumulate_contact_force(
        self,
        dt: float,
        _iter: int,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        particle_forces: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        particle_stiff: wp.array[wp.vec3] = None,
    ):
        """
        Evaluates contact forces and the diagonal of the Hessian for implicit time integration.

        This method launches kernels to compute contact forces and Hessian contributions
        based on broad-phase collision candidates computed in frame_begin().

        Args:
            dt (float): Time step.
            state_in (State): Current simulation state (input).
            state_out (State): Next simulation state (output).
            contacts (Contacts): Contact data structure containing contact information.
            particle_forces (wp.array): Output array for computed contact forces.
            particle_q_prev (wp.array): Previous positions (optional, for velocity-based damping).
            particle_stiff (wp.array): Optional stiffness array for particles.
        """
        thickness = 2.0 * self.radius
        self.contact_hessian_diags.zero_()
        if self.stiff_vf > 0:
            wp.launch(
                handle_vertex_triangle_contacts_kernel,
                dim=len(state_in.particle_q),
                inputs=[
                    thickness,
                    self.stiff_vf,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.broad_phase_vf,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ee > 0:
            wp.launch(
                handle_edge_edge_contacts_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ee,
                    state_in.particle_q,
                    self.model.edge_indices,
                    self.broad_phase_ee,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.stiff_ef > 0:
            wp.launch(
                solve_untangling_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    thickness,
                    self.stiff_ef,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.broad_phase_ef,
                    particle_stiff,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.icm_tri_indices is not None:
            # R2-3 rigid untangling. Guarded on a Python attribute that is
            # ``None`` unless ``enable_rigid_untangling`` ran, so on the default
            # path nothing below is even queued -- and the branch is resolved at
            # capture time, so the captured graph stays fixed.
            icm_body_q = (
                state_out.body_q
                if self.integrate_with_external_rigid_solver
                else state_in.body_q
            )
            if _iter == 0:
                # The gripper pose is constant within a substep, so the
                # candidate list only has to be rebuilt once per substep; the
                # query box padding covers the cloth's motion across iterations.
                self.icm_stats.zero_()
                self._update_rigid_untangling_query(state_in.particle_q, icm_body_q)
            wp.launch(
                solve_rigid_untangling_kernel,
                dim=self.model.edge_indices.shape[0],
                inputs=[
                    self.icm_thickness,
                    self.icm_stiff_factor,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.icm_pos,
                    self.icm_tri_indices,
                    self.icm_broad_phase,
                    particle_stiff,
                    self.icm_capacity,
                    1 if _iter == 0 else 0,
                ],
                outputs=[
                    particle_forces,
                    self.contact_hessian_diags,
                    self.icm_stats,
                    self.icm_stats_total,
                ],
                device=self.model.device,
            )

        wp.launch(
            kernel=eval_body_contact_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                state_in.particle_q,
                # body-particle contact
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                self.shape_contact_offset,
                self.shape_hardening_eps,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.material_ke,
                self.particle_contact_ke,
                self.model.shape_material_mu,
                self.model.shape_body,
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                state_in.body_q if self.integrate_with_external_rigid_solver else None,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                self.anchor_kt_ratio,
                # The solver calls this once per iteration; advancing the anchor
                # every time would drag it 20x per substep and destroy the very
                # persistence it exists for.
                1 if _iter == 0 else 0,
                self.anchor_local,
                self.anchor_shape,
                self.avbd_dual_k,
                self.lambda_t,
                self.lambda_shape,
            ],
            outputs=[particle_forces, self.contact_hessian_diags],
            device=self.model.device,
        )

        if self.feature_vertex_shape is not None:
            body_q = (
                state_out.body_q
                if self.integrate_with_external_rigid_solver
                else state_in.body_q
            )
            body_q_prev = state_in.body_q
            if _iter == 0:
                self._update_rigid_feature_queries(
                    state_in.particle_q, body_q_prev, body_q)
            else:
                self._update_rigid_feature_edge_query(state_in.particle_q)
            wp.launch(
                eval_rigid_vertex_cloth_face_contacts_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.feature_vertex_shape,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    state_in.particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )
            wp.launch(
                eval_rigid_edge_cloth_edge_contacts_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.feature_edge_shape,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    state_in.particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                ],
                outputs=[particle_forces, self.contact_hessian_diags],
                device=self.model.device,
            )

        if self.tri_sdf_slot_shape is not None and self.tri_sdf_compliant:
            _tri_sdf_body_q = (
                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q
            )
            _n_pairs = self.tri_sdf_slots * int(self.model.tri_count)
            # T14 iteration stride: k = 1 is the launch this stack always issued
            # (straight into the RHS, no cache touched); k > 1 evaluates on every
            # k-th iteration into the cache and folds the cache in below.
            _k = self.tri_sdf_every
            _out_f = particle_forces
            _out_h = self.contact_hessian_diags
            _do_eval = True
            # T14 HOLD: iteration 0 searches and records the plane, the rest ride it.
            _hold_mode = 0
            if self.tri_sdf_hold:
                _hold_mode = 1 if _iter == 0 else 2
            if _k > 1 or self.tri_sdf_cache_always:
                _out_f = self.tri_sdf_cache_f
                _out_h = self.tri_sdf_cache_h
                if self.tri_sdf_every_tail == 2:
                    _do_eval = (_iter == 0) or ((_iter % _k) == (_k - 1))
                else:
                    _do_eval = (_iter % _k) == 0
                    if (self.tri_sdf_every_tail == 1
                            and _iter == self.nonlinear_iterations - 1):
                        _do_eval = True
                if _do_eval:
                    self.tri_sdf_cache_f.zero_()
                    self.tri_sdf_cache_h.zero_()
            if _do_eval:
                if self.tri_sdf_par and _hold_mode != 2:
                    self._t14_par_search(state_out.particle_q, _tri_sdf_body_q, _n_pairs)
                if self.tri_sdf_bp_enabled and _iter == 0:
                    # T14: rebuild the candidate list once per substep.  The blade
                    # pose is fixed inside a substep and the cloth's motion over the
                    # 20 iterations is covered by ``tri_sdf_bp_slack``, so the list
                    # built here is valid for every consumer of this substep --
                    # including the reaction pass, which runs after the solve.
                    self._t14_broadphase(state_out.particle_q, _tri_sdf_body_q, _n_pairs)
                for _bp_dim, _bp_mode in self._t14_launch_plan(_n_pairs):
                    with _t14_prof.section("tri_sdf_force"
                                          if _hold_mode != 2
                                          else "tri_sdf_force_held"):
                        wp.launch(
                            eval_tri_sdf_contact_kernel,
                            dim=_bp_dim,
                            inputs=[
                                state_out.particle_q,
                                self.model.tri_indices,
                                int(self.model.tri_count),
                                self.tri_sdf_stiffness,
                                self.tri_sdf_data,
                                self.tri_sdf_base,
                                self.tri_sdf_nx,
                                self.tri_sdf_ny,
                                self.tri_sdf_nz,
                                self.tri_sdf_origin,
                                self.tri_sdf_voxel,
                                self.tri_sdf_bg,
                                self.tri_sdf_slot_shape,
                                self.model.shape_body,
                                self.model.shape_transform,
                                state_out.body_q if self.integrate_with_external_rigid_solver else state_in.body_q,
                                self.tri_sdf_h,
                                self.tri_sdf_max_correction,
                                self.tri_sdf_refine,
                                self.shape_hardening_eps,
                                self.model.particle_radius,
                                # R13f friction inputs (inert unless R13F_SDF_FRICTION)
                                particle_q_prev,
                                state_in.body_q if self.integrate_with_external_rigid_solver else state_out.body_q,
                                self.model.shape_material_mu,
                                self.model.soft_contact_mu,
                                self.friction_epsilon,
                                dt,
                                # R13g frozen contact point (inert unless R13G_SDF_FREEZE)
                                self.tri_sdf_w,
                                (1 if _iter == 0 else 0)
                                if self.tri_sdf_resolve_every <= 0
                                else (1 if (_iter % self.tri_sdf_resolve_every) == 0 else 0),
                                # R16-A2' exact backend (inert unless R16_SDF_EXACT)
                                self.tri_sdf_mesh_id,
                                self.tri_sdf_mesh_max_dist,
                                # T8 anchor (inert unless tri_sdf_anchor_kt_ratio > 0)
                                self.tri_sdf_anchor_p,
                                self.tri_sdf_anchor_valid,
                                self.tri_sdf_anchor_w,
                                self.tri_sdf_anchor_kt_ratio,
                                # T13: seeding + return mapping on iteration 0, evaluated in
                                # the previous pose (= previous substep's converged state).
                                1 if _iter == 0 else 0,
                                self.tri_sdf_anchor_dbg,
                                self.tri_sdf_anchor_dbg2,
                                # T14 broad phase (inert unless T14_SDF_BROADPHASE)
                                self.tri_sdf_cand_idx,
                                self.tri_sdf_cand_count,
                                self.tri_sdf_bp_overflow,
                                _bp_mode,
                                # T14 round 4 (iii) per-seed scratch (inert unless T14_SDF_PAR)
                                self.tri_sdf_par_best,
                                self.tri_sdf_par_w,
                                self.tri_sdf_par_n,
                                # T14 round 3 hold (inert unless T14_SDF_HOLD)
                                self.tri_sdf_hold_valid,
                                self.tri_sdf_hold_w,
                                self.tri_sdf_hold_p,
                                self.tri_sdf_hold_n,
                                _hold_mode,
                                self.tri_sdf_hold_diag,
                                # T15-A nearest-triangle table (inert unless T15_SDF_GRIDEXACT)
                                self.tri_sdf_gx,
                                # T16 w/load diagnostic (inert unless T16_W_DIAG)
                                self.tri_sdf_w_diag,
                            ],
                            outputs=[_out_f, _out_h],
                            device=self.model.device,
                        )
            if _k > 1 or self.tri_sdf_cache_always:
                # every iteration, including the ones that skipped the search
                wp.launch(
                    add_tri_sdf_cache_kernel,
                    dim=int(self.model.particle_count),
                    inputs=[self.tri_sdf_cache_f, self.tri_sdf_cache_h],
                    outputs=[particle_forces, self.contact_hessian_diags],
                    device=self.model.device,
                )
            self.tri_sdf_w_valid = True
            # R13f: the reaction pass runs outside the solver loop and gets no
            # dt / previous state of its own.  Stash exactly what this pass used
            # so both halves of the contact see identical inputs; no public
            # signature changes, so the consumer repo is untouched.
            self._r13f_dt = dt
            self._r13f_particle_q_prev = particle_q_prev
            self._r13f_body_q_prev = (
                state_in.body_q if self.integrate_with_external_rigid_solver else state_out.body_q
            )


    def _t15_build_gridexact(self, device, verts_list, inds_list, meshes, pad, bake_max_dist, voxel):
        """T15-A: bake the nearest-triangle grid and the mesh pseudo-normals.

        Grid geometry mirrors the SDF bake line for line (same ``lo``, same
        ``n``, same padding), so the two tables index the same cells;
        ``T15_SDF_GX_VOXEL`` overrides the spacing for the accuracy/memory
        sweep and leaves the SDF bake alone.  Shapes whose (geometry, grid) key
        matches -- the four identical blades -- share ONE stored table.

        Returns a string for the backend self-certification print.
        """
        import os

        gx = SdfGridExact()

        def _dummy():
            gx.face = wp.zeros(1, dtype=wp.int32, device=device)
            gx.vnrm = wp.zeros(1, dtype=wp.vec3, device=device)
            gx.enrm = wp.zeros(1, dtype=wp.vec3, device=device)
            gx.fbase = wp.zeros(1, dtype=wp.int32, device=device)
            gx.vbase = wp.zeros(1, dtype=wp.int32, device=device)
            gx.ebase = wp.zeros(1, dtype=wp.int32, device=device)
            gx.nx = wp.zeros(1, dtype=wp.int32, device=device)
            gx.ny = wp.zeros(1, dtype=wp.int32, device=device)
            gx.nz = wp.zeros(1, dtype=wp.int32, device=device)
            gx.org = wp.zeros(1, dtype=wp.vec3, device=device)
            gx.inv_voxel = 1.0
            gx.ring_off = wp.zeros(2, dtype=wp.int32, device=device)
            gx.ring_idx = wp.zeros(1, dtype=wp.int32, device=device)
            gx.robase = wp.zeros(1, dtype=wp.int32, device=device)
            gx.ribase = wp.zeros(1, dtype=wp.int32, device=device)

        _tol0 = float(os.environ.get("T16_SDF_ARGMIN_TOL", "0") or 0.0)
        _soft0 = float(os.environ.get("T16_SDF_ARGMIN_SOFT", "0") or 0.0)
        if _soft0 != 0.0:
            # SOFT and TOL are mutually exclusive; SOFT wins.
            _tolnote = (
                f" + T16 ARGMIN_SOFT tau={_soft0:g} "
                f"(= {_soft0 * float(self.tri_sdf_h) * 1.0e6:g} um; softmin over the 4 seeds, "
                f"then refine FROM the blend)"
                + (" + SOFT_GATE (descent guard gn > tau*h)"
                   if os.environ.get("T16_SDF_SOFT_GATE", "0") not in ("", "0") else "")
                + (f" [TOL r={_tol0:g} IGNORED]" if _tol0 != 0.0 else "")
            )
        elif _tol0 != 0.0:
            _tolnote = f" + T16 ARGMIN_TOL r={_tol0:g} (= {_tol0 * float(self.tri_sdf_h) * 1000.0:g} mm)"
        else:
            _tolnote = ""
        if os.environ.get("T15_SDF_GRIDEXACT", "0") in ("", "0"):
            _dummy()
            self.tri_sdf_gx = gx
            return _tolnote

        # T15-A: the face table has its OWN spacing, fixed at 0.25 mm.  It is not
        # tied to the SDF bake's ``voxel`` (nor to a task yaml's tri_sdf_voxel):
        # the measured accuracy/speed optimum is 0.25 mm -- at 1 mm the 8-corner
        # candidate set misses the true nearest face 55% of the time on this
        # blade (p99 |dbest| 0.25 mm ~ the working penetration), and at 0.125 mm
        # the 332 MB table stops fitting in cache and the query gets SLOWER.
        # ``T15_SDF_GX_VOXEL`` still overrides, for the accuracy ladder.
        gx_voxel = float(os.environ.get("T15_SDF_GX_VOXEL", "") or _T15_GX_DEFAULT_VOXEL)
        face_blocks, vn_blocks, en_blocks = [], [], []
        ro_blocks, ri_blocks = [], []
        fbase, vbase, ebase, robase, ribase = [], [], [], [], []
        gnx, gny, gnz, gorg = [], [], [], []
        seen = {}
        ftot = vtot = etot = rotot = ritot = 0
        for vertices, indices, mesh in zip(verts_list, inds_list, meshes):
            lo = vertices.min(axis=0) - pad
            hi = vertices.max(axis=0) + pad
            n = np.maximum(np.ceil((hi - lo) / gx_voxel).astype(np.int32) + 1, 2)
            key = _t15_gx_key(vertices, indices, gx_voxel, pad, bake_max_dist, lo, n)
            if key not in seen:
                face, vn, en = _t15_gx_load(key)
                if face is None:
                    count = int(n[0]) * int(n[1]) * int(n[2])
                    _t0 = __import__("time").perf_counter()
                    d_face = wp.zeros(count, dtype=wp.int32, device=device)
                    wp.launch(
                        bake_shape_face_kernel,
                        dim=(int(n[0]), int(n[1]), int(n[2])),
                        inputs=[
                            mesh.id,
                            wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
                            float(gx_voxel),
                            int(n[0]),
                            int(n[1]),
                            int(n[2]),
                            0,
                            float(bake_max_dist),
                        ],
                        outputs=[d_face],
                        device=device,
                    )
                    wp.synchronize_device()
                    face = d_face.numpy()
                    del d_face
                    vn, en = _t15_pseudo_normals(vertices, indices)
                    print(
                        f"[collision] T15 gx bake: {count} cells "
                        f"({count * 4 / 1.0e6:.1f} MB) + {len(vn)} vertex / {len(en)} edge "
                        f"pseudo-normals in "
                        f"{__import__('time').perf_counter() - _t0:.2f} s",
                        flush=True,
                    )
                    _t15_gx_store(key, face, vn, en)
                else:
                    print(
                        f"[collision] T15 gx CACHE HIT: {len(face)} cells from {key[:12]}",
                        flush=True,
                    )
                r_off, r_idx = _t15_vertex_one_ring(vertices, indices)
                seen[key] = (ftot, vtot, etot, rotot, ritot)
                face_blocks.append(np.ascontiguousarray(face, dtype=np.int32))
                vn_blocks.append(np.ascontiguousarray(vn, dtype=np.float32))
                en_blocks.append(np.ascontiguousarray(en, dtype=np.float32))
                ro_blocks.append(np.ascontiguousarray(r_off, dtype=np.int32))
                ri_blocks.append(np.ascontiguousarray(r_idx, dtype=np.int32))
                ftot += len(face)
                vtot += len(vn)
                etot += len(en)
                rotot += len(r_off)
                ritot += len(r_idx)
            fb, vb, eb, rob, rib = seen[key]
            fbase.append(fb)
            vbase.append(vb)
            ebase.append(eb)
            robase.append(rob)
            ribase.append(rib)
            gnx.append(int(n[0]))
            gny.append(int(n[1]))
            gnz.append(int(n[2]))
            gorg.append(np.asarray(lo, dtype=np.float32))

        if not face_blocks:
            _dummy()
            self.tri_sdf_gx = gx
            return _tolnote

        gx.face = wp.array(np.concatenate(face_blocks), dtype=wp.int32, device=device)
        gx.vnrm = wp.array(np.concatenate(vn_blocks), dtype=wp.vec3, device=device)
        gx.enrm = wp.array(np.concatenate(en_blocks), dtype=wp.vec3, device=device)
        gx.fbase = wp.array(np.asarray(fbase, dtype=np.int32), dtype=wp.int32, device=device)
        gx.vbase = wp.array(np.asarray(vbase, dtype=np.int32), dtype=wp.int32, device=device)
        gx.ebase = wp.array(np.asarray(ebase, dtype=np.int32), dtype=wp.int32, device=device)
        gx.nx = wp.array(np.asarray(gnx, dtype=np.int32), dtype=wp.int32, device=device)
        gx.ny = wp.array(np.asarray(gny, dtype=np.int32), dtype=wp.int32, device=device)
        gx.nz = wp.array(np.asarray(gnz, dtype=np.int32), dtype=wp.int32, device=device)
        gx.org = wp.array(np.asarray(gorg, dtype=np.float32), dtype=wp.vec3, device=device)
        gx.inv_voxel = 1.0 / gx_voxel
        gx.ring_off = wp.array(np.concatenate(ro_blocks), dtype=wp.int32, device=device)
        gx.ring_idx = wp.array(np.concatenate(ri_blocks), dtype=wp.int32, device=device)
        gx.robase = wp.array(np.asarray(robase, dtype=np.int32), dtype=wp.int32, device=device)
        gx.ribase = wp.array(np.asarray(ribase, dtype=np.int32), dtype=wp.int32, device=device)
        self.tri_sdf_gx = gx
        self.tri_sdf_gx_voxel = gx_voxel
        _miss = int((np.concatenate(face_blocks) < 0).sum())
        _ring = os.environ.get("T15_SDF_GX_RING", "0") not in ("", "0")
        _rmean = (ritot / max(rotot - len(ro_blocks), 1)) if ritot else 0.0
        return (
            f" + T15 GRIDEXACT (8-corner face table, gx_voxel={gx_voxel * 1000:g} mm, "
            f"{len(face_blocks)} distinct table(s), {ftot} cells "
            f"[{ftot * 4 / 1.0e6:.2f} MB], {_miss} empty)"
            + (f" + RING (winner face's vertex one-ring, {_rmean:.1f} faces/face avg, "
               f"{ritot} entries [{ritot * 4 / 1.0e6:.2f} MB])" if _ring else "")
            + _tolnote
        )

    def read_w_diag(self, reset: bool = True):
        """T16 diagnostic accumulator as a dict; optionally zero it.

        Call it per frame (``reset=True``) to correlate the contact-point
        statistics with a per-frame quantity such as ``sign_ok``.  Layout is
        documented on ``_T16_W_DIAG`` in kernels.py.
        """
        d = self.tri_sdf_w_diag.numpy()
        n = max(float(d[0]), 1.0)
        out = {
            "pairs": float(d[0]),
            "centroid": float(d[1]), "vertex_w": float(d[2]), "other_w": float(d[3]),
            "mean_maxw": float(d[4]) / n, "mean_wdot": float(d[5]) / n,
            "sum_fn": float(d[6]), "max_fn": float(d[9]),
            "mean_depth_mm": float(d[10]) / n * 1000.0,
            "hist_maxw": [float(x) for x in d[12:22]],
            "feat_vertex": float(d[22]), "feat_edge": float(d[23]),
            "feat_face": float(d[24]), "feat_miss": float(d[25]),
        }
        if reset:
            self.tri_sdf_w_diag.zero_()
        return out

    def accumulate_tri_sdf_reaction(
        self,
        particle_q: wp.array[wp.vec3],
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
    ):
        """Equal-and-opposite wrench of the compliant triangle contact.

        Call AFTER :meth:`accumulate_body_reaction` (which zeroes ``body_f``) so
        both channels land in the same buffer.
        """
        if self.tri_sdf_slot_shape is None or not self.tri_sdf_compliant:
            return
        if self.tri_sdf_par and not self.tri_sdf_hold:
            self._t14_par_search(
                particle_q, body_q, self.tri_sdf_slots * int(self.model.tri_count)
            )
        for _bp_dim, _bp_mode in self._t14_launch_plan(self.tri_sdf_slots * int(self.model.tri_count)):
            with _t14_prof.section("tri_sdf_reaction"):
                wp.launch(
                    accumulate_tri_sdf_reaction_kernel,
                    dim=_bp_dim,
                    inputs=[
                        particle_q,
                        self.model.tri_indices,
                        int(self.model.tri_count),
                        self.tri_sdf_stiffness,
                        self.tri_sdf_data,
                        self.tri_sdf_base,
                        self.tri_sdf_nx,
                        self.tri_sdf_ny,
                        self.tri_sdf_nz,
                        self.tri_sdf_origin,
                        self.tri_sdf_voxel,
                        self.tri_sdf_bg,
                        self.tri_sdf_slot_shape,
                        self.model.shape_body,
                        self.model.shape_transform,
                        body_q,
                        body_com,
                        body_enabled,
                        self.tri_sdf_h,
                        self.tri_sdf_max_correction,
                        self.tri_sdf_refine,
                        self.shape_hardening_eps,
                        self.model.particle_radius,
                        # R13f friction inputs (inert unless R13F_SDF_FRICTION)
                        self._r13f_particle_q_prev if self._r13f_particle_q_prev is not None else particle_q,
                        self._r13f_body_q_prev if self._r13f_body_q_prev is not None else body_q,
                        self.model.shape_material_mu,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        float(self._r13f_dt),
                        # R13g: the point the force pass used (inert unless flag).
                        # Before the first force pass there is no such point yet, so the
                        # reaction resolves it itself (frame 0 would otherwise read the
                        # zero vector as a barycentric point and land inside the shape).
                        self.tri_sdf_w,
                        0 if self.tri_sdf_w_valid else 1,
                        # R16-A2' exact backend (inert unless R16_SDF_EXACT)
                        self.tri_sdf_mesh_id,
                        self.tri_sdf_mesh_max_dist,
                        # T8 anchor, read only (inert unless tri_sdf_anchor_kt_ratio > 0)
                        self.tri_sdf_anchor_p,
                        self.tri_sdf_anchor_valid,
                        self.tri_sdf_anchor_w,
                        self.tri_sdf_anchor_kt_ratio,
                        # T14 broad phase (inert unless T14_SDF_BROADPHASE)
                        self.tri_sdf_cand_idx,
                        self.tri_sdf_cand_count,
                        self.tri_sdf_bp_overflow,
                        _bp_mode,
                        # T14 round 4 (iii) per-seed scratch (inert unless T14_SDF_PAR)
                    self.tri_sdf_par_best,
                    self.tri_sdf_par_w,
                    self.tri_sdf_par_n,
                    # T14 round 3 hold (inert unless T14_SDF_HOLD): the reaction rides
                        # the SAME plane the substep's last force launch used.
                        self.tri_sdf_hold_valid,
                        self.tri_sdf_hold_w,
                        self.tri_sdf_hold_p,
                        self.tri_sdf_hold_n,
                        2 if self.tri_sdf_hold else 0,
                        # T15-A nearest-triangle table (inert unless T15_SDF_GRIDEXACT)
                        self.tri_sdf_gx,
                    ],
                    outputs=[body_f],
                    device=self.model.device,
                )

    def accumulate_body_reaction(
        self,
        dt: float,
        particle_q_prev: wp.array[wp.vec3],
        particle_q: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
        max_force: float = 0.0,
    ):
        """Accumulate cloth contact reaction wrenches onto rigid bodies.

        Evaluates the particle-body contact force once at the given (typically
        converged, post-step) particle positions and adds the equal-and-opposite
        wrench per body into ``body_f`` (world frame, about the body COM — the
        :attr:`newton.State.body_f` convention). Feed the result to an external
        rigid solver for two-way cloth-rigid coupling.

        Args:
            dt: Time step used for the contact damping/friction terms.
            particle_q_prev: Particle positions at the start of the step.
            particle_q: Particle positions after the cloth solve.
            contacts: Contacts filled by the collision pipeline for this step.
            body_q: Current body transforms.
            body_q_prev: Body transforms at the start of the step.
            body_enabled: Per-body int mask; only bodies with 1 receive forces.
            body_f: Output wrench accumulator (not zeroed here).
            max_force: If > 0, per-body wrenches are uniformly scaled so the
                linear force magnitude stays below this bound.
        """
        wp.launch(
            kernel=accumulate_body_reaction_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                particle_q_prev,
                particle_q,
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                self.model.particle_radius,
                self.shape_contact_offset,
                self.shape_hardening_eps,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                self.shape_contact_ke,
                self.material_ke,
                self.particle_contact_ke,
                self.model.shape_material_mu,
                self.model.shape_body,
                body_q,
                body_q_prev,
                self.model.body_qd,
                self.model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                self.anchor_kt_ratio,
                self.anchor_local,
                self.anchor_shape,
                self.avbd_dual_k,
                self.lambda_t,
                self.lambda_shape,
                body_enabled,
            ],
            outputs=[body_f],
            device=self.model.device,
        )
        if self.feature_vertex_shape is not None:
            self._update_rigid_feature_queries(particle_q, body_q_prev, body_q)
            wp.launch(
                accumulate_rigid_vertex_cloth_face_reaction_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_shape,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                    self.model.shape_body,
                    body_q,
                    self.model.body_com,
                    body_enabled,
                ],
                outputs=[body_f],
                device=self.model.device,
            )
            wp.launch(
                accumulate_rigid_edge_cloth_edge_reaction_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    dt,
                    self.feature_contact_radius,
                    self.shape_contact_ke,
                    self.model.soft_contact_kd,
                    particle_q_prev,
                    particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_shape,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                    self.model.shape_body,
                    body_q,
                    self.model.body_com,
                    body_enabled,
                ],
                outputs=[body_f],
                device=self.model.device,
            )
        if max_force > 0.0:
            wp.launch(
                kernel=clamp_body_wrench_kernel,
                dim=body_f.shape[0],
                inputs=[max_force],
                outputs=[body_f],
                device=self.model.device,
            )

    def enable_contact_projection(
        self,
        slack: float = 2.0e-3,
        iterations: int = 3,
        relaxation: float = 1.0,
        friction_scale: float = 0.0,
        interleaved: bool = False,
    ) -> None:
        """Enable the post-solve position-level non-penetration pass.

        After the implicit solve, cloth positions are projected so the
        penetration depth of particle-shape and rigid-feature contacts is
        bounded by ``slack``; the applied displacement is folded into the
        velocities. Penalty forces stay in charge of the reaction channel —
        evaluate reactions AFTER calling :meth:`project_contacts` so the
        residual (= slack) keeps feeding force back to two-way coupled bodies.

        Args:
            slack: Residual depth [m] left for the penalty force channel.
            iterations: Jacobi projection iterations per substep.
            relaxation: Under-relaxation factor for the averaged corrections.
            friction_scale: Multiplies ``shape_material_mu`` for the position-level
                Coulomb branch. 0 keeps the pass normal-only (previous behaviour).
            interleaved: Run one sweep inside each solver iteration
                (:meth:`project_contacts_iteration`) instead of a post-solve pass.
                This is the PBD arrangement: elasticity and contact alternate to a
                self-consistent state, rather than contact patching a converged
                elastic solve. The post-solve entry point becomes a no-op.
        """
        device = self.model.device
        self.projection_slack = max(float(slack), 0.0)
        self.projection_iterations = max(int(iterations), 0)
        self.projection_relaxation = float(relaxation)
        self.projection_friction_scale = max(float(friction_scale), 0.0)
        self.projection_interleaved = bool(interleaved)
        self.proj_delta = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_weight = wp.zeros(self.model.particle_count, dtype=float, device=device)
        self.proj_accum = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_accum_kept = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.proj_contact_share = wp.zeros(self.model.particle_count, dtype=float, device=device)
        # Positions as the solve left them, before the last sweep moved them.
        # With slack = 0 the projected state carries no penetration and so no
        # penalty reaction; this snapshot is where the reaction has to be read
        # instead (ke x the depth the solve settled at = the load the jaw is
        # actually pressing with).
        self.proj_pre_q = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        # Which shapes the projection acts on. It exists to fix gripper-cloth
        # penetration; applying it to the table as well makes the cloth resist
        # sliding as it spreads (measured: 0.22 of flatten coverage). Default
        # keeps every shape so behaviour is unchanged unless a caller narrows it.
        if getattr(self, "projection_shape_enabled", None) is None:
            self.projection_shape_enabled = wp.ones(
                self.model.shape_count, dtype=int, device=device
            )
        print(
            "[collision] contact projection: "
            f"slack={self.projection_slack:g} m, iterations={self.projection_iterations}, "
            f"friction_scale={self.projection_friction_scale:g}, "
            f"mode={'interleaved' if self.projection_interleaved else 'post'}",
            flush=True,
        )

    def enable_material_contact_stiffness(self, modulus: float, thickness: float) -> None:
        """Derive the contact stiffness from the cloth material instead of a constant.

        ``k_i = modulus * A_i / thickness`` with ``A_i`` the particle's tributary
        area on the rest mesh (each incident triangle contributes a third of its
        area). A per-shape constant ``ke`` is mesh-coupled: the penalty is
        per-particle, so refining the cloth 9k -> 40k multiplies the particle
        count by ~4 and the total normal load with it, and the constant has to be
        retuned per mesh. ``sum_i A_i`` is the cloth's area whatever the
        tessellation, so this form holds the load invariant under remeshing and
        leaves ONE parameter with physical units (a transverse compression
        modulus [Pa]) instead of a per-configuration knob.

        Args:
            modulus: E_t [Pa]. <= 0 disables and restores the per-shape constant.
            thickness: cloth thickness t0 [m] (asset meta ``thickness``).
        """
        modulus = float(modulus)
        if modulus <= 0.0 or thickness <= 0.0:
            self.material_ke = 0.0
            return
        pos = self.model.particle_q.numpy().astype(np.float64)
        tris = self.model.tri_indices.numpy().astype(np.int64)
        e1 = pos[tris[:, 1]] - pos[tris[:, 0]]
        e2 = pos[tris[:, 2]] - pos[tris[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        tributary = np.zeros(self.model.particle_count, dtype=np.float64)
        for corner in range(3):
            np.add.at(tributary, tris[:, corner], area / 3.0)
        ke = modulus * tributary / float(thickness)
        self.particle_contact_ke.assign(ke.astype(np.float32))
        self.material_ke = modulus
        nz = tributary[tributary > 0.0]
        print(
            "[collision] material contact stiffness: "
            f"E_t={modulus:g} Pa, t0={thickness * 1000:g} mm, "
            f"cloth area={tributary.sum():.5f} m^2, "
            f"A_i mean={nz.mean() * 1e6:.3f} mm^2 -> k_i mean={ke[tributary > 0.0].mean():.1f} N/m "
            f"(range {ke[tributary > 0.0].min():.1f}..{ke.max():.1f})",
            flush=True,
        )

    def enable_triangle_sdf_contacts(
        self,
        shape_ids,
        half_thickness: float,
        voxel: float = 1.0e-3,
        pad: float = 6.0e-3,
        refine_steps: int = 3,
        max_correction: float = 5.0e-3,
        compliant: bool = False,
        modulus: float = 0.0,
        thickness: float = 0.0,
        anchor_kt_ratio: float = 0.0,
    ) -> None:
        """Enable E3 triangle-level contact against a baked per-shape SDF.

        Vertex-sphere detection is blind to a blade crossing the INTERIOR of a
        cloth triangle, which is why ``particle_radius`` had to be inflated to
        the mesh aperture (8 mm for a 7.5 mm p99 aperture) -- a contact model
        coupled to the tessellation. This constrains the whole triangle instead:
        ``min_{x in tri} SDF_shape(x) >= half_thickness``. Nothing in the kernel
        reads ``particle_radius`` or an edge length, so the constraint is mesh
        independent.

        The SDF is baked once per shape (rigid bodies, so the local-frame field
        never changes) on a dense ``voxel`` grid padded by ``pad``; a 45x71x22 mm
        finger at 1 mm is ~0.2 M voxels, i.e. under a megabyte.

        Args:
            shape_ids: Shapes to constrain against (typically the gripper fingers).
            half_thickness: h [m] -- the cloth half-thickness the surface is held
                off the solid by. This is the ONLY length scale in the model.
            voxel: SDF grid spacing [m].
            pad: Grid padding around the shape bounds [m]; triangles further than
                this from the shape are rejected by one compare.
            refine_steps: Projected steepest-descent steps in barycentric space
                after the 3-vertex + centroid seeding.
            max_correction: Per-sweep displacement cap [m]. Bounds the response
                to a triangle that is already deep inside (e.g. right after a
                teleporting reset) instead of launching it.
            compliant: Run the constraint as a PENALTY FORCE into the implicit
                solve (and its equal-and-opposite onto the body) instead of as a
                position projection. The projection form removes exactly the
                overlap the penalty channel needs, so the two compete for the
                same quantity and the reaction collapses once the collision
                radius is retired; the compliant form has ONE mechanism carrying
                both the geometry and the force.
            modulus: E_t [Pa] for the compliant form. ``k_tri = E_t * A_tri / t0``
                with A_tri the REST triangle area, so sum_tri A_tri is the cloth's
                area regardless of tessellation and the load is mesh independent.
            thickness: t0 [m] for that stiffness -- the CLOTH's transverse
                thickness (a material property; the asset meta carries it), not
                the numerical shell ``half_thickness``. 0 keeps the legacy
                ``t0 = 2*half_thickness``, which is the same number whenever the
                shell IS the cloth's half-thickness (every run to date). Passing
                it explicitly is what keeps an h ladder from silently rescaling
                the stiffness by ``2h/t0`` when the shell is raised.
        """
        shape_ids = [int(s) for s in shape_ids]
        if not shape_ids:
            return
        device = self.model.device
        shape_scale = self.model.shape_scale.numpy()

        blocks, bases, dims, origins, slots, meshes = [], [], [], [], [], []
        # T15-A: the source geometry per slot, kept for the nearest-triangle bake
        gx_verts, gx_inds = [], []
        # T14: the blade's own bounds in the SHAPE-LOCAL frame -- the very frame
        # the contact kernel transforms the cloth triangles into.  Rigid shape, so
        # these never change and are computed once here.
        aabb_lo, aabb_hi = [], []
        total = 0
        # ``bg`` is the out-of-grid sentinel only. The bake itself queries out to
        # ``bake_max_dist`` so the SOLID INTERIOR carries a true negative
        # distance: querying only to the pad would leave anything deeper than the
        # pad reading +pad, i.e. a deeply penetrated triangle would look free and
        # the constraint would release it.
        bg = 1.0e3
        bake_max_dist = 0.05
        for shape in shape_ids:
            source = self.model.shape_source[shape]
            vertices = np.asarray(source.vertices, dtype=np.float32)
            indices = np.asarray(source.indices, dtype=np.int32).reshape(-1)
            if not len(vertices) or not len(indices):
                continue
            vertices = vertices * np.asarray(shape_scale[shape], dtype=np.float32)
            mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=device),
                indices=wp.array(indices, dtype=wp.int32, device=device),
            )
            aabb_lo.append(vertices.min(axis=0))
            aabb_hi.append(vertices.max(axis=0))
            lo = vertices.min(axis=0) - pad
            hi = vertices.max(axis=0) + pad
            n = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int32) + 1, 2)
            count = int(n[0]) * int(n[1]) * int(n[2])
            # T14 round 5: bake cache.  At 1 mm the bake is a rounding error, but
            # 0.25 mm is 41.8 M cells per finger and 0.125 mm is 334 M -- baked
            # from scratch on every process start that would dominate every
            # measurement in the campaign.  The key covers everything the field
            # depends on (geometry, scale, grid, query cutoff), so a hit is the
            # same array the bake would have produced; ``T14_SDF_BAKE_VERIFY=1``
            # re-bakes and asserts that bitwise.
            _ck = _t14_bake_key(vertices, indices, voxel, pad, bake_max_dist, lo, n)
            _cached = _t14_bake_load(_ck)
            if _cached is not None and not _t14_bake_verify():
                blocks.append(_cached)
                print(f"[collision] tri-SDF bake CACHE HIT shape {shape}: "
                      f"{count} cells ({count * 4 / 1.0e6:.1f} MB) from {_ck[:12]}",
                      flush=True)
            else:
                grid = wp.zeros(count, dtype=float, device=device)
                _t_bake = __import__("time").perf_counter()
                wp.launch(
                    bake_shape_sdf_kernel,
                    dim=(int(n[0]), int(n[1]), int(n[2])),
                    inputs=[
                        mesh.id,
                        wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
                        float(voxel),
                        int(n[0]),
                        int(n[1]),
                        int(n[2]),
                        0,
                        bake_max_dist,
                    ],
                    outputs=[grid],
                    device=device,
                )
                _b = grid.numpy()
                wp.synchronize_device()
                _dt_bake = __import__("time").perf_counter() - _t_bake
                print(f"[collision] tri-SDF bake shape {shape}: {count} cells "
                      f"({count * 4 / 1.0e6:.1f} MB) in {_dt_bake:.2f} s", flush=True)
                if _cached is not None:
                    same = bool(np.array_equal(_cached, _b))
                    print(f"[collision] tri-SDF bake CACHE VERIFY shape {shape}: "
                          f"bitwise identical = {same}", flush=True)
                    if not same:
                        raise RuntimeError("tri-SDF bake cache mismatch")
                else:
                    _t14_bake_store(_ck, _b)
                blocks.append(_b)
                del grid
            bases.append(total)
            dims.append(n)
            origins.append(lo)
            slots.append(shape)
            meshes.append(mesh)
            gx_verts.append(vertices)
            gx_inds.append(indices)
            total += count

        if not blocks:
            return
        dims = np.asarray(dims, dtype=np.int32)
        # R16-A2': keep the shape meshes alive (a wp.Mesh releases its BVH when
        # collected) and hand their ids to the kernels.  The exact backend
        # queries THESE -- the very meshes the bake above sampled -- so the two
        # backends differ by the discretisation and nothing else.
        self.tri_sdf_meshes = meshes
        self.tri_sdf_mesh_id = wp.array(
            np.asarray([m.id for m in meshes], dtype=np.uint64), dtype=wp.uint64, device=device
        )
        self.tri_sdf_mesh_max_dist = float(bake_max_dist)
        self.tri_sdf_data = wp.array(np.concatenate(blocks).astype(np.float32), dtype=float, device=device)
        self.tri_sdf_base = wp.array(np.asarray(bases, dtype=np.int32), dtype=int, device=device)
        self.tri_sdf_nx = wp.array(dims[:, 0].copy(), dtype=int, device=device)
        self.tri_sdf_ny = wp.array(dims[:, 1].copy(), dtype=int, device=device)
        self.tri_sdf_nz = wp.array(dims[:, 2].copy(), dtype=int, device=device)
        self.tri_sdf_origin = wp.array(np.asarray(origins, dtype=np.float32), dtype=wp.vec3, device=device)
        self.tri_sdf_slot_shape = wp.array(np.asarray(slots, dtype=np.int32), dtype=int, device=device)
        self.tri_sdf_voxel = float(voxel)
        self.tri_sdf_bg = bg
        self.tri_sdf_h = float(half_thickness)
        # T14 measurement knob: the refinement-step ladder is timed against the
        # SAME task yaml, so the count is overridable from the environment rather
        # than by editing a config.  Unset = whatever the caller passed.
        _t14_refine = __import__("os").environ.get("T14_SDF_REFINE")
        if _t14_refine is not None:
            refine_steps = int(_t14_refine)
        self.tri_sdf_refine = max(int(refine_steps), 0)
        self.tri_sdf_max_correction = float(max_correction)
        self.tri_sdf_slots = len(slots)
        self.tri_sdf_compliant = bool(compliant)
        _gx_note = self._t15_build_gridexact(
            device, gx_verts, gx_inds, meshes, float(pad), float(bake_max_dist), float(voxel)
        )
        _exact = int(__import__("os").environ.get("R16_SDF_EXACT", "0"))
        print(
            "[collision] tri-SDF query backend = "
            + ("EXACT mesh (analytic closest point, no grid)" if _exact else "VOXEL grid (trilinear)")
            + (" + PAR (4 seeds on 4 threads, chain 7 -> 1+refine)"
               if __import__("os").environ.get("T14_SDF_PAR", "0") not in ("", "0") else "")
            + ({1: " + HOLD=1 (plane held for the substep, no query in iters>=1)",
                2: " + HOLD=2 (bary point held, 1 query/iter instead of 7)"}
               .get(int(__import__("os").environ.get("T14_SDF_HOLD", "0") or 0), ""))
            + ({1: " + EDGE_EXACT=1 (exact normal at edge-flagged pairs)",
                2: " + EDGE_EXACT=2 (exact depth+normal at edge-flagged pairs)",
                3: " + EDGE_EXACT=3 (exact depth+normal at all near pairs)"}
               .get(int(__import__("os").environ.get("T14_SDF_EDGE_EXACT", "0") or 0), ""))
            + _gx_note
            + f": {len(slots)} shape(s), voxel={voxel * 1000:g} mm, h={half_thickness * 1000:g} mm, "
            + f"grid={total} cells ({total * 4 / 1.0e6:.2f} MB)",
            flush=True,
        )
        pos_rest = self.model.particle_q.numpy().astype(np.float64)
        tris = self.model.tri_indices.numpy().astype(np.int64)
        e1 = pos_rest[tris[:, 1]] - pos_rest[tris[:, 0]]
        e2 = pos_rest[tris[:, 2]] - pos_rest[tris[:, 0]]
        tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        t0 = float(thickness) if thickness > 0.0 else max(float(half_thickness) * 2.0, 1.0e-9)
        k_tri = (float(modulus) * tri_area / t0) if modulus > 0.0 else np.zeros_like(tri_area)
        self.tri_sdf_stiffness = wp.array(k_tri.astype(np.float32), dtype=float, device=device)
        # R13g: storage for the frozen barycentric contact point.  Always
        # allocated (the kernels take it as an argument either way); it is only
        # read/written when R13G_SDF_FREEZE is set, so the default path is
        # bit-identical and only pays one array allocation.
        self.tri_sdf_w = wp.zeros(
            self.tri_sdf_slots * int(self.model.tri_count), dtype=wp.vec3, device=device
        )
        self.tri_sdf_w_valid = False
        # T8: tangential anchor state, keyed by the same (slot, triangle) index
        # the kernels are launched over -- one anchor per contactING PAIR, which
        # is what "per-(triangle, shape)" means here.  Allocated unconditionally
        # for the same reason as tri_sdf_w: the kernels take it either way, and
        # with ratio 0 nothing reads or writes it.
        self.tri_sdf_anchor_kt_ratio = float(anchor_kt_ratio)
        self.tri_sdf_anchor_p = wp.zeros(
            self.tri_sdf_slots * int(self.model.tri_count), dtype=wp.vec3, device=device
        )
        self.tri_sdf_anchor_valid = wp.zeros(
            self.tri_sdf_slots * int(self.model.tri_count), dtype=wp.int32, device=device
        )
        # T12: the barycentric coordinates the anchor was seeded on, so the
        # tangential spring tracks one material point instead of the
        # closest-point search's per-iteration winner.
        self.tri_sdf_anchor_w = wp.zeros(
            self.tri_sdf_slots * int(self.model.tri_count), dtype=wp.vec3, device=device
        )
        # T12 验法探针：T12_ANCHOR_DBG=1 时按对写出诊断量，否则哑数组（内核跳过写）。
        import os as _os_t12
        _dbg_n = (
            self.tri_sdf_slots * int(self.model.tri_count)
            if _os_t12.environ.get("T12_ANCHOR_DBG") else 1
        )
        # --- T14 round 4 (iii): per-seed parallel search -----------------------
        # Scratch for 4 seeds per pair plus the shared centroid cull.  Allocated
        # unconditionally (the kernels take the arguments either way); with
        # T14_SDF_PAR unset nothing reads or writes them.
        self.tri_sdf_par = _os_t12.environ.get("T14_SDF_PAR", "0") not in ("", "0")
        _pn = self.tri_sdf_slots * int(self.model.tri_count)
        _pa = 4 * _pn if self.tri_sdf_par else 1
        self.tri_sdf_par_best = wp.zeros(_pa, dtype=float, device=device)
        self.tri_sdf_par_w = wp.zeros(_pa, dtype=wp.vec3, device=device)
        self.tri_sdf_par_n = wp.zeros(_pa, dtype=wp.vec3, device=device)
        self.tri_sdf_par_g_best = wp.zeros(_pn if self.tri_sdf_par else 1, dtype=float, device=device)
        self.tri_sdf_par_g_n = wp.zeros(_pn if self.tri_sdf_par else 1, dtype=wp.vec3, device=device)

        # --- T14 round 3: per-substep held contact geometry -------------------
        # ``T14_SDF_HOLD=1`` resolves the closest point + normal ONCE per substep
        # (iteration 0's existing search) and holds them in the blade's local
        # frame for the rest of the substep, so iterations 1..n-1 and the
        # reaction pass evaluate a plane distance instead of 7 serial mesh-BVH
        # queries.  Legitimate because the blade pose is constant inside a
        # substep and the cloth moves ~1 mm over it -- the same standing rule the
        # rigid-feature channel runs under.
        # 1 = plane hold (no mesh query at all in iterations >= 1)
        # 2 = fallback: hold only the barycentric point, one mesh query per
        #     iteration at that point instead of the 7-query search.
        self.tri_sdf_hold = int(_os_t12.environ.get("T14_SDF_HOLD", "0") or 0) > 0
        self.tri_sdf_hold_kind = int(_os_t12.environ.get("T14_SDF_HOLD", "0") or 0)
        _hn = self.tri_sdf_slots * int(self.model.tri_count)
        self.tri_sdf_hold_valid = wp.zeros(_hn, dtype=wp.int32, device=device)
        self.tri_sdf_hold_w = wp.zeros(_hn, dtype=wp.vec3, device=device)
        self.tri_sdf_hold_p = wp.zeros(_hn, dtype=wp.vec3, device=device)
        self.tri_sdf_hold_n = wp.zeros(_hn, dtype=wp.vec3, device=device)
        # T14 HOLD falsifier accumulator (T14_SDF_HOLD_DIAG=1): runs the full
        # search alongside every held evaluation and accumulates the error.
        # Diagnostic only -- the force still uses the held value.
        self.tri_sdf_hold_diag = wp.zeros(10, dtype=float, device=device)
        # T16 diagnostic accumulator (layout documented on _T16_W_DIAG).  Always
        # allocated (the kernel takes it either way); with T16_W_DIAG unset
        # nothing reads or writes it.
        self.tri_sdf_w_diag = wp.zeros(32, dtype=float, device=device)
        if _os_t12.environ.get("T16_W_DIAG", "0") not in ("", "0"):
            import atexit as _atexit

            def _dump_w_diag(_a=self.tri_sdf_w_diag):
                try:
                    d = _a.numpy()
                except Exception:  # noqa: BLE001
                    return
                n = max(d[0], 1.0)
                print(
                    "\n[T16 w-diag] pairs=%d  centroid=%d (%.2f%%)  vertex=%d (%.2f%%)  "
                    "edge/interior=%d (%.2f%%)"
                    % (d[0], d[1], 100 * d[1] / n, d[2], 100 * d[2] / n, d[3], 100 * d[3] / n),
                    flush=True,
                )
                print(
                    "[T16 w-diag] mean max(w)=%.6f  mean w.w=%.6f  (1/3 = 载荷均摊, 1 = 集中到单顶点)"
                    % (d[4] / n, d[5] / n),
                    flush=True,
                )
                fn = max(d[6], 1.0e-30)
                print(
                    "[T16 w-diag] sum f_n=%.4f N  载荷加权 max(w)=%.6f  载荷加权 w.w=%.6f  "
                    "max f_n=%.4f N  mean depth=%.6f mm  max max(w)=%.6f"
                    % (d[6], d[7] / fn, d[8] / fn, d[9], d[10] / n * 1000.0, d[11]),
                    flush=True,
                )
                print(
                    "[T16 w-diag] max(w) 直方图 [1/3..1] 10 格: "
                    + " ".join("%d" % x for x in d[12:22]),
                    flush=True,
                )
                nf = d[22] + d[23] + d[24] + d[25]
                if nf > 0:
                    print(
                        "[T16 w-diag] 接触点最近特征: 顶点=%d (%.2f%%)  棱=%d (%.2f%%)  "
                        "面内=%d (%.2f%%)  未命中=%d"
                        % (d[22], 100 * d[22] / nf, d[23], 100 * d[23] / nf,
                           d[24], 100 * d[24] / nf, d[25]),
                        flush=True,
                    )

            _atexit.register(_dump_w_diag)

        if self.tri_sdf_hold and not _exact:
            print(
                "[collision] WARNING: T14_SDF_HOLD needs the EXACT backend "
                "(R16_SDF_EXACT=1); the voxel path has no cached normal. HOLD is "
                "still applied but the held normal is whatever the exact branch "
                "would have produced -- do not trust this combination.",
                flush=True,
            )
        # --- T14 iteration stride --------------------------------------------
        # The expensive part of this channel is not the pair count, it is that
        # the closest-point search is redone on EVERY Newton iteration: 20 per
        # substep plus the reaction pass.  ``T14_SDF_EVERY = k`` evaluates it on
        # every k-th iteration into a per-particle cache and folds that cache
        # into the RHS on all of them, so the contact term is held rather than
        # dropped on the skipped iterations.  k = 1 (default) never touches the
        # cache and issues exactly the launch this stack always issued.
        #
        # k divides 0, so iteration 0 always evaluates -- which is where the T8
        # anchor is seeded and return-mapped, and where the broad phase (if on)
        # rebuilds its list.  Those states are therefore untouched by the
        # stride.  The reaction pass is outside the loop and still runs once per
        # substep.
        self.tri_sdf_every = max(1, int(_os_t12.environ.get("T14_SDF_EVERY", "1")))
        # Control knob: route k = 1 through the cache too.  Mathematically the
        # same as the direct path (evaluate every iteration, then add), so a run
        # with it on separates "the cache plumbing is wrong" from "lagging the
        # force by one iteration is what breaks the grasp".
        if _os_t12.environ.get("T14_SDF_CACHE_ALWAYS", "0") not in ("", "0"):
            self.tri_sdf_cache_always = True
        else:
            self.tri_sdf_cache_always = False
        # Which iterations of the stride actually evaluate.
        #   0 (default): _iter % k == 0  -> 0, k, 2k, ...  The LAST iteration of
        #     a substep is then a held one, and that is the iterate whose solve
        #     produces the positions the substep ends on.
        #   2: {0} U {i : i % k == k-1}.  Same count as mode 1 at k=2 but a
        #     different set (odd iterations); the only variant of the three that
        #     survived the lift gate at k=2, and only barely (see the commit).
        #   1: additionally force the LAST iteration of the substep, i.e. evaluate
        #     on {0, k, 2k, ...} U {n_iter-1}.  Iteration 0 is where the T8 anchor
        #     is seeded; iteration n_iter-1 is the solve that produces the
        #     positions the substep ends on.  Measured: with mode 0 the grasp is
        #     destroyed at k=2 (4 attempts, all FAIL, cloth never leaves the
        #     table); with mode 1 it holds at k=2.
        self.tri_sdf_every_tail = int(_os_t12.environ.get("T14_SDF_EVERY_TAIL", "0") or 0)
        self.tri_sdf_cache_f = wp.zeros(int(self.model.particle_count), dtype=wp.vec3, device=device)
        self.tri_sdf_cache_h = wp.zeros(int(self.model.particle_count), dtype=wp.mat33, device=device)
        if self.tri_sdf_every > 1:
            print(
                f"[collision] tri-SDF ITERATION STRIDE k={self.tri_sdf_every}: "
                f"force/Hessian re-evaluated on iterations 0, {self.tri_sdf_every}, "
                f"{2 * self.tri_sdf_every}, ... and held (cached) in between",
                flush=True,
            )
        # --- T14 broad phase -------------------------------------------------
        # Allocated unconditionally (the kernels take the arguments either way);
        # with T14_SDF_BROADPHASE unset nothing reads or writes them and the
        # launches keep their full slots*tri_count dim, so the OFF path is the
        # code it was plus one array allocation.
        _n_pairs = self.tri_sdf_slots * int(self.model.tri_count)
        self.tri_sdf_bp_enabled = _os_t12.environ.get("T14_SDF_BROADPHASE", "0") not in ("", "0")
        _cap = int(_os_t12.environ.get("T14_SDF_BP_CAPACITY", "16384"))
        self.tri_sdf_bp_capacity = max(1, min(_cap, _n_pairs))
        # Motion budget the list is held over: the gather runs once per substep
        # and the list is reused by the substep's 20 Newton iterations and the
        # reaction pass, over which the cloth keeps moving.  8 mm at 60 Hz /
        # 10 substeps is 4.8 m/s of cloth travel inside one substep.
        self.tri_sdf_bp_slack = float(_os_t12.environ.get("T14_SDF_BP_SLACK", "0.008"))
        # Measurement knob ONLY: 0 drops the overflow safety net so the candidate
        # pass can be timed on its own.  Never set it in a run whose physics you
        # intend to trust -- an overflowing list would then silently lose pairs.
        self.tri_sdf_bp_fallback = _os_t12.environ.get("T14_SDF_BP_FALLBACK", "1") not in ("", "0")
        self.tri_sdf_bp_aabb_lo = wp.array(
            np.asarray(aabb_lo, dtype=np.float32), dtype=wp.vec3, device=device
        )
        self.tri_sdf_bp_aabb_hi = wp.array(
            np.asarray(aabb_hi, dtype=np.float32), dtype=wp.vec3, device=device
        )
        self.tri_sdf_cand_idx = wp.zeros(self.tri_sdf_bp_capacity, dtype=wp.int32, device=device)
        self.tri_sdf_cand_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.tri_sdf_bp_overflow = wp.zeros(1, dtype=wp.int32, device=device)
        # T14 verification counters, host-read at the end of a run only:
        # [0] gathered  [1] cull-pass  [2] cull-pass AND NOT gathered  [3] in contact
        self.tri_sdf_bp_selfcheck = _os_t12.environ.get("T14_SDF_BP_SELFCHECK", "0") not in ("", "0")
        self.tri_sdf_bp_stats = wp.zeros(4, dtype=wp.int32, device=device)
        self.tri_sdf_bp_stats_hist = []
        if self.tri_sdf_bp_enabled:
            print(
                "[collision] tri-SDF BROAD PHASE on: "
                f"{_n_pairs} pairs -> candidate list capacity {self.tri_sdf_bp_capacity}, "
                f"slack={self.tri_sdf_bp_slack * 1000:g} mm, "
                f"selfcheck={'on' if self.tri_sdf_bp_selfcheck else 'off'}",
                flush=True,
            )
        self.tri_sdf_anchor_dbg = wp.zeros(_dbg_n, dtype=wp.vec4, device=device)
        # T13 逐对向量诊断：同一开关，形状 [n_pairs, 20]（关闭时 (1, 20) 哑数组）。
        self.tri_sdf_anchor_dbg2 = wp.zeros((_dbg_n, 20), dtype=float, device=device)
        if self.tri_sdf_anchor_kt_ratio > 0.0:
            print(
                "[collision] tri-SDF TRUE STATIC friction (T8 anchor, T13 map@prev-pose+blend-tangent): "
                f"kt_ratio={self.tri_sdf_anchor_kt_ratio:g} "
                f"(kt = kt_ratio * k_tri; predicted pre-slip at mu=0.7, depth=0.3mm: "
                f"{0.7 * 0.3 / self.tri_sdf_anchor_kt_ratio * 1000:.0f} um)",
                flush=True,
            )
        if self.tri_sdf_compliant:
            print(
                "[collision] triangle SDF contact is COMPLIANT: "
                f"E_t={modulus:g} Pa, t0={t0 * 1000:g} mm, "
                f"cloth area={tri_area.sum():.5f} m^2, "
                f"k_tri mean={k_tri.mean():.1f} N/m (range {k_tri.min():.1f}..{k_tri.max():.1f})",
                flush=True,
            )
        # Jacobi accumulators. The position-projection pass owns the same three
        # buffers; allocate only if it did not (the two are independent switches).
        if getattr(self, "proj_delta", None) is None:
            self.proj_delta = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
            self.proj_weight = wp.zeros(self.model.particle_count, dtype=float, device=device)
            self.proj_accum = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        if getattr(self, "projection_relaxation", None) is None:
            self.projection_relaxation = 1.0
        print(
            "[collision] triangle SDF contacts: "
            f"{len(slots)} shapes, {total} voxels @ {voxel * 1000:g} mm, "
            f"h={half_thickness * 1000:g} mm, refine={self.tri_sdf_refine}, "
            f"max_corr={max_correction * 1000:g} mm",
            flush=True,
        )

    def _tri_sdf_sweep(self, particle_q: wp.array[wp.vec3], body_q: wp.array[wp.transform]):
        """One Jacobi sweep of the triangle-level SDF constraint."""
        wp.launch(
            project_tri_sdf_kernel,
            dim=self.tri_sdf_slots * self.model.tri_count,
            inputs=[
                particle_q,
                self.model.tri_indices,
                int(self.model.tri_count),
                self.tri_sdf_data,
                self.tri_sdf_base,
                self.tri_sdf_nx,
                self.tri_sdf_ny,
                self.tri_sdf_nz,
                self.tri_sdf_origin,
                self.tri_sdf_voxel,
                self.tri_sdf_bg,
                self.tri_sdf_slot_shape,
                self.model.shape_body,
                self.model.shape_transform,
                body_q,
                self.tri_sdf_h,
                self.tri_sdf_max_correction,
                self.tri_sdf_refine,
            ],
            outputs=[self.proj_delta, self.proj_weight],
            device=self.model.device,
        )
        wp.launch(
            apply_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[self.projection_relaxation, self.model.particle_flags],
            outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
            device=self.model.device,
        )

    def set_projection_shapes(self, shape_indices) -> None:
        """Restrict the position projection to the given shapes.

        Args:
            shape_indices: Shapes the projection may move particles out of.
                Everything else keeps only its penalty contact.
        """
        import numpy as np

        mask = np.zeros(self.model.shape_count, dtype=np.int32)
        idx = np.asarray(list(shape_indices), dtype=np.int32)
        if len(idx):
            mask[idx] = 1
        self.projection_shape_enabled = wp.array(mask, dtype=int, device=self.model.device)
        print(f"[collision] contact projection limited to {int(mask.sum())} shapes", flush=True)

    def accumulate_projection_impulse(
        self,
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_com: wp.array[wp.vec3],
        body_enabled: wp.array[int],
        body_f: wp.array[wp.spatial_vector],
        dt: float,
        max_force: float = 0.0,
    ):
        """Feed the last projection pass back to the rigid side as an impulse.

        Call AFTER the reaction accumulation (which zeroes ``body_f``). With this
        the rigid side feels the position solve directly, so the projection no
        longer has to leave residual penetration behind just to keep a penalty
        force alive -- see :meth:`enable_contact_projection`.
        """
        if self.projection_iterations <= 0:
            return
        self.proj_contact_share.zero_()
        wp.launch(
            count_particle_contacts_kernel,
            dim=self.body_contact_max,
            inputs=[
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
            ],
            outputs=[self.proj_contact_share],
            device=self.model.device,
        )
        wp.launch(
            accumulate_projection_impulse_kernel,
            dim=self.body_contact_max,
            inputs=[
                dt,
                self.proj_accum_kept,
                self.model.particle_mass,
                self.proj_contact_share,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.model.shape_body,
                body_q,
                body_com,
                body_enabled,
            ],
            outputs=[body_f],
            device=self.model.device,
        )
        # Re-clamp: the reaction pass already clamped its own contribution, and
        # a projection impulse spike must not slip past that guard.
        if max_force > 0.0:
            wp.launch(
                clamp_body_wrench_kernel,
                dim=len(body_f),
                inputs=[max_force],
                outputs=[body_f],
                device=self.model.device,
            )

    def project_contacts_iteration(
        self,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        dt: float,
        body_q_prev: wp.array[wp.transform] | None = None,
    ):
        """One projection sweep, to be called INSIDE the solver's iteration loop.

        Same three kernels as :meth:`project_contacts`, run once instead of
        ``projection_iterations`` times: the solver's own loop supplies the
        iteration count, so elasticity and contact alternate the way a PBD
        solver does. Velocities are NOT touched here -- the solver derives the
        whole step's velocity from ``(x_out - x_prev)/dt`` after the loop, which
        already contains the projected displacement.

        No-op unless :meth:`enable_contact_projection` was called with
        ``interleaved=True``.
        """
        if self.projection_interleaved and self.projection_iterations > 0:
            self._project_sweep(particle_q, particle_q_prev, contacts, body_q, body_q_prev)
        # E3 triangle-level SDF constraint: an independent switch, so it can run
        # on the simplified stack (position projection off, E0c) without
        # dragging the particle-radius projection back in.
        if self.tri_sdf_slot_shape is not None and not self.tri_sdf_compliant:
            # compliant mode carries the constraint as a force instead (see
            # accumulate_contact_force), so there is no position sweep here.
            self._tri_sdf_sweep(particle_q, body_q)

    def _project_sweep(
        self,
        particle_q: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        body_q_prev: wp.array[wp.transform] | None,
    ):
        """One Jacobi sweep: accumulate corrections from all contact types, apply."""
        self.proj_pre_q.assign(particle_q)
        wp.launch(
            project_body_particle_contacts_kernel,
            dim=self.body_contact_max,
            inputs=[
                self.projection_slack,
                self.projection_friction_scale,
                self.model.soft_contact_mu,
                particle_q,
                particle_q_prev,
                self.proj_accum,
                self.model.particle_radius,
                contacts.soft_contact_particle,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.model.shape_body,
                self.model.shape_material_mu,
                self.projection_shape_enabled,
                body_q,
                body_q if body_q_prev is None else body_q_prev,
            ],
            outputs=[self.proj_delta, self.proj_weight],
            device=self.model.device,
        )
        if self.feature_vertex_shape is not None:
            wp.launch(
                project_rigid_vertex_cloth_face_kernel,
                dim=len(self.feature_local_pos),
                inputs=[
                    self.projection_slack,
                    self.feature_contact_radius,
                    particle_q_prev,
                    particle_q,
                    self.model.tri_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_vertex_weight,
                    self.feature_broad_phase_vf,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
            wp.launch(
                project_rigid_edge_cloth_edge_kernel,
                dim=len(self.feature_edge_indices),
                inputs=[
                    self.projection_slack,
                    self.feature_contact_radius,
                    particle_q_prev,
                    particle_q,
                    self.model.edge_indices,
                    self.feature_pos_prev,
                    self.feature_pos,
                    self.feature_edge_indices,
                    self.feature_edge_weight,
                    self.feature_broad_phase_ee,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
        wp.launch(
            apply_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.projection_relaxation,
                self.model.particle_flags,
            ],
            outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
            device=self.model.device,
        )

    def project_contacts(
        self,
        particle_q: wp.array[wp.vec3],
        particle_qd: wp.array[wp.vec3],
        particle_q_prev: wp.array[wp.vec3],
        contacts: Contacts,
        body_q: wp.array[wp.transform],
        dt: float,
        body_q_prev: wp.array[wp.transform] | None = None,
    ):
        """Run the position-level projection pass (see :meth:`enable_contact_projection`).

        Call after the cloth solve and before reaction accumulation. Reuses
        the rigid-feature broad-phase candidates from the solve iterations
        (positions drift sub-millimetre per substep, covered by the search
        margin).

        Args:
            particle_q: Converged particle positions (modified in place).
            particle_qd: Particle velocities (corrected in place).
            particle_q_prev: Particle positions at the start of the substep.
            contacts: Contacts filled by the collision pipeline for this step.
            body_q: Current body transforms (the pose the cloth solved against).
            dt: Substep time step.
        """
        if self.projection_iterations <= 0 or self.projection_interleaved:
            return
        for _ in range(self.projection_iterations):
            wp.launch(
                project_body_particle_contacts_kernel,
                dim=self.body_contact_max,
                inputs=[
                    self.projection_slack,
                    self.projection_friction_scale,
                    self.model.soft_contact_mu,
                    particle_q,
                    particle_q_prev,
                    self.proj_accum,
                    self.model.particle_radius,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_count,
                    contacts.soft_contact_max,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    self.model.shape_body,
                    self.model.shape_material_mu,
                    self.projection_shape_enabled,
                    body_q,
                    body_q if body_q_prev is None else body_q_prev,
                ],
                outputs=[self.proj_delta, self.proj_weight],
                device=self.model.device,
            )
            if self.feature_vertex_shape is not None:
                wp.launch(
                    project_rigid_vertex_cloth_face_kernel,
                    dim=len(self.feature_local_pos),
                    inputs=[
                        self.projection_slack,
                        self.feature_contact_radius,
                        particle_q_prev,
                        particle_q,
                        self.model.tri_indices,
                        self.feature_pos_prev,
                        self.feature_pos,
                        self.feature_vertex_weight,
                        self.feature_broad_phase_vf,
                    ],
                    outputs=[self.proj_delta, self.proj_weight],
                    device=self.model.device,
                )
                wp.launch(
                    project_rigid_edge_cloth_edge_kernel,
                    dim=len(self.feature_edge_indices),
                    inputs=[
                        self.projection_slack,
                        self.feature_contact_radius,
                        particle_q_prev,
                        particle_q,
                        self.model.edge_indices,
                        self.feature_pos_prev,
                        self.feature_pos,
                        self.feature_edge_indices,
                        self.feature_edge_weight,
                        self.feature_broad_phase_ee,
                    ],
                    outputs=[self.proj_delta, self.proj_weight],
                    device=self.model.device,
                )
            wp.launch(
                apply_contact_projection_kernel,
                dim=self.model.particle_count,
                inputs=[
                    self.projection_relaxation,
                    self.model.particle_flags,
                ],
                outputs=[self.proj_delta, self.proj_weight, particle_q, self.proj_accum],
                device=self.model.device,
            )
        wp.launch(
            finalize_contact_projection_kernel,
            dim=self.model.particle_count,
            inputs=[dt],
            outputs=[self.proj_accum, particle_qd, self.proj_accum_kept],
            device=self.model.device,
        )

    def contact_hessian_diagonal(self):
        """Return diagonal of contact Hessian for preconditioning.
        Note:
            Should be called after `accumulate_contact_force()`.
        """
        return self.contact_hessian_diags

    def hessian_multiply(self, x: wp.array[wp.vec3]):
        """Computes the Hessian-vector product for implicit integration."""
        wp.launch(
            hessian_multiply_kernel,
            dim=self.model.particle_count,
            inputs=[self.contact_hessian_diags, x],
            outputs=[self.Hx],
            device=self.model.device,
        )
        return self.Hx

    def linear_iteration_end(self, dx: wp.array[wp.vec3]):
        """No post-solve contact projection; contacts live in the Newton solve."""
        pass

    def frame_end(self, pos: wp.array[wp.vec3], vel: wp.array[wp.vec3], dt: float):
        """Apply post-processing"""
