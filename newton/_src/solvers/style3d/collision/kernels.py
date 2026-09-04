# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    HARDENING_DEN_MIN,
    _eval_body_particle_contact_banded,
    compute_projected_isotropic_friction,
)

# R13f: Coulomb friction on the COMPLIANT tri-SDF channel.  The compliant SDF
# force law is purely normal (F = k_tri*(h-SDF) along grad SDF), and the 1b/1c
# stacks zero the vertex penalty channel (shape_contact_ke -> 0) where all of
# the R2-R11 tangential physics lives, so those stacks run frictionless at the
# gripper.  This flag ports the SAME law the vertex channel uses -- the
# IPC-mollified isotropic Coulomb force of
# ``compute_projected_isotropic_friction(mu, f_n, n, u_rel, eps_u)`` with the
# same ``mu = sqrt(soft_contact_mu * shape_material_mu[shape])`` mixing -- onto
# the triangle contact point.  Nothing new is invented.
# 0 = OFF and the kernels run the pre-R13f statements untouched (bit-identical).
_R13F_SDF_FRICTION = wp.constant(int(__import__("os").environ.get("R13F_SDF_FRICTION", "0")))

# R13g: LUMPED Hessian for the compliant tri-SDF contact.  The triangle contact
# is one constraint acting at a barycentric point, so its exact 9x9 Hessian is
# the rank-1 block ``k (w (x) n)(w (x) n)^T``; its diagonal blocks are
# ``w_i^2 k nn``.  The solver stores contacts as a PER-PARTICLE diagonal only
# (``contact_hessian_diags`` feeds both the Jacobi preconditioner AND
# ``hessian_multiply``), so the off-diagonal ``w_i w_j`` coupling is dropped and
# the operator under-counts the stiffness of the mode that actually carries the
# load -- the three vertices moving together along n.  That mode's true
# stiffness is ``k (sum_i w_i)^2 = k``; the stored diagonal gives ``k sum_i
# w_i^2``, which is 3x too soft at the centroid.  Row-sum (mass-lumping)
# consistency uses ``sum_j H_ij = k w_i nn`` instead, which reproduces the
# collective stiffness exactly and never under-counts.  Measured effect: with
# w_i^2 the normal solve stalls at a permanent residual (sum of rhs_z = +0.145 x
# cloth weight at E_t=6500 Pa, invariant under 20/40/60 nonlinear and 10/40
# linear iterations), so the cloth sits 14.5% deeper than equilibrium and every
# f(depth) rides on an inflated normal load.  0 = OFF, statements untouched.
_R13G_SDF_HESS = wp.constant(int(__import__("os").environ.get("R13G_SDF_HESS", "0")))

# R13g: FREEZE the compliant tri-SDF contact point inside a substep.
# ``tri_sdf_closest`` re-runs a DISCRETE minimum search (centroid + 3 vertex
# seeds, then projected steepest descent) on every Newton iteration, and the
# whole normal load of the triangle is applied at whichever single point wins.
# On a nominally flat contact the winner is decided by micro-noise, so the load
# is winner-take-all: a vertex that wins all six of its triangles receives ~34x
# its own weight, gets pushed off, and the next iteration hands the load to a
# different vertex.  The Newton iteration therefore never converges -- measured
# per-particle |residual| stays at 6.3e-5 N and |dx| at 7e-8 m for 200
# iterations -- and because ``best`` is a MIN over the re-drawn candidates the
# stall is biased deep: the contact delivers 1.145x the cloth weight at rest and
# 2.4-3.3x once the cloth slides.  Every f(depth) rides on that, friction
# (mu*f_n) included.
# The remedy is the treatment this file already uses for every other channel --
# resolve the contact geometry ONCE per substep and hold it for the iterations
# (collision.py uses ``_iter == 0`` for the rigid feature queries, the friction
# anchor advance and the untangling candidate list).  The force law is
# untouched: same k_tri*(h - SDF) along grad SDF, same point, only the
# barycentric location stops being re-searched mid-solve.
# 0 = OFF: ``tri_sdf_closest`` runs exactly as before, bit-identical.
_R13G_SDF_FREEZE = wp.constant(int(__import__("os").environ.get("R13G_SDF_FREEZE", "0")))

# R16-A2': EXACT query backend for the compliant tri-SDF contact.
# The contact geometry is queried from a 1 mm voxel bake read back with
# trilinear interpolation.  The measured working regime is ENTIRELY SUB-VOXEL --
# the cloth's median penetration into the blade is 0.193 mm, i.e. a fifth of a
# cell -- and trilinear interpolation rounds every edge and corner over a
# cell-sized neighbourhood.  That is the textbook signature the root-cause chain
# read off the field: the force core under-states the intrusion by 3.4-4x and
# mis-points the gradient by 1.9x, and it does so exactly where the cloth enters
# (side walls 91.3% + blade tip 6.6%, the edge-dense regions), while missing
# nothing (0/240000 misses).  So the defect is in the QUERY, not in the penalty
# formulation.  This backend evaluates distance and gradient against the shape's
# own triangle mesh -- the same closed mesh the bake itself queried -- so both
# are exact to machine precision, with no grid, no interpolation and no
# discretisation memory.  The force law, the hardening law, the friction law and
# the search algorithm are untouched: ONLY the field evaluation changes.
# 0 = OFF: the voxel statements run exactly as before.
_R16_SDF_EXACT = wp.constant(int(__import__("os").environ.get("R16_SDF_EXACT", "0")))


@wp.func
def triangle_normal(A: wp.vec3, B: wp.vec3, C: wp.vec3):
    n = wp.cross(B - A, C - A)
    ln = wp.length(n)
    return wp.vec3(0.0) if ln < 1.0e-12 else (n / ln)


@wp.func
def triangle_barycentric(A: wp.vec3, B: wp.vec3, C: wp.vec3, P: wp.vec3):
    v0 = A - C
    v1 = B - C
    v2 = P - C
    dot00 = wp.dot(v0, v0)
    dot01 = wp.dot(v0, v1)
    dot02 = wp.dot(v0, v2)
    dot11 = wp.dot(v1, v1)
    dot12 = wp.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    invDenom = 0.0 if wp.abs(denom) < 1.0e-12 else 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return wp.vec3(u, v, 1.0 - u - v)


@wp.func
def particle_contact_stiffness(
    material_ke: float,
    particle_contact_ke: wp.array[float],
    shape_contact_ke: wp.array[float],
    particle_idx: int,
    shape_idx: int,
):
    """Contact stiffness for one particle-shape pair.

    ``material_ke <= 0`` keeps the per-shape constant (stock behaviour, and the
    multiplication-free path, so the default is bit-identical). Above zero the
    stiffness comes from the CLOTH MATERIAL instead:

        k_i = E_t * A_i / t0

    with ``A_i`` the particle's tributary area on the rest mesh, ``t0`` the cloth
    thickness and ``E_t`` a transverse compression modulus [Pa]. A per-shape
    constant is mesh-coupled -- refining 9k -> 40k multiplies the particle count
    by ~4 and so multiplies the total normal load by ~4 -- while sum_i A_i is the
    cloth's area no matter how it is tessellated, so this form keeps the load
    invariant under remeshing. The gate is a uniform scalar, so there is no
    divergence and the branch is resolved identically for every thread.
    """
    if material_ke > 0.0:
        # a shape whose constant was zeroed has the stock per-particle penalty
        # switched OFF (e.g. the gripper, once the compliant triangle contact
        # carries that load); the material branch must honour it too, or the two
        # channels double-count.
        if shape_contact_ke[shape_idx] <= 0.0:
            return 0.0
        return particle_contact_ke[particle_idx]
    return shape_contact_ke[shape_idx]


@wp.func
def contact_band_radius(
    particle_radius: wp.array[float],
    shape_contact_offset: wp.array[float],
    particle_idx: int,
    shape_idx: int,
):
    """Range band of the particle-shape penalty ``f = ke * (band - d)`` [m].

    E4.  The stock law reads ``particle_radius``, which is simultaneously the
    geometric probe radius (how far the cloth's collision shell sticks out) and
    the force band.  Retiring the shell to the cloth's physical half-thickness
    therefore also collapses the normal load and, with it, mu*N -- measured
    15.4 N (r=8 mm) -> 8.5 N (4 mm) -> 0.8 N (0.5 mm), doc GRIPPER-CONTACT 5(25).

    ``shape_contact_offset[shape] > 0`` overrides the band for that shape only
    (PhysX ``contact_offset``); the geometric stop stays whatever the geometry
    channel says (``particle_radius`` for the vertex projection, the triangle
    constraint's half-thickness for the SDF one).  The default array is all
    zeros, so every shape takes the stock branch and the default path is
    bit-identical.  The value is a per-shape scalar, so the branch is uniform
    across the warp.
    """
    band = shape_contact_offset[shape_idx]
    if band > 0.0:
        return band
    return particle_radius[particle_idx]


@wp.func
def contact_hardening_inv_dmax(
    shape_hardening_eps: wp.array[float],
    t0: float,
    shape_idx: int,
):
    """``1/d_max`` for the hardening compression law, per shape.  0 = off.

    ``p(eps) = k0*eps/(1 - eps/eps_max)`` (doc GRIPPER-CONTACT 2.2) is a
    TRANSVERSE COMPRESSION law for the fabric, so its reference length is the
    cloth's own thickness ``t0``, not the penalty band:

        eps = delta / t0,   d_max = eps_max * t0,   eps_max in (0, 1)

    R13 (2026-08-28) corrects R9 here.  R9 used the band as the compressible
    layer, on the stated premise that "the geometric stop sits at the cloth's
    physical half-thickness and everything between it and the band is the
    padding the force law is allowed to squeeze".  R12 measured that premise
    false: with a 6 mm band the closest cloth particle during a HOLD sits
    2.02 mm from the blade and every one of the 112 loaded contacts lies
    3.75-6.00 mm out, i.e. the gripper never reaches the cloth at all and the
    real compression is zero.  Referencing the strain to the band therefore
    put the operating point at 39% of d_max (hardening factor 1.65, law
    effectively dormant) while a 5.26 mm "graze" contact sat PAST d_max and
    took the 20x cap.  Both are artefacts of the wrong reference length.

    ``t0`` comes from the cloth itself (2 x particle_radius; the asset meta
    carries ``thickness: 1e-3`` for blue_tshirt_9k), so the law is mesh- and
    band-independent and the asymptote lands where de Jong's incompressible
    core does: single-layer compression <= 0.32*t0, two layers in the jaw
    <= 0.6 mm (archive cloth-pinch-contact-research.md).

    PER SHAPE, not global: the table is a half-space the cloth is meant to lie
    on, and an asymptote there would turn resting on the table into a wall.
    The default array is all zeros, so every shape returns 0.0 and the force
    law takes its stock linear branch -- bit-identical default path.  The value
    is a per-shape scalar, so the branch is uniform across the warp and there
    is no host-side decision (CUDA graph safe).
    """
    eps_max = shape_hardening_eps[shape_idx]
    if eps_max > 0.0 and t0 > 0.0:
        return 1.0 / (eps_max * t0)
    return 0.0


@wp.func
def evaluate_body_particle_contact_banded(
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    body_particle_contact_ke: float,
    body_particle_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    contact_radius: float,
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    dt: float,
    hardening_inv_dmax: float,
):
    """``evaluate_body_particle_contact`` with an explicit force band.

    Same per-shape mu mixing; the band comes in as a scalar instead of being
    read from ``particle_radius``.  Friction rides on ``f_n = ke * depth`` inside
    the stock law, so widening the band widens the friction band with it -- N and
    mu*N act over the same interval by construction.
    """
    shape_index = contact_shape[contact_index]
    mixed_mu = wp.sqrt(friction_mu * shape_material_mu[shape_index])
    return _eval_body_particle_contact_banded(
        particle_pos,
        particle_prev_pos,
        contact_index,
        body_particle_contact_ke,
        body_particle_contact_kd,
        mixed_mu,
        friction_epsilon,
        contact_radius,
        shape_body,
        body_q,
        body_q_prev,
        body_qd,
        body_com,
        contact_shape,
        contact_body_pos,
        contact_body_vel,
        contact_normal,
        dt,
        hardening_inv_dmax,
    )


@wp.func
def combine_contact_stiffness(stiff_factor: float, stiff_0: float, stiff_1: float):
    if stiff_0 <= 1.0e-12 and stiff_1 <= 1.0e-12:
        return 0.0
    if stiff_0 <= 1.0e-12:
        return stiff_factor * stiff_1
    if stiff_1 <= 1.0e-12:
        return stiff_factor * stiff_0
    denom = stiff_0 + stiff_1
    return stiff_factor * (stiff_0 * stiff_1) / denom


@wp.kernel
def hessian_multiply_kernel(
    hessian_diags: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    # outputs
    Hx: wp.array[wp.vec3],
):
    tid = wp.tid()
    Hx[tid] = hessian_diags[tid] * x[tid]


@wp.kernel
def transform_rigid_feature_vertices_kernel(
    local_pos: wp.array[wp.vec3],
    vertex_shape: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    world_pos: wp.array[wp.vec3],
):
    vid = wp.tid()
    shape = vertex_shape[vid]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = wp.transform_multiply(body_q[body], X_ws)
    world_pos[vid] = wp.transform_point(X_ws, local_pos[vid])


@wp.func
def _previous_side_normal(
    current_delta: wp.vec3,
    previous_delta: wp.vec3,
    fallback: wp.vec3,
):
    n = previous_delta
    ln = wp.length(n)
    if ln > 1.0e-9:
        return n / ln
    n = current_delta
    ln = wp.length(n)
    if ln > 1.0e-9:
        return n / ln
    ln = wp.length(fallback)
    if ln > 1.0e-9:
        return fallback / ln
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def _feature_normal_force(
    gap: float,
    relative_translation: wp.vec3,
    normal: wp.vec3,
    radius: float,
    soft_ke: float,
    soft_kd: float,
    weight: float,
    dt: float,
):
    depth = radius - gap
    if depth <= 0.0 or weight <= 0.0:
        return float(0.0), float(0.0)
    stiffness = soft_ke * weight
    force = stiffness * depth
    rel_n = wp.dot(normal, relative_translation)
    if rel_n < 0.0 and dt > 0.0:
        damping = soft_kd * weight / dt
        force -= damping * rel_n
        stiffness += damping
    return force, stiffness


@wp.kernel
def eval_rigid_vertex_cloth_face_contacts_kernel(
    dt: float,
    radius: float,
    vertex_shape: wp.array[int],
    shape_contact_ke: wp.array[float],
    soft_kd: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    vertex_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    vid = wp.tid()
    soft_ke = shape_contact_ke[vertex_shape[vid]]
    rp = rigid_pos[vid]
    rp_prev = rigid_pos_prev[vid]
    weight = vertex_weight[vid]
    count = broad_phase[0, vid]
    for i in range(count):
        tri = broad_phase[i + 1, vid]
        i0 = tri_indices[tri, 0]
        i1 = tri_indices[tri, 1]
        i2 = tri_indices[tri, 2]
        a = cloth_pos[i0]
        b = cloth_pos[i1]
        c = cloth_pos[i2]
        cp, bary, _feature = triangle_closest_point(a, b, c, rp)
        if bary[0] <= 1.0e-5 or bary[1] <= 1.0e-5 or bary[2] <= 1.0e-5:
            continue
        cp_prev = (
            cloth_pos_prev[i0] * bary[0]
            + cloth_pos_prev[i1] * bary[1]
            + cloth_pos_prev[i2] * bary[2]
        )
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(b - a, c - a))
        gap = wp.dot(n, cp - rp)
        relative_translation = (cp - cp_prev) - (rp - rp_prev)
        force_mag, stiffness = _feature_normal_force(
            gap,
            relative_translation,
            n,
            radius,
            soft_ke,
            soft_kd,
            weight,
            dt,
        )
        if force_mag <= 0.0:
            continue
        force = n * force_mag
        hess = stiffness * wp.outer(n, n)
        wp.atomic_add(forces, i0, force * bary[0])
        wp.atomic_add(forces, i1, force * bary[1])
        wp.atomic_add(forces, i2, force * bary[2])
        wp.atomic_add(hessians, i0, hess * bary[0] * bary[0])
        wp.atomic_add(hessians, i1, hess * bary[1] * bary[1])
        wp.atomic_add(hessians, i2, hess * bary[2] * bary[2])


@wp.kernel
def eval_rigid_edge_cloth_edge_contacts_kernel(
    dt: float,
    radius: float,
    edge_shape: wp.array[int],
    shape_contact_ke: wp.array[float],
    soft_kd: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    cloth_edge_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    rigid_edge_indices: wp.array2d[int],
    edge_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    eid = wp.tid()
    weight = edge_weight[eid]
    if weight <= 0.0:
        return
    soft_ke = shape_contact_ke[edge_shape[eid]]
    rv0 = rigid_edge_indices[eid, 2]
    rv1 = rigid_edge_indices[eid, 3]
    r0 = rigid_pos[rv0]
    r1 = rigid_pos[rv1]
    r0_prev = rigid_pos_prev[rv0]
    r1_prev = rigid_pos_prev[rv1]
    count = broad_phase[0, eid]
    for i in range(count):
        ce = broad_phase[i + 1, eid]
        cv0 = cloth_edge_indices[ce, 2]
        cv1 = cloth_edge_indices[ce, 3]
        c0 = cloth_pos[cv0]
        c1 = cloth_pos[cv1]
        st = wp.closest_point_edge_edge(r0, r1, c0, c1, 1.0e-6)
        s = st[0]
        t = st[1]
        if s <= 1.0e-5 or s >= 1.0 - 1.0e-5 or t <= 1.0e-5 or t >= 1.0 - 1.0e-5:
            continue
        rp = wp.lerp(r0, r1, s)
        cp = wp.lerp(c0, c1, t)
        rp_prev = wp.lerp(r0_prev, r1_prev, s)
        cp_prev = wp.lerp(cloth_pos_prev[cv0], cloth_pos_prev[cv1], t)
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(r1 - r0, c1 - c0))
        gap = wp.dot(n, cp - rp)
        relative_translation = (cp - cp_prev) - (rp - rp_prev)
        force_mag, stiffness = _feature_normal_force(
            gap,
            relative_translation,
            n,
            radius,
            soft_ke,
            soft_kd,
            weight,
            dt,
        )
        if force_mag <= 0.0:
            continue
        force = n * force_mag
        hess = stiffness * wp.outer(n, n)
        w0 = 1.0 - t
        w1 = t
        wp.atomic_add(forces, cv0, force * w0)
        wp.atomic_add(forces, cv1, force * w1)
        wp.atomic_add(hessians, cv0, hess * w0 * w0)
        wp.atomic_add(hessians, cv1, hess * w1 * w1)


@wp.kernel
def summarize_feature_query_kernel(
    query_results: wp.array2d[int],
    capacity: int,
    max_slot: int,
    feature_stats: wp.array[int],
):
    tid = wp.tid()
    count = query_results[0, tid]
    wp.atomic_max(feature_stats, max_slot, count)
    if count >= capacity:
        wp.atomic_add(feature_stats, max_slot + 1, 1)


@wp.kernel
def accumulate_rigid_vertex_cloth_face_reaction_kernel(
    dt: float,
    radius: float,
    shape_contact_ke: wp.array[float],
    soft_kd: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    vertex_shape: wp.array[int],
    vertex_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_enabled: wp.array[int],
    body_f: wp.array[wp.spatial_vector],
):
    vid = wp.tid()
    shape = vertex_shape[vid]
    body = shape_body[shape]
    if body < 0 or body_enabled[body] == 0:
        return
    soft_ke = shape_contact_ke[shape]
    rp = rigid_pos[vid]
    rp_prev = rigid_pos_prev[vid]
    count = broad_phase[0, vid]
    for i in range(count):
        tri = broad_phase[i + 1, vid]
        i0 = tri_indices[tri, 0]
        i1 = tri_indices[tri, 1]
        i2 = tri_indices[tri, 2]
        a = cloth_pos[i0]
        b = cloth_pos[i1]
        c = cloth_pos[i2]
        cp, bary, _feature = triangle_closest_point(a, b, c, rp)
        if bary[0] <= 1.0e-5 or bary[1] <= 1.0e-5 or bary[2] <= 1.0e-5:
            continue
        cp_prev = (
            cloth_pos_prev[i0] * bary[0]
            + cloth_pos_prev[i1] * bary[1]
            + cloth_pos_prev[i2] * bary[2]
        )
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(b - a, c - a))
        force_mag, _stiffness = _feature_normal_force(
            wp.dot(n, cp - rp),
            (cp - cp_prev) - (rp - rp_prev),
            n,
            radius,
            soft_ke,
            soft_kd,
            vertex_weight[vid],
            dt,
        )
        if force_mag <= 0.0:
            continue
        reaction = -n * force_mag
        com_world = wp.transform_point(body_q[body], body_com[body])
        torque = wp.cross(rp - com_world, reaction)
        wp.atomic_add(body_f, body, wp.spatial_vector(reaction, torque))


@wp.kernel
def accumulate_rigid_edge_cloth_edge_reaction_kernel(
    dt: float,
    radius: float,
    shape_contact_ke: wp.array[float],
    soft_kd: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    cloth_edge_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    rigid_edge_indices: wp.array2d[int],
    edge_shape: wp.array[int],
    edge_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_enabled: wp.array[int],
    body_f: wp.array[wp.spatial_vector],
):
    eid = wp.tid()
    weight = edge_weight[eid]
    if weight <= 0.0:
        return
    shape = edge_shape[eid]
    body = shape_body[shape]
    if body < 0 or body_enabled[body] == 0:
        return
    soft_ke = shape_contact_ke[shape]
    rv0 = rigid_edge_indices[eid, 2]
    rv1 = rigid_edge_indices[eid, 3]
    r0 = rigid_pos[rv0]
    r1 = rigid_pos[rv1]
    r0_prev = rigid_pos_prev[rv0]
    r1_prev = rigid_pos_prev[rv1]
    count = broad_phase[0, eid]
    for i in range(count):
        ce = broad_phase[i + 1, eid]
        cv0 = cloth_edge_indices[ce, 2]
        cv1 = cloth_edge_indices[ce, 3]
        c0 = cloth_pos[cv0]
        c1 = cloth_pos[cv1]
        st = wp.closest_point_edge_edge(r0, r1, c0, c1, 1.0e-6)
        s = st[0]
        t = st[1]
        if s <= 1.0e-5 or s >= 1.0 - 1.0e-5 or t <= 1.0e-5 or t >= 1.0 - 1.0e-5:
            continue
        rp = wp.lerp(r0, r1, s)
        cp = wp.lerp(c0, c1, t)
        rp_prev = wp.lerp(r0_prev, r1_prev, s)
        cp_prev = wp.lerp(cloth_pos_prev[cv0], cloth_pos_prev[cv1], t)
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(r1 - r0, c1 - c0))
        force_mag, _stiffness = _feature_normal_force(
            wp.dot(n, cp - rp),
            (cp - cp_prev) - (rp - rp_prev),
            n,
            radius,
            soft_ke,
            soft_kd,
            weight,
            dt,
        )
        if force_mag <= 0.0:
            continue
        reaction = -n * force_mag
        com_world = wp.transform_point(body_q[body], body_com[body])
        torque = wp.cross(rp - com_world, reaction)
        wp.atomic_add(body_f, body, wp.spatial_vector(reaction, torque))


@wp.func
def eval_body_particle_contact_anchored(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    contact_ke: float,
    contact_kd: float,
    friction_mu: float,
    kt_ratio: float,
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    update_anchor: int,
    anchor_local: wp.array[wp.vec3],
    anchor_shape: wp.array[int],
    dt: float,
):
    """S1: normal penalty + STATIC friction as a spring to a persistent anchor.

    The stock particle-rigid law (``_compute_body_particle_contact_force``)
    feeds this substep's relative displacement into an IPC-mollified friction
    term, i.e. pure viscosity with the history cleared every substep — a pinched
    bite therefore ratchets out of the jaw at a steady creep rate. Here the
    tangential force is instead a spring toward an anchor stored in the SHAPE's
    frame (so it rides along with the finger), clamped to the Coulomb cone by
    elastoplastic return mapping: inside the cone the contact truly sticks, and
    once the cone is exceeded the anchor is dragged so the offset lands exactly
    on it.

    ``update_anchor`` must be 1 for exactly one of the two passes that evaluate
    this contact (the force pass), and 0 for the reaction pass, so the anchor
    state advances once per substep.
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])
    n = contact_normal[contact_index]
    depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if depth <= 0.0:
        # Guardrail 1: no contact -> the anchor must follow freely, otherwise a
        # stale anchor keeps pulling and welds the cloth to the finger.
        if update_anchor != 0:
            anchor_shape[particle_index] = -1
        return wp.vec3(0.0), wp.mat33(0.0)

    f_n = depth * contact_ke
    force = n * f_n
    hessian = contact_ke * wp.outer(n, n)

    # Approach damping, same as the stock law. Dropping it lets a closing jaw
    # drive straight through (measured -5.8 mm during the close).
    bx_prev = bx
    if body_q_prev:
        X_wb_prev = wp.transform_identity()
        if body_index >= 0:
            X_wb_prev = body_q_prev[body_index]
        bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
    relative_translation = (particle_pos - particle_prev_pos) - (bx - bx_prev)
    if wp.dot(n, relative_translation) < 0.0:
        damping_hessian = (contact_kd / dt) * wp.outer(n, n)
        hessian = hessian + damping_hessian
        force = force - damping_hessian * relative_translation

    mu = wp.sqrt(friction_mu * shape_material_mu[shape_index])
    if mu <= 0.0 or kt_ratio <= 0.0:
        return force, hessian

    # (Re)seed the anchor whenever this particle is not already anchored to
    # this shape. Stored in body frame so finger motion carries it along.
    if anchor_shape[particle_index] != shape_index:
        if update_anchor == 0:
            return force, hessian
        anchor_local[particle_index] = wp.transform_point(wp.transform_inverse(X_wb), particle_pos)
        anchor_shape[particle_index] = shape_index

    a_world = wp.transform_point(X_wb, anchor_local[particle_index])
    offset = particle_pos - a_world
    offset_t = offset - n * wp.dot(n, offset)

    kt = contact_ke * kt_ratio
    cone = mu * f_n
    slip_max = cone / kt
    lo = wp.length(offset_t)

    if lo > slip_max:
        # Slipping: clamp the force to the cone and drag the anchor up to it.
        if lo > 1.0e-12:
            force = force - offset_t * (cone / lo)
            if update_anchor != 0:
                dragged = a_world + offset_t * ((lo - slip_max) / lo)
                anchor_local[particle_index] = wp.transform_point(
                    wp.transform_inverse(X_wb), dragged
                )
    else:
        # Sticking: finite-stiffness spring, and the tangential stiffness enters
        # the Hessian so the implicit solve sees it.
        force = force - offset_t * kt
        hessian = hessian + kt * (wp.identity(n=3, dtype=float) - wp.outer(n, n))

    return force, hessian


@wp.func
def eval_body_particle_contact_avbd(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    contact_ke: float,
    contact_kd: float,
    friction_mu: float,
    dual_k: float,
    particle_radius: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    update_dual: int,
    lambda_t: wp.array[wp.vec3],
    lambda_shape: wp.array[int],
    dt: float,
):
    """S3: friction as an accumulated tangential Lagrange multiplier (AVBD).

    Ported from the rigid-rigid side's dual update
    (:func:`update_duals_body_body_contacts`) onto body-particle contacts. The
    tangential multiplier is advanced by dual ascent every solver iteration and
    clamped onto the Coulomb cone:

        lambda_t <- clamp_cone(lambda_t + k * u_t,  mu * lambda_n)

    Unlike a finite-stiffness spring (S1), the force here is the multiplier
    itself, so ``dual_k`` only sets how fast it converges -- 20 iterations of a
    small step still reach the cone. That is what makes it insensitive to the
    stiffness choice that leaves S1 stuck between "too soft to hold" and "stiff
    enough to disturb the normal solve".
    """
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    if body_index >= 0:
        X_wb = body_q[body_index]

    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])
    n = contact_normal[contact_index]
    depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if depth <= 0.0:
        if update_dual != 0:
            lambda_shape[particle_index] = -1
            lambda_t[particle_index] = wp.vec3(0.0)
        return wp.vec3(0.0), wp.mat33(0.0)

    lam_n = depth * contact_ke
    force = n * lam_n
    hessian = contact_ke * wp.outer(n, n)

    bx_prev = bx
    if body_q_prev:
        X_wb_prev = wp.transform_identity()
        if body_index >= 0:
            X_wb_prev = body_q_prev[body_index]
        bx_prev = wp.transform_point(X_wb_prev, contact_body_pos[contact_index])
    relative_translation = (particle_pos - particle_prev_pos) - (bx - bx_prev)
    if wp.dot(n, relative_translation) < 0.0:
        damping_hessian = (contact_kd / dt) * wp.outer(n, n)
        hessian = hessian + damping_hessian
        force = force - damping_hessian * relative_translation

    mu = wp.sqrt(friction_mu * shape_material_mu[shape_index])
    if mu <= 0.0 or dual_k <= 0.0:
        return force, hessian

    if lambda_shape[particle_index] != shape_index:
        if update_dual == 0:
            return force, hessian
        lambda_t[particle_index] = wp.vec3(0.0)
        lambda_shape[particle_index] = shape_index

    u_t = relative_translation - n * wp.dot(n, relative_translation)
    lam_t = lambda_t[particle_index] + u_t * (dual_k * contact_ke)
    cone = mu * lam_n
    lt = wp.length(lam_t)
    if lt > cone and lt > 0.0:
        lam_t = lam_t * (cone / lt)
    if update_dual != 0:
        lambda_t[particle_index] = lam_t

    force = force - lam_t
    # The tangential stiffness is left OUT of the Hessian here. Note this was
    # NOT the cure it was hypothesised to be: dropping it still loses the grip
    # (f=126/152) and still penetrates during the close, same as with it in. The
    # shared failure of S1/S2/S3 lies elsewhere -- most likely that this branch
    # REPLACES the stock viscous friction rather than adding to it, and the
    # stock term is what the working baseline actually grips with.
    return force, hessian


@wp.kernel
def eval_body_contact_kernel(
    # inputs
    dt: float,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    # E4: per-shape penalty band (0 = use particle_radius, stock path)
    shape_contact_offset: wp.array[float],
    # R9: per-shape hardening eps_max (0 = stock linear law, bit-identical)
    shape_hardening_eps: wp.array[float],
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    shape_contact_ke: wp.array[float],
    material_ke: float,
    particle_contact_ke: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    # S1 static-friction anchors; kt_ratio <= 0 keeps the stock viscous law
    anchor_kt_ratio: float,
    anchor_update: int,
    anchor_local: wp.array[wp.vec3],
    anchor_shape: wp.array[int],
    avbd_dual_k: float,
    lambda_t: wp.array[wp.vec3],
    lambda_shape: wp.array[int],
    # outputs: particle force and hessian
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    t_id = wp.tid()

    particle_body_contact_count = wp.min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        particle_idx = soft_contact_particle[t_id]
        shape_idx = contact_shape[t_id]
        if avbd_dual_k > 0.0:
            f_v, h_v = eval_body_particle_contact_avbd(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
                soft_contact_kd,
                friction_mu,
                avbd_dual_k,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                contact_shape,
                contact_body_pos,
                contact_normal,
                1,
                lambda_t,
                lambda_shape,
                dt,
            )
            wp.atomic_add(forces, particle_idx, f_v)
            wp.atomic_add(hessians, particle_idx, h_v)
            return
        if anchor_kt_ratio > 0.0:
            f_a, h_a = eval_body_particle_contact_anchored(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
                soft_contact_kd,
                friction_mu,
                anchor_kt_ratio,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                contact_shape,
                contact_body_pos,
                contact_normal,
                anchor_update,          # 1 only on the substep's first iteration
                anchor_local,
                anchor_shape,
                dt,
            )
            wp.atomic_add(forces, particle_idx, f_a)
            wp.atomic_add(hessians, particle_idx, h_a)
            return
        body_contact_force, body_contact_hessian = evaluate_body_particle_contact_banded(
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            contact_band_radius(particle_radius, shape_contact_offset, particle_idx, shape_idx),
            shape_material_mu,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
            contact_hardening_inv_dmax(
                shape_hardening_eps,
                2.0 * particle_radius[particle_idx],   # R13: t0 = cloth thickness
                shape_idx,
            ),
        )
        wp.atomic_add(forces, particle_idx, body_contact_force)
        wp.atomic_add(hessians, particle_idx, body_contact_hessian)


@wp.kernel
def accumulate_body_reaction_kernel(
    # inputs
    dt: float,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    # body-particle contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array[float],
    # E4: per-shape penalty band (0 = use particle_radius, stock path)
    shape_contact_offset: wp.array[float],
    # R9: per-shape hardening eps_max (0 = stock linear law, bit-identical)
    shape_hardening_eps: wp.array[float],
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    shape_contact_ke: wp.array[float],
    material_ke: float,
    particle_contact_ke: wp.array[float],
    shape_material_mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    # S1 anchors: READ-ONLY here (update_anchor=0) so the state advances once
    anchor_kt_ratio: float,
    anchor_local: wp.array[wp.vec3],
    anchor_shape: wp.array[int],
    avbd_dual_k: float,
    lambda_t: wp.array[wp.vec3],
    lambda_shape: wp.array[int],
    body_enabled: wp.array[int],
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """Accumulate the reaction wrench of particle-body contacts onto bodies.

    Re-evaluates the same contact force as :func:`eval_body_contact_kernel`
    (at the converged particle positions) and adds the equal-and-opposite
    wrench to ``body_f`` — world frame, referenced at the body COM, matching
    :attr:`newton.State.body_f` — so an external rigid solver can consume it
    as an applied force (two-way cloth-rigid coupling).

    ``body_enabled`` gates which bodies receive the reaction (e.g. only free
    rigid objects, not kinematically driven robot links).
    """
    t_id = wp.tid()

    particle_body_contact_count = wp.min(contact_max, contact_count[0])

    if t_id < particle_body_contact_count:
        shape_idx = contact_shape[t_id]
        body_idx = shape_body[shape_idx]
        if body_idx < 0 or body_enabled[body_idx] == 0:
            return
        particle_idx = soft_contact_particle[t_id]
        if avbd_dual_k > 0.0:
            f_v, _h_v = eval_body_particle_contact_avbd(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
                soft_contact_kd,
                friction_mu,
                avbd_dual_k,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                contact_shape,
                contact_body_pos,
                contact_normal,
                0,
                lambda_t,
                lambda_shape,
                dt,
            )
            reaction_v = -f_v
            X_wb_v = body_q[body_idx]
            cp_v = wp.transform_point(X_wb_v, contact_body_pos[t_id])
            com_v = wp.transform_point(X_wb_v, body_com[body_idx])
            wp.atomic_add(
                body_f,
                body_idx,
                wp.spatial_vector(reaction_v, wp.cross(cp_v - com_v, reaction_v)),
            )
            return
        if anchor_kt_ratio > 0.0:
            f_a, _h_a = eval_body_particle_contact_anchored(
                particle_idx,
                pos[particle_idx],
                pos_prev[particle_idx],
                t_id,
                particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
                soft_contact_kd,
                friction_mu,
                anchor_kt_ratio,
                particle_radius,
                shape_material_mu,
                shape_body,
                body_q,
                body_q_prev,
                contact_shape,
                contact_body_pos,
                contact_normal,
                0,                      # read-only: the force pass owns updates
                anchor_local,
                anchor_shape,
                dt,
            )
            reaction_a = -f_a
            X_wb_a = body_q[body_idx]
            cp_a = wp.transform_point(X_wb_a, contact_body_pos[t_id])
            com_a = wp.transform_point(X_wb_a, body_com[body_idx])
            wp.atomic_add(
                body_f,
                body_idx,
                wp.spatial_vector(reaction_a, wp.cross(cp_a - com_a, reaction_a)),
            )
            return
        body_contact_force, _hessian = evaluate_body_particle_contact_banded(
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            particle_contact_stiffness(material_ke, particle_contact_ke, shape_contact_ke, particle_idx, shape_idx),
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            contact_band_radius(particle_radius, shape_contact_offset, particle_idx, shape_idx),
            shape_material_mu,
            shape_body,
            body_q,
            body_q_prev,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
            contact_hardening_inv_dmax(
                shape_hardening_eps,
                2.0 * particle_radius[particle_idx],   # R13: t0 = cloth thickness
                shape_idx,
            ),
        )
        reaction = -body_contact_force
        X_wb = body_q[body_idx]
        contact_point = wp.transform_point(X_wb, contact_body_pos[t_id])
        com_world = wp.transform_point(X_wb, body_com[body_idx])
        torque = wp.cross(contact_point - com_world, reaction)
        wp.atomic_add(body_f, body_idx, wp.spatial_vector(reaction, torque))


@wp.kernel
def clamp_body_wrench_kernel(
    max_force: float,
    # outputs (in-place)
    body_f: wp.array[wp.spatial_vector],
):
    """Uniformly scale a body wrench so its linear force stays under its cap.

    Explosion guard for penalty-force feedback: a deep-penetration spike scaled
    down keeps its direction, so behaviour degrades gracefully.
    """
    tid = wp.tid()
    f = body_f[tid]
    force = wp.vec3(f[0], f[1], f[2])
    mag = wp.length(force)
    if mag > max_force:
        body_f[tid] = f * (max_force / mag)


@wp.kernel
def handle_vertex_triangle_contacts_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    broad_phase_vf: wp.array2d[int],
    static_diags: wp.array[float],
    # outputs
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    vid = wp.tid()

    x0 = pos[vid]
    force0 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    vert_stiff = static_diags[vid]
    is_collided = wp.int32(0)

    count = broad_phase_vf[0, vid]
    for i in range(count):
        fid = broad_phase_vf[i + 1, vid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])
        x1 = pos[face[0]]
        x2 = pos[face[1]]
        x3 = pos[face[2]]
        tri_normal = triangle_normal(x1, x2, x3)
        dist = wp.dot(x0 - x1, tri_normal)
        p = x0 - tri_normal * dist
        bary_coord = triangle_barycentric(x1, x2, x3, p)

        if wp.abs(dist) > thickness:
            continue
        if bary_coord[0] < 0.0 or bary_coord[1] < 0.0 or bary_coord[2] < 0.0:
            continue  # is outside triangle

        face_stiff = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = combine_contact_stiffness(stiff_factor, vert_stiff, face_stiff)
        if stiff <= 0.0:
            continue

        force = stiff * tri_normal * (thickness - wp.abs(dist)) * wp.sign(dist)
        hess = stiff * wp.outer(tri_normal, tri_normal)

        force0 += force
        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        hess0 += hess
        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])
        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, vid, force0)
        wp.atomic_add(hessian_diags, vid, hess0)


@wp.kernel
def handle_edge_edge_contacts_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[int],
    broad_phase_ee: wp.array2d[int],
    static_diags: wp.array[float],
    # outputs
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    eid = wp.tid()
    edge0 = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    x0 = pos[edge0[0]]
    x1 = pos[edge0[1]]
    len0 = wp.length(x0 - x1)

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge0[0]] + static_diags[edge0[1]]) / 2.0
    is_collided = wp.int32(0)

    count = broad_phase_ee[0, eid]
    for i in range(count):
        idx = broad_phase_ee[i + 1, eid]
        edge1 = wp.vec4i(edge_indices[idx, 2], edge_indices[idx, 3], edge_indices[idx, 0], edge_indices[idx, 1])
        x2, x3 = pos[edge1[0]], pos[edge1[1]]
        edge_edge_parallel_epsilon = wp.float32(1e-5)

        st = wp.closest_point_edge_edge(x0, x1, x2, x3, edge_edge_parallel_epsilon)
        s, t = st[0], st[1]

        if (s <= 0) or (s >= 1) or (t <= 0) or (t >= 1):
            continue

        c1 = wp.lerp(x0, x1, s)
        c2 = wp.lerp(x2, x3, t)
        dir = c1 - c2
        dist = wp.length(dir)
        limited_thickness = thickness

        len1 = wp.length(x2 - x3)
        avg_len = (len0 + len1) * 0.5
        if edge0[2] == edge1[0] or edge0[3] == edge1[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge0[2] == edge1[1] or edge0[3] == edge1[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        if edge1[2] == edge0[0] or edge1[3] == edge0[0]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)
        elif edge1[2] == edge0[1] or edge1[3] == edge0[1]:
            limited_thickness = wp.min(limited_thickness, avg_len * 0.5)

        if 1e-6 < dist < limited_thickness:
            stiff_1 = (static_diags[edge1[0]] + static_diags[edge1[1]]) / 2.0
            stiff = combine_contact_stiffness(stiff_factor, stiff_0, stiff_1)
            if stiff <= 0.0:
                continue

            dir = wp.normalize(dir)
            force = stiff * dir * (limited_thickness - dist)
            hess = stiff * wp.outer(dir, dir)

            force0 += force * (1.0 - s)
            force1 += force * s
            wp.atomic_add(forces, edge1[0], -force * (1.0 - t))
            wp.atomic_add(forces, edge1[1], -force * t)

            hess0 += hess * (1.0 - s) * (1.0 - s)
            hess1 += hess * s * s
            wp.atomic_add(hessian_diags, edge1[0], hess * (1.0 - t) * (1.0 - t))
            wp.atomic_add(hessian_diags, edge1[1], hess * t * t)
            is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge0[0], force0)
        wp.atomic_add(forces, edge0[1], force1)
        wp.atomic_add(hessian_diags, edge0[0], hess0)
        wp.atomic_add(hessian_diags, edge0[1], hess1)


@wp.func
def intersection_gradient_vector(R: wp.vec3, E: wp.vec3, N: wp.vec3):
    """
    Reference: Resolving Surface Collisions through Intersection Contour Minimization, Pascal Volino & Magnenat-Thalmann, 2006.

    Args:
        R: The direction of the intersection segment
        E: Direction vector of the edge
        N: The normals of the polygons
    """
    dot_EN = wp.dot(E, N)
    if wp.abs(dot_EN) > 1e-6:
        return R - 2.0 * N * wp.dot(E, R) / dot_EN
    else:
        return R


@wp.kernel
def solve_untangling_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    broad_phase_ef: wp.array2d[int],
    static_diags: wp.array[float],
    # outputs
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
):
    eid = wp.tid()
    edge = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    v0 = pos[edge[0]]
    v1 = pos[edge[1]]

    # Skip invalid edge
    len0 = wp.length(v0 - v1)
    if len0 < 5e-4:
        return

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge[0]] + static_diags[edge[1]]) / 2.0
    is_collided = wp.int32(0)

    # Edge direction
    E = wp.normalize(v0 - v1)
    N2 = wp.vec3(0.0) if edge[2] < 0 else triangle_normal(v0, v1, pos[edge[2]])
    N3 = wp.vec3(0.0) if edge[3] < 0 else triangle_normal(v0, v1, pos[edge[3]])

    count = broad_phase_ef[0, eid]
    for i in range(count):
        fid = broad_phase_ef[i + 1, eid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])

        if face[0] == edge[0] or face[0] == edge[1]:
            continue
        if face[1] == edge[0] or face[1] == edge[1]:
            continue
        if face[2] == edge[0] or face[2] == edge[1]:
            continue

        x0 = pos[face[0]]
        x1 = pos[face[1]]
        x2 = pos[face[2]]
        face_normal = wp.cross(x1 - x0, x2 - x1)
        normal_len = wp.length(face_normal)
        if normal_len < 1e-8:
            continue  # invalid triangle

        face_normal = wp.normalize(face_normal)
        d1 = wp.dot(face_normal, v0 - x0)
        d2 = wp.dot(face_normal, v1 - x0)
        if d1 * d2 >= 0.0:
            continue  # on same side

        d1, d2 = wp.abs(d1), wp.abs(d2)
        hit_point = (v0 * d2 + v1 * d1) / (d2 + d1)
        bary_coord = triangle_barycentric(x0, x1, x2, hit_point)

        if (bary_coord[0] < 1e-2) or (bary_coord[1] < 1e-2) or (bary_coord[2] < 1e-2):
            continue  # hit outside

        G = wp.vec3(0.0)

        if edge[2] >= 0:
            R = wp.cross(face_normal, N2)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[2]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if edge[3] >= 0:
            R = wp.cross(face_normal, N3)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[3]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if wp.length(G) < 1.0e-12:
            continue
        G = wp.normalize(G)

        # Can be precomputed
        stiff_1 = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = combine_contact_stiffness(stiff_factor, stiff_0, stiff_1)
        if stiff <= 0.0:
            continue
        disp = 2.0 * thickness

        force = stiff * G * disp
        hess = stiff * wp.outer(G, G)
        edge_bary = wp.vec2(d2, d1) / (d1 + d2)

        force0 += force * edge_bary[0]
        force1 += force * edge_bary[1]
        hess0 += hess * edge_bary[0] * edge_bary[0]
        hess1 += hess * edge_bary[1] * edge_bary[1]

        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])

        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge[0], force0)
        wp.atomic_add(forces, edge[1], force1)
        wp.atomic_add(hessian_diags, edge[0], hess0)
        wp.atomic_add(hessian_diags, edge[1], hess1)


########################################################################################################################
#########################   Rigid untangling (ICM: cloth edge x rigid triangle)   ######################################
########################################################################################################################
# R2-3.  Volino's Intersection Contour Minimisation with the "face" side taken
# from a RIGID mesh instead of the cloth.  ``solve_untangling_kernel`` above is
# the stock cloth-cloth version; what it implements -- once the cloth is ALREADY
# through, shrink the intersection contour until it is gone -- has no
# counterpart for cloth-vs-gripper.  The penalty and projection channels only
# know how to push a particle towards the NEAREST surface, which for a blade
# thinner than the penetration depth is the wrong side half of the time, and
# neither of them sees a particle that sits outside the shell while its EDGE
# threads straight through the plate.
#
# Signed distance cannot express "which way is out" for a thin plate; the
# intersection CONTOUR can, because its length is zero exactly when the cloth is
# untangled.  The gradient of that length w.r.t. the crossing edge is Volino's
# ``intersection_gradient_vector`` (already used above), so this kernel is the
# same law with ``rigid_pos``/``rigid_tri_indices`` on the face side.
#
# Default OFF.  Nothing is allocated and no kernel is launched unless
# ``Collision.enable_rigid_untangling`` has been called, so the default path
# never reaches this code.
#
# Asymmetric by construction: the reaction on the rigid body is NOT accumulated.
# ICM is a topological recovery force, not a contact force -- the normal load and
# its reaction stay with the penalty channel (2.4).  Feeding a recovery impulse
# back into MuJoCo would show up as a fake grip force.


@wp.kernel
def solve_rigid_untangling_kernel(
    thickness: float,
    stiff_factor: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    edge_indices: wp.array2d[int],
    rigid_pos: wp.array[wp.vec3],
    rigid_tri_indices: wp.array2d[int],
    broad_phase: wp.array2d[int],
    static_diags: wp.array[float],
    capacity: int,
    count_stats: int,
    # outputs
    forces: wp.array[wp.vec3],
    hessian_diags: wp.array[wp.mat33],
    stats: wp.array[int],
    stats_total: wp.array[int],
):
    eid = wp.tid()
    edge = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    v0 = pos[edge[0]]
    v1 = pos[edge[1]]

    # Skip invalid edge
    len0 = wp.length(v0 - v1)
    if len0 < 5e-4:
        return

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge[0]] + static_diags[edge[1]]) / 2.0
    is_collided = wp.int32(0)
    pair_count = wp.int32(0)

    # Edge direction and the normals of the cloth faces sharing this edge.
    E = wp.normalize(v0 - v1)
    N2 = wp.vec3(0.0) if edge[2] < 0 else triangle_normal(v0, v1, pos[edge[2]])
    N3 = wp.vec3(0.0) if edge[3] < 0 else triangle_normal(v0, v1, pos[edge[3]])

    count = broad_phase[0, eid]
    for i in range(count):
        fid = broad_phase[i + 1, eid]
        face = wp.vec3i(
            rigid_tri_indices[fid, 0],
            rigid_tri_indices[fid, 1],
            rigid_tri_indices[fid, 2],
        )
        x0 = rigid_pos[face[0]]
        x1 = rigid_pos[face[1]]
        x2 = rigid_pos[face[2]]
        face_normal = wp.cross(x1 - x0, x2 - x1)
        normal_len = wp.length(face_normal)
        if normal_len < 1e-8:
            continue  # invalid triangle

        face_normal = face_normal / normal_len
        d1 = wp.dot(face_normal, v0 - x0)
        d2 = wp.dot(face_normal, v1 - x0)
        if d1 * d2 >= 0.0:
            continue  # on same side

        d1, d2 = wp.abs(d1), wp.abs(d2)
        hit_point = (v0 * d2 + v1 * d1) / (d2 + d1)
        bary_coord = triangle_barycentric(x0, x1, x2, hit_point)

        if (bary_coord[0] < 1e-2) or (bary_coord[1] < 1e-2) or (bary_coord[2] < 1e-2):
            continue  # hit outside

        # Counted here, BEFORE the gradient can degenerate: this is the number
        # the unit scene reports as "still tangled", so it must not depend on
        # whether the force ends up non-zero.
        pair_count += 1

        G = wp.vec3(0.0)

        if edge[2] >= 0:
            R = wp.cross(face_normal, N2)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[2]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if edge[3] >= 0:
            R = wp.cross(face_normal, N3)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[3]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if wp.length(G) < 1.0e-12:
            continue
        G = wp.normalize(G)

        stiff = combine_contact_stiffness(stiff_factor, stiff_0, 0.0)
        if stiff <= 0.0:
            continue
        disp = 2.0 * thickness

        force = stiff * G * disp
        hess = stiff * wp.outer(G, G)
        edge_bary = wp.vec2(d2, d1) / (d1 + d2)

        force0 += force * edge_bary[0]
        force1 += force * edge_bary[1]
        hess0 += hess * edge_bary[0] * edge_bary[0]
        hess1 += hess * edge_bary[1] * edge_bary[1]

        is_collided = 1

    if count_stats != 0:
        # [1] = broad-phase candidates seen. Without it a zero in [0] is
        # ambiguous: "the kernel looked and found nothing tangled" and "the
        # broad phase handed it nothing to look at" read the same.
        wp.atomic_max(stats, 1, count)
        # ``stats_total`` is never zeroed: [0] = crossings summed over the whole
        # episode, [1] = the largest number in one substep, [2] = the largest
        # candidate list, [3] = how many times a candidate list hit the cap.
        # [2]/[3] separate "nothing was tangled" from "the broad phase truncated
        # the list before the tangled triangle was reached" -- with a 20k-triangle
        # finger those are very different failures and read the same in [0].
        wp.atomic_max(stats_total, 1, pair_count)
        wp.atomic_max(stats_total, 2, count)
        if count >= capacity:
            wp.atomic_add(stats_total, 3, 1)
        if pair_count > 0:
            wp.atomic_add(stats, 0, pair_count)
            wp.atomic_add(stats_total, 0, pair_count)

    if is_collided != 0:
        wp.atomic_add(forces, edge[0], force0)
        wp.atomic_add(forces, edge[1], force1)
        wp.atomic_add(hessian_diags, edge[0], hess0)
        wp.atomic_add(hessian_diags, edge[1], hess1)


########################################################################################################################
##############################################    Contact projection    ###############################################
########################################################################################################################
# Position-level non-penetration pass (PBD-style), run after the implicit
# solve. Penalty forces remain the force/reaction channel; projection only
# bounds the residual penetration depth to ``slack``. Constraints from a
# kinematically over-closing gripper are infeasible by construction — the
# Jacobi average then leaves a residual depth whose penalty reaction pushes
# the (two-way coupled) fingers back until the system becomes feasible.


@wp.kernel
def project_body_particle_contacts_kernel(
    slack: float,
    friction_scale: float,
    friction_mu: float,
    pos: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],
    accum: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_material_mu: wp.array[float],
    shape_enabled: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    # outputs
    delta: wp.array[wp.vec3],
    delta_weight: wp.array[float],
):
    """Project particles out of rigid shapes down to a residual depth of ``slack``.

    With ``friction_scale > 0`` the same pass also resolves Coulomb friction at
    the position level (XPBD style): the tangential slip accumulated over this
    substep, measured RELATIVE to the shape's own motion, is cancelled outright
    while it stays inside the cone, and clamped back onto the cone once it
    leaves. This is the half our original projection was missing -- PBD engines
    solve normal non-penetration and friction in the SAME iteration loop, and
    friction is the part that actually consumes the iterations.

    Cone half-width is ``mu * |normal correction applied this pass|``, the XPBD
    form. It has to be the correction and not the standing overlap: the
    correction is what a squeeze produces each pass and what a contact at rest
    stops producing, so friction fades on its own when the jaw opens. Riding the
    cone on the standing overlap instead glues the cloth to the finger forever
    whenever ``slack > 0``, since slack guarantees the overlap never reaches
    zero (measured: cloth still hanging 3-14 mm off the open jaw, 198/198
    frames penetrating). The corollary is that static friction and slack are
    incompatible -- run this branch with ``slack = 0`` and take the reaction
    force from the pre-projection state instead.
    """
    t_id = wp.tid()
    count = wp.min(contact_max, contact_count[0])
    if t_id >= count:
        return
    particle = soft_contact_particle[t_id]
    shape = contact_shape[t_id]
    if shape_enabled[shape] == 0:
        return
    body = shape_body[shape]
    bx_local = contact_body_pos[t_id]
    bx = bx_local
    if body >= 0:
        bx = wp.transform_point(body_q[body], bx_local)
    n = contact_normal[t_id]
    gap = wp.dot(n, pos[particle] - bx) - particle_radius[particle]
    if gap >= -slack:
        return
    dn = -slack - gap
    corr = n * dn

    if friction_scale > 0.0:
        # Same mixing rule as the penalty channel (geometric mean of the global
        # and per-shape coefficients), so both contact channels agree on mu.
        mu = wp.sqrt(friction_mu * shape_material_mu[shape]) * friction_scale
        if mu > 0.0:
            # Contact-point motion of the shape over the substep; without it a
            # closing gripper would read its own approach as cloth slip.
            bx_prev = bx_local
            if body >= 0:
                bx_prev = wp.transform_point(body_q_prev[body], bx_local)
            # NOTE: subtracting ``accum`` here (to measure slip against the
            # unprojected solve) makes it strictly worse -- that position is
            # already buried in the jaw, so its tangential part is huge.
            # Measured: -4.5 mm without, -7.5 mm with. Keep the projected pos.
            rel = (pos[particle] - pos_prev[particle]) - (bx - bx_prev)
            rel_t = rel - n * wp.dot(n, rel)
            slip = wp.length(rel_t)
            if slip > 1.0e-9:
                limit = mu * dn
                if slip <= limit:
                    corr = corr - rel_t                      # stick: cancel it
                else:
                    corr = corr - rel_t * (limit / slip)     # slide: clamp to cone

    wp.atomic_add(delta, particle, corr)
    wp.atomic_add(delta_weight, particle, 1.0)


@wp.kernel
def project_rigid_vertex_cloth_face_kernel(
    slack: float,
    radius: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    vertex_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    # outputs
    delta: wp.array[wp.vec3],
    delta_weight: wp.array[float],
):
    """Push cloth faces off rigid feature vertices down to a residual depth of ``slack``.

    Mirrors :func:`eval_rigid_vertex_cloth_face_contacts_kernel` (same
    candidates and side-memory normal), but emits position corrections
    instead of penalty forces.
    """
    vid = wp.tid()
    if vertex_weight[vid] <= 0.0:
        return
    rp = rigid_pos[vid]
    rp_prev = rigid_pos_prev[vid]
    count = broad_phase[0, vid]
    for i in range(count):
        tri = broad_phase[i + 1, vid]
        i0 = tri_indices[tri, 0]
        i1 = tri_indices[tri, 1]
        i2 = tri_indices[tri, 2]
        a = cloth_pos[i0]
        b = cloth_pos[i1]
        c = cloth_pos[i2]
        cp, bary, _feature = triangle_closest_point(a, b, c, rp)
        if bary[0] <= 1.0e-5 or bary[1] <= 1.0e-5 or bary[2] <= 1.0e-5:
            continue
        cp_prev = (
            cloth_pos_prev[i0] * bary[0]
            + cloth_pos_prev[i1] * bary[1]
            + cloth_pos_prev[i2] * bary[2]
        )
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(b - a, c - a))
        violation = (radius - slack) - wp.dot(n, cp - rp)
        if violation <= 0.0:
            continue
        denom = bary[0] * bary[0] + bary[1] * bary[1] + bary[2] * bary[2]
        correction = n * (violation / wp.max(denom, 1.0e-6))
        wp.atomic_add(delta, i0, correction * bary[0])
        wp.atomic_add(delta, i1, correction * bary[1])
        wp.atomic_add(delta, i2, correction * bary[2])
        wp.atomic_add(delta_weight, i0, 1.0)
        wp.atomic_add(delta_weight, i1, 1.0)
        wp.atomic_add(delta_weight, i2, 1.0)


@wp.kernel
def project_rigid_edge_cloth_edge_kernel(
    slack: float,
    radius: float,
    cloth_pos_prev: wp.array[wp.vec3],
    cloth_pos: wp.array[wp.vec3],
    cloth_edge_indices: wp.array2d[int],
    rigid_pos_prev: wp.array[wp.vec3],
    rigid_pos: wp.array[wp.vec3],
    rigid_edge_indices: wp.array2d[int],
    edge_weight: wp.array[float],
    broad_phase: wp.array2d[int],
    # outputs
    delta: wp.array[wp.vec3],
    delta_weight: wp.array[float],
):
    """Push cloth edges off rigid feature edges down to a residual depth of ``slack``."""
    eid = wp.tid()
    if edge_weight[eid] <= 0.0:
        return
    rv0 = rigid_edge_indices[eid, 2]
    rv1 = rigid_edge_indices[eid, 3]
    r0 = rigid_pos[rv0]
    r1 = rigid_pos[rv1]
    r0_prev = rigid_pos_prev[rv0]
    r1_prev = rigid_pos_prev[rv1]
    count = broad_phase[0, eid]
    for i in range(count):
        ce = broad_phase[i + 1, eid]
        cv0 = cloth_edge_indices[ce, 2]
        cv1 = cloth_edge_indices[ce, 3]
        c0 = cloth_pos[cv0]
        c1 = cloth_pos[cv1]
        st = wp.closest_point_edge_edge(r0, r1, c0, c1, 1.0e-6)
        s = st[0]
        t = st[1]
        if s <= 1.0e-5 or s >= 1.0 - 1.0e-5 or t <= 1.0e-5 or t >= 1.0 - 1.0e-5:
            continue
        rp = wp.lerp(r0, r1, s)
        cp = wp.lerp(c0, c1, t)
        rp_prev = wp.lerp(r0_prev, r1_prev, s)
        cp_prev = wp.lerp(cloth_pos_prev[cv0], cloth_pos_prev[cv1], t)
        n = _previous_side_normal(cp - rp, cp_prev - rp_prev, wp.cross(r1 - r0, c1 - c0))
        violation = (radius - slack) - wp.dot(n, cp - rp)
        if violation <= 0.0:
            continue
        w0 = 1.0 - t
        w1 = t
        denom = w0 * w0 + w1 * w1
        correction = n * (violation / wp.max(denom, 1.0e-6))
        wp.atomic_add(delta, cv0, correction * w0)
        wp.atomic_add(delta, cv1, correction * w1)
        wp.atomic_add(delta_weight, cv0, 1.0)
        wp.atomic_add(delta_weight, cv1, 1.0)


@wp.kernel
def apply_contact_projection_kernel(
    relaxation: float,
    particle_flags: wp.array[wp.int32],
    # outputs (in-place)
    delta: wp.array[wp.vec3],
    delta_weight: wp.array[float],
    pos: wp.array[wp.vec3],
    accum: wp.array[wp.vec3],
):
    """Apply Jacobi-averaged corrections in place and reset the accumulators."""
    tid = wp.tid()
    w = delta_weight[tid]
    if w > 0.0 and (particle_flags[tid] & ParticleFlags.ACTIVE):
        d = delta[tid] * (relaxation / wp.max(w, 1.0))
        pos[tid] = pos[tid] + d
        accum[tid] = accum[tid] + d
    delta[tid] = wp.vec3(0.0)
    delta_weight[tid] = 0.0


@wp.kernel
def finalize_contact_projection_kernel(
    dt: float,
    # outputs (in-place)
    accum: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    accum_kept: wp.array[wp.vec3],
):
    """Fold the applied projection displacement into velocities (PBD-consistent).

    ``accum_kept`` keeps the substep's total correction after ``accum`` is
    cleared, so the impulse pass can still feed it back to the rigid side --
    that pass has to run after the reaction accumulation zeroes ``body_f``.
    """
    tid = wp.tid()
    if dt > 0.0:
        vel[tid] = vel[tid] + accum[tid] / dt
    accum_kept[tid] = accum[tid]
    accum[tid] = wp.vec3(0.0)


@wp.kernel
def count_particle_contacts_kernel(
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    # outputs
    per_particle: wp.array[float],
):
    """Contacts per particle, so a shared correction can be split between them."""
    t_id = wp.tid()
    if t_id >= wp.min(contact_max, contact_count[0]):
        return
    wp.atomic_add(per_particle, soft_contact_particle[t_id], 1.0)


@wp.kernel
def accumulate_projection_impulse_kernel(
    dt: float,
    accum_kept: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    per_particle: wp.array[float],
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_enabled: wp.array[int],
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """Feed the position projection back to the rigid side as an impulse.

    A position-level solve moves only the cloth; in a co-sim the gripper cannot
    feel it unless the correction is converted to a force. Newton's third law
    then comes from the projection itself rather than from residual penetration,
    which is what lets the ``slack`` (kept only to keep the penalty channel
    alive) go to zero.
    """
    t_id = wp.tid()
    if t_id >= wp.min(contact_max, contact_count[0]):
        return
    shape = contact_shape[t_id]
    body = shape_body[shape]
    if body < 0 or body_enabled[body] == 0:
        return
    particle = soft_contact_particle[t_id]
    share = per_particle[particle]
    if share <= 0.0 or dt <= 0.0:
        return
    n = contact_normal[t_id]
    dn = wp.dot(n, accum_kept[particle])
    if dn <= 0.0:
        return                      # only outward pushes carry a reaction
    f = n * (particle_mass[particle] * dn / (dt * dt) / share)
    reaction = -f
    X_wb = body_q[body]
    cp = wp.transform_point(X_wb, contact_body_pos[t_id])
    com = wp.transform_point(X_wb, body_com[body])
    wp.atomic_add(body_f, body, wp.spatial_vector(reaction, wp.cross(cp - com, reaction)))


# ---------------------------------------------------------------------------
# E3: triangle-level rigid contact against a precomputed shape SDF.
#
# Vertex-sphere detection cannot see a blade crossing the INTERIOR of a cloth
# triangle; the stack has been hiding that by inflating ``particle_radius``
# until the sphere covers the mesh aperture, which couples the contact model to
# the cloth tessellation. These kernels replace that with the mesh-independent
# form: constrain min_{x in tri} SDF_shape(x) >= h, h = cloth half-thickness.
# The shape SDF is baked ONCE per shape (rigid), sampled trilinearly, and the
# per-triangle minimum is found by 3 vertices + centroid seeding followed by a
# few projected steepest-descent steps in barycentric coordinates. The
# correction is distributed to the three vertices by barycentric weight, so one
# constraint covers vertex-face, edge-face and face-vertex at once.
# ---------------------------------------------------------------------------


@wp.func
def sdf_grid_sample(
    sdf: wp.array(dtype=float),
    base: int,
    nx: int,
    ny: int,
    nz: int,
    origin: wp.vec3,
    inv_voxel: float,
    bg: float,
    p: wp.vec3,
):
    """Trilinear lookup in a dense per-shape SDF grid (shape-local frame).

    Returns ``bg`` (a large positive number) outside the grid, so a triangle
    that never enters the shape's neighbourhood costs one compare.
    """
    q = (p - origin) * inv_voxel
    fx = q[0]
    fy = q[1]
    fz = q[2]
    if fx < 0.0:
        return bg
    if fy < 0.0:
        return bg
    if fz < 0.0:
        return bg
    ix = int(fx)
    iy = int(fy)
    iz = int(fz)
    if ix >= nx - 1:
        return bg
    if iy >= ny - 1:
        return bg
    if iz >= nz - 1:
        return bg
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)
    s0 = base + (ix * ny + iy) * nz + iz
    s1 = base + (ix * ny + iy + 1) * nz + iz
    s2 = base + ((ix + 1) * ny + iy) * nz + iz
    s3 = base + ((ix + 1) * ny + iy + 1) * nz + iz
    c00 = sdf[s0] * (1.0 - tz) + sdf[s0 + 1] * tz
    c01 = sdf[s1] * (1.0 - tz) + sdf[s1 + 1] * tz
    c10 = sdf[s2] * (1.0 - tz) + sdf[s2 + 1] * tz
    c11 = sdf[s3] * (1.0 - tz) + sdf[s3 + 1] * tz
    c0 = c00 * (1.0 - ty) + c01 * ty
    c1 = c10 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tx) + c1 * tx


@wp.func
def sdf_grid_gradient(
    sdf: wp.array(dtype=float),
    base: int,
    nx: int,
    ny: int,
    nz: int,
    origin: wp.vec3,
    inv_voxel: float,
    bg: float,
    voxel: float,
    p: wp.vec3,
):
    """Central-difference gradient of the SDF grid (unit-length if non-degenerate)."""
    e = voxel
    gx = sdf_grid_sample(sdf, base, nx, ny, nz, origin, inv_voxel, bg, p + wp.vec3(e, 0.0, 0.0)) - sdf_grid_sample(
        sdf, base, nx, ny, nz, origin, inv_voxel, bg, p - wp.vec3(e, 0.0, 0.0)
    )
    gy = sdf_grid_sample(sdf, base, nx, ny, nz, origin, inv_voxel, bg, p + wp.vec3(0.0, e, 0.0)) - sdf_grid_sample(
        sdf, base, nx, ny, nz, origin, inv_voxel, bg, p - wp.vec3(0.0, e, 0.0)
    )
    gz = sdf_grid_sample(sdf, base, nx, ny, nz, origin, inv_voxel, bg, p + wp.vec3(0.0, 0.0, e)) - sdf_grid_sample(
        sdf, base, nx, ny, nz, origin, inv_voxel, bg, p - wp.vec3(0.0, 0.0, e)
    )
    g = wp.vec3(gx, gy, gz)
    ln = wp.length(g)
    if ln < 1.0e-12:
        return wp.vec3(0.0, 0.0, 1.0)
    return g / ln


@wp.kernel
def bake_shape_sdf_kernel(
    mesh: wp.uint64,
    origin: wp.vec3,
    voxel: float,
    nx: int,
    ny: int,
    nz: int,
    base: int,
    max_dist: float,
    # outputs
    sdf: wp.array(dtype=float),
):
    """One-time signed-distance bake of a rigid shape onto a dense grid.

    Signed by ``mesh_query_point_sign_parity`` so points inside the solid are
    unambiguous (the shape meshes are closed thick solids).
    """
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) * voxel, float(j) * voxel, float(k) * voxel)
    d = max_dist
    q = wp.mesh_query_point_sign_parity(mesh, p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = wp.length(p - cp) * q.sign
    sdf[base + (i * ny + j) * nz + k] = d


@wp.kernel
def project_tri_sdf_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    sdf: wp.array(dtype=float),
    sdf_base: wp.array(dtype=int),
    sdf_nx: wp.array(dtype=int),
    sdf_ny: wp.array(dtype=int),
    sdf_nz: wp.array(dtype=int),
    sdf_origin: wp.array(dtype=wp.vec3),
    voxel: float,
    bg: float,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    half_thickness: float,
    max_correction: float,
    refine_steps: int,
    # outputs
    delta: wp.array(dtype=wp.vec3),
    delta_weight: wp.array(dtype=float),
):
    """Constrain min_{x in tri} SDF_shape(x) >= h and share the fix barycentrically.

    One thread per (shape slot, cloth triangle). Mesh-independent: nothing here
    reads ``particle_radius`` or the triangle's edge lengths.
    """
    tid = wp.tid()
    slot = tid / tri_count
    t = tid - slot * tri_count

    shape = slot_shape[slot]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)

    i0 = tri_indices[t, 0]
    i1 = tri_indices[t, 1]
    i2 = tri_indices[t, 2]
    a = wp.transform_point(X_sw, pos[i0])
    b = wp.transform_point(X_sw, pos[i1])
    c = wp.transform_point(X_sw, pos[i2])

    base = sdf_base[slot]
    nx = sdf_nx[slot]
    ny = sdf_ny[slot]
    nz = sdf_nz[slot]
    org = sdf_origin[slot]
    inv_voxel = 1.0 / voxel

    # seed: three vertices + centroid
    third = 1.0 / 3.0
    w = wp.vec3(third, third, third)
    p = a * third + b * third + c * third
    best = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, p)
    da = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, a)
    if da < best:
        best = da
        w = wp.vec3(1.0, 0.0, 0.0)
    db = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, b)
    if db < best:
        best = db
        w = wp.vec3(0.0, 1.0, 0.0)
    dc = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, c)
    if dc < best:
        best = dc
        w = wp.vec3(0.0, 0.0, 1.0)
    if best >= bg:
        return

    # projected steepest descent in barycentric coordinates (scale-free steps)
    step = float(0.5)
    for _s in range(refine_steps):
        p = a * w[0] + b * w[1] + c * w[2]
        g = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
        ga = wp.dot(g, a)
        gb = wp.dot(g, b)
        gc = wp.dot(g, c)
        m = (ga + gb + gc) * third
        gw = wp.vec3(ga - m, gb - m, gc - m)
        gn = wp.length(gw)
        if gn > 1.0e-12:
            cand = w - gw * (step / gn)
            cand = wp.vec3(wp.max(cand[0], 0.0), wp.max(cand[1], 0.0), wp.max(cand[2], 0.0))
            s = cand[0] + cand[1] + cand[2]
            if s > 1.0e-9:
                cand = cand / s
                pc = a * cand[0] + b * cand[1] + c * cand[2]
                dcand = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, pc)
                if dcand < best:
                    best = dcand
                    w = cand
        step = step * 0.5

    if best >= half_thickness:
        return

    p = a * w[0] + b * w[1] + c * w[2]
    n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
    push = wp.min(half_thickness - best, max_correction)
    n_world = wp.transform_vector(X_ws, n_local)
    denom = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    scale = push / wp.max(denom, 1.0e-6)
    if w[0] > 0.0:
        wp.atomic_add(delta, i0, n_world * (scale * w[0]))
        wp.atomic_add(delta_weight, i0, w[0])
    if w[1] > 0.0:
        wp.atomic_add(delta, i1, n_world * (scale * w[1]))
        wp.atomic_add(delta_weight, i1, w[1])
    if w[2] > 0.0:
        wp.atomic_add(delta, i2, n_world * (scale * w[2]))
        wp.atomic_add(delta_weight, i2, w[2])


@wp.func
def tri_sdf_closest(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    sdf: wp.array(dtype=float),
    base: int,
    nx: int,
    ny: int,
    nz: int,
    org: wp.vec3,
    inv_voxel: float,
    voxel: float,
    bg: float,
    refine_steps: int,
):
    """Triangle's minimum-SDF point: 3-vertex + centroid seeding, then projected
    steepest descent in barycentric coordinates. Returns (w, best)."""
    third = 1.0 / 3.0
    w = wp.vec3(third, third, third)
    p = a * third + b * third + c * third
    best = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, p)
    da = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, a)
    if da < best:
        best = da
        w = wp.vec3(1.0, 0.0, 0.0)
    db = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, b)
    if db < best:
        best = db
        w = wp.vec3(0.0, 1.0, 0.0)
    dc = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, c)
    if dc < best:
        best = dc
        w = wp.vec3(0.0, 0.0, 1.0)
    step = float(0.5)
    for _s in range(refine_steps):
        p = a * w[0] + b * w[1] + c * w[2]
        g = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
        ga = wp.dot(g, a)
        gb = wp.dot(g, b)
        gc = wp.dot(g, c)
        m = (ga + gb + gc) * third
        gw = wp.vec3(ga - m, gb - m, gc - m)
        gn = wp.length(gw)
        if gn > 1.0e-12:
            cand = w - gw * (step / gn)
            cand = wp.vec3(wp.max(cand[0], 0.0), wp.max(cand[1], 0.0), wp.max(cand[2], 0.0))
            s = cand[0] + cand[1] + cand[2]
            if s > 1.0e-9:
                cand = cand / s
                pc = a * cand[0] + b * cand[1] + c * cand[2]
                dcand = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, pc)
                if dcand < best:
                    best = dcand
                    w = cand
        step = step * 0.5
    return w, best


@wp.func
def sdf_mesh_query(mesh: wp.uint64, max_dist: float, bg: float, p: wp.vec3):
    """Exact signed distance and unit gradient of a closed shape mesh at ``p``.

    Returns ``(d, n)``: ``d`` negative inside the solid, ``n`` the unit gradient
    of that field (the outward direction).  Both are exact -- the closest point
    on the mesh is found by the BVH and the distance is a length, so there is no
    interpolation and therefore none of the edge rounding a voxel field carries.
    ``d = bg`` when the nearest surface is beyond ``max_dist`` (the caller's
    rejection radius), which is the same sentinel the grid path returns off-grid.
    """
    d = bg
    n = wp.vec3(0.0, 0.0, 1.0)
    q = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        delta = p - cp
        ln = wp.length(delta)
        if ln < 1.0e-9:
            # exactly on the surface: the distance gradient is the face normal
            d = 0.0
            n = wp.mesh_eval_face_normal(mesh, q.face)
        else:
            d = ln * q.sign
            n = delta * (q.sign / ln)
    return d, n


@wp.func
def tri_sdf_closest_mesh(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    mesh: wp.uint64,
    bg: float,
    cull: float,
    refine_steps: int,
):
    """``tri_sdf_closest`` on the exact field: same search, exact evaluations.

    Seeding (3 vertices + centroid) and the projected steepest descent in
    barycentric space are line-for-line the grid version's; only the field
    lookups are exact.  One rejection is added ahead of them, and it is
    conservative rather than heuristic: a signed distance field is 1-Lipschitz
    and every point of the triangle lies within ``reach`` of the centroid, so
    ``SDF(centroid) > cull + reach`` implies ``min_tri SDF > cull``.  The query
    radius that decides it is a few mm, so a triangle nowhere near the blade
    costs one BVH root test.
    """
    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    w = wp.vec3(third, third, third)
    # radius that keeps every point of the triangle resolvable once the centroid
    # is inside the cull band: |SDF(x)| <= SDF(g) + reach <= cull + 2*reach.
    rq = cull + 2.0 * reach
    # Each query returns distance AND gradient, and the descent only ever needs
    # the gradient AT THE CURRENT BEST POINT -- which is the point whose query
    # produced ``best``.  So carry its normal along instead of re-querying: the
    # evaluated points are exactly the same, 7 queries per triangle instead of
    # 11, and the caller gets the contact normal for free.
    best, nbest = sdf_mesh_query(mesh, cull + reach, bg, g)
    if best >= bg:
        return w, bg, nbest
    da, na = sdf_mesh_query(mesh, rq, bg, a)
    if da < best:
        best = da
        w = wp.vec3(1.0, 0.0, 0.0)
        nbest = na
    db, nb = sdf_mesh_query(mesh, rq, bg, b)
    if db < best:
        best = db
        w = wp.vec3(0.0, 1.0, 0.0)
        nbest = nb
    dc, nc = sdf_mesh_query(mesh, rq, bg, c)
    if dc < best:
        best = dc
        w = wp.vec3(0.0, 0.0, 1.0)
        nbest = nc
    step = float(0.5)
    for _s in range(refine_steps):
        gvec = nbest
        ga = wp.dot(gvec, a)
        gb = wp.dot(gvec, b)
        gc = wp.dot(gvec, c)
        m = (ga + gb + gc) * third
        gw = wp.vec3(ga - m, gb - m, gc - m)
        gn = wp.length(gw)
        if gn > 1.0e-12:
            cand = w - gw * (step / gn)
            cand = wp.vec3(wp.max(cand[0], 0.0), wp.max(cand[1], 0.0), wp.max(cand[2], 0.0))
            sm = cand[0] + cand[1] + cand[2]
            if sm > 1.0e-9:
                cand = cand / sm
                pc = a * cand[0] + b * cand[1] + c * cand[2]
                dcand, ncand = sdf_mesh_query(mesh, rq, bg, pc)
                if dcand < best:
                    best = dcand
                    w = cand
                    nbest = ncand
        step = step * 0.5
    return w, best, nbest


@wp.kernel
def eval_tri_sdf_contact_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    tri_stiffness: wp.array(dtype=float),
    sdf: wp.array(dtype=float),
    sdf_base: wp.array(dtype=int),
    sdf_nx: wp.array(dtype=int),
    sdf_ny: wp.array(dtype=int),
    sdf_nz: wp.array(dtype=int),
    sdf_origin: wp.array(dtype=wp.vec3),
    voxel: float,
    bg: float,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    half_thickness: float,
    max_depth: float,
    refine_steps: int,
    # R13c: hardening compression law on the compliant SDF force (0 = off)
    shape_hardening_eps: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    # R13f: Coulomb friction on this channel (inert unless _R13F_SDF_FRICTION)
    pos_prev: wp.array(dtype=wp.vec3),
    body_q_prev: wp.array(dtype=wp.transform),
    shape_material_mu: wp.array(dtype=float),
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
    # R13g: frozen contact point (inert unless _R13G_SDF_FREEZE)
    tri_sdf_w: wp.array(dtype=wp.vec3),
    resolve_w: int,
    # R16-A2': per-slot shape mesh for the exact backend (inert unless
    # _R16_SDF_EXACT); ``mesh_max_dist`` only bounds the single gradient query.
    sdf_mesh: wp.array(dtype=wp.uint64),
    mesh_max_dist: float,
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessians: wp.array(dtype=wp.mat33),
):
    """COMPLIANT triangle-level contact: a penalty force into the implicit solve.

    The position-projection form of this constraint fights the penalty channel:
    the projection removes exactly the overlap the penalty needs to produce a
    force, so the reaction collapses when the collision radius is retired
    (measured: 8.5 N at r=4 mm -> 0.8 N at r=0.5 mm, and raising ke does not
    bring it back). Here the SAME constraint carries the force:

        F = k_tri * (h - min_{x in tri} SDF)      along the SDF gradient
        k_tri = E_t * A_tri / t0                  (material, not a constant)

    so geometry and reaction come from one mechanism and nothing competes for the
    overlap. The compression that produces the force is the cloth's own
    transverse compliance, not an inflated collision radius; sum_tri A_tri is the
    cloth's area whatever the tessellation, so the load is mesh independent.
    """
    tid = wp.tid()
    slot = tid / tri_count
    t = tid - slot * tri_count

    shape = slot_shape[slot]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)

    i0 = tri_indices[t, 0]
    i1 = tri_indices[t, 1]
    i2 = tri_indices[t, 2]
    a = wp.transform_point(X_sw, pos[i0])
    b = wp.transform_point(X_sw, pos[i1])
    c = wp.transform_point(X_sw, pos[i2])

    base = sdf_base[slot]
    nx = sdf_nx[slot]
    ny = sdf_ny[slot]
    nz = sdf_nz[slot]
    org = sdf_origin[slot]
    inv_voxel = 1.0 / voxel

    w = wp.vec3(0.0)
    best = float(0.0)
    n_exact = wp.vec3(0.0, 0.0, 1.0)
    if _R16_SDF_EXACT != 0:
        # R16-A2': same search, exact field.  The freeze switch is a grid-path
        # remedy for the winner-take-all redraw and is not combined with it.
        w, best, n_exact = tri_sdf_closest_mesh(a, b, c, sdf_mesh[slot], bg, half_thickness, refine_steps)
    elif _R13G_SDF_FREEZE == 0:
        w, best = tri_sdf_closest(a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps)
    else:
        # Resolve the barycentric contact point once per substep, then hold it.
        if resolve_w != 0:
            w, best = tri_sdf_closest(a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps)
            tri_sdf_w[tid] = w
        else:
            w = tri_sdf_w[tid]
            best = sdf_grid_sample(
                sdf, base, nx, ny, nz, org, inv_voxel, bg, a * w[0] + b * w[1] + c * w[2]
            )
    if best >= half_thickness:
        return

    depth = wp.min(half_thickness - best, max_depth)
    p = a * w[0] + b * w[1] + c * w[2]
    n_local = wp.vec3(0.0, 0.0, 1.0)
    if _R16_SDF_EXACT != 0:
        # the gradient at the winning point, carried out of the search
        n_local = n_exact
    else:
        n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
    n_world = wp.transform_vector(X_ws, n_local)
    k = tri_stiffness[t]
    f = n_world * (k * depth)
    nn = wp.outer(n_world, n_world) * k
    # R13c: hardening compression law INSIDE the SDF force core.  R9 hung the
    # law on the vertex penalty kernel only; the compliant SDF stack zeroes that
    # channel (shape_contact_ke -> 0), so the law never ran.  Same closed form as
    # _compute_body_particle_contact_force, with the strain referenced to the
    # cloth thickness t0 = 2*particle_radius (the R13 gate-1 fix), averaged over
    # the triangle's three vertices (uniform radius => exact).  eps_max = 0
    # (default) keeps the three statements above untouched: bit-identical OFF.
    t0_h = (2.0 / 3.0) * (particle_radius[i0] + particle_radius[i1] + particle_radius[i2])
    hardening_inv_dmax = contact_hardening_inv_dmax(shape_hardening_eps, t0_h, shape)
    if hardening_inv_dmax > 0.0:
        den = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
        inv_den = 1.0 / den
        f = n_world * (k * depth * inv_den)
        nn = wp.outer(n_world, n_world) * (k * inv_den * inv_den)
    # R13f: Coulomb friction, same law/mixing as the vertex penalty channel.
    # f_n is the normal load this triangle actually carries (post-hardening);
    # u_rel is the relative translation of the coincident material points over
    # the substep -- the cloth contact point minus the shape point under it.
    # OFF branch (_R13F_SDF_FRICTION == 0) leaves f/nn exactly as above.
    if _R13F_SDF_FRICTION != 0:
        mu = wp.sqrt(friction_mu * shape_material_mu[shape])
        if mu > 0.0 and dt > 0.0:
            f_n = k * depth
            if hardening_inv_dmax > 0.0:
                den_f = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
                f_n = f_n / den_f
            a_p = wp.transform_point(X_sw, pos_prev[i0])
            b_p = wp.transform_point(X_sw, pos_prev[i1])
            c_p = wp.transform_point(X_sw, pos_prev[i2])
            # cloth displacement of the contact point, in world
            dp_cloth = wp.transform_vector(
                X_ws, (a * w[0] + b * w[1] + c * w[2]) - (a_p * w[0] + b_p * w[1] + c_p * w[2])
            )
            # displacement of the SHAPE material point that sits under it
            p_world = wp.transform_point(X_ws, p)
            X_ws_prev = shape_transform[shape]
            if body >= 0:
                X_ws_prev = body_q_prev[body] * shape_transform[shape]
            dp_body = p_world - wp.transform_point(X_ws_prev, p)
            u_rel = dp_cloth - dp_body
            eps_u = friction_epsilon * dt
            f_t, k_t = compute_projected_isotropic_friction(mu, f_n, n_world, u_rel, eps_u)
            f = f + f_t
            nn = nn + k_t
    # R13g: OFF keeps the exact w_i^2 diagonal blocks; ON uses the row-sum lump.
    hw = wp.vec3(w[0] * w[0], w[1] * w[1], w[2] * w[2])
    if _R13G_SDF_HESS != 0:
        hw = w
    if w[0] > 0.0:
        wp.atomic_add(forces, i0, f * w[0])
        wp.atomic_add(hessians, i0, nn * hw[0])
    if w[1] > 0.0:
        wp.atomic_add(forces, i1, f * w[1])
        wp.atomic_add(hessians, i1, nn * hw[1])
    if w[2] > 0.0:
        wp.atomic_add(forces, i2, f * w[2])
        wp.atomic_add(hessians, i2, nn * hw[2])


@wp.kernel
def accumulate_tri_sdf_reaction_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    tri_stiffness: wp.array(dtype=float),
    sdf: wp.array(dtype=float),
    sdf_base: wp.array(dtype=int),
    sdf_nx: wp.array(dtype=int),
    sdf_ny: wp.array(dtype=int),
    sdf_nz: wp.array(dtype=int),
    sdf_origin: wp.array(dtype=wp.vec3),
    voxel: float,
    bg: float,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_enabled: wp.array(dtype=int),
    half_thickness: float,
    max_depth: float,
    refine_steps: int,
    # R13c: same hardening law as the force kernel (0 = off, bit-identical)
    shape_hardening_eps: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    # R13f: same friction as the force kernel (inert unless _R13F_SDF_FRICTION)
    pos_prev: wp.array(dtype=wp.vec3),
    body_q_prev: wp.array(dtype=wp.transform),
    shape_material_mu: wp.array(dtype=float),
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
    # R13g: same frozen contact point the force kernel used (inert unless flag)
    tri_sdf_w: wp.array(dtype=wp.vec3),
    resolve_w: int,
    # R16-A2': per-slot shape mesh for the exact backend (inert unless
    # _R16_SDF_EXACT); ``mesh_max_dist`` only bounds the single gradient query.
    sdf_mesh: wp.array(dtype=wp.uint64),
    mesh_max_dist: float,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Equal-and-opposite wrench of the compliant triangle contact onto the body.

    Re-evaluates the same force at the converged positions and adds -F to the
    rigid side at the contact point (world frame, referenced at the body COM),
    matching accumulate_body_reaction_kernel's convention. This is the reaction
    channel the position-projection form could not provide.
    """
    tid = wp.tid()
    slot = tid / tri_count
    t = tid - slot * tri_count

    shape = slot_shape[slot]
    body = shape_body[shape]
    if body < 0:
        return
    if body_enabled[body] == 0:
        return
    X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)

    i0 = tri_indices[t, 0]
    i1 = tri_indices[t, 1]
    i2 = tri_indices[t, 2]
    a = wp.transform_point(X_sw, pos[i0])
    b = wp.transform_point(X_sw, pos[i1])
    c = wp.transform_point(X_sw, pos[i2])

    base = sdf_base[slot]
    nx = sdf_nx[slot]
    ny = sdf_ny[slot]
    nz = sdf_nz[slot]
    org = sdf_origin[slot]
    inv_voxel = 1.0 / voxel

    w = wp.vec3(0.0)
    best = float(0.0)
    n_exact = wp.vec3(0.0, 0.0, 1.0)
    if _R16_SDF_EXACT != 0:
        # R16-A2': identical query to the force pass, so the two halves of the
        # contact cannot disagree about where or how deep it is.
        w, best, n_exact = tri_sdf_closest_mesh(a, b, c, sdf_mesh[slot], bg, half_thickness, refine_steps)
    elif _R13G_SDF_FREEZE == 0:
        w, best = tri_sdf_closest(a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps)
    else:
        # the point the force kernel actually used this substep.  resolve_w != 0
        # means the force pass has not run yet (``tri_sdf_w`` is still the zero
        # vector, which is NOT a barycentric point), so resolve it here instead.
        if resolve_w != 0:
            w, best = tri_sdf_closest(a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps)
        else:
            w = tri_sdf_w[tid]
            best = sdf_grid_sample(
                sdf, base, nx, ny, nz, org, inv_voxel, bg, a * w[0] + b * w[1] + c * w[2]
            )
    if best >= half_thickness:
        return

    depth = wp.min(half_thickness - best, max_depth)
    p_local = a * w[0] + b * w[1] + c * w[2]
    n_local = wp.vec3(0.0, 0.0, 1.0)
    if _R16_SDF_EXACT != 0:
        n_local = n_exact
    else:
        n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p_local)
    n_world = wp.transform_vector(X_ws, n_local)
    f_n = tri_stiffness[t] * depth
    t0_h = (2.0 / 3.0) * (particle_radius[i0] + particle_radius[i1] + particle_radius[i2])
    hardening_inv_dmax = contact_hardening_inv_dmax(shape_hardening_eps, t0_h, shape)
    if hardening_inv_dmax > 0.0:
        den = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
        f_n = f_n / den
    reaction = n_world * (-f_n)
    p_world = wp.transform_point(X_ws, p_local)
    # R13f: the tangential half of the same contact, equal and opposite.
    # Re-evaluated with the identical inputs as the force kernel so the two
    # sides cannot disagree.  OFF branch leaves ``reaction`` untouched.
    if _R13F_SDF_FRICTION != 0:
        mu = wp.sqrt(friction_mu * shape_material_mu[shape])
        if mu > 0.0 and dt > 0.0:
            a_p = wp.transform_point(X_sw, pos_prev[i0])
            b_p = wp.transform_point(X_sw, pos_prev[i1])
            c_p = wp.transform_point(X_sw, pos_prev[i2])
            dp_cloth = wp.transform_vector(
                X_ws, p_local - (a_p * w[0] + b_p * w[1] + c_p * w[2])
            )
            X_ws_prev = body_q_prev[body] * shape_transform[shape]
            dp_body = p_world - wp.transform_point(X_ws_prev, p_local)
            u_rel = dp_cloth - dp_body
            f_t, _k_t = compute_projected_isotropic_friction(
                mu, f_n, n_world, u_rel, friction_epsilon * dt
            )
            reaction = reaction - f_t
    com = wp.transform_point(body_q[body], body_com[body])
    wp.atomic_add(body_f, body, wp.spatial_vector(reaction, wp.cross(p_world - com, reaction)))
