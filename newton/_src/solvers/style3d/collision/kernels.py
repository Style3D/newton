# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point
from newton._src.solvers.vbd.rigid_vbd_kernels import (
    evaluate_body_particle_contact,
)


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
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    shape_contact_ke: wp.array[float],
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
                shape_contact_ke[shape_idx],
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
                shape_contact_ke[shape_idx],
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
        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_idx,
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            shape_contact_ke[shape_idx],
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
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
    soft_contact_particle: wp.array[int],
    contact_count: wp.array[int],
    contact_max: int,
    shape_contact_ke: wp.array[float],
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
                shape_contact_ke[shape_idx],
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
                shape_contact_ke[shape_idx],
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
        body_contact_force, _hessian = evaluate_body_particle_contact(
            particle_idx,
            pos[particle_idx],
            pos_prev[particle_idx],
            t_id,
            shape_contact_ke[shape_idx],
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
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
