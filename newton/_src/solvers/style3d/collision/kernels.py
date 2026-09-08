# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.geometry import ParticleFlags
from newton._src.geometry.kernels import (
    TRI_CONTACT_FEATURE_EDGE_AB,
    TRI_CONTACT_FEATURE_EDGE_AC,
    TRI_CONTACT_FEATURE_EDGE_BC,
    TRI_CONTACT_FEATURE_VERTEX_A,
    TRI_CONTACT_FEATURE_VERTEX_B,
    TRI_CONTACT_FEATURE_VERTEX_C,
    triangle_closest_point,
)
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

# T12: 锚用播种时的重心坐标（材料点）当参照，1 = 修后（默认），0 = 修前
# （每次牛顿迭代重搜的最近点）。留 0 这条只为做同二进制 A/B 与留档，不要在生产里设。
_T12_ANCHOR_SEEDW = wp.constant(int(__import__("os").environ.get("T12_ANCHOR_SEEDW", "1")))

# T14: per-substep broad phase for the tri-SDF contact channel.
# The contact kernel is launched over slots x tri_count (4 blades x 18564
# triangles = 74k) on EVERY Newton iteration (20 per substep), and a triangle
# nowhere near a blade still pays one BVH root test in
# ``tri_sdf_closest_mesh``'s centroid cull -- 20 times per substep, plus once
# more in the reaction pass.  With the flag ON the pairs that can possibly
# contact are gathered ONCE per substep into ``cand_idx`` and the 21 launches
# only spawn threads for those.
#
# The gather test is a PROVABLE SUPERSET of that centroid cull, not a heuristic:
# ``tri_sdf_closest_mesh`` rejects a triangle iff its centroid has no mesh point
# within ``h + reach`` (reach = the centroid's distance to the furthest of the
# three vertices).  The blade mesh is contained in its own AABB, so
#     dist(centroid, AABB) > h + reach  =>  dist(centroid, mesh) > h + reach
# and the pair provably early-exits.  The gather therefore keeps every pair with
# ``dist(centroid, AABB) <= h + reach + slack``; ``slack`` covers the cloth
# motion across the substep's 20 iterations, over which the list is held fixed.
# 0 = OFF: every statement below is dead code and the kernels are byte-for-byte
# the launches they were.
#
# MEASURED (4070 Ti Super, t12_flatten_a_mu050_s42, 666 frames, R16 exact +
# R13f friction + T8 anchor), and the reason this ships OFF:
#
#   gather is correct   0 pairs dropped in 6660 substeps (selfcheck kernel);
#                       candidates med 0 / max 1197 / mean 385 of 74256 pairs,
#                       of which med 0 / max 351 pass the real centroid cull.
#   but it is SLOWER    wall 160.7 s OFF -> 183.1 s ON (cap 16384 + fallback),
#                       170.0 s ON with the list sized to all pairs (no
#                       fallback launch).  Per force launch: 0.350 ms OFF,
#                       0.508 ms ON at the SAME dim, 0.477 ms ON at dim 2048.
#
# The cost the gather removes is not the cost that matters.  A launch in which
# EVERY pair fails the cull measures 0.057 ms (settle phase, no gripper near the
# cloth), i.e. the far-triangle cull is ~16% of the 46.6 s the force kernel
# spends over an episode; the other ~84% is the 7 exact mesh queries the NEAR
# triangles run, which the gather keeps by construction.  And packing those
# queries into the first ~12 warps costs MORE than leaving them scattered over
# ~2300: the per-launch time is the same (0.48 / 0.51 ms) whether the launch is
# 2048 threads or 74256, so the regression is where the work sits, not how many
# threads are spawned -- the mesh-BVH traversal is latency bound and the full
# scan was hiding that latency across the whole grid for free.
#
# Keep the flag for the ledger; the head to attack is the near-triangle query
# count (refine steps / query backend), not the pair count.
_T14_SDF_BROADPHASE = wp.constant(int(__import__("os").environ.get("T14_SDF_BROADPHASE", "0")))

# T14 round 3: hold the CONTACT GEOMETRY for a whole substep.
# The blade pose is constant inside a substep (co-sim writes body_q once per
# substep, and the 20 Newton iterations never touch it) and the cloth moves at
# most a millimetre or two over those iterations, so the closest point and the
# surface normal can be resolved ONCE and held in the shape's local frame --
# which is what this stack already does for the rigid-feature channel (contact
# list built once per substep, normal frozen for the substep).
#
# Iteration 0 runs the full ``tri_sdf_closest_mesh`` search exactly as before
# and records, per (slot, triangle): the barycentric point ``hold_w``, the blade
# surface point ``hold_p`` and the outward normal ``hold_n``, both in shape
# local.  Iterations 1..19 and the reaction pass then evaluate
#     best = dot(hold_n, x(w) - hold_p)
# -- a plane distance, no BVH query at all.  The force law, the hardening law,
# the friction law and the T8 anchor read the same statements they always did;
# only where ``best`` and the normal come from changes.
#
# This is the lever the first two rounds missed: the cost is 7 SERIAL mesh-BVH
# queries per near triangle x 21 launches per substep, a latency chain, which is
# why neither culling the far pairs (round 1) nor lagging the force (round 2)
# touched it.  HOLD takes the chain from 21 per substep to 1.
# 0 = OFF: every statement below is dead code.
# Requires the exact backend (R16_SDF_EXACT); the voxel path has no cached
# normal and is left alone.
_T14_SDF_HOLD = wp.constant(int(__import__("os").environ.get("T14_SDF_HOLD", "0")))

# T14 HOLD falsifier: on every HELD evaluation, ALSO run the full search and
# accumulate how far the held value is from it.  Pure diagnostic -- the force
# still uses the held value, so the run's physics is the HOLD run's physics; it
# just costs the search back.  This is what separates "the cache is wired wrong"
# from "the approximation is not second order in this regime".
_T14_SDF_HOLD_DIAG = wp.constant(int(__import__("os").environ.get("T14_SDF_HOLD_DIAG", "0")))

# T14 round 4 (i): skip a refinement query that provably cannot improve.
# The projected descent clamps the candidate to the simplex and renormalises;
# on a FLAT contact the winner is a vertex and the projected gradient points at
# that vertex, so ``cand`` lands exactly back on it -- (1+d,-x,-y) -> clamp ->
# (1+d,0,0) -> normalise -> (1,0,0), exactly.  The query point is then bitwise
# the point that already produced ``best``, so ``dcand < best`` is false and the
# query is dead weight.  Guard is on the QUERY POINT, not on ``cand``: the
# centroid seed's stored point is ``(a+b+c)/3`` while the barycentric rebuild is
# ``a/3+b/3+c/3``, which are not bitwise equal, so comparing ``cand`` to ``w``
# would be wrong there.  Bit-identical: the skipped query's result was discarded.
_T14_SDF_SKIPREF = wp.constant(int(__import__("os").environ.get("T14_SDF_SKIPREF", "0")))

# T14 round 4 (ii): tighter query radius for the vertex and refinement queries.
# A signed distance field is 1-Lipschitz, so every point p of the triangle obeys
#     |SDF(p)| <= |SDF(g)| + |p - g| <= |SDF(g)| + reach
# and a search radius of |SDF(g)| + reach therefore still contains the true
# closest point of EVERY triangle point -- the queries return the same closest
# point, just after pruning more of the BVH.  The old radius ``cull + 2*reach``
# is a worst case over ``SDF(g) <= cull``; in contact ``|SDF(g)| ~ h`` and the
# new radius is about half the old.  Bit-identical.
_T14_SDF_RQ = wp.constant(int(__import__("os").environ.get("T14_SDF_RQ", "0")))

# T14 round 4 (iii): give each of the four seeds its own thread.
# The serial search is a DEPENDENCY CHAIN of up to 7 mesh-BVH queries: centroid,
# three vertices, then three refinement steps that each need the previous
# winner's normal.  Rounds 1-3 established this kernel is latency bound, so the
# chain length -- not the query count -- is what costs.  Splitting the seeds
# across four threads makes each thread's chain 1 (its own seed) + refine_steps,
# and the force kernel takes the minimum of the four.
#
# NOT bit-identical, and deliberately so: the serial version refines only from
# the BEST seed, the parallel one refines from all four and takes the min.  The
# thread that owns the serial winner runs the identical statements from the
# identical start (``tri_sdf_refine_from`` is shared), so its result IS the
# serial result -- and the min over four can only be <= that.  The reported
# distance is therefore never SHALLOWER than today's, only equal or deeper,
# which is the safe direction for a penetration constraint.
_T14_SDF_PAR = wp.constant(int(__import__("os").environ.get("T14_SDF_PAR", "0")))

# T14 round 5: refine from EVERY seed, not just the best one.
# This is round 4's (iii) finding carried over to the voxel path.  The serial
# search picks the best of {centroid, a, b, c} and then descends only from that
# one; on compliant cloth that under-resolves the minimum (measured on the T13-C
# bench: refining all four and taking the min raised the pull-out force 1.20 ->
# 3.30 N and cut the creep 0.696 -> 0.088 mm/s).  On the EXACT backend that cost
# 36% because each extra descent is three more mesh-BVH queries; on the voxel
# backend a sample is a trilinear grid read, so the same improvement is close to
# free and does not even need the four threads -- one thread runs the four
# descents in sequence.
# 0 = OFF: the descent loop below is the single-seed one it always was.
_T14_SDF_ALLSEED = wp.constant(int(__import__("os").environ.get("T14_SDF_ALLSEED", "0")))

# T14 round 5 (3): fall back to the exact normal only where the grid's is wrong.
# A true signed distance field has |grad| == 1 everywhere except on the medial
# axis and where the surface is not smooth; the central difference collapses
# toward 0 across a gradient discontinuity, so ``| ||g|| - 1 | > 0.1`` flags an
# edge/corner for free -- the six samples were already taken for the gradient.
# Measured on the real blade at 0.25 mm (300k triangles, in contact n=177829):
#   detector fires on 15.54% of pairs
#   fired    : normal angle error vs exact  p50 21.05  p99 44.69  max 88.56 deg
#   not fired:                              p50  0.00  p99  8.69  max 25.63 deg
#   overall p99 40.45 -> 8.69 deg if the fired set falls back
# So this trades one on-surface exact query on ~1/6 of the contacting pairs for
# a 4.6x better normal tail.  DEPTH still comes from the grid -- only the normal
# direction is replaced, because depth already converges with the grid (p99
# 0.053 mm at 0.25 mm) while the edge normal does not (p99 44 deg at 0.25 mm,
# 43 deg at 0.125 mm).
# Modes: 3 = HYBRID -- no detector at all: every pair that reaches the force
#   law pays one exact query at its winner point and takes BOTH depth and
#   normal from it, so the quantities the force law consumes are the exact
#   ones and only the SEARCH stays on the grid.  Motivated by the voxel
#   tier's (2) regression on the mu=0.35 edge seeds (7/15 vs exact 12/12).
# 1 = take only the NORMAL from the fallback query (depth stays the
#   grid's, which is what the accuracy harness says already converges);
#   2 = take BOTH the normal and the DEPTH from it, for the case where the
#   grid's rounded tip also costs bite at the blade edge.
# 0 = OFF: the single ``sdf_grid_gradient`` call below is all that is emitted.
_T14_SDF_EDGE_EXACT = wp.constant(int(__import__("os").environ.get("T14_SDF_EDGE_EXACT", "0")))



# T15-A: O(1) EXACT closest-point backend for the tri-SDF contact channel.
#
# The exact backend (R16_SDF_EXACT) answers every query with
# ``wp.mesh_query_point_sign_normal`` -- a BVH descent over the 21302-face blade
# mesh.  ``tri_sdf_closest_mesh`` issues 7 of them per near triangle and they
# form a DEPENDENCY CHAIN (each refinement step needs the previous winner's
# normal), so the force kernel is latency bound: 0.369 ms/launch against
# 0.066 ms for the same kernel reading a trilinear voxel table.  T14 established
# that the query COUNT is locked by the force law (the deepest point on the
# triangle must be re-searched every Newton iteration) and that the blade cannot
# be decimated (4000 faces -> Hausdorff p99 0.26 mm, the working penetration
# depth).  So the only head left is the COST OF ONE QUERY.
#
# This backend replaces the tree walk with a table lookup that is still EXACT:
#   bake  -- on the same dense grid the SDF is baked on, store the id of the
#            NEAREST TRIANGLE at every grid point (int32), plus the mesh's
#            angle-weighted vertex pseudo-normals and edge pseudo-normals
#            (Baerentzen & Aanaes 2005, the same construction
#            ``mesh_query_point_sign_normal`` signs with).
#   query -- read the 8 corner face ids of the cell holding p, de-duplicate,
#            run the analytic point-triangle closest point (Ericson 5.1.5) on
#            each, keep the minimum, and sign it with the winning FEATURE's
#            pseudo-normal (face / edge / vertex).  Distance and normal are then
#            the same floating-point expressions the BVH path evaluates -- the
#            grid only chooses WHICH triangles to test.
#
# Not an approximation of the field, but an approximation of the CANDIDATE SET:
# the true nearest triangle of p is assumed to be the nearest triangle of one of
# the 8 corners.  That holds wherever the surface is locally resolved by the
# grid and can fail near the medial axis or on features thinner than a cell; the
# T15 harness measures exactly that, and it is the only thing that can differ
# from the BVH answer.
# 0 = OFF: ``sdf_query`` calls ``sdf_mesh_query`` and nothing below is reached.
_T15_SDF_GRIDEXACT = wp.constant(int(__import__("os").environ.get("T15_SDF_GRIDEXACT", "0")))

# T15-A (ii): two-stage candidate set.  Stage 1 is the 8 cell corners above;
# stage 2 tests ONLY the winner face's VERTEX ONE-RING (the faces sharing at
# least one vertex with it, precomputed as a CSR, ~12 per face on the blade).
# Rationale: when the grid cell is finer than a mesh triangle, a point whose
# true nearest face is missed by the corners still has a corner whose nearest
# face is ADJACENT to the true one, so one ring closes the gap.  Cost is a
# bounded number of extra point-triangle tests, no BVH, no branch divergence
# beyond the ring length.  This is NOT the r=1 (4x4x4 = 64 grid point)
# neighbourhood -- that reads 8x the table; this reads none of it.
# 0 = OFF: stage 2 is dead code and the query is the 8-corner one.
_T15_SDF_GX_RING = wp.constant(int(__import__("os").environ.get("T15_SDF_GX_RING", "0")))

# T16-lite: TOLERANCE on the closest-point search's argmin.
#
# On a nominally FLAT contact ``min_{x in tri} SDF(x)`` is DEGENERATE: every
# point of a cloth triangle parallel to the blade face is the same distance
# away, so which point wins is decided by the last bit of the arithmetic.  The
# T8 anchor seeds its material point from that winner, so the whole tangential
# law is applied at an essentially arbitrary point -- this is the winner-take-all
# pathology ``_R13G_SDF_FREEZE``'s comment block already describes, and it is
# what made two backends that agree on the DEPTH to 6.7e-9 mm disagree on the
# CONTACT POINT in 50.98% of flat triangles (26444/121968 of them with the depth
# bitwise identical and the point moved across the whole triangle).
#
# The minimal cure is to stop treating a tie as a win: a candidate replaces the
# incumbent only if it is deeper by more than ``r * h``.  With r = 1e-3 and
# h = 0.5 mm that is 0.5 um -- far below any physical depth (median penetration
# 0.193 mm) and far above the float32 noise that currently decides the tie.  On
# a flat contact the four seeds are then all "equal" and the CENTROID (the first
# seed) wins deterministically; on a tilted contact the genuinely deepest vertex
# still wins by millimetres.
# 0 = OFF: the strict ``<`` this file always used, bit-identical.
_T16_SDF_ARGMIN_TOL = wp.constant(float(__import__("os").environ.get("T16_SDF_ARGMIN_TOL", "0")))

# T16 SOFT: make the choice of contact point CONTINUOUS instead of a switch.
#
# ``_T16_SDF_ARGMIN_TOL`` removed the coin flip but replaced it with a THRESHOLD:
# under load the four candidates' depths hover right around r*h, so the winner
# still jumps 1-2 mm across the triangle, only now driven by physical noise
# crossing the threshold rather than by float noise.  Measured on the bench that
# shows up as same-arm repeat scatter going from bitwise-identical to 12%, the
# low-load delivery ratio turning NEGATIVE and the first-frame slip changing sign.
#
# SOFT replaces the hard pick of a STARTING POINT with a softmin over the four
# seeds, and then runs the SAME descent from there:
#     alpha_i  ~  exp(-(d_i - d_min) / (tau * h))          i = centroid, a, b, c
#     w_soft   =  sum_i alpha_i * w_i     (already barycentric: sum alpha = 1)
#     w        =  refine(w_soft)          the same 3 projected-gradient steps
# and the depth AND the normal are evaluated at that final point, so the force
# law and the anchor read one self-consistent point.
#
# Only the four SEEDS enter the blend, deliberately.  They average EXACTLY to the
# centroid ((wg + wa + wb + wc)/4 = (1/3,1/3,1/3)) and their depths are continuous
# in the triangle's pose, so the blend is continuous everywhere and lands on the
# centroid on a flat contact for both query backends.  Feeding the REFINED point
# into the blend instead re-injects the very arbitrary pick this is removing --
# measured: it leaves 50.941% of flat triangles disagreeing between backends,
# against 0.173% for the seeds-only blend.  The refinement is not dropped, it is
# moved: it now starts FROM w_soft, so the blade-tip regime keeps the interior
# minimum it needs (on a flat contact the projected gradient is zero and the
# descent does not move, so (1a) is unaffected).
#   flat contact  (spreads << tau*h) -> all candidates equal weight -> w_soft is
#     the centroid, and the two query backends agree bitwise;
#   tilted contact (spreads >> tau*h) -> the deepest vertex takes essentially all
#     the weight -> same answer as today;
#   in between      -> a continuous interpolation, with no threshold to cross.
# The depth this reports is at most O(tau*h) shallower than the true minimum
# (2 um at tau = 4e-3, h = 0.5 mm), i.e. ~1% of the working penetration.
# Mutually exclusive with the TOL knob: when SOFT is non-zero, TOL is forced to 0.
# 0 = OFF: every statement below is dead code.
_T16_SDF_ARGMIN_SOFT = wp.constant(float(__import__("os").environ.get("T16_SDF_ARGMIN_SOFT", "0")))


# T16 diagnostic: where does the contact point sit, and how is the load shared?
# Pure accumulator, the force law never reads it.  Answers the question the
# tolerance experiment raised -- is the side effect coming from "the material
# point sits at the centroid, so the load is split 1/3 each" rather than from the
# threshold switch itself.  Layout of ``w_diag`` (32 floats, atomically summed
# over every pair that reaches the force law):
#   0 n | 1 centroid | 2 vertex | 3 edge/interior
#   4 sum max(w) | 5 sum (w.w) | 6 sum f_n | 7 sum f_n*max(w) | 8 sum f_n*(w.w)
#   9 max f_n | 10 sum depth | 11 max max(w)
#   12..21 histogram of max(w) over [1/3, 1] in 10 equal bins
# ``T16_W_DIAG=2`` adds the CLOSEST FEATURE of the contact point (22 vertex,
# 23 edge, 24 face interior, 25 no-hit).  It is classified from an EXTRA BVH
# query at the contact point -- backend independent, so the BVH and GRIDEXACT
# arms are read on the same ruler, and it never touches the force path.
_T16_W_DIAG = wp.constant(int(__import__("os").environ.get("T16_W_DIAG", "0")))

# T16 SOFT gate: the DESCENT is a second noise amplifier, independent of the seed
# pick.  Its guard is ``gn > 1e-12`` where ``gn = |projected barycentric
# gradient|`` in METRES: on a flat contact ga = gb = gc exactly, so gn should be
# 0, but in float32 it is ~1e-9 m of pure rounding -- a thousand times the guard.
# The step is then ``cand = w - gw*(step/gn)``: the DIRECTION is noise but it is
# renormalised to a FULL 0.5 step, and ``dcand < best`` accepts on any
# improvement at all.  Measured: blending the seeds and then descending still
# leaves 33-40 full-width jumps in the tilt sweep and only 61-63% of flat
# triangles on the centroid, with the two backends disagreeing.
# This gate compares gn against the SAME length scale the softmin uses (tau*h),
# so a step is taken only when the barycentric gradient is real rather than
# rounding.  0 = OFF: the guard is the ``1e-12`` it always was, bit-identical.
_T16_SDF_SOFT_GATE = wp.constant(int(__import__("os").environ.get("T16_SDF_SOFT_GATE", "0")))

# T17: per-SHAPE CAP on the total tangent stiffness the hardening law hands to
# the explicitly-coupled rigid side.
#
# The hardening law (``contact_hardening_inv_dmax``) multiplies BOTH the load
# and its slope: f = k*d/den, so the tangent stiffness of one contact is
#
#     dF/dd = k / den^2,   den = max(1 - d/d_max, HARDENING_DEN_MIN)
#
# i.e. up to 1/0.05^2 = 400x the linear k at the incompressible core.  The cloth
# solver is implicit and does not care, but the finger is coupled EXPLICITLY:
# the reaction wrench is evaluated once per substep and held constant inside the
# 2 ms MuJoCo step, so the finger sees a spring integrated with forward Euler and
# the stability condition is the explicit one, K*dt^2/m < 4.  With m = 0.025 kg
# (the MJCF link mass) and dt = 2 ms that is K < 25 kN/m, while T10-C measured
# the wall region at ~6.4 MN/m -- 250x over -- and the finger force blew up to
# 85-550 N against a 6 N effort limit (doc GRIPPER-CONTACT.md T10-C).
#
# The cap keeps the law but bounds what it can add:
#
#     Sigma(shape) = sum_i k_i / den_i^2                  (uncapped, measured)
#     s            = min(1, k_max(shape) / Sigma_prev)    (one substep of lag)
#     1/den_eff    = 1 + (1/den - 1) * s
#
# so the LINEAR part (1/den = 1) is never scaled -- s only shrinks the EXCESS the
# hardening adds -- and s = 1 reproduces the stock hardening bit for bit.
# ``k_max`` is set per shape from the host as c * m_body / dt^2 (see
# ``set_shape_hardening_kcap``); c = 1 is a quarter of the explicit limit.
#
# 1 = cap the TOTAL tangent sum (the literal budget: everything the shape's
#     contacts hand the rigid side must fit under k_max).
# 2 = cap the hardening EXCESS only, Sigma_x = sum_i (k_i/den_i^2 - k_i).  Use
#     when the LINEAR sum alone already exceeds k_max (many contacts): mode 1
#     then drives s to ~0 and switches the law off, while mode 2 still budgets
#     the extra stiffness against the same explicit limit.
# 0 = OFF: not one statement below runs and the hardening law is the R13c one,
#     bit-identical.
_T17_SDF_HARDEN_KCAP = wp.constant(int(__import__("os").environ.get("T17_SDF_HARDEN_KCAP", "0")))
# T18 诊断开关（力律永不读它们，默认 0 ⇒ 逐位不变）：
#   T18_ANCHOR_DBG_TAIL=1  anchor_dbg/anchor_dbg2 改写在 substep 的「最后一次
#                          Newton 迭代」而不是第 0 次，这样读到的是真正被施加
#                          的力，而不是迭代 0 的试探值（FRICTION_AUDIT §F 的
#                          两个候选路径要靠它二选一）。
#   fric_diag              逐 shape 的 tri-SDF 力累加（P-9：compliant 栈把
#                          shape_contact_ke 置 0，探针的 fric_n 恒 0，读不到
#                          摩擦状态）。由 fric_diag_accum 门控。
_T18_ANCHOR_DBG_TAIL = wp.constant(int(__import__("os").environ.get("T18_ANCHOR_DBG_TAIL", "0")))
# T18 力律开关（默认 0 ⇒ 逐位不变）。
# 是**位掩码**，三位彼此独立（0 = 全关 = 逐位不变）：
#   bit0 (1)  ① 锚的切向 Hessian 行和集总（主修，见散射块的注释）
#             ② 播种改在当前位姿上做 ⇒ 新接触第一子步 f_t 严格为 0
#             ③ 迭代 >=1 才进接触的对不再拿零力切向刚度（审计 C-1）
#   bit1 (2)  塑性拖动量不得超过本 substep 真实相对切向位移（审计 B1 的守卫）
#   bit2 (4)  法向罚力块同样做行和集总（同一条论证：法向也是三角形级弹簧，
#             对角块用 w^2 时把「三点一起法向压缩」这个模态的刚度低估 3 倍）
# 台架实测的推荐值 = 5（bit0 + bit2）。
_T18_FIX_RAW = int(__import__("os").environ.get("T18_SDF_ANCHOR_FIX", "0"))
_T18_SDF_ANCHOR_FIX = wp.constant(_T18_FIX_RAW)
_T18_FIX_TAN = wp.constant(1 if (_T18_FIX_RAW & 1) else 0)
_T18_FIX_DRAG = wp.constant(1 if (_T18_FIX_RAW & 2) else 0)
_T18_FIX_NRM = wp.constant(1 if (_T18_FIX_RAW & 4) else 0)


@wp.struct
class SdfGridExact:
    """Per-shape nearest-triangle grid and mesh pseudo-normals for T15-A.

    All arrays are concatenations over the DISTINCT shape meshes; the per-slot
    ``*base`` arrays index into them, so four blades built from one source mesh
    store one grid and one normal set.  ``face[i] < 0`` marks a grid point with
    no triangle inside the bake radius.  With ``T15_SDF_GRIDEXACT`` unset none
    of this is read (length-1 dummies are still allocated so the struct is a
    valid kernel argument).
    """

    face: wp.array(dtype=wp.int32)
    vnrm: wp.array(dtype=wp.vec3)
    enrm: wp.array(dtype=wp.vec3)
    fbase: wp.array(dtype=wp.int32)
    vbase: wp.array(dtype=wp.int32)
    ebase: wp.array(dtype=wp.int32)
    nx: wp.array(dtype=wp.int32)
    ny: wp.array(dtype=wp.int32)
    nz: wp.array(dtype=wp.int32)
    org: wp.array(dtype=wp.vec3)
    inv_voxel: float
    # T15-A (ii) vertex one-ring of each face, CSR (inert unless _T15_SDF_GX_RING)
    ring_off: wp.array(dtype=wp.int32)
    ring_idx: wp.array(dtype=wp.int32)
    robase: wp.array(dtype=wp.int32)
    ribase: wp.array(dtype=wp.int32)


@wp.kernel
def bake_shape_face_kernel(
    mesh: wp.uint64,
    origin: wp.vec3,
    voxel: float,
    nx: int,
    ny: int,
    nz: int,
    base: int,
    max_dist: float,
    # outputs
    face: wp.array(dtype=wp.int32),
):
    """One-time nearest-triangle bake of a rigid shape onto a dense grid.

    Sign is not needed here (the query re-derives it from the pseudo-normals),
    so this uses the cheapest of the three point queries.
    """
    i, j, k = wp.tid()
    p = origin + wp.vec3(float(i) * voxel, float(j) * voxel, float(k) * voxel)
    f = int(-1)
    q = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    if q.result:
        f = q.face
    face[base + (i * ny + j) * nz + k] = f


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


@wp.func
def sdf_grid_gradient_mag(
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
    """``sdf_grid_gradient`` plus the UNNORMALISED magnitude.

    Same six samples and the same unit vector, statement for statement; the only
    addition is returning ``||g|| / (2*voxel)``, which is the field's local
    |grad| and equals 1 wherever the true SDF is smooth.  Kept as a separate
    function so the default path's ``sdf_grid_gradient`` is not touched at all.
    """
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
        return wp.vec3(0.0, 0.0, 1.0), 0.0
    return g / ln, ln / (2.0 * e)


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
    dg = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, p)
    best = dg
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
    tol: float,
    tol_soft: float,
):
    """Triangle's minimum-SDF point: 3-vertex + centroid seeding, then projected
    steepest descent in barycentric coordinates. Returns (w, best).

    ``tol`` is T16-lite's argmin tolerance (``r * h``); 0 is the strict ``<``
    this function always used and is bit-identical."""
    third = 1.0 / 3.0
    w = wp.vec3(third, third, third)
    p = a * third + b * third + c * third
    dg = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, p)
    best = dg
    da = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, a)
    if da < best - tol:
        best = da
        w = wp.vec3(1.0, 0.0, 0.0)
    db = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, b)
    if db < best - tol:
        best = db
        w = wp.vec3(0.0, 1.0, 0.0)
    dc = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, c)
    if dc < best - tol:
        best = dc
        w = wp.vec3(0.0, 0.0, 1.0)
    if _T16_SDF_ARGMIN_SOFT != 0.0:
        # same construction as the exact path: blend the four seeds, then let the
        # descent below start from the blend.
        w = _t16_softmin_w(
            wp.vec3(third, third, third), dg,
            wp.vec3(1.0, 0.0, 0.0), da,
            wp.vec3(0.0, 1.0, 0.0), db,
            wp.vec3(0.0, 0.0, 1.0), dc,
            tol_soft,
        )
        best = sdf_grid_sample(
            sdf, base, nx, ny, nz, org, inv_voxel, bg,
            a * w[0] + b * w[1] + c * w[2],
        )
    n_start = int(1)
    if _T14_SDF_ALLSEED != 0 and _T16_SDF_ARGMIN_SOFT == 0.0:
        # ALLSEED re-seeds the loop per seed, which would discard the blend;
        # SOFT takes precedence and runs the single descent from w_soft.
        n_start = 4
    for _k in range(n_start):
        # OFF: exactly one pass starting from the winning seed -- the loop body
        # below is then the statements this function always ran.
        # ON: four passes, one per seed, keeping the running best across them.
        wk = w
        bk = best
        if _T14_SDF_ALLSEED != 0:
            if _k == 0:
                wk = wp.vec3(third, third, third)
                bk = dg
            if _k == 1:
                wk = wp.vec3(1.0, 0.0, 0.0)
                bk = da
            if _k == 2:
                wk = wp.vec3(0.0, 1.0, 0.0)
                bk = db
            if _k == 3:
                wk = wp.vec3(0.0, 0.0, 1.0)
                bk = dc
        step = float(0.5)
        for _s in range(refine_steps):
            p = a * wk[0] + b * wk[1] + c * wk[2]
            g = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
            ga = wp.dot(g, a)
            gb = wp.dot(g, b)
            gc = wp.dot(g, c)
            m = (ga + gb + gc) * third
            gw = wp.vec3(ga - m, gb - m, gc - m)
            gn_gate = float(1.0e-12)
            if _T16_SDF_SOFT_GATE != 0:
                gn_gate = wp.max(1.0e-12, tol_soft)
            gn = wp.length(gw)
            if gn > gn_gate:
                cand = wk - gw * (step / gn)
                cand = wp.vec3(wp.max(cand[0], 0.0), wp.max(cand[1], 0.0), wp.max(cand[2], 0.0))
                sm = cand[0] + cand[1] + cand[2]
                if sm > 1.0e-9:
                    cand = cand / sm
                    pc = a * cand[0] + b * cand[1] + c * cand[2]
                    dcand = sdf_grid_sample(sdf, base, nx, ny, nz, org, inv_voxel, bg, pc)
                    if dcand < bk - tol:
                        bk = dcand
                        wk = cand
            step = step * 0.5
        if bk < best - tol:
            best = bk
            w = wk
        if _T14_SDF_ALLSEED == 0:
            best = bk
            w = wk
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
def _t16_soft_weight(d: float, dmin: float, tau: float):
    """exp(-(d - dmin)/tau), underflowing cleanly to 0 for a candidate that is
    out of range (d = bg) or simply much shallower."""
    x = (d - dmin) / tau
    if x > 60.0:
        return 0.0
    return wp.exp(-x)


@wp.func
def _t16_softmin_w(
    w0: wp.vec3, d0: float,
    w1: wp.vec3, d1: float,
    w2: wp.vec3, d2: float,
    w3: wp.vec3, d3: float,
    tau: float,
):
    """Softmin over the FOUR seeds.  Returns the blended barycentric point.

    ``sum_i alpha_i == 1`` and every ``w_i`` sums to 1, so the blend is already a
    valid barycentric coordinate -- no renormalisation is needed.
    """
    dmin = wp.min(d0, wp.min(d1, wp.min(d2, d3)))
    e0 = _t16_soft_weight(d0, dmin, tau)
    e1 = _t16_soft_weight(d1, dmin, tau)
    e2 = _t16_soft_weight(d2, dmin, tau)
    e3 = _t16_soft_weight(d3, dmin, tau)
    ssum = e0 + e1 + e2 + e3
    if ssum < 1.0e-30:
        return w0
    inv = 1.0 / ssum
    return (w0 * e0 + w1 * e1 + w2 * e2 + w3 * e3) * inv


@wp.func
def sdf_mesh_query_grid(
    mesh: wp.uint64,
    gx: SdfGridExact,
    slot: int,
    max_dist: float,
    bg: float,
    p: wp.vec3,
):
    """``sdf_mesh_query``'s answer, found by table lookup instead of BVH descent.

    Same contract, same units, same sentinel: returns ``(d, n)`` with ``d``
    negative inside the solid and ``n`` the unit outward gradient, and ``d = bg``
    when nothing is within ``max_dist``.  The closest point is computed
    analytically on the candidate triangles (so the distance is a length, exact
    to machine precision, exactly as the BVH path's is); only the CANDIDATE SET
    comes from the grid -- the union of the nearest triangles of the 8 corners of
    the cell holding ``p``.
    """
    d = bg
    n = wp.vec3(0.0, 0.0, 1.0)
    nx = gx.nx[slot]
    ny = gx.ny[slot]
    nz = gx.nz[slot]
    q = (p - gx.org[slot]) * gx.inv_voxel
    if q[0] < 0.0:
        return d, n
    if q[1] < 0.0:
        return d, n
    if q[2] < 0.0:
        return d, n
    ix = int(q[0])
    iy = int(q[1])
    iz = int(q[2])
    if ix >= nx - 1:
        return d, n
    if iy >= ny - 1:
        return d, n
    if iz >= nz - 1:
        return d, n

    c000 = gx.fbase[slot] + (ix * ny + iy) * nz + iz
    best = float(3.0e38)
    cpb = wp.vec3(0.0)
    fb = int(-1)
    featb = int(0)
    for m in range(8):
        di = m / 4
        dj = (m - di * 4) / 2
        dk = m - di * 4 - dj * 2
        f = gx.face[c000 + (di * ny + dj) * nz + dk]
        if f >= 0:
            # de-duplicate against the corners already tested (the 8 corners of
            # a cell in contact usually share one or two faces)
            dup = int(0)
            for m2 in range(m):
                di2 = m2 / 4
                dj2 = (m2 - di2 * 4) / 2
                dk2 = m2 - di2 * 4 - dj2 * 2
                if gx.face[c000 + (di2 * ny + dj2) * nz + dk2] == f:
                    dup = 1
            if dup == 0:
                # NB: ``mesh_get_point`` takes a FACE-VERTEX index and does the
                # ``indices[]`` dereference itself; ``mesh_get_index`` is only
                # needed where the POINT id itself is wanted (the vertex
                # pseudo-normal lookup below).
                v0 = wp.mesh_get_point(mesh, f * 3 + 0)
                v1 = wp.mesh_get_point(mesh, f * 3 + 1)
                v2 = wp.mesh_get_point(mesh, f * 3 + 2)
                cp, bary, feat = triangle_closest_point(v0, v1, v2, p)
                dd = wp.length(p - cp)
                if dd < best:
                    best = dd
                    cpb = cp
                    fb = f
                    featb = feat
    if fb < 0:
        return d, n

    if _T15_SDF_GX_RING != 0:
        # stage 2: the winner's VERTEX ONE-RING.  Can only lower ``best``, so it
        # runs before the max_dist rejection.
        ro = gx.robase[slot]
        ri = gx.ribase[slot]
        r0 = gx.ring_off[ro + fb]
        r1 = gx.ring_off[ro + fb + 1]
        fw = fb
        for r in range(r0, r1):
            f2 = gx.ring_idx[ri + r]
            if f2 != fw:
                u0 = wp.mesh_get_point(mesh, f2 * 3 + 0)
                u1 = wp.mesh_get_point(mesh, f2 * 3 + 1)
                u2 = wp.mesh_get_point(mesh, f2 * 3 + 2)
                cp2, bary2, feat2 = triangle_closest_point(u0, u1, u2, p)
                dd2 = wp.length(p - cp2)
                if dd2 < best:
                    best = dd2
                    cpb = cp2
                    fb = f2
                    featb = feat2

    if best > max_dist:
        return d, n

    # Sign from the CLOSEST FEATURE's pseudo-normal.  On a closed mesh the
    # angle-weighted vertex normal and the two-face edge normal make the
    # in/out test exact for a point whose closest feature is that vertex/edge
    # (Baerentzen & Aanaes 2005); the face-interior case is the face normal.
    pn = wp.mesh_eval_face_normal(mesh, fb)
    if featb == TRI_CONTACT_FEATURE_VERTEX_A:
        pn = gx.vnrm[gx.vbase[slot] + wp.mesh_get_index(mesh, fb * 3 + 0)]
    elif featb == TRI_CONTACT_FEATURE_VERTEX_B:
        pn = gx.vnrm[gx.vbase[slot] + wp.mesh_get_index(mesh, fb * 3 + 1)]
    elif featb == TRI_CONTACT_FEATURE_VERTEX_C:
        pn = gx.vnrm[gx.vbase[slot] + wp.mesh_get_index(mesh, fb * 3 + 2)]
    elif featb == TRI_CONTACT_FEATURE_EDGE_AB:
        pn = gx.enrm[gx.ebase[slot] + fb * 3 + 0]
    elif featb == TRI_CONTACT_FEATURE_EDGE_AC:
        pn = gx.enrm[gx.ebase[slot] + fb * 3 + 1]
    elif featb == TRI_CONTACT_FEATURE_EDGE_BC:
        pn = gx.enrm[gx.ebase[slot] + fb * 3 + 2]

    delta = p - cpb
    sgn = 1.0
    if wp.dot(delta, pn) < 0.0:
        sgn = -1.0
    if best < 1.0e-9:
        # exactly on the surface: same convention as ``sdf_mesh_query``
        d = 0.0
        n = wp.mesh_eval_face_normal(mesh, fb)
    else:
        d = best * sgn
        n = delta * (sgn / best)
    return d, n


@wp.func
def sdf_query(
    mesh: wp.uint64,
    gx: SdfGridExact,
    slot: int,
    max_dist: float,
    bg: float,
    p: wp.vec3,
):
    """Backend switch for the exact field evaluation.  OFF = ``sdf_mesh_query``
    verbatim, so the default path emits the same statements it always did."""
    d = bg
    n = wp.vec3(0.0, 0.0, 1.0)
    if _T15_SDF_GRIDEXACT != 0:
        d, n = sdf_mesh_query_grid(mesh, gx, slot, max_dist, bg, p)
    else:
        d, n = sdf_mesh_query(mesh, max_dist, bg, p)
    return d, n


@wp.func
def tri_sdf_refine_from(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    mesh: wp.uint64,
    gx: SdfGridExact,
    slot: int,
    bg: float,
    tol: float,
    gate: float,
    rq: float,
    w: wp.vec3,
    best: float,
    nbest: wp.vec3,
    p_best: wp.vec3,
    refine_steps: int,
):
    """The projected steepest descent, lifted out of ``tri_sdf_closest_mesh``.

    Extracted verbatim so the serial search and the T14 (iii) per-seed parallel
    search run the SAME statements from their respective starting points -- the
    only way to make "seed s*'s parallel thread reproduces the serial result"
    an identity rather than a hope.  ``@wp.func`` is inlined, and the bitwise
    probe (scratchpad/t14_search_equiv.py) confirms the extraction changed
    nothing.
    """
    third = 1.0 / 3.0
    step = float(0.5)
    for _s in range(refine_steps):
        gvec = nbest
        ga = wp.dot(gvec, a)
        gb = wp.dot(gvec, b)
        gc = wp.dot(gvec, c)
        m = (ga + gb + gc) * third
        gw = wp.vec3(ga - m, gb - m, gc - m)
        gn = wp.length(gw)
        if gn > wp.max(1.0e-12, gate):
            cand = w - gw * (step / gn)
            cand = wp.vec3(wp.max(cand[0], 0.0), wp.max(cand[1], 0.0), wp.max(cand[2], 0.0))
            sm = cand[0] + cand[1] + cand[2]
            if sm > 1.0e-9:
                cand = cand / sm
                pc = a * cand[0] + b * cand[1] + c * cand[2]
                probe = int(1)
                if _T14_SDF_SKIPREF != 0:
                    if pc[0] == p_best[0]:
                        if pc[1] == p_best[1]:
                            if pc[2] == p_best[2]:
                                probe = 0
                if probe != 0:
                    dcand, ncand = sdf_query(mesh, gx, slot, rq, bg, pc)
                    if dcand < best - tol:
                        best = dcand
                        w = cand
                        nbest = ncand
                        if _T14_SDF_SKIPREF != 0:
                            p_best = pc
        step = step * 0.5
    return w, best, nbest


@wp.func
def tri_sdf_closest_mesh(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    mesh: wp.uint64,
    gx: SdfGridExact,
    slot: int,
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
    # T16-lite: a candidate must be deeper by more than this to unseat the
    # incumbent.  0 (default) reproduces the strict '<' exactly.
    tol = _T16_SDF_ARGMIN_TOL * cull
    gate = float(0.0)
    if _T16_SDF_SOFT_GATE != 0:
        gate = _T16_SDF_ARGMIN_SOFT * cull
    if _T16_SDF_ARGMIN_SOFT != 0.0:
        # SOFT and TOL are mutually exclusive; SOFT wins and the seed/refine
        # comparisons go back to the strict '<' they always were.
        tol = 0.0
    best, nbest = sdf_query(mesh, gx, slot, cull + reach, bg, g)
    if best >= bg:
        return w, bg, nbest
    dg = best
    if _T14_SDF_RQ != 0:
        rq = wp.abs(best) + reach + 1.0e-6
    # the point that actually produced ``best`` (see _T14_SDF_SKIPREF)
    p_best = g
    da, na = sdf_query(mesh, gx, slot, rq, bg, a)
    if da < best - tol:
        best = da
        w = wp.vec3(1.0, 0.0, 0.0)
        nbest = na
        if _T14_SDF_SKIPREF != 0:
            p_best = a
    db, nb = sdf_query(mesh, gx, slot, rq, bg, b)
    if db < best - tol:
        best = db
        w = wp.vec3(0.0, 1.0, 0.0)
        nbest = nb
        if _T14_SDF_SKIPREF != 0:
            p_best = b
    dc, nc = sdf_query(mesh, gx, slot, rq, bg, c)
    if dc < best - tol:
        best = dc
        w = wp.vec3(0.0, 0.0, 1.0)
        nbest = nc
        if _T14_SDF_SKIPREF != 0:
            p_best = c
    if _T16_SDF_ARGMIN_SOFT != 0.0:
        # softmin over the four seeds -> a CONTINUOUS starting point, then the
        # same descent from there.  One extra field evaluation (at the blend).
        w_s = _t16_softmin_w(
            wp.vec3(third, third, third), dg,
            wp.vec3(1.0, 0.0, 0.0), da,
            wp.vec3(0.0, 1.0, 0.0), db,
            wp.vec3(0.0, 0.0, 1.0), dc,
            _T16_SDF_ARGMIN_SOFT * cull,
        )
        p_s = a * w_s[0] + b * w_s[1] + c * w_s[2]
        d_s, n_s = sdf_query(mesh, gx, slot, rq, bg, p_s)
        return tri_sdf_refine_from(
            a, b, c, mesh, gx, slot, bg, tol, gate, rq, w_s, d_s, n_s, p_s, refine_steps
        )
    return tri_sdf_refine_from(
        a, b, c, mesh, gx, slot, bg, tol, gate, rq, w, best, nbest, p_best, refine_steps
    )


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
    # T8: true static friction on this channel.  ``anchor_kt_ratio <= 0`` (the
    # default) skips every statement below and leaves the IPC law in charge, so
    # the OFF path is bit-identical.  ``anchor_seed`` is 1 only on the first
    # Newton iteration of a substep: the anchor is a per-STEP state, not a
    # per-iteration one, and re-mapping it inside the loop would let the spring
    # chase its own solution.
    anchor_p: wp.array(dtype=wp.vec3),
    anchor_valid: wp.array(dtype=wp.int32),
    # T12: barycentric coordinates of the MATERIAL point the anchor was seeded
    # on.  Without it the spring is measured against whatever point this Newton
    # iteration's closest-point search happened to win, which on a flat blade
    # face is not the same point twice (see the seeding block below).
    anchor_w: wp.array(dtype=wp.vec3),
    anchor_kt_ratio: float,
    anchor_seed: int,
    # T12 验法：每对写出 (|slip_t|, 是否在锥上, |搜索点-播种材料点|, f_n)。
    # 纯诊断，力律不读它；关闭时传长度 1 的哑数组。
    anchor_dbg: wp.array(dtype=wp.vec4),
    # T13 验法：逐对向量诊断，形状 [n_pairs, 20]，关闭时传 (1, 20) 哑数组。
    # 与 anchor_dbg 同条件写（anchor_seed != 0 且锚开启），力律不读它。
    # 0-2 n_world | 3 f_n | 4-6 f_t(钳制后, 世界) | 7-9 slip_t(世界, m)
    # 10-12 搜索点世界坐标 | 13-15 播种材料点世界坐标 | 16 播种点自身有符号距离
    # 17 newly_seeded | 18 slot | 19 cone(=mu*f_n)
    anchor_dbg2: wp.array2d(dtype=float),
    # T14 broad phase (inert unless _T14_SDF_BROADPHASE): the pair index is
    # read out of the candidate list instead of being the thread id.
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    bp_overflow: wp.array(dtype=wp.int32),
    bp_mode: int,
    # T14 round 4 (iii): per-seed scratch, 4 entries per pair (inert unless
    # _T14_SDF_PAR).  The force kernel then only picks the minimum.
    par_best: wp.array(dtype=float),
    par_w: wp.array(dtype=wp.vec3),
    par_n: wp.array(dtype=wp.vec3),
    # T14 round 3: per-substep held contact geometry (inert unless
    # _T14_SDF_HOLD).  ``hold_mode`` 1 = search and record (iteration 0),
    # 2 = evaluate the held plane (iterations >= 1 and the reaction pass).
    hold_valid: wp.array(dtype=wp.int32),
    hold_w: wp.array(dtype=wp.vec3),
    hold_p: wp.array(dtype=wp.vec3),
    hold_n: wp.array(dtype=wp.vec3),
    hold_mode: int,
    # T14 HOLD falsifier accumulator (inert unless _T14_SDF_HOLD_DIAG):
    # [0] n [1] sum|db| [2] max|db| [3] contact-state disagreements
    # [4] sum|db| in contact [5] n in contact [6] max|db| in contact
    # [7] signed sum (held - true) in contact [8] sum|dn| [9] sum|dw|
    hold_diag: wp.array(dtype=float),
    # T15-A: nearest-triangle grid + pseudo-normals for the O(1) exact
    # backend (inert unless _T15_SDF_GRIDEXACT).
    gx: SdfGridExact,
    # T16 diagnostic accumulator (inert unless _T16_W_DIAG)
    w_diag: wp.array(dtype=float),
    # T17 hardening stiffness cap (inert unless _T17_SDF_HARDEN_KCAP).
    # ``harden_ksum`` is the per-shape accumulator for THIS substep's uncapped
    # tangent sum, written only when ``kcap_accum != 0`` (iteration 0, so each
    # contact counts once per SUBSTEP, not once per Newton iteration);
    # ``harden_kscale`` is the scale computed from the PREVIOUS substep's sum by
    # ``update_shape_hardening_kcap_kernel``.
    harden_ksum: wp.array(dtype=float),
    harden_ksum_lin: wp.array(dtype=float),
    harden_kscale: wp.array(dtype=float),
    kcap_accum: int,
    # T18 仪器（纯输出）。``anchor_tail`` 在 substep 的最后一次 Newton 迭代上
    # 为 1，其余为 0；``fric_diag`` 是逐 shape 的 12 槽力累加，只在
    # ``fric_diag_accum != 0`` 时写。两者都不进任何力/Hessian 表达式。
    # T18 模式 2：上一 substep 这一对的材料点（shape 局部系），用来算本
    # substep 的**真实**相对切向位移。模式 <2 时既不读也不写。
    anchor_qp: wp.array(dtype=wp.vec3),
    anchor_tail: int,
    fric_diag: wp.array(dtype=float),
    fric_diag_accum: int,
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
    pair = tid
    if _T14_SDF_BROADPHASE != 0:
        # T14: ON path only.  ``bp_mode`` 1 = consume the candidate list built
        # this substep; 0 = the full-scan fallback launch, which no-ops unless
        # the list overflowed.  Both launches keep a FIXED dim and read the
        # overflow flag on the device, so the pair stays CUDA-graph capturable.
        if bp_mode != 0:
            if bp_overflow[0] != 0:
                return
            if tid >= cand_count[0]:
                return
            pair = cand_idx[tid]
        else:
            if bp_overflow[0] == 0:
                return
    slot = pair / tri_count
    t = pair - slot * tri_count

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
    held = int(0)
    if _T14_SDF_HOLD != 0:
        if hold_mode == 2:
            held = 1
    if held != 0:
        # T14 HOLD, iterations >= 1 (and the reaction pass): the closest point
        # and the normal were resolved at iteration 0 of THIS substep, in the
        # blade's own frame, and the blade has not moved since -- so the only
        # thing left to evaluate is how far the (moved) cloth point now sits
        # from that plane.  No mesh query.
        if hold_valid[pair] == 0:
            return
        w = hold_w[pair]
        q_hold = a * w[0] + b * w[1] + c * w[2]
        if _T14_SDF_HOLD == 2:
            # HOLD=2: hold only the barycentric point; still ask the mesh where
            # the surface is, but ONCE at that point instead of the 7-query
            # seed+descent search.  Keeps the exact distance and the exact
            # normal, drops only the re-optimisation of WHERE on the triangle
            # the closest point sits.
            best, n_exact = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, q_hold)
        else:
            n_exact = hold_n[pair]
            best = wp.dot(n_exact, q_hold - hold_p[pair])
        if _T14_SDF_HOLD_DIAG != 0:
            # what the un-held code would have said at this same iterate
            w_t, best_t, n_t = tri_sdf_closest_mesh(
                a, b, c, sdf_mesh[slot], gx, slot, bg, half_thickness, refine_steps
            )
            if best_t < bg:
                d = wp.abs(best - best_t)
                wp.atomic_add(hold_diag, 0, 1.0)
                wp.atomic_add(hold_diag, 1, d)
                wp.atomic_max(hold_diag, 2, d)
                c_hold = wp.where(best < half_thickness, 1.0, 0.0)
                c_true = wp.where(best_t < half_thickness, 1.0, 0.0)
                if c_hold != c_true:
                    wp.atomic_add(hold_diag, 3, 1.0)
                if best_t < half_thickness:
                    wp.atomic_add(hold_diag, 4, d)
                    wp.atomic_add(hold_diag, 5, 1.0)
                    wp.atomic_max(hold_diag, 6, d)
                    # depth error signed: held minus true (>0 = held reports
                    # LESS penetration, i.e. the force is under-reported)
                    wp.atomic_add(hold_diag, 7, best - best_t)
                    wp.atomic_add(hold_diag, 8, wp.length(n_exact - n_t))
                    wp.atomic_add(hold_diag, 9, wp.length(w_t - w))
    elif _T14_SDF_PAR != 0:
        # T14 (iii): the four seeds were refined in their own threads; take the
        # minimum.  Strict '<' so ties keep the lowest seed index and the pick
        # stays deterministic.
        base_i = pair * 4
        best = par_best[base_i]
        w = par_w[base_i]
        n_exact = par_n[base_i]
        d1 = par_best[base_i + 1]
        if d1 < best:
            best = d1
            w = par_w[base_i + 1]
            n_exact = par_n[base_i + 1]
        d2 = par_best[base_i + 2]
        if d2 < best:
            best = d2
            w = par_w[base_i + 2]
            n_exact = par_n[base_i + 2]
        d3 = par_best[base_i + 3]
        if d3 < best:
            best = d3
            w = par_w[base_i + 3]
            n_exact = par_n[base_i + 3]
    elif _R16_SDF_EXACT != 0:
        # R16-A2': same search, exact field.  The freeze switch is a grid-path
        # remedy for the winner-take-all redraw and is not combined with it.
        w, best, n_exact = tri_sdf_closest_mesh(a, b, c, sdf_mesh[slot], gx, slot, bg, half_thickness, refine_steps)
    elif _R13G_SDF_FREEZE == 0:
        w, best = tri_sdf_closest(
                a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps,
                wp.where(_T16_SDF_ARGMIN_SOFT != 0.0, 0.0, _T16_SDF_ARGMIN_TOL * half_thickness),
                _T16_SDF_ARGMIN_SOFT * half_thickness,
            )
    else:
        # Resolve the barycentric contact point once per substep, then hold it.
        if resolve_w != 0:
            w, best = tri_sdf_closest(
                a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps,
                wp.where(_T16_SDF_ARGMIN_SOFT != 0.0, 0.0, _T16_SDF_ARGMIN_TOL * half_thickness),
                _T16_SDF_ARGMIN_SOFT * half_thickness,
            )
            tri_sdf_w[pair] = w
        else:
            w = tri_sdf_w[pair]
            best = sdf_grid_sample(
                sdf, base, nx, ny, nz, org, inv_voxel, bg, a * w[0] + b * w[1] + c * w[2]
            )
    if _T14_SDF_HOLD != 0:
        if hold_mode == 1:
            # iteration 0: record the plane the rest of the substep rides on.
            # The gate is "the query HIT" (best < mesh_max_dist), not "in
            # contact" (best < h): a pair a few millimetres out at iteration 0
            # can be driven into the shell by the solve, and it must be on the
            # list to be seen.  A miss stores 0 and the pair is off for the
            # substep -- the same per-substep contact-set rule the rigid feature
            # channel already runs under.
            if best < mesh_max_dist:
                hold_valid[pair] = 1
                hold_w[pair] = w
                hold_n[pair] = n_exact
                hold_p[pair] = (a * w[0] + b * w[1] + c * w[2]) - n_exact * best
            else:
                hold_valid[pair] = 0
    if best >= half_thickness:
        # T8: the pair separated -> drop the anchor.  Seeding/holding is keyed on
        # GEOMETRIC contact (best <= h), not on the force band, so a pair that
        # leaves the shell starts clean the next time it comes back.
        if anchor_kt_ratio > 0.0 and anchor_seed != 0:
            anchor_valid[pair] = 0
        return

    depth = wp.min(half_thickness - best, max_depth)
    p = a * w[0] + b * w[1] + c * w[2]
    n_local = wp.vec3(0.0, 0.0, 1.0)
    if _R16_SDF_EXACT != 0:
        # the gradient at the winning point, carried out of the search
        n_local = n_exact
    else:
        if _T14_SDF_EDGE_EXACT == 3:
            # HYBRID: unconditional.  Start from the grid gradient so a missed
            # query still leaves a usable normal, then overwrite with the exact
            # depth and normal at the winner point.
            n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
            d_h, n_h = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, p)
            if d_h < bg:
                n_local = n_h
                depth = wp.max(wp.min(half_thickness - d_h, max_depth), 0.0)
        elif _T14_SDF_EDGE_EXACT != 0:
            # depth stays the grid's; only the DIRECTION falls back, and only
            # where the grid says its own gradient is not unit length.
            ng_e, gmag_e = sdf_grid_gradient_mag(
                sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p
            )
            n_local = ng_e
            if wp.abs(gmag_e - 1.0) > 0.1:
                d_e, n_e = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, p)
                if d_e < bg:
                    n_local = n_e
                    if _T14_SDF_EDGE_EXACT == 2:
                        # mode 2: the depth comes from the exact field too.
                        # Clamped at 0 because the exact field can put this
                        # point outside the shell that the grid put inside;
                        # a negative depth would flip the force outward.
                        depth = wp.max(wp.min(half_thickness - d_e, max_depth), 0.0)
        else:
            n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p)
    n_world = wp.transform_vector(X_ws, n_local)
    k = tri_stiffness[t]
    f = n_world * (k * depth)
    nn = wp.outer(n_world, n_world) * k
    if _T16_W_DIAG != 0:
        # pure accumulator; nothing below reads it
        wmax = wp.max(w[0], wp.max(w[1], w[2]))
        wdot = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
        fn_d = k * depth
        third_d = 1.0 / 3.0
        wp.atomic_add(w_diag, 0, 1.0)
        if wp.abs(w[0] - third_d) < 1.0e-6 and wp.abs(w[1] - third_d) < 1.0e-6:
            wp.atomic_add(w_diag, 1, 1.0)
        elif wmax > 1.0 - 1.0e-6:
            wp.atomic_add(w_diag, 2, 1.0)
        else:
            wp.atomic_add(w_diag, 3, 1.0)
        wp.atomic_add(w_diag, 4, wmax)
        wp.atomic_add(w_diag, 5, wdot)
        wp.atomic_add(w_diag, 6, fn_d)
        wp.atomic_add(w_diag, 7, fn_d * wmax)
        wp.atomic_add(w_diag, 8, fn_d * wdot)
        wp.atomic_max(w_diag, 9, fn_d)
        wp.atomic_add(w_diag, 10, depth)
        wp.atomic_max(w_diag, 11, wmax)
        bin_i = int((wmax - third_d) * (10.0 / (1.0 - third_d)))
        if bin_i < 0:
            bin_i = 0
        if bin_i > 9:
            bin_i = 9
        wp.atomic_add(w_diag, 12 + bin_i, 1.0)
        if _T16_W_DIAG >= 2:
            # closest FEATURE of the contact point, from one extra BVH query.
            # Diagnostic only; the force above is already computed.
            q_f = wp.mesh_query_point_sign_normal(
                sdf_mesh[slot], a * w[0] + b * w[1] + c * w[2], mesh_max_dist
            )
            if q_f.result:
                bu = q_f.u
                bv = q_f.v
                bw = 1.0 - bu - bv
                bmin = wp.min(bw, wp.min(bu, bv))
                bmax = wp.max(bw, wp.max(bu, bv))
                if bmax > 1.0 - 1.0e-4:
                    wp.atomic_add(w_diag, 22, 1.0)      # vertex
                elif bmin < 1.0e-4:
                    wp.atomic_add(w_diag, 23, 1.0)      # edge
                else:
                    wp.atomic_add(w_diag, 24, 1.0)      # face interior
            else:
                wp.atomic_add(w_diag, 25, 1.0)
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
        if _T17_SDF_HARDEN_KCAP != 0:
            # T17: measure this substep's UNCAPPED tangent sum, then apply the
            # scale the PREVIOUS substep's sum earned.  Measuring the uncapped
            # value keeps the loop open (the scale never feeds its own input).
            if kcap_accum != 0:
                k_tan = k * inv_den * inv_den
                if _T17_SDF_HARDEN_KCAP >= 2:
                    k_tan = k_tan - k       # mode 2: budget the EXCESS only
                wp.atomic_add(harden_ksum, shape, k_tan)
                # T17 diagnostic: the LINEAR sum of the SAME contacts, so the
                # budget can be compared against what the stack carries with the
                # hardening law switched off.  Never read by the force law.
                wp.atomic_add(harden_ksum_lin, shape, k)
            inv_den = 1.0 + (inv_den - 1.0) * harden_kscale[shape]
        f = n_world * (k * depth * inv_den)
        nn = wp.outer(n_world, n_world) * (k * inv_den * inv_den)
    # R13f: Coulomb friction, same law/mixing as the vertex penalty channel.
    # f_n is the normal load this triangle actually carries (post-hardening);
    # u_rel is the relative translation of the coincident material points over
    # the substep -- the cloth contact point minus the shape point under it.
    # OFF branch (_R13F_SDF_FRICTION == 0) leaves f/nn exactly as above.
    # T12-b: the anchor's tangential force is evaluated at the SEEDED material
    # point, so it must also be distributed with the SEEDED weights -- adding it
    # into ``f`` would spread it with this iteration's search weights, i.e. apply
    # it at a different point of the triangle.
    f_t_a = wp.vec3(0.0, 0.0, 0.0)
    nn_t_a = wp.identity(n=3, dtype=float) * 0.0
    aw_out = w
    # T18 P-9 仪器：法向力在摩擦律改写 ``f`` 之前先记下来（IPC 分支会把 f_t
    # 折进 ``f``）。这几个变量只被 fric_diag 的累加块读。
    fn_diag = wp.length(f)
    ft_diag = wp.vec3(0.0, 0.0, 0.0)
    cone_diag = float(0.0)
    slip_diag = float(0.0)
    oncone_diag = float(0.0)
    if anchor_kt_ratio > 0.0:
        # T8: Coulomb stick-slip via a tangential anchor, replacing the IPC
        # regularisation on this channel.
        #
        # Why: `compute_projected_isotropic_friction` is an IPC-type regularised
        # Coulomb law -- correct while SLIDING (f_t = mu*N against the motion),
        # but it has no sticking state.  Under any load below the cone it creeps
        # at v = eps*(1 - sqrt(1 - rho)), rho = load/(mu*N); with eps = 1e-2 m/s
        # that is 1.3 mm/s at rho = 0.25, i.e. 6.5 mm over a 5 s carry (measured:
        # pack's left hand drifts 19-34 mm during B5 at mu_eff 0.70 and drops the
        # strap in 2/3 runs).  Lowering eps only postpones it (v ~ eps^0.47).
        #
        # The anchor is the material point of the SHAPE that the cloth contact
        # point was sitting on when contact formed, stored in the shape's local
        # frame so rigid motion carries it for free.  While the spring force
        # stays inside the cone the contact sticks exactly (zero drift); on the
        # cone it slides at f_t = mu*N and the anchor is dragged along, which is
        # the standard return mapping.
        mu = wp.sqrt(friction_mu * shape_material_mu[shape])
        if mu > 0.0:
            f_n = k * depth
            if hardening_inv_dmax > 0.0:
                den_f = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
                f_n = f_n / den_f
                if _T17_SDF_HARDEN_KCAP != 0:
                    # T17: the friction cone rides the load the normal law
                    # actually applied, so it takes the same scaled denominator.
                    f_n = k * depth * (
                        1.0 + (1.0 / den_f - 1.0) * harden_kscale[shape]
                    )
            # T12 fix: the spring must measure the displacement of a MATERIAL
            # point of the cloth, so the barycentric coordinates are seeded WITH
            # the anchor and reused; the closest-point search keeps supplying
            # only ``best`` and the normal.
            #
            # Why (T12, 16 runs, flatten): ``w`` is re-searched by
            # ``tri_sdf_closest_mesh`` on every call (centroid + 3 vertices, then
            # refinement, strict ``<``).  With the cloth lying flat on the blade
            # face the four candidates are equally deep, so the winner flips on
            # float noise and ``a*w0+b*w1+c*w2`` moves by a triangle edge
            # (2-7 mm) with the cloth perfectly still.  That fake slip is orders
            # of magnitude past the cone, so f_t is clamped to mu*N with its
            # direction set by the jump: a per-step kick of arbitrary direction
            # whose mean is ~0, not friction.  Measured consequence: the fold is
            # squeezed OUT of the jaw over the last 20 mm of closing
            # (jawpts 32->21->12->4->0 with the anchor vs 38->33->22->15->17
            # without) and grasp fails 12/12.
            #
            # The IPC branch below never had this problem because it evaluates
            # ``pos`` and ``pos_prev`` at the SAME ``w``, so the jump cancels;
            # this is the accumulating form of that same rule.
            seeded = anchor_valid[pair]
            aw = anchor_w[pair]
            if seeded == 0:
                aw = w
            if _T12_ANCHOR_SEEDW == 0:
                aw = w          # 修前行为：参照点跟着每次迭代的搜索赢家跑
            # cloth material point, in the shape's local frame (a/b/c already are)
            q_loc = a * aw[0] + b * aw[1] + c * aw[2]
            kt = k * anchor_kt_ratio
            cone = mu * f_n
            if anchor_seed != 0:
                # T13 fix3': seeding and the elastoplastic RETURN MAPPING happen
                # once per substep, on iteration 0, but evaluated in the PREVIOUS
                # body pose -- i.e. on the previous substep's converged state
                # (pos is still x_prev here, so x_prev with body_q_prev is a
                # consistent configuration).  The original code did this with
                # body_q (the NEW pose): the trial slip then already contained
                # the blade's substep displacement delta, and whenever
                # delta > slip_max the anchor was dragged back by (delta -
                # slip_max) every substep regardless of load -- a velocity-driven
                # ratchet (bench: cloth slipped at 96-99 % of plate speed at
                # rho = 1/6; task: fold slid out at 80 mm/s during the lift).
                # Mapping on the last Newton iteration instead (first attempt)
                # removed the ratchet but turned un-converged iterates into
                # permanent drift (bench: 0.9 mm/s creep at rho = 1/6 on a static
                # plate).  The previous converged state has neither problem.
                X_ws_p = shape_transform[shape]
                if body >= 0:
                    X_ws_p = body_q_prev[body] * shape_transform[shape]
                X_sw_p = wp.transform_inverse(X_ws_p)
                q_prev_loc = (
                    wp.transform_point(X_sw_p, pos[i0]) * aw[0]
                    + wp.transform_point(X_sw_p, pos[i1]) * aw[1]
                    + wp.transform_point(X_sw_p, pos[i2]) * aw[2]
                )
                if seeded == 0:
                    anchor_w[pair] = w
                    seed_p = q_prev_loc
                    if _T18_FIX_TAN != 0:
                        # T18-B2：播种点取**当前位姿**下的材料点，也就是这一
                        # 迭代的接触判定所用的那个配置，于是 slip(x_prev) == 0、
                        # 第一子步 f_t 严格为 0。原来播在上一位姿上，slip 恒等于
                        # 刀自身这一 substep 的位移（合爪 30 mm/s、dt=2 ms 时
                        # 60 um，而 slip_max 只有 90 um）⇒ 新接触一形成就已经
                        # 在锥的 2/3 处（FRICTION_AUDIT §C 的播种项）。
                        # 帧一致性：本 substep 的「当前位姿」正是下一 substep 的
                        # 「上一位姿」，所以下一次返回映射读到的 q_prev_loc 与
                        # 这里播下的锚在同一个位姿里，没有引入新的错位。
                        seed_p = q_loc
                    anchor_p[pair] = seed_p
                    anchor_valid[pair] = 1
                    seeded = 1
                else:
                    sp_loc = q_prev_loc - anchor_p[pair]
                    sp_w = wp.transform_vector(X_ws_p, sp_loc)
                    sp_t = sp_w - n_world * wp.dot(sp_w, n_world)
                    lp = wp.length(sp_t)
                    if kt * lp > cone and kt > 1.0e-12 and lp > 1.0e-12:
                        # sliding at the end of the previous substep: park the
                        # anchor slip_max behind the material point.
                        keep_len = cone / kt
                        if _T18_FIX_DRAG != 0:
                            # T18-B1 守卫：塑性流动量 (lp - slip_max) 必须由本
                            # substep **真实发生过的**相对切向位移解释；解释不了
                            # 的部分不拖。真滑动时 |du_t| >> 塑性增量，守卫是
                            # no-op；「没滑却被判在锥外」时锚就不动，不会把一次
                            # 暂态攒成永久偏移。
                            du_loc = q_prev_loc - anchor_qp[pair]
                            du_w = wp.transform_vector(X_ws_p, du_loc)
                            du_t = du_w - n_world * wp.dot(du_w, n_world)
                            moved = wp.max(wp.dot(du_t, sp_t) / lp, 0.0)
                            drag = wp.min(lp - keep_len, moved)
                            keep_len = lp - drag
                        keep = wp.transform_vector(X_sw_p, sp_t * (keep_len / lp))
                        anchor_p[pair] = q_prev_loc - keep
                if _T18_FIX_DRAG != 0:
                    anchor_qp[pair] = q_prev_loc
            slip_loc = wp.vec3(0.0, 0.0, 0.0)
            if seeded != 0:
                slip_loc = q_loc - anchor_p[pair]
            slip_w = wp.transform_vector(X_ws, slip_loc)
            slip_t = slip_w - n_world * wp.dot(slip_w, n_world)
            f_t = slip_t * (-kt)
            m = wp.length(f_t)
            dbg_when = anchor_seed
            if _T18_ANCHOR_DBG_TAIL != 0:
                # T18: 读最后一次迭代（真正被施加的力），而不是迭代 0 的试探值。
                dbg_when = anchor_tail
            if dbg_when != 0 and anchor_dbg.shape[0] > 1:
                q_search = a * w[0] + b * w[1] + c * w[2]
                anchor_dbg[pair] = wp.vec4(
                    wp.length(slip_t),
                    wp.where(m > cone, 1.0, 0.0),
                    wp.length(q_search - q_loc),
                    f_n,
                )
            if m > cone:
                # sliding: clamp to the cone.  The anchor itself is advanced on
                # the next substep's iteration 0 (see above), never inside the
                # Newton loop and never on a trial or un-converged state.
                if m > 1.0e-12:
                    f_t = f_t * (cone / m)
                f_t_a = f_t
                # T12-b FIX: the sliding branch MUST return a tangential
                # stiffness.  The old comment ("no tangential stiffness, same as
                # the IPC law's s >= 1 branch") was wrong: IPC's sliding branch
                # returns K = mu*N/|u| * (I - n n^T), it is not zero.  With K = 0
                # the cloth vertex's only tangential stiffness is inertia
                # (~2 N/m) plus in-plane stretch, so a 0.1 N cone force moves it
                # ~0.1 mm in one Newton step, overshoots the anchor, the force
                # flips, the next iteration walks back -- 20 iterations end mid
                # oscillation, the return mapping then parks the anchor exactly
                # on the cone boundary and any micro-motion leaves the cone
                # again.  Measured: 70-90% of pairs on the cone at rho ~ 0.05,
                # material point moving 0.1-0.3 mm per substep against the blade.
                # That is oscillation, not sliding, and it pumps energy into the
                # fold until it is squeezed out of the jaw.
                # Using kt (the sticking stiffness) rather than the consistent
                # cone/|slip_t|: the two coincide at the cone boundary and kt is
                # the stiffer, unconditionally stable choice.  IPC-consistent
                # alternative, one line:
                #     nn_t_a = (wp.identity(n=3, dtype=float)
                #               - wp.outer(n_world, n_world)) * (cone / ln_t)
                # T13 fix4: CONSISTENT tangent on the cone, cone/|slip_t| (the IPC
                # sliding-branch form), not kt.  With kt the Newton step for a pair
                # that must catch up delta = v*dt to the blade is only
                # cone/kt = slip_max (~0.04-0.1 mm) per iteration: 20 iterations
                # cover 0.8 mm, so at lift speed (0.26 mm/substep) the cloth
                # converges but at fling speed (2-4 mm/substep) it lags, and the
                # converged-state return map then parks the anchor behind the
                # lagging cloth every substep -- the same ratchet, now caused by
                # non-convergence (measured: fix3 held the lift, lost the fold in
                # the first fling swing, 17 -> 0 particles in 16 frames while IPC
                # held 13-14).  cone/|slip_t| equals kt exactly at the cone
                # boundary (|slip_t| = slip_max), so the tangent is continuous.
                # T13 fix6: blend.  Near the cone (|slip_t| <= 3*slip_max) keep
                # the stiff kt tangent -- on a soft, non-converged cloth the soft
                # cone/|slip| tangent let the iterates wander and the return map
                # turned that into creep (bench, spec cloth: 0.69 -> 8.66 mm/s).
                # Far from the cone (fast blade, |slip_t| = v*dt >> slip_max) the
                # tangent must soften or Newton cannot catch up (fix4 finding).
                # 3*cone/|slip_t| equals kt at |slip_t| = 3*slip_max (continuous)
                # and still gives a catch-up step of |slip_t|/3 per iteration.
                ln_t = wp.length(slip_t)
                k_cone = kt
                if ln_t > 1.0e-12:
                    k_cone = wp.min(kt, 3.0 * cone / ln_t)
                nn_t_a = (wp.identity(n=3, dtype=float) - wp.outer(n_world, n_world)) * k_cone
                oncone_diag = 1.0
            else:
                f_t_a = f_t
                # sticking: the full tangential spring, projected off the normal.
                nn_t_a = (wp.identity(n=3, dtype=float) - wp.outer(n_world, n_world)) * kt
                if _T18_FIX_TAN != 0 and seeded == 0:
                    # T18-C-1：迭代 0 不接触、迭代 k>=1 才进接触的对没有锚，
                    # slip 恒 0、f_t 恒 0，却照样拿到一个 kt(=3k) 的切向刚度，
                    # 而且不受锥约束——合爪把布卷进钳口的那些帧里，它悄悄把布
                    # 的切向刚化。零力就不该有刚度。
                    nn_t_a = wp.identity(n=3, dtype=float) * 0.0
            ft_diag = f_t_a
            cone_diag = cone
            slip_diag = wp.length(slip_t)
            # T13 逐对向量诊断（纯输出）。f_t_a 已是钳制后的最终切向力，
            # best_seed 用与内核同一个后端查询函数对播种材料点 q_loc 再查一次。
            if dbg_when != 0 and anchor_dbg2.shape[0] > 1:
                q_search_w = wp.transform_point(X_ws, a * w[0] + b * w[1] + c * w[2])
                q_seed_w = wp.transform_point(X_ws, q_loc)
                best_seed = float(bg)
                if _R16_SDF_EXACT != 0:
                    d_seed, n_seed = sdf_query(
                        sdf_mesh[slot], gx, slot, half_thickness + 0.01, bg, q_loc
                    )
                    best_seed = d_seed
                else:
                    best_seed = sdf_grid_sample(
                        sdf, base, nx, ny, nz, org, inv_voxel, bg, q_loc
                    )
                anchor_dbg2[pair, 0] = n_world[0]
                anchor_dbg2[pair, 1] = n_world[1]
                anchor_dbg2[pair, 2] = n_world[2]
                anchor_dbg2[pair, 3] = f_n
                anchor_dbg2[pair, 4] = f_t_a[0]
                anchor_dbg2[pair, 5] = f_t_a[1]
                anchor_dbg2[pair, 6] = f_t_a[2]
                anchor_dbg2[pair, 7] = slip_t[0]
                anchor_dbg2[pair, 8] = slip_t[1]
                anchor_dbg2[pair, 9] = slip_t[2]
                anchor_dbg2[pair, 10] = q_search_w[0]
                anchor_dbg2[pair, 11] = q_search_w[1]
                anchor_dbg2[pair, 12] = q_search_w[2]
                anchor_dbg2[pair, 13] = q_seed_w[0]
                anchor_dbg2[pair, 14] = q_seed_w[1]
                anchor_dbg2[pair, 15] = q_seed_w[2]
                anchor_dbg2[pair, 16] = best_seed
                anchor_dbg2[pair, 17] = wp.where(seeded == 0, 1.0, 0.0)
                anchor_dbg2[pair, 18] = float(slot)
                anchor_dbg2[pair, 19] = cone
            if anchor_tail != 0 and anchor_dbg2.shape[0] > 1:
                # T18：同一跑里再记一份「最后一次 Newton 迭代」的切向力与偏移。
                # 迭代 0 的那一份是试探值（pos 还是 x_prev）；这一份才是这个
                # substep 真正被施加的力。两份放同一个数组的不同槽位，就能
                # 逐对相减，不必比两次跑（GPU 原子序不可复现）。
                anchor_dbg2[pair, 20] = f_t_a[0]
                anchor_dbg2[pair, 21] = f_t_a[1]
                anchor_dbg2[pair, 22] = f_t_a[2]
                anchor_dbg2[pair, 23] = wp.length(slip_t)
            aw_out = aw
    elif _R13F_SDF_FRICTION != 0:
        mu = wp.sqrt(friction_mu * shape_material_mu[shape])
        if mu > 0.0 and dt > 0.0:
            f_n = k * depth
            if hardening_inv_dmax > 0.0:
                den_f = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
                f_n = f_n / den_f
                if _T17_SDF_HARDEN_KCAP != 0:
                    # T17: the friction cone rides the load the normal law
                    # actually applied, so it takes the same scaled denominator.
                    f_n = k * depth * (
                        1.0 + (1.0 / den_f - 1.0) * harden_kscale[shape]
                    )
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
            ft_diag = f_t
            cone_diag = mu * f_n
    if fric_diag_accum != 0 and fric_diag.shape[0] > 1:
        # T18 P-9 仪器：逐 shape 12 槽
        # 0 Sigma|f_t| | 1 Sigma f_n | 2 受载对数 | 3 Sigma cone | 4 在锥上的对数
        # 5 Sigma|slip_t| | 6 Sigma slip_max | 7-9 Sigma f_t（矢量和，世界系）
        # 10 Sigma depth | 11 max depth
        # 由调用侧在每个 substep 的第 0 次迭代清零、最后一次迭代累加，
        # 因此读到的就是「这一 substep 实际施加」的那一份。
        b12 = 12 * shape
        wp.atomic_add(fric_diag, b12 + 0, wp.length(ft_diag))
        wp.atomic_add(fric_diag, b12 + 1, fn_diag)
        wp.atomic_add(fric_diag, b12 + 2, 1.0)
        wp.atomic_add(fric_diag, b12 + 3, cone_diag)
        wp.atomic_add(fric_diag, b12 + 4, oncone_diag)
        wp.atomic_add(fric_diag, b12 + 5, slip_diag)
        if anchor_kt_ratio > 0.0 and k > 0.0:
            wp.atomic_add(fric_diag, b12 + 6, cone_diag / (k * anchor_kt_ratio))
        wp.atomic_add(fric_diag, b12 + 7, ft_diag[0])
        wp.atomic_add(fric_diag, b12 + 8, ft_diag[1])
        wp.atomic_add(fric_diag, b12 + 9, ft_diag[2])
        wp.atomic_add(fric_diag, b12 + 10, depth)
        wp.atomic_max(fric_diag, b12 + 11, depth)
    # R13g: OFF keeps the exact w_i^2 diagonal blocks; ON uses the row-sum lump.
    hw = wp.vec3(w[0] * w[0], w[1] * w[1], w[2] * w[2])
    if _R13G_SDF_HESS != 0:
        hw = w
    if _T18_FIX_NRM != 0:
        # T18 bit2：法向罚力块与切向块是同一条论证——f = k*depth*n 也按 w_i
        # 散射、Hessian 按 w_i^2 散射，而只有对角块进求解器，于是「三个顶点
        # 一起沿法向压缩」这个模态在对角上只剩 k*sum w^2（形心处 = k/3）。
        # 台架实测：只集总切向时零载残余 |Fz| 从 7.7 N 降到 0.34/0.09 N，
        # 法向也集总后再降到 0.06/0.03 N（IPC 对照 0.05/0.10 N），
        # 法向载荷 N 本身不变（11.46 vs 11.44）。
        hw = w
    # T12-b: anchor tangential half, distributed with the SEEDED weights.
    # aw_out == w whenever the anchor is off, so the OFF path is unchanged
    # (f_t_a and nn_t_a are then identically zero and add nothing).
    haw = wp.vec3(aw_out[0] * aw_out[0], aw_out[1] * aw_out[1], aw_out[2] * aw_out[2])
    if _R13G_SDF_HESS != 0:
        haw = aw_out
    if _T18_FIX_TAN != 0:
        # T18 主修：锚是一个**三角形级**的切向弹簧
        # E = 1/2 kt |sum_i aw_i x_i - p_anchor|^2。它的精确 Hessian 块是
        # kt * aw_i * aw_j；求解器只把 i=j 的对角块送进 contact_hessian_diags，
        # 三个顶点之间的耦合被丢掉。用 aw_i^2 做对角时，这个弹簧真正抵抗的模态
        # （三点一起切向平移）在对角上只剩 kt*sum aw^2（形心处 = kt/3），比真
        # 刚度 kt 小 3 倍 ⇒ Newton 每步过冲 3 倍。实测后果：同一 substep 内
        # 迭代 0 与最后一次迭代的 f_t 逐对反号（96 对里 94 对，cos 中位 -0.9986），
        # |slip_t| 在 0.99~1.46 倍 slip_max 之间来回，|f_t| 恒等于 0.9 倍满锥，
        # 与真实切向载荷无关；返回映射每 substep 又把锚重新停回锥面，把这个
        # 极限环锁死。行和集总 sum_j kt*aw_i*aw_j = kt*aw_i（aw >= 0、sum aw = 1）
        # 是保 SPD 的标准集总，对角正好复现该模态的真刚度。
        # 只动切向块：法向的 hw 不碰（法向通道实测对此不敏感，N 变化 <1.5%）。
        haw = aw_out
    if aw_out[0] > 0.0:
        wp.atomic_add(forces, i0, f_t_a * aw_out[0])
        wp.atomic_add(hessians, i0, nn_t_a * haw[0])
    if aw_out[1] > 0.0:
        wp.atomic_add(forces, i1, f_t_a * aw_out[1])
        wp.atomic_add(hessians, i1, nn_t_a * haw[1])
    if aw_out[2] > 0.0:
        wp.atomic_add(forces, i2, f_t_a * aw_out[2])
        wp.atomic_add(hessians, i2, nn_t_a * haw[2])
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
    # T8: same anchor state as the force kernel, READ ONLY here.  The reaction
    # pass must report the force the solver actually applied, so it re-reads the
    # anchor rather than advancing it (advancing twice would halve the drift).
    anchor_p: wp.array(dtype=wp.vec3),
    anchor_valid: wp.array(dtype=wp.int32),
    anchor_w: wp.array(dtype=wp.vec3),
    anchor_kt_ratio: float,
    # T14 broad phase (inert unless _T14_SDF_BROADPHASE): the pair index is
    # read out of the candidate list instead of being the thread id.
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    bp_overflow: wp.array(dtype=wp.int32),
    bp_mode: int,
    # T14 round 4 (iii): per-seed scratch, 4 entries per pair (inert unless
    # _T14_SDF_PAR).  The force kernel then only picks the minimum.
    par_best: wp.array(dtype=float),
    par_w: wp.array(dtype=wp.vec3),
    par_n: wp.array(dtype=wp.vec3),
    # T14 round 3: per-substep held contact geometry (inert unless
    # _T14_SDF_HOLD).  ``hold_mode`` 1 = search and record (iteration 0),
    # 2 = evaluate the held plane (iterations >= 1 and the reaction pass).
    hold_valid: wp.array(dtype=wp.int32),
    hold_w: wp.array(dtype=wp.vec3),
    hold_p: wp.array(dtype=wp.vec3),
    hold_n: wp.array(dtype=wp.vec3),
    hold_mode: int,
    # T15-A: nearest-triangle grid + pseudo-normals for the O(1) exact
    # backend (inert unless _T15_SDF_GRIDEXACT).
    gx: SdfGridExact,
    # T17 hardening stiffness cap, READ ONLY here (inert unless
    # _T17_SDF_HARDEN_KCAP): the reaction must ride the same scaled denominator
    # the force kernel used, or action and reaction disagree.
    harden_kscale: wp.array(dtype=float),
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
    pair = tid
    if _T14_SDF_BROADPHASE != 0:
        # T14: ON path only.  ``bp_mode`` 1 = consume the candidate list built
        # this substep; 0 = the full-scan fallback launch, which no-ops unless
        # the list overflowed.  Both launches keep a FIXED dim and read the
        # overflow flag on the device, so the pair stays CUDA-graph capturable.
        if bp_mode != 0:
            if bp_overflow[0] != 0:
                return
            if tid >= cand_count[0]:
                return
            pair = cand_idx[tid]
        else:
            if bp_overflow[0] == 0:
                return
    slot = pair / tri_count
    t = pair - slot * tri_count

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
    held = int(0)
    if _T14_SDF_HOLD != 0:
        if hold_mode == 2:
            held = 1
    if held != 0:
        # T14 HOLD: read the very plane the last force launch of this substep
        # rode on, so the two halves of the contact still cannot disagree.
        if hold_valid[pair] == 0:
            return
        w = hold_w[pair]
        q_hold = a * w[0] + b * w[1] + c * w[2]
        if _T14_SDF_HOLD == 2:
            best, n_exact = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, q_hold)
        else:
            n_exact = hold_n[pair]
            best = wp.dot(n_exact, q_hold - hold_p[pair])
    elif _T14_SDF_PAR != 0:
        # T14 (iii): the four seeds were refined in their own threads; take the
        # minimum.  Strict '<' so ties keep the lowest seed index and the pick
        # stays deterministic.
        base_i = pair * 4
        best = par_best[base_i]
        w = par_w[base_i]
        n_exact = par_n[base_i]
        d1 = par_best[base_i + 1]
        if d1 < best:
            best = d1
            w = par_w[base_i + 1]
            n_exact = par_n[base_i + 1]
        d2 = par_best[base_i + 2]
        if d2 < best:
            best = d2
            w = par_w[base_i + 2]
            n_exact = par_n[base_i + 2]
        d3 = par_best[base_i + 3]
        if d3 < best:
            best = d3
            w = par_w[base_i + 3]
            n_exact = par_n[base_i + 3]
    elif _R16_SDF_EXACT != 0:
        # R16-A2': identical query to the force pass, so the two halves of the
        # contact cannot disagree about where or how deep it is.
        w, best, n_exact = tri_sdf_closest_mesh(a, b, c, sdf_mesh[slot], gx, slot, bg, half_thickness, refine_steps)
    elif _R13G_SDF_FREEZE == 0:
        w, best = tri_sdf_closest(
                a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps,
                wp.where(_T16_SDF_ARGMIN_SOFT != 0.0, 0.0, _T16_SDF_ARGMIN_TOL * half_thickness),
                _T16_SDF_ARGMIN_SOFT * half_thickness,
            )
    else:
        # the point the force kernel actually used this substep.  resolve_w != 0
        # means the force pass has not run yet (``tri_sdf_w`` is still the zero
        # vector, which is NOT a barycentric point), so resolve it here instead.
        if resolve_w != 0:
            w, best = tri_sdf_closest(
                a, b, c, sdf, base, nx, ny, nz, org, inv_voxel, voxel, bg, refine_steps,
                wp.where(_T16_SDF_ARGMIN_SOFT != 0.0, 0.0, _T16_SDF_ARGMIN_TOL * half_thickness),
                _T16_SDF_ARGMIN_SOFT * half_thickness,
            )
        else:
            w = tri_sdf_w[pair]
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
        if _T14_SDF_EDGE_EXACT == 3:
            # HYBRID: unconditional.  Start from the grid gradient so a missed
            # query still leaves a usable normal, then overwrite with the exact
            # depth and normal at the winner point.
            n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p_local)
            d_h, n_h = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, p_local)
            if d_h < bg:
                n_local = n_h
                depth = wp.max(wp.min(half_thickness - d_h, max_depth), 0.0)
        elif _T14_SDF_EDGE_EXACT != 0:
            # depth stays the grid's; only the DIRECTION falls back, and only
            # where the grid says its own gradient is not unit length.
            ng_e, gmag_e = sdf_grid_gradient_mag(
                sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p_local
            )
            n_local = ng_e
            if wp.abs(gmag_e - 1.0) > 0.1:
                d_e, n_e = sdf_query(sdf_mesh[slot], gx, slot, mesh_max_dist, bg, p_local)
                if d_e < bg:
                    n_local = n_e
                    if _T14_SDF_EDGE_EXACT == 2:
                        # mode 2: the depth comes from the exact field too.
                        # Clamped at 0 because the exact field can put this
                        # point outside the shell that the grid put inside;
                        # a negative depth would flip the force outward.
                        depth = wp.max(wp.min(half_thickness - d_e, max_depth), 0.0)
        else:
            n_local = sdf_grid_gradient(sdf, base, nx, ny, nz, org, inv_voxel, bg, voxel, p_local)
    n_world = wp.transform_vector(X_ws, n_local)
    f_n = tri_stiffness[t] * depth
    t0_h = (2.0 / 3.0) * (particle_radius[i0] + particle_radius[i1] + particle_radius[i2])
    hardening_inv_dmax = contact_hardening_inv_dmax(shape_hardening_eps, t0_h, shape)
    if hardening_inv_dmax > 0.0:
        den = wp.max(1.0 - depth * hardening_inv_dmax, HARDENING_DEN_MIN)
        f_n = f_n / den
        if _T17_SDF_HARDEN_KCAP != 0:
            f_n = tri_stiffness[t] * depth * (
                1.0 + (1.0 / den - 1.0) * harden_kscale[shape]
            )
    reaction = n_world * (-f_n)
    p_world = wp.transform_point(X_ws, p_local)
    # R13f: the tangential half of the same contact, equal and opposite.
    # Re-evaluated with the identical inputs as the force kernel so the two
    # sides cannot disagree.  OFF branch leaves ``reaction`` untouched.
    if anchor_kt_ratio > 0.0:
        # T8 mirror of the force kernel's anchor branch, minus every write.
        mu = wp.sqrt(friction_mu * shape_material_mu[shape])
        if mu > 0.0 and anchor_valid[pair] != 0:
            # T12: same seeded material point as the force kernel (p_local uses
            # THIS iteration's winner and would reintroduce the jump).
            aw = anchor_w[pair]
            slip_w = wp.transform_vector(
                X_ws, (a * aw[0] + b * aw[1] + c * aw[2]) - anchor_p[pair]
            )
            slip_t = slip_w - n_world * wp.dot(slip_w, n_world)
            f_t = slip_t * (-tri_stiffness[t] * anchor_kt_ratio)
            cone = mu * f_n
            m = wp.length(f_t)
            if m > cone and m > 1.0e-12:
                f_t = f_t * (cone / m)
            reaction = reaction - f_t
    elif _R13F_SDF_FRICTION != 0:
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


@wp.kernel
def update_shape_hardening_kcap_kernel(
    harden_ksum: wp.array(dtype=float),
    harden_ksum_lin: wp.array(dtype=float),
    harden_kmax: wp.array(dtype=float),
    harden_ksum_last: wp.array(dtype=float),
    harden_kdiag: wp.array(dtype=float),
    harden_kscale: wp.array(dtype=float),
):
    """T17: turn the previous substep's tangent sum into this substep's scale.

    One thread per shape, launched once per substep BEFORE the first force
    evaluation, so the sum it reads is complete (iteration 0 of the previous
    substep wrote all of it) and the accumulator starts this substep at zero.
    Fixed dim, no host read-back: CUDA-graph safe.

    ``harden_kmax[shape] <= 0`` (the default for every shape) leaves the scale at
    1.0, which reproduces the stock hardening law exactly.
    """
    i = wp.tid()
    total = harden_ksum[i]
    lin = harden_ksum_lin[i]
    kmax = harden_kmax[i]
    s = 1.0
    if kmax > 0.0 and total > kmax:
        s = kmax / total
    harden_kscale[i] = s
    harden_ksum_last[i] = total
    harden_ksum[i] = 0.0
    harden_ksum_lin[i] = 0.0
    # diagnostic, 8 slots per shape:
    # 0 max Sigma_hard | 1 min s | 2 substeps | 3 max Sigma_lin
    # 4 sum Sigma_hard | 5 sum Sigma_lin | 6 loaded substeps | 7 spare
    if kmax > 0.0:
        wp.atomic_max(harden_kdiag, 8 * i + 0, total)
        wp.atomic_min(harden_kdiag, 8 * i + 1, s)
        wp.atomic_add(harden_kdiag, 8 * i + 2, 1.0)
        wp.atomic_max(harden_kdiag, 8 * i + 3, lin)
        if total > 0.0:
            wp.atomic_add(harden_kdiag, 8 * i + 4, total)
            wp.atomic_add(harden_kdiag, 8 * i + 5, lin)
            wp.atomic_add(harden_kdiag, 8 * i + 6, 1.0)


@wp.kernel
def add_tri_sdf_cache_kernel(
    cache_f: wp.array(dtype=wp.vec3),
    cache_h: wp.array(dtype=wp.mat33),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessians: wp.array(dtype=wp.mat33),
):
    """T14: fold the cached tri-SDF contact force/Hessian into this iteration.

    Used only when ``T14_SDF_EVERY > 1``.  The expensive kernel then runs on
    every k-th Newton iteration and writes its result into a per-particle cache
    instead of accumulating straight into the solver's RHS; this kernel adds the
    cache in on EVERY iteration, including the skipped ones, so the contact term
    is present in every solve -- it is held at the value the last evaluated
    iterate produced rather than dropped.

    One thread per particle and the launches are serialised on the stream, so
    the read-modify-write needs no atomic (unlike the contact kernel, which has
    many triangles landing on the same particle).
    """
    tid = wp.tid()
    forces[tid] = forces[tid] + cache_f[tid]
    hessians[tid] = hessians[tid] + cache_h[tid]


@wp.func
def _point_aabb_dist2(p: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    """Squared distance from ``p`` to the axis-aligned box; 0 when inside."""
    dx = wp.max(wp.max(lo[0] - p[0], p[0] - hi[0]), 0.0)
    dy = wp.max(wp.max(lo[1] - p[1], p[1] - hi[1]), 0.0)
    dz = wp.max(wp.max(lo[2] - p[2], p[2] - hi[2]), 0.0)
    return dx * dx + dy * dy + dz * dz


@wp.kernel
def tri_sdf_broadphase_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
    half_thickness: float,
    slack: float,
    capacity: int,
    anchor_valid: wp.array(dtype=wp.int32),
    anchor_kt_ratio: float,
    # outputs
    cand_idx: wp.array(dtype=wp.int32),
    cand_count: wp.array(dtype=wp.int32),
    bp_overflow: wp.array(dtype=wp.int32),
):
    """Gather the (slot, triangle) pairs that can possibly contact this substep.

    One thread per pair, once per substep.  ``cand_count``/``bp_overflow`` are
    zeroed by the caller; on overflow the flag is raised and the consumers fall
    back to a full scan, so the capacity is a performance knob only.

    A pair that is NOT gathered also has its tangential anchor dropped here:
    the contact kernel is what normally clears it (``best >= h`` -> invalidate),
    and a pair off the list never reaches that statement, so an anchor would
    otherwise survive with a stale reference frame.
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

    a = wp.transform_point(X_sw, pos[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, pos[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, pos[tri_indices[t, 2]])

    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    r = half_thickness + reach + slack

    if _point_aabb_dist2(g, aabb_lo[slot], aabb_hi[slot]) <= r * r:
        k = wp.atomic_add(cand_count, 0, 1)
        if k < capacity:
            cand_idx[k] = tid
        else:
            bp_overflow[0] = 1
    else:
        if anchor_kt_ratio > 0.0 and anchor_valid[tid] != 0:
            anchor_valid[tid] = 0


@wp.kernel
def tri_sdf_bp_selfcheck_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
    half_thickness: float,
    slack: float,
    sdf_mesh: wp.array(dtype=wp.uint64),
    bg: float,
    # outputs: [0] gathered, [1] cull-pass, [2] cull-pass AND NOT gathered,
    #          [3] in geometric contact (best < h at the centroid seed)
    stats: wp.array(dtype=wp.int32),
):
    """Diagnostic: does the AABB gather ever drop a pair the cull would keep?

    Runs the gather test and the REAL centroid cull side by side on every pair.
    ``stats[2]`` is the falsification counter -- it must be 0, by the containment
    argument above; anything else means the gather is not a superset and the ON
    path is not equivalent.  Off the verification path this kernel is not
    launched at all.
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

    a = wp.transform_point(X_sw, pos[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, pos[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, pos[tri_indices[t, 2]])

    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    r = half_thickness + reach + slack
    gathered = _point_aabb_dist2(g, aabb_lo[slot], aabb_hi[slot]) <= r * r

    # the cull exactly as tri_sdf_closest_mesh runs it
    best, _n = sdf_mesh_query(sdf_mesh[slot], half_thickness + reach, bg, g)
    culled_in = best < bg

    if gathered:
        wp.atomic_add(stats, 0, 1)
    if culled_in:
        wp.atomic_add(stats, 1, 1)
        if not gathered:
            wp.atomic_add(stats, 2, 1)
        if best < half_thickness:
            wp.atomic_add(stats, 3, 1)


@wp.kernel
def tri_sdf_par_cull_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    sdf_mesh: wp.array(dtype=wp.uint64),
    bg: float,
    half_thickness: float,
    # T15-A: nearest-triangle grid + pseudo-normals for the O(1) exact
    # backend (inert unless _T15_SDF_GRIDEXACT).
    gx: SdfGridExact,
    # outputs
    g_best: wp.array(dtype=float),
    g_n: wp.array(dtype=wp.vec3),
):
    """T14 (iii) pass 1: the centroid cull, once per pair.

    Kept in its own launch so a triangle nowhere near the blade still costs
    exactly ONE query -- the same as today.  Splitting the seeds without this
    would make the far case cost four.
    """
    pair = wp.tid()
    slot = pair / tri_count
    t = pair - slot * tri_count
    shape = slot_shape[slot]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, pos[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, pos[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, pos[tri_indices[t, 2]])
    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    d, n = sdf_query(sdf_mesh[slot], gx, slot, half_thickness + reach, bg, g)
    g_best[pair] = d
    g_n[pair] = n


@wp.kernel
def tri_sdf_par_seed_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    tri_count: int,
    slot_shape: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    sdf_mesh: wp.array(dtype=wp.uint64),
    bg: float,
    half_thickness: float,
    refine_steps: int,
    g_best: wp.array(dtype=float),
    g_n: wp.array(dtype=wp.vec3),
    # T15-A: nearest-triangle grid + pseudo-normals for the O(1) exact
    # backend (inert unless _T15_SDF_GRIDEXACT).
    gx: SdfGridExact,
    # outputs, 4 per pair
    par_best: wp.array(dtype=float),
    par_w: wp.array(dtype=wp.vec3),
    par_n: wp.array(dtype=wp.vec3),
):
    """T14 (iii) pass 2: one thread per (pair, seed), each refining from its own.

    Seed 0 is the centroid and reuses pass 1's query, so it spends none; seeds
    1-3 spend one each.  Then ``refine_steps`` from there, through the SAME
    ``tri_sdf_refine_from`` the serial search uses.
    """
    tid = wp.tid()
    pair = tid / 4
    seed = tid - pair * 4
    slot = pair / tri_count
    t = pair - slot * tri_count

    bg_g = g_best[pair]
    if bg_g >= bg:
        par_best[tid] = bg
        return

    shape = slot_shape[slot]
    body = shape_body[shape]
    X_ws = shape_transform[shape]
    if body >= 0:
        X_ws = body_q[body] * shape_transform[shape]
    X_sw = wp.transform_inverse(X_ws)
    a = wp.transform_point(X_sw, pos[tri_indices[t, 0]])
    b = wp.transform_point(X_sw, pos[tri_indices[t, 1]])
    c = wp.transform_point(X_sw, pos[tri_indices[t, 2]])
    third = 1.0 / 3.0
    g = (a + b + c) * third
    reach = wp.max(wp.length(a - g), wp.max(wp.length(b - g), wp.length(c - g)))
    rq = half_thickness + 2.0 * reach
    if _T14_SDF_RQ != 0:
        rq = wp.abs(bg_g) + reach + 1.0e-6

    w = wp.vec3(third, third, third)
    p0 = g
    best = bg_g
    nb = g_n[pair]
    if seed == 1:
        w = wp.vec3(1.0, 0.0, 0.0)
        p0 = a
        best, nb = sdf_query(sdf_mesh[slot], gx, slot, rq, bg, a)
    if seed == 2:
        w = wp.vec3(0.0, 1.0, 0.0)
        p0 = b
        best, nb = sdf_query(sdf_mesh[slot], gx, slot, rq, bg, b)
    if seed == 3:
        w = wp.vec3(0.0, 0.0, 1.0)
        p0 = c
        best, nb = sdf_query(sdf_mesh[slot], gx, slot, rq, bg, c)
    if best >= bg:
        par_best[tid] = bg
        return

    w, best, nb = tri_sdf_refine_from(
        a, b, c, sdf_mesh[slot], gx, slot, bg, _T16_SDF_ARGMIN_TOL * half_thickness,
        0.0, rq, w, best, nb, p0, refine_steps
    )
    par_best[tid] = best
    par_w[tid] = w
    par_n[tid] = nb
