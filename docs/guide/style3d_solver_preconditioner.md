# Style3D Solver Preconditioner Changes

This note documents the Style3D simulator changes that add an optional coarse translation preconditioner to the PCG
linear solve and expose it through cloth examples. It is meant to accompany the test-branch code changes so collaborators
can review the behavior without reconstructing the design from the diff alone.

## Summary

The Style3D solver still follows the existing projective-dynamics loop:

```text
init step
for each nonlinear iteration:
    build rhs from inertia, stretch, bend, dragging, and contacts
    build the local Jacobi preconditioner
    solve the linearized displacement system with PCG
    apply dx to the current particle positions
update velocity
```

The new path adds an optional PCG preconditioner stage after Jacobi. It estimates one coarse per-axis translation
correction per connected cloth component and adds that correction to the PCG preconditioned residual.

Direct solver construction:

```python
solver = newton.solvers.SolverStyle3D(
    model,
    iterations=5,
    linear_iterations=50,
    enable_translation_preconditioner=True,
)
```

Example CLI flag:

```bash
--style3d-translation-preconditioner
```

The flag is available on `cloth_style3d_bag_drop`, `cloth_style3d_garment_drop`, and `cloth_style3d_two_cloths`.

## Motivation

The existing Jacobi preconditioner is local: each particle is corrected from its own diagonal approximation. That is cheap
and robust, but it does not directly capture low-frequency modes where a connected cloth piece moves almost as one body.

The new preconditioner adds a cheap component-level translation correction for those modes. It does not change the target
linear system; it changes only the PCG preconditioner.

## Linear Solver Hook

`PcgSolver.solve()` now accepts:

```python
preconditioner: Callable | None = None
```

The callback is invoked once per PCG iteration immediately after the Jacobi update:

```text
z = inv_M * r
preconditioner(r, z)
```

The callback receives the current residual `r` and the mutable preconditioned residual `z`. This keeps the PCG solver
generic while letting Style3D add cloth-specific coarse correction logic.

The existing collision Hessian-vector product remains separate and still enters through `additional_multiplier`, using
`collision.hessian_multiply()`.

## Component Grouping

`SolverStyle3D` builds a static particle-to-component map in the constructor using union-find over `model.tri_indices`:

```text
for each triangle (a, b, c):
    union(a, b)
    union(b, c)
    union(c, a)
```

Particles in different `particle_world` entries are not merged. This avoids cross-world coarse solves in models that
contain more than one world or disconnected cloth object.

The solver allocates the coarse buffers once:

```text
_translation_particle_component
_translation_coarse_rhs
_translation_coarse_diag
_translation_zero_contact_hessian
```

Because the component mapping is static, recreate the solver if a future workflow mutates cloth topology after solver
construction.

## Preconditioner Math

For each active particle, the accumulate kernel adds the residual to the component RHS:

```text
coarse_rhs[component] += residual[particle]
```

It also accumulates a diagonal approximation per world axis:

```text
mass_diag = particle_mass / dt^2

coarse_diag[component].x += mass_diag + contact_hessian.xx
coarse_diag[component].y += mass_diag + contact_hessian.yy
coarse_diag[component].z += mass_diag + contact_hessian.zz
```

The apply kernel computes one translation correction per component:

```text
correction.axis = coarse_rhs[component].axis / coarse_diag[component].axis
z[particle] += correction
```

Each axis checks for a positive denominator before division. Inactive particles are ignored. The accumulation uses
floating-point atomics, so tiny device- or schedule-dependent numerical differences are expected.

## Contact Coupling

At the start of each Style3D step, the solver resets:

```text
_translation_preconditioner_dt = dt
_translation_contact_hessian_diags = _translation_zero_contact_hessian
```

When collision is enabled, `collision.accumulate_contact_force()` computes contact forces and contact Hessian diagonals.
The solver then uses:

```text
collision.contact_hessian_diagonal()
```

as the contact stiffness contribution for the coarse diagonal. The full contact Hessian-vector product remains in the PCG
matrix-vector product through `collision.hessian_multiply()`.

## Example Usage

Bag:

```bash
uv run -m newton.examples cloth_style3d_bag_drop \
    --device cuda:0 \
    --solver-iterations 5 \
    --linear-iterations 50 \
    --style3d-translation-preconditioner
```

Garment:

```bash
uv run -m newton.examples cloth_style3d_garment_drop \
    --device cuda:0 \
    --style3d-translation-preconditioner
```

Two-cloth diagnostic:

```bash
uv run -m newton.examples cloth_style3d_two_cloths \
    --device cuda:0 \
    --style3d-translation-preconditioner
```

`example_cloth_style3d_two_cloths.py` creates two disconnected cloth grids in one model. It is useful for checking that
component grouping is per connected cloth object:

```text
low cloth particles  -> one component
high cloth particles -> another component
```

Each cloth should receive its own coarse translation correction.

## CUDA Graph Capture

When enabled, the translation preconditioner adds two Warp kernel launches per PCG iteration:

```text
_accumulate_translation_preconditioner_kernel
_apply_translation_preconditioner_kernel
```

The buffers are allocated in the solver constructor, and launch dimensions are based on fixed particle and component
counts. This should be compatible with CUDA graph capture when the surrounding example has fixed control flow.

For debugging, use:

```bash
--no-cuda-graph
```

For steady-state timing on CUDA, omit `--no-cuda-graph` and compare replay time after the initial capture frame. The first
captured frame includes graph setup cost.

## Validation

Unit test target:

```bash
uv run pytest newton/tests/test_solver_style3d.py
```

Manual smoke tests:

```bash
uv run -m newton.examples cloth_style3d_bag_drop \
    --device cuda:0 \
    --style3d-translation-preconditioner \
    --no-cuda-graph
```

```bash
uv run -m newton.examples cloth_style3d_two_cloths \
    --device cuda:0 \
    --style3d-translation-preconditioner \
    --no-cuda-graph
```

Useful comparisons:

```text
with and without --style3d-translation-preconditioner
with and without --no-cuda-graph
different --linear-iterations values
collision enabled vs. --no-style3d-self-collision
```

The main correctness checks are finite particle positions and velocities, stable contact behavior, and no cross-component
coupling between disconnected cloths.

## Review Note

The test branch includes `test_step_uses_configured_linear_iterations`, which records calls into the linear solver and
expects the configured `linear_iterations` value to be passed on every nonlinear iteration.

Reviewers should check `SolverStyle3D.step()` carefully. If the non-preconditioned path still uses a nonlinear-iteration
ramp such as:

```text
min(nonlinear_iter + 1, 10)
```

then `linear_iterations` is not honored literally on that path, and the test expectation and implementation need to be
reconciled before merging. This document only calls out the review point; it does not change solver behavior.

## Limitations

The coarse translation preconditioner is intentionally narrow:

```text
it corrects translation modes only
it does not model component rotation
it does not replace the local Jacobi preconditioner
it uses only diagonal mass and contact stiffness
it assumes fixed cloth topology after solver construction
it ignores inactive particles
```

Because the correction is added to the PCG preconditioned residual, it can change convergence behavior even though it does
not change the target linear system. Keep it behind the explicit flag when comparing against existing baselines.

## Files Covered

Core implementation:

```text
newton/_src/solvers/style3d/linear_solver.py
newton/_src/solvers/style3d/solver_style3d.py
```

Examples:

```text
style3d/examples/example_cloth_style3d_bag_drop.py
style3d/examples/example_cloth_style3d_garment_drop.py
style3d/examples/example_cloth_style3d_two_cloths.py
```

Tests:

```text
newton/tests/test_solver_style3d.py
```
