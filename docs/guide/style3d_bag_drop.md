# Style3D Bag Drop Tuning

This guide documents the dedicated Style3D bag drop example for local material and contact tuning.

## Related Code

The bag asset import and test flow is split across these files:

```text
scripts/newton_asset_probe.py
newton/examples/cloth/_style3d_asset_probe.py
newton/examples/cloth/example_cloth_style3d.py
newton/examples/cloth/example_cloth_style3d_fold_probe.py
newton/examples/cloth/example_cloth_style3d_bag_drop.py
docs/guide/style3d_asset_probe.md
docs/guide/style3d_bag_drop.md
```

The new dedicated entry point is:

```bash
uv run -m newton.examples cloth_style3d_bag_drop --device cuda:0
```

It loads the default bag asset:

```text
newton/examples/assets/style3d_probe/bag/nonwoven_small_6/nonwoven_small_6.obj
```

Pass another OBJ/USD asset as the positional argument to test a different bag.

## What The Example Does

`cloth_style3d_bag_drop` uses the same asset import helpers as `scripts/newton_asset_probe.py`, then runs a normal
Newton example viewer loop:

1. Parse OBJ/USD mesh data.
2. Use OBJ `vt` data as Style3D panel coordinates when available.
3. Repair negative-area panel winding unless `--no-style3d-fix-panel-winding` is set.
4. Clean duplicate triangles and extra non-manifold faces by default.
5. Build a Style3D cloth model with a ground plane.
6. Advance the bag by `--view-substeps` Style3D solver substeps per rendered viewer frame.

The default bag is a connected 3D mesh, so it appears sewn because the OBJ vertex topology is already connected. UV/panel
vertices can still be separate for Style3D material coordinates.

## Time Step Controls

The example follows the official Style3D viewer pattern:

```text
frame_dt = 1 / view_fps
sim_dt = frame_dt / view_substeps
```

With defaults:

```text
view_fps = 60
view_substeps = 10
sim_dt = 1 / 60 / 10 = 0.0016667 s
```

If the machine renders only 10 viewer FPS, it still uses `sim_dt = 0.0016667 s`; it simply advances fewer viewer frames
per real second. One real second then advances only `10 / 60 = 0.1667 s` of simulation time, so the motion looks 6 times
slower in wall-clock time.

Use `--view-substeps` to reduce each physics step size. Use `--view-fps` to choose how much simulation time one viewer
frame represents.

## Low Stiffness Freefall Check

Use this to verify gravity and freefall speed with minimal internal stiffness and self-collision disabled. This is not a
recommended ground-contact setup; the bag will be very soft after impact.

```bash
uv run -m newton.examples cloth_style3d_bag_drop \
    --device cuda:0 \
    --start-height 0.5 \
    --view-fps 60 \
    --view-substeps 10 \
    --solver-iterations 3 \
    --linear-iterations 50 \
    --cloth-density 1.0 \
    --particle-radius 0.0015 \
    --tri-aniso-ke 1,1,0.01 \
    --tri-ka 0 \
    --tri-kd 0 \
    --edge-aniso-ke 0,0,0 \
    --edge-kd 0 \
    --soft-contact-margin 0.0001 \
    --soft-contact-ke 1.0 \
    --soft-contact-kd 0 \
    --soft-contact-mu 0 \
    --no-style3d-self-collision \
    --style3d-collision-radius 0 \
    --style3d-collision-stiff-vf 0 \
    --style3d-collision-stiff-ee 0 \
    --style3d-collision-stiff-ef 0 \
    --no-cuda-graph
```

Before ground contact, compare the vertical displacement to:

```text
z(t) = z0 - 0.5 * 9.81 * t^2
v(t) = -9.81 * t
```

Remember that `t` is simulation time, not wall-clock time when rendering is slower than `--view-fps`.

## High Stiffness Bag Setup

This matches the current high-stiffness bag tuning path and keeps Style3D self-collision enabled with reduced
self-collision stiffness. Start here when tuning bag shape preservation.

```bash
uv run -m newton.examples cloth_style3d_bag_drop \
    --device cuda:0 \
    --start-height 0.5 \
    --view-fps 60 \
    --view-substeps 10 \
    --solver-iterations 3 \
    --linear-iterations 50 \
    --cloth-density 1.0 \
    --particle-radius 0.0015 \
    --tri-aniso-ke 100000,100000,1000 \
    --tri-ka 500 \
    --tri-kd 1e-4 \
    --edge-aniso-ke 5e-6,5e-6,5e-6 \
    --edge-kd 1e-2 \
    --soft-contact-margin 0.0015 \
    --soft-contact-ke 10 \
    --soft-contact-kd 1e-6 \
    --soft-contact-mu 0.2 \
    --style3d-collision-radius 0.0005 \
    --style3d-collision-stiff-vf 0.01 \
    --style3d-collision-stiff-ee 0.005 \
    --style3d-collision-stiff-ef 0.05 \
    --no-cuda-graph
```

If ground impact launches the bag upward, try this contact-first stabilization variant before raising material stiffness
further:

```bash
uv run -m newton.examples cloth_style3d_bag_drop \
    --device cuda:0 \
    --start-height 0.2 \
    --view-fps 60 \
    --view-substeps 30 \
    --solver-iterations 8 \
    --linear-iterations 200 \
    --cloth-density 1.0 \
    --particle-radius 0.001 \
    --tri-aniso-ke 1000,1000,50 \
    --tri-ka 50 \
    --tri-kd 1e-5 \
    --edge-aniso-ke 1e-6,1e-6,1e-6 \
    --edge-kd 1e-3 \
    --soft-contact-margin 0.001 \
    --soft-contact-ke 5 \
    --soft-contact-kd 1e-3 \
    --soft-contact-mu 0.4 \
    --style3d-collision-radius 0.0005 \
    --style3d-collision-stiff-vf 0.005 \
    --style3d-collision-stiff-ee 0.002 \
    --style3d-collision-stiff-ef 0.01 \
    --no-cuda-graph
```

## Parameter Groups

Public Newton/Style3D physics controls:

```text
--solver-iterations
--linear-iterations
--cloth-density
--particle-radius
--tri-aniso-ke
--tri-ka
--tri-kd
--edge-aniso-ke
--edge-kd
--soft-contact-margin
--soft-contact-ke
--soft-contact-kd
--soft-contact-mu
--style3d-collision-radius
--style3d-collision-stiff-vf
--style3d-collision-stiff-ee
--style3d-collision-stiff-ef
```

Example/probe controls:

```text
asset
--scale
--start-height
--view-fps
--view-substeps
--style3d-clean-nonmanifold
--no-style3d-fix-panel-winding
--style3d-sew-distance
--style3d-sew-ke
--style3d-sew-kd
--no-cuda-graph
--show-particles
```

The old soft contact radius option is not exposed in this example. Ground/body contact detection uses
`--soft-contact-margin` through `newton.CollisionPipeline`, and the cloth particle size is controlled by
`--particle-radius`.

## Tuning Notes

If the bag collapses into a flat pile after impact, increase material stiffness gradually and keep Style3D self-collision
enabled. If the bag explodes or bounces too high, lower `--soft-contact-ke`, raise `--soft-contact-kd`, reduce
self-collision stiffness, or increase `--view-substeps`.

If the bag penetrates the ground, increase `--view-substeps` first. Then increase `--soft-contact-margin` or
`--soft-contact-ke` in small steps.

For slow machines, do not infer physical speed from wall-clock FPS. Log or compute against simulation time:
`simulation_seconds = rendered_frames / view_fps`.
