# Plastic Bag Drawstring Examples

This note documents the plastic bag, drawstring, and dustbin cloth examples:

```text
newton/examples/cloth/example_cloth_vbd_plastic_bag_drawstring.py
newton/examples/cloth/example_cloth_style3d_plastic_bag_drawstring.py
newton/_src/solvers/style3d/collision/kernels.py
```

Both examples load the same asset family:

```text
newton/examples/assets/style3d_probe/bag/plastic_bag/
```

The bag is imported as one cloth mesh, the drawstring is imported as a second cloth strip, and the dustbin is imported as
a static rigid mesh. The drawstring mesh is welded before import so close endpoints become connected.

## VBD Example

Run the VBD setup with the tuned defaults:

```bash
uv run -m newton.examples cloth_vbd_plastic_bag_drawstring \
    --device cuda:0
```

The defaults match this explicit setup:

```bash
uv run -m newton.examples cloth_vbd_plastic_bag_drawstring \
    --device cuda:0 \
    --no-fix-mouth \
    --bag-particle-radius 0.005 \
    --drawstring-particle-radius 0.005 \
    --soft-contact-ke 5e4 \
    --soft-contact-kd 1.0 \
    --dustbin-contact-ke 5e5 \
    --dustbin-contact-kd 50 \
    --rigid-body-particle-contact-buffer-size 16384
```

The VBD example drives selected drawstring particles along a parabolic trajectory. The driven particles are made
kinematic by clearing `ParticleFlags.ACTIVE` and setting mass and inverse mass to zero. VBD then solves particle
constraints, self contact, and body-particle contacts around those kinematic handles.

### Increasing VBD Stiffness

Use these parameters when the plastic bag is too stretchy:

```text
--bag-tri-ke
--bag-tri-ka
--bag-edge-ke
--bag-spring-ke
```

Use these parameters when the drawstring is too stretchy or bends too much:

```text
--drawstring-tri-ke
--drawstring-tri-ka
--drawstring-edge-ke
--drawstring-spring-ke
```

Start by increasing the drawstring first, because the pull trajectory is applied to the drawstring handles:

```bash
uv run -m newton.examples cloth_vbd_plastic_bag_drawstring \
    --device cuda:0 \
    --drawstring-tri-ke 5e4 \
    --drawstring-tri-ka 5e4 \
    --drawstring-edge-ke 5e-2 \
    --drawstring-spring-ke 5e4
```

If the bag stretches too much while the drawstring pulls it, increase the bag stiffness more conservatively:

```bash
uv run -m newton.examples cloth_vbd_plastic_bag_drawstring \
    --device cuda:0 \
    --bag-tri-ke 1e4 \
    --bag-tri-ka 1e4 \
    --bag-edge-ke 5e-3
```

If higher stiffness causes oscillation or instability, increase solver work before raising stiffness further:

```bash
--solver-iterations 20
--view-substeps 8
```

## Style3D Example

Run the Style3D setup with the tuned defaults:

```bash
uv run -m newton.examples cloth_style3d_plastic_bag_drawstring \
    --device cuda:0 \
    --no-cuda-graph
```

The defaults match this explicit setup:

```bash
uv run -m newton.examples cloth_style3d_plastic_bag_drawstring \
    --device cuda:0 \
    --no-cuda-graph \
    --style3d-panel-scale 1.0 \
    --soft-contact-ke 10000 \
    --soft-contact-kd 0.1 \
    --soft-contact-mu 0.8 \
    --soft-contact-radius 0.006 \
    --soft-contact-margin 0.012 \
    --bag-particle-radius 0.006 \
    --drawstring-particle-radius 0.004 \
    --solver-iterations 12 \
    --linear-iterations 24 \
    --style3d-collision-radius 0.005 \
    --style3d-collision-stiff-vf 0.02 \
    --style3d-collision-stiff-ee 0.01 \
    --style3d-collision-stiff-ef 0.04 \
    --no-fix-mouth \
    --pull-duration 2.0
```

The Style3D example also uses kinematic drawstring handles. The difference is that Style3D cloth self contact is solved
through the Style3D collision path using vertex-face, edge-edge, and edge-face contact terms. Body contact with the
dustbin is generated through `CollisionPipeline` and applied as particle-body soft contact forces.

## Why `kernels.py` Was Changed

Style3D self contact combines the two contacting sides using stiffness values derived from the solver diagonal. The
original expression was:

```text
stiff_factor * stiff_0 * stiff_1 / (stiff_0 + stiff_1)
```

That is fine when both sides are active cloth. It breaks down when fixed or kinematic particles are involved:

- if both sides have zero effective stiffness, the expression becomes `0 / 0` and can produce NaNs;
- if one side has zero effective stiffness, the combined stiffness becomes zero, so contact with that kinematic side is
  effectively disabled.

The drawstring pull uses kinematic handles, so the selected drawstring particles are exactly this zero-stiffness case.
Without the kernel change, those handles can be pulled through the bag without the bag receiving a useful self-contact
response.

The helper now treats a one-sided zero-stiffness contact as contact against a kinematic obstacle:

```text
if both sides are zero:
    skip the contact
if side 0 is zero:
    use side 1 stiffness
if side 1 is zero:
    use side 0 stiffness
otherwise:
    use the original combined stiffness
```

This keeps the original behavior for active cloth-cloth contacts, prevents `0 / 0`, and allows kinematic drawstring
handles and fixed bag-mouth particles to participate in Style3D self contact as static obstacles.

## Tuning Notes

For Style3D bag/drawstring self contact, tune:

```text
--style3d-collision-radius
--style3d-collision-stiff-vf
--style3d-collision-stiff-ee
--style3d-collision-stiff-ef
```

For dustbin contact, tune:

```text
--soft-contact-ke
--soft-contact-kd
--soft-contact-mu
--soft-contact-radius
--soft-contact-margin
```

For VBD body-particle contact buffer pressure around the dustbin, tune:

```text
--rigid-body-particle-contact-buffer-size
```

