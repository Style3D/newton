# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Viewer, camera, and sensor helpers for Newton teleoperation examples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

from .newton_ik_teleop import (
    Button,
    _as_np3,
    quat_mul_xyzw,
    quat_rotate_xyzw,
    quat_wxyz_to_xyzw,
    transform_point_pose_xyzw,
)


@wp.kernel
def offset_camera_ray_origins_kernel(camera_rays: wp.array4d[wp.vec3f], near_clip: float):
    camera_index, py, px = wp.tid()
    ray_direction = camera_rays[camera_index, py, px, 1]
    camera_rays[camera_index, py, px, 0] = ray_direction * wp.float32(near_clip)


@dataclass(frozen=True)
class CameraFrameSpec:
    """A named camera frame attached to a model body."""

    name: str
    parent_body_name: str | None
    local_pos: tuple[float, float, float]
    local_quat_wxyz: tuple[float, float, float, float]
    fov_y_deg: float


def _parse_float_sequence(text: str | None, expected: int, *, default: Sequence[float]) -> tuple[float, ...]:
    if text is None:
        return tuple(float(v) for v in default)
    values = tuple(float(part) for part in text.replace(",", " ").split())
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(values)} from {text!r}")
    return values


def load_mjcf_camera_frame(mjcf_path: str | Path, camera_name: str) -> CameraFrameSpec:
    """Load a named MJCF camera's local frame and parent body name."""

    path = Path(mjcf_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF has no worldbody: {path}")

    def visit(element: ET.Element, parent_body_name: str | None) -> CameraFrameSpec | None:
        current_parent = element.get("name") if element.tag == "body" else parent_body_name
        if element.tag == "camera" and element.get("name") == camera_name:
            pos = _parse_float_sequence(element.get("pos"), 3, default=(0.0, 0.0, 0.0))
            quat = _parse_float_sequence(element.get("quat"), 4, default=(1.0, 0.0, 0.0, 0.0))
            fov = float(element.get("fovy", element.get("fov", 45.0)))
            return CameraFrameSpec(
                name=camera_name,
                parent_body_name=parent_body_name,
                local_pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                local_quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                fov_y_deg=fov,
            )
        for child in element:
            found = visit(child, current_parent)
            if found is not None:
                return found
        return None

    found = visit(worldbody, None)
    if found is None:
        raise ValueError(f"Could not find MJCF camera {camera_name!r} in {path}")
    return found


def set_viewer_camera_look_at(
    viewer,
    position: Sequence[float],
    target: Sequence[float],
    *,
    up: Sequence[float] | None = None,
    fov: float | None = None,
) -> None:
    """Set a Newton viewer camera from a world position and look-at target."""

    camera = getattr(viewer, "camera", None)
    position_np = _as_np3(position)
    target_np = _as_np3(target)
    if camera is not None and hasattr(camera, "look_at"):
        camera.pos = camera._as_vec3(position_np)
        if up is not None:
            try:
                camera.look_at(target_np, up=_as_np3(up))
            except TypeError:
                camera.look_at(target_np)
        else:
            camera.look_at(target_np)
        if fov is not None and hasattr(camera, "fov"):
            camera.fov = float(fov)
        if hasattr(viewer, "_camera_dirty"):
            viewer._camera_dirty = True
        return

    direction = target_np - position_np
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        pitch = 0.0
        yaw = 0.0
    else:
        direction /= norm
        pitch = float(np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
        yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(*map(float, position_np)), pitch, yaw)


def ghost_model_bodies(
    viewer,
    model: newton.Model,
    *,
    body_start: int,
    body_end: int,
    alpha: float = 0.22,
    color: Sequence[float] = (0.62, 0.72, 0.78),
) -> None:
    """Ghost render shapes attached to ``body_start:body_end``."""

    if model is None or getattr(model, "shape_body", None) is None:
        return

    alpha = float(np.clip(alpha, 0.02, 1.0))
    body_start = int(body_start)
    body_end = int(body_end)
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int32)
    if shape_body.size == 0:
        return
    mask = (shape_body >= body_start) & (shape_body < body_end)
    if not np.any(mask):
        return

    if getattr(model, "shape_color", None) is not None:
        colors = np.asarray(model.shape_color.numpy(), dtype=np.float32)
        colors[mask] = np.asarray(color, dtype=np.float32)[:3]
        model.shape_color.assign(colors)

    shape_instances = getattr(viewer, "_shape_instances", None)
    if shape_instances:
        for batch in shape_instances.values():
            materials = getattr(batch, "materials", None)
            model_shapes = getattr(batch, "model_shapes", None)
            if materials is None or model_shapes is None:
                continue
            material_np = np.asarray(materials.numpy(), dtype=np.float32)
            changed = False
            for local_index, shape_index in enumerate(model_shapes):
                if 0 <= int(shape_index) < len(mask) and mask[int(shape_index)]:
                    material_np[local_index, 2] = -alpha
                    changed = True
            if changed:
                materials.assign(material_np)

    if hasattr(viewer, "model_changed"):
        viewer.model_changed = True


class TeleopEndEffectorVisualizer:
    """Draw a small TCP point and RGB axes for a teleoperated end effector."""

    def __init__(
        self,
        viewer,
        *,
        body_index: int,
        local_offset: Sequence[float] = (0.0, 0.0, 0.0),
        axis_length: float = 0.08,
        point_radius: float = 0.006,
        point_alpha: float = 0.45,
        line_width: float = 0.008,
        name: str = "/teleop/eef",
        device=None,
    ) -> None:
        self.viewer = viewer
        self.body_index = int(body_index)
        self.local_offset = _as_np3(local_offset)
        self.axis_length = float(axis_length)
        self.point_radius = float(point_radius)
        self.point_alpha = float(np.clip(point_alpha, 0.02, 1.0))
        self.line_width = float(line_width)
        self.name = name
        self.device = device
        self._axis_colors = wp.array(
            [wp.vec3(1.0, 0.05, 0.05), wp.vec3(0.05, 0.9, 0.1), wp.vec3(0.1, 0.35, 1.0)],
            dtype=wp.vec3,
            device=device,
        )
        self._point_color = wp.array([wp.vec3(1.0, 0.15, 0.1)], dtype=wp.vec3, device=device)
        self._point_material = wp.array(
            [wp.vec4(0.35, 0.0, -self.point_alpha, 0.0)],
            dtype=wp.vec4,
            device=device,
        )

    def pose_from_state(self, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
        body_q = np.asarray(state.body_q.numpy(), dtype=np.float64)
        pose = body_q[self.body_index]
        q_xyzw = pose[3:7]
        pos = transform_point_pose_xyzw(pose, self.local_offset)
        return pos, q_xyzw

    def render(self, state: newton.State, *, hidden: bool = False) -> None:
        if hidden:
            self.viewer.log_lines(f"{self.name}/axes", None, None, None, hidden=True)
            if hasattr(self.viewer, "log_shapes"):
                point_xform = wp.array(
                    [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                    dtype=wp.transform,
                    device=self.device,
                )
                self.viewer.log_shapes(
                    f"{self.name}/point",
                    newton.GeoType.SPHERE,
                    self.point_radius,
                    point_xform,
                    colors=self._point_color,
                    materials=self._point_material,
                    hidden=True,
                )
            else:
                self.viewer.log_points(f"{self.name}/point", None, hidden=True)
            return

        pos, q_xyzw = self.pose_from_state(state)
        axes = np.eye(3, dtype=np.float64)
        starts_np = np.repeat(pos[None, :], 3, axis=0).astype(np.float32)
        ends_np = np.asarray(
            [pos + quat_rotate_xyzw(q_xyzw, axis) * self.axis_length for axis in axes],
            dtype=np.float32,
        )
        starts = wp.array(starts_np, dtype=wp.vec3, device=self.device)
        ends = wp.array(ends_np, dtype=wp.vec3, device=self.device)
        self.viewer.log_lines(
            f"{self.name}/axes",
            starts,
            ends,
            self._axis_colors,
            width=self.line_width,
            hidden=False,
        )
        if hasattr(self.viewer, "log_shapes"):
            point_xform = wp.array(
                [wp.transform(wp.vec3(*map(float, pos)), wp.quat_identity())],
                dtype=wp.transform,
                device=self.device,
            )
            self.viewer.log_shapes(
                f"{self.name}/point",
                newton.GeoType.SPHERE,
                self.point_radius,
                point_xform,
                colors=self._point_color,
                materials=self._point_material,
                hidden=False,
            )
        else:
            point_np = pos.reshape(1, 3).astype(np.float32)
            point = wp.array(point_np, dtype=wp.vec3, device=self.device)
            self.viewer.log_points(
                f"{self.name}/point",
                point,
                self.point_radius,
                colors=self._point_color,
                hidden=False,
            )


class TeleopCameraRig:
    """Reusable camera modes for teleoperation examples."""

    MODES = ("rear", "wrist", "free")

    def __init__(
        self,
        viewer,
        *,
        base_body_index: int,
        wrist_body_index: int,
        tool_offset: Sequence[float] = (0.0, 0.0, 0.0),
        initial_mode: str = "rear",
        rear_position_offset: Sequence[float] = (-0.75, 0.0, 0.90),
        rear_target_offset: Sequence[float] = (0.35, 0.0, 0.25),
        wrist_position_offset: Sequence[float] = (-0.0735, 0.0078, 0.0384),
        wrist_quat_wxyz: Sequence[float] = (0.1228, 0.6964, -0.6964, -0.1228),
        wrist_fov: float = 43.23,
        switch_keys: Sequence[str] = ("1", "2", "3"),
        cycle_gamepad_button: str = "x",
        print_help: bool = True,
    ) -> None:
        if initial_mode not in self.MODES:
            raise ValueError(f"initial_mode must be one of {self.MODES}, got {initial_mode!r}")
        self.viewer = viewer
        self.base_body_index = int(base_body_index)
        self.wrist_body_index = int(wrist_body_index)
        self.tool_offset = _as_np3(tool_offset)
        self.mode = initial_mode
        self.rear_position_offset = _as_np3(rear_position_offset)
        self.rear_target_offset = _as_np3(rear_target_offset)
        self.wrist_position_offset = _as_np3(wrist_position_offset)
        self.wrist_quat_xyzw = quat_wxyz_to_xyzw(np.asarray(wrist_quat_wxyz, dtype=np.float64))
        self.wrist_fov = float(wrist_fov)
        self.switch_keys = tuple(switch_keys)
        self.cycle_gamepad_button = cycle_gamepad_button
        self._key_buttons = {key: Button() for key in self.switch_keys}
        self._last_gamepad_events_id: int | None = None
        if print_help:
            print("[NewtonTeleop] Camera: 1 rear, 2 wrist, 3 free; gamepad X cycles camera", flush=True)

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown camera mode {mode!r}")
        if mode != self.mode:
            self.mode = mode
            print(f"[NewtonTeleop] Camera mode: {self.mode}", flush=True)

    def cycle(self) -> None:
        index = (self.MODES.index(self.mode) + 1) % len(self.MODES)
        self.set_mode(self.MODES[index])

    def update(self, state: newton.State, *, gamepad_controller=None) -> None:
        for key, mode in zip(self.switch_keys, self.MODES, strict=False):
            if self._key_buttons[key].update(self._key_down(key)):
                self.set_mode(mode)

        events = getattr(gamepad_controller, "last_button_events", {})
        events_id = id(events)
        if (
            self.cycle_gamepad_button
            and events_id != self._last_gamepad_events_id
            and events.get(self.cycle_gamepad_button, False)
        ):
            self.cycle()
        self._last_gamepad_events_id = events_id

        if self.mode == "free":
            return

        body_q = np.asarray(state.body_q.numpy(), dtype=np.float64)
        if self.mode == "rear":
            base_pose = body_q[self.base_body_index]
            base_pos = base_pose[:3]
            base_q = base_pose[3:7]
            camera_pos = base_pos + quat_rotate_xyzw(base_q, self.rear_position_offset)
            target = base_pos + quat_rotate_xyzw(base_q, self.rear_target_offset)
            if 0 <= self.wrist_body_index < len(body_q):
                tool_pos = transform_point_pose_xyzw(body_q[self.wrist_body_index], self.tool_offset)
                target = 0.65 * tool_pos + 0.35 * target
            set_viewer_camera_look_at(self.viewer, camera_pos, target)
            return

        wrist_pose = body_q[self.wrist_body_index]
        wrist_pos = transform_point_pose_xyzw(wrist_pose, self.wrist_position_offset)
        wrist_q = quat_mul_xyzw(wrist_pose[3:7], self.wrist_quat_xyzw)
        forward = quat_rotate_xyzw(wrist_q, np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
        up = quat_rotate_xyzw(wrist_q, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        set_viewer_camera_look_at(
            self.viewer,
            wrist_pos,
            wrist_pos + forward,
            up=up,
            fov=self.wrist_fov,
        )

    def _key_down(self, key: str) -> bool:
        try:
            return bool(self.viewer.is_key_down(key))
        except Exception:
            return False


class TeleopWristCameraPreview:
    """Render a wrist-mounted sensor camera into the viewer image panel."""

    def __init__(
        self,
        viewer,
        model: newton.Model,
        *,
        body_index: int,
        local_offset: Sequence[float] = (0.0, 0.0, 0.0),
        local_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        fov_y_deg: float = 43.23,
        width: int = 256,
        height: int = 256,
        name: str = "wrist/color",
        load_textures: bool = True,
        near_clip_m: float = 0.02,
    ) -> None:
        self.viewer = viewer
        self.model = model
        self.body_index = int(body_index)
        self.local_offset = _as_np3(local_offset)
        self.local_quat_xyzw = quat_wxyz_to_xyzw(np.asarray(local_quat_wxyz, dtype=np.float64))
        self.fov_y_deg = float(fov_y_deg)
        self.width = int(width)
        self.height = int(height)
        self.name = name
        self.near_clip_m = max(0.0, float(near_clip_m))
        self.sensor = SensorTiledCamera(model=self.model, load_textures=load_textures)
        self.camera_rays = self.sensor.utils.compute_pinhole_camera_rays(
            self.width,
            self.height,
            math.radians(self.fov_y_deg),
        )
        if self.near_clip_m > 0.0:
            wp.launch(
                offset_camera_ray_origins_kernel,
                dim=(1, self.height, self.width),
                inputs=[self.camera_rays, self.near_clip_m],
                device=self.model.device,
            )
        self.color_image = self.sensor.utils.create_color_image_output(self.width, self.height, camera_count=1)

    def camera_pose_from_state(self, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
        body_q = np.asarray(state.body_q.numpy(), dtype=np.float64)
        wrist_pose = body_q[self.body_index]
        pos = transform_point_pose_xyzw(wrist_pose, self.local_offset)
        quat_xyzw = quat_mul_xyzw(wrist_pose[3:7], self.local_quat_xyzw)
        return pos, quat_xyzw

    def camera_transforms_from_state(self, state: newton.State) -> wp.array:
        pos, quat_xyzw = self.camera_pose_from_state(state)
        camera_tf = wp.transformf(
            wp.vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
            wp.quatf(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3])),
        )
        world_count = int(getattr(self.model, "world_count", 1))
        return wp.array([[camera_tf] * world_count], dtype=wp.transformf, device=self.model.device)

    def render(self, state: newton.State) -> None:
        if hasattr(self.model, "bvh_refit_shapes"):
            self.model.bvh_refit_shapes(state)
        if hasattr(self.model, "bvh_refit_particles"):
            self.model.bvh_refit_particles(state)
        self.sensor.update(
            state,
            self.camera_transforms_from_state(state),
            self.camera_rays,
            color_image=self.color_image,
            clear_data=SensorTiledCamera.GRAY_CLEAR_DATA,
        )
        color_rgba = self.sensor.utils.to_rgba_from_color(self.color_image)
        self.viewer.log_image(self.name, color_rgba)
