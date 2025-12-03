########################################################################################################################
#   Company:        Zhejiang Linctex Digital Technology Ltd.(Style3D)                                                  #
#   Copyright:      All rights reserved by Linctex                                                                     #
#   Description:    Style3D examples                                                                                   #
#   Author:         Wenchao Huang (physhuangwenchao@gmail.com)                                                         #
#   Date:           2025/11/28                                                                                         #
########################################################################################################################

import os
import sys

# sys.path.append("D:/Desktop/SimulatorSDK/build/lib/Debug")
sys.path.append("D:/Desktop/SimulatorSDK/build/lib/RelWithDebInfo")

from datetime import datetime

import numpy as np
import polyscope as ps
import style3dsim as sim
import warp as wp
from pxr import Usd, UsdGeom

import newton.examples
from style3d import Viewer


def log_callback(file_name: str, func_name: str, line: int, level: sim.LogLevel, message: str):
    formatted_time = datetime.now().strftime("%H:%M:%S")
    if level == sim.LogLevel.INFO:
        print(f"[{formatted_time}][info]: {message}")
    elif level == sim.LogLevel.ERROR:
        print(f"[{formatted_time}][error]: {message}")
    elif level == sim.LogLevel.WARNING:
        print(f"[{formatted_time}][warning]: {message}")
    elif level == sim.LogLevel.DEBUG:
        print(f"[{formatted_time}][debug]: {message}")


def generate_square_cloth_np(nx: int, ny: int, width: float, height: float):
    x = np.linspace(-width / 2, width / 2, nx)
    y = np.linspace(height, height - width, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pos = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])

    # verts = sim.Arr3f()
    # uvcoords = sim.Arr2f()
    # verts.reserve(nx * ny)
    # uvcoords.reserve(nx * ny)

    # for i in range(nx * ny):
    # 	verts.push_back(sim.Vec3f(pos[i][0], pos[i][1], pos[i][2]))
    # 	uvcoords.push_back(sim.Vec2f(pos[i][0], pos[i][1]))

    # faces = sim.Arr3i()
    # faces.reserve(2 * (nx - 1) * (ny - 1))
    faces_list = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = v0 + ny
            v2 = v0 + 1
            v3 = v2 + ny
            # faces.push_back(sim.Vec3i(v0, v1, v2))
            # faces.push_back(sim.Vec3i(v1, v3, v2))
            faces_list.append([v0, v1, v2])
            faces_list.append([v1, v3, v2])

    faces_np = np.array(faces_list)
    return pos, faces_np, pos[:, 0:2]


def LoadUsd(file_path, root_path):
    usd_stage = Usd.Stage.Open(file_path)
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(root_path))
    indices = usd_geom.GetFaceVertexIndicesAttr().Get()
    normals = usd_geom.GetNormalsAttr().Get()
    points = usd_geom.GetPointsAttr().Get()
    return [indices, points, normals]


if __name__ == "__main__":
    # Set log callback
    sim.set_log_callback(log_callback)

    # Login
    if os.path.exists("key.txt"):
        with open("key.txt", encoding="utf-8") as f:
            lines = f.read().splitlines()
            username = lines[0].strip()
            password = lines[1].strip()
    else:
        username = input("User Name: ")
        password = input("Password: ")
    sim.login(username, password, True, None)

    # Create viewer
    viewer = Viewer()

    # Create world
    world = sim.World()
    world_attrib = sim.WorldAttrib()
    world_attrib.gravity = sim.Vec3f(0, -10, 0)
    world_attrib.enable_gpu = True
    world.set_attrib(world_attrib)

    # Load bunny mesh
    bunny_usd_path = newton.examples.get_asset("bunny.usd")
    [mesh_indices, mesh_points, mesh_normals] = LoadUsd(bunny_usd_path, "/root/bunny")
    mesh_collider_verts = np.array(mesh_points).reshape(-1, 3)
    mesh_collider_tris = np.array(mesh_indices).reshape(-1, 3)

    # create mesh collider
    collider_sim_mesh = sim.Mesh(mesh_collider_tris, mesh_collider_verts)
    mesh_collider = sim.MeshCollider(collider_sim_mesh.get_triangles(), collider_sim_mesh.get_positions())
    collider_attrib = sim.ColliderAttrib()
    collider_attrib.collision_gap = 0.005  # unit m
    collider_attrib.dynamic_friction = 0.3
    collider_attrib.static_friction = 0.6
    mesh_collider.set_attrib(collider_attrib)
    mesh_collider.attach(world)  # attach to sim world
    # Render mesh
    collider_render_mesh = ps.register_surface_mesh("collider", mesh_collider_verts, mesh_collider_tris)

    # Create solidified cloth bunny using bunny mesh
    solidified_cloth_bunny_verts = collider_sim_mesh.get_positions()
    solidified_cloth_bunny_verts[:, 1] += 2.0  # move to higher place
    solidified_cloth_bunny_verts[:, 0] += 1.0
    solidified_cloth_bunny_tris = collider_sim_mesh.get_triangles()
    solidified_cloth_bunny = sim.Cloth(
        tris=solidified_cloth_bunny_tris, verts=solidified_cloth_bunny_verts, keep_wrinkles=True
    )
    cloth_bunny_attrib = sim.ClothAttrib()
    cloth_bunny_attrib.stretch_stiff = sim.Vec3f(150, 150, 150)
    cloth_bunny_attrib.bend_stiff = sim.Vec3f(1e-5, 1e-5, 1e-5)
    cloth_bunny_attrib.density = 0.2
    cloth_bunny_attrib.thickness = 0.005
    cloth_bunny_attrib.static_friction = 0.03
    cloth_bunny_attrib.dynamic_friction = 0.06
    cloth_bunny_attrib.pressure = 3.0
    solidified_cloth_bunny.set_attrib(cloth_bunny_attrib)
    solidified_cloth_bunny.attach(world)  # attach to sim world
    # solidify all verts
    solidify_stiffs = np.full(solidified_cloth_bunny.get_vert_num(), 0.1, dtype=float)
    solidify_vert_ints = np.arange(0, solidified_cloth_bunny.get_vert_num())
    solidified_cloth_bunny.solidify(world, solidify_stiffs, solidify_vert_ints)
    # Render mesh
    bunny_render_mesh = ps.register_surface_mesh("bunny", solidified_cloth_bunny_verts, solidified_cloth_bunny_tris)
    bunny_render_mesh.set_selection_mode("faces_only")

    # Create square cloth
    [verts, faces, uvcoords] = generate_square_cloth_np(100, 100, 0.5, 1.0)
    # cloth = sim.Cloth(faces, verts, uvcoords)
    cloth = sim.Cloth(faces, verts)
    cloth_attrib = sim.ClothAttrib()
    cloth_attrib.stretch_stiff = sim.Vec3f(120, 100, 80)
    cloth_attrib.bend_stiff = sim.Vec3f(1e-6, 1e-6, 1e-6)
    cloth_attrib.density = 0.2
    cloth_attrib.static_friction = 0.03
    cloth_attrib.dynamic_friction = 0.03
    cloth.set_attrib(cloth_attrib)

    pin_flags = np.empty(2, dtype=np.bool_)
    pin_flags.fill(True)
    pin_vert_indices = np.empty(2, dtype=np.int_)
    pin_vert_indices[0] = 0
    pin_vert_indices[1] = 99
    cloth.set_pin(pin_flags, pin_vert_indices)  # pin two verts
    cloth.attach(world)  # attach to sim world

    cloths_verts = cloth.get_positions()
    cloths_normals = cloth.get_normals()
    cloths_triangles = cloth.get_triangles()

    pin_vert_p = cloths_verts[pin_vert_indices, :]  # store init pin vert positions
    cloth_render_mesh = ps.register_surface_mesh("cloth", verts, faces)
    cloth_render_mesh.set_selection_mode("faces_only")

    # Create rigid body 0
    transform_0 = sim.Transform()
    transform_0.scale = sim.Vec3f(1, 1, 1)
    transform_0.rotation = sim.Quat(0, 0, 0, 1)
    transform_0.translation = sim.Vec3f(-1, 2, 0)

    rigid_body_0 = sim.RigidBody(collider_sim_mesh, transform_0)
    rigid_body_0.set_collision_group(-1)
    rigid_body_0.set_collision_mask(-1)
    rigid_body_0.attach(world)
    rigid_body_0_render_entity = ps.register_surface_mesh("rigid_body_0", mesh_collider_verts, mesh_collider_tris)
    rigid_body_0_render_entity.set_position([-1, 2, 0])

    # Create rigid body 1
    transform_1 = sim.Transform()
    transform_1.scale = sim.Vec3f(1, 1, 1)
    transform_1.rotation = sim.Quat(0, 0, 0, 1)
    transform_1.translation = sim.Vec3f(0, 2, 2)

    rigid_body_1 = sim.RigidBody(collider_sim_mesh, transform_1)
    rigid_body_1.set_collision_group(-1)
    rigid_body_1.set_collision_mask(-1)
    rigid_body_1.attach(world)
    rigid_body_1_render_entity = ps.register_surface_mesh("rigid_body_1", mesh_collider_verts, mesh_collider_tris)
    rigid_body_1_render_entity.set_position([0, 2, 2])

    # Add cloth <-> dragged-points spring
    draggedPoints = sim.DraggedPoints([0.0, 0.0, 0.0])
    draggedPoints.attach(world)

    class DragSpring:
        data = None

    drag_spring = DragSpring()

    def update():
        world.step_sim()
        if world.fetch_sim(-1):
            # update cloth
            cur_cloths_verts = cloth.get_positions()
            cloth_render_mesh.update_vertex_positions(cur_cloths_verts)

            # update bunny cloth
            cur_bunny_verts = solidified_cloth_bunny.get_positions()
            bunny_render_mesh.update_vertex_positions(cur_bunny_verts)

            # update collider
            # cur_collider_verts = np.array(mesh_collider_verts)
            # collider_render_mesh.update_vertex_positions(cur_collider_verts)

            # update rigid body 0
            transform = rigid_body_0.get_transform()
            translation = wp.vec3(transform.translation.x, transform.translation.y, transform.translation.z)
            rotation = wp.quat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
            transform_wp = wp.transform()
            wp.transform_set_translation(transform_wp, translation)
            wp.transform_set_rotation(transform_wp, rotation)
            mat = wp.transform_to_matrix(transform_wp)
            mat4x4 = [mat[0], mat[1], mat[2], mat[3]]
            rigid_body_0_render_entity.set_transform(mat4x4)

            # update rigid body 1
            transform = rigid_body_1.get_transform()
            translation = wp.vec3(transform.translation.x, transform.translation.y, transform.translation.z)
            rotation = wp.quat(transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
            transform_wp = wp.transform()
            wp.transform_set_translation(transform_wp, translation)
            wp.transform_set_rotation(transform_wp, rotation)
            mat = wp.transform_to_matrix(transform_wp)
            mat4x4 = [mat[0], mat[1], mat[2], mat[3]]
            rigid_body_1_render_entity.set_transform(mat4x4)
        viewer.sim_frames += 1
        viewer.sim_time += 0.01

    def on_pick(pick_result: ps.PickResult):
        if pick_result is not None:
            if pick_result.is_hit:
                if pick_result.structure_name == cloth_render_mesh.get_name():
                    index = pick_result.structure_data["index"]
                    bary_coord = pick_result.structure_data["bary_coords"]
                    draggedPoints.set_positions([pick_result.position])
                    drag_spring.data = sim.Springs(cloth, [index], [bary_coord], draggedPoints)
                    drag_spring.data.attach(world)
                elif pick_result.structure_name == bunny_render_mesh.get_name():
                    index = pick_result.structure_data["index"]
                    bary_coord = pick_result.structure_data["bary_coords"]
                    draggedPoints.set_positions([pick_result.position])
                    drag_spring.data = sim.Springs(solidified_cloth_bunny, [index], [bary_coord], draggedPoints)
                    drag_spring.data.attach(world)

    def on_drag(drag_pos: tuple[float, float, float]):
        draggedPoints.set_positions([drag_pos])

    def on_release_drag():
        if drag_spring.data is not None:
            drag_spring.data.detach()
            drag_spring.data = None

    viewer.set_user_update(update)
    viewer.set_on_release_drag(on_release_drag)
    viewer.set_on_drag(on_drag)
    viewer.set_on_pick(on_pick)
    viewer.run()
