# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import os
import unittest

import numpy as np
import warp as wp
import warp.examples

from newton import ModelBuilder
from newton.sim.graph_coloring import (
    ColoringAlgorithm,
    construct_trimesh_graph_edges,
    convert_to_color_groups,
    validate_graph_coloring,
)
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, assert_np_equal, get_test_devices


def create_lattice_grid(N):
    size = 10
    position = (0, 0)

    X = np.linspace(-0.5 * size + position[0], 0.5 * size + position[0], N)
    Y = np.linspace(-0.5 * size + position[1], 0.5 * size + position[1], N)

    X, Y = np.meshgrid(X, Y)

    Z = []
    for _i in range(N):
        Z.append(np.linspace(0, size, N))

    Z = np.array(Z)

    vs = []
    for i, j in itertools.product(range(N), range(N)):
        vs.append(wp.vec3((X[i, j], Y[i, j], Z[i, j])))

    fs = []
    for i, j in itertools.product(range(0, N - 1), range(0, N - 1)):
        vId = j + i * N

        if (j + i) % 2:
            fs.extend(
                [
                    vId,
                    vId + N + 1,
                    vId + 1,
                ]
            )
            fs.extend(
                [
                    vId,
                    vId + N,
                    vId + N + 1,
                ]
            )
        else:
            fs.extend(
                [
                    vId,
                    vId + N,
                    vId + 1,
                ]
            )
            fs.extend(
                [
                    vId + N,
                    vId + N + 1,
                    vId + 1,
                ]
            )

    return vs, fs


def color_lattice_grid(num_x, num_y):
    colors = []
    for _ in range(4):
        colors.append([])

    for xi in range(num_x + 1):
        for yi in range(num_y + 1):
            node_dx = yi * (num_x + 1) + xi

            a = 1 if xi % 2 else 0
            b = 1 if yi % 2 else 0

            c = b * 2 + a

            colors[c].append(node_dx)

    color_groups = [np.array(group) for group in colors]

    return color_groups


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_coloring_trimesh(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    with wp.ScopedDevice(device):
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        builder = ModelBuilder()

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=[wp.vec3(p) for p in vertices],
            indices=faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        model = builder.finalize()

        particle_colors = wp.empty(shape=(model.particle_count), dtype=int, device="cpu")

        edge_indices_cpu = wp.array(model.edge_indices.numpy()[:, 2:], dtype=int, device="cpu")

        # coloring without bending
        num_colors_greedy = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.GREEDY.value,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, particle_colors],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )

        num_colors_mcs = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu.__ctype__(),
            ColoringAlgorithm.MCS.value,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, particle_colors],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )

        # coloring with bending
        edge_indices_cpu_with_bending = construct_trimesh_graph_edges(model.edge_indices, True)
        num_colors_greedy = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.GREEDY.value,
            particle_colors.__ctype__(),
        )
        wp.context.runtime.core.wp_balance_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_greedy,
            1.1,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu_with_bending, particle_colors],
            dim=edge_indices_cpu_with_bending.shape[0],
            device="cpu",
        )

        num_colors_mcs = wp.context.runtime.core.wp_graph_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            ColoringAlgorithm.MCS.value,
            particle_colors.__ctype__(),
        )
        max_min_ratio = wp.context.runtime.core.wp_balance_coloring(
            model.particle_count,
            edge_indices_cpu_with_bending.__ctype__(),
            num_colors_mcs,
            1.1,
            particle_colors.__ctype__(),
        )
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu_with_bending, particle_colors],
            dim=edge_indices_cpu_with_bending.shape[0],
            device="cpu",
        )

        color_categories_balanced = convert_to_color_groups(num_colors_mcs, particle_colors)

        color_sizes = np.array([c.shape[0] for c in color_categories_balanced], dtype=np.float32)
        test.assertTrue(np.max(color_sizes) / np.min(color_sizes) <= max_min_ratio)

        # test if the color balance can quit from equilibrium
        builder = ModelBuilder()

        vs, fs = create_lattice_grid(100)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=vs,
            indices=fs,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        builder.color(include_bending=True)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_combine_coloring(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    with wp.ScopedDevice(device):
        builder1 = ModelBuilder()
        usd_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        builder1.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            scale=1.0,
            vertices=[wp.vec3(p) for p in vertices],
            indices=faces.flatten(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
        )

        builder1.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
        )
        builder1.color()

        builder2 = ModelBuilder()
        builder2.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            # to include bending in coloring
            edge_ke=100000,
            fix_left=True,
        )
        builder2.color()

        builder3 = ModelBuilder()
        builder3.add_cloth_grid(
            pos=wp.vec3(0.0, 4.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=50,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            fix_left=True,
        )

        builder3.set_coloring(
            color_lattice_grid(50, 100),
        )

        builder1.add_builder(builder2)
        builder1.add_builder(builder3)

        model = builder2.finalize()

        particle_number_colored = np.full((model.particle_count), -1, dtype=int)
        particle_colors = np.full((model.particle_count), -1, dtype=int)
        for color, color_group in enumerate(model.particle_color_groups):
            particle_number_colored[color_group.numpy()] += 1
            particle_colors[color_group.numpy()] = color

        # all particles has been colored exactly once
        assert_np_equal(particle_number_colored, 0)

        edge_indices_cpu = wp.array(model.edge_indices.numpy()[:, 2:], dtype=int, device="cpu")
        wp.launch(
            kernel=validate_graph_coloring,
            inputs=[edge_indices_cpu, wp.array(particle_colors, dtype=int, device="cpu")],
            dim=edge_indices_cpu.shape[0],
            device="cpu",
        )


devices = get_test_devices()


class TestColoring(unittest.TestCase):
    pass


add_function_test(TestColoring, "test_coloring_trimesh", test_coloring_trimesh, devices=devices, check_output=False)
add_function_test(TestColoring, "test_combine_coloring", test_combine_coloring, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
