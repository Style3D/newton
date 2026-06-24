# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


import newton.utils
from newton.examples.cloth.example_cloth_style3d import Example
from style3d.viewer import ViewerNewton

if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="null")  # disable viewer in newton.examples.init, we will create viewer by ourself.

    # Parse arguments and initialize viewer
    _, args = newton.examples.init(parser)
    viewer = ViewerNewton(newton.Axis.Z)  # replace the default viewer with our own viewer.

    # Create example and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
