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

import ursina

# import ursina.shaders.unlit_shader
# Modified based on ursina.shaders.unlit_shader
floor_shader = ursina.Shader(
    name="floor_shader",
    language=ursina.Shader.GLSL,
    vertex="""
#version 140

uniform vec2 texture_scale;
uniform vec2 texture_offset;
uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Color;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoords;
out vec4 vertex_color;

void main()
{
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
}
""",
    fragment="""
#version 140

uniform vec4 p3d_ColorScale;
uniform sampler2D p3d_Texture0;

in vec2 texcoords;
in vec4 vertex_color;

out vec4 fragColor;

void main()
{
    vec4 color = texture(p3d_Texture0, texcoords) * p3d_ColorScale * vertex_color;

    if (color.r + color.g + color.b > 2.5f)
        discard;
    else
        fragColor = vec4(1, 1, 1, 0.3f);
}
""",
    default_input={
        "texture_scale": ursina.Vec2(1, 1),
        "texture_offset": ursina.Vec2(0, 0),
    },
)


def create_floor_entity(**kwargs):
    floor = ursina.Entity(model="plane", texture="white_cube", shader=floor_shader, **kwargs)
    floor.texture.filtering = "bilinear"
    floor.double_sided_setter(True)
    return floor


if __name__ == "__main__":
    from ursina import EditorCamera

    app = ursina.Ursina()
    EditorCamera()
    create_floor_entity()
    app.run()
