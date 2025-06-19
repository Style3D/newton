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

# import ursina.shaders.basic_lighting_shader
# Modified based on ursina.shaders.basic_lighting_shader
cloth_shader = ursina.Shader(
    name="cloth_shader",
    language=ursina.Shader.GLSL,
    vertex="""
#version 140

uniform mat4 p3d_ViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;
out vec3 world_pos;
out vec3 camera_pos;
out vec3 world_normal;

void main()
{
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    world_normal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    camera_pos = vec3(inverse(p3d_ViewMatrix) * vec4(0, 0, 0, 1));
    world_pos = vec3(p3d_ModelMatrix * p3d_Vertex);
    texcoord = p3d_MultiTexCoord0;
}
""",
    fragment="""
#version 140

uniform vec4 p3d_ColorScale;
uniform sampler2D p3d_Texture0;

in vec2 texcoord;
in vec3 world_pos;
in vec3 camera_pos;
in vec3 world_normal;

out vec4 fragColor;

void main()
{
    vec3 N = world_normal;
    vec3 V = normalize(camera_pos - world_pos);
    vec3 L = V;
    vec3 H = normalize(V + L);
    float diff = abs(dot(N, H));

    float gamma = 2.2f;
    float ambient = 0.1f;
    vec3 base_color = vec3(0.37f, 0.5f, 1.0f);

    if (dot(N, V) < 0.0f)
    {
        vec3 color = 1.0f - base_color;
        base_color = vec3(color.y, color.x, color.z);
    }

    fragColor = vec4(ambient + base_color * diff, 1);
    fragColor.rgb = pow(fragColor.rgb, vec3(1.0f / gamma));
}
""",
)


if __name__ == "__main__":
    app = ursina.Ursina()
    ursina.EditorCamera()
    sphere = ursina.Entity(model="sphere", shader=cloth_shader)
    app.run()
