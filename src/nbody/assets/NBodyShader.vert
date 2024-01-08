/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Description:
 *    Vertex shader for NBody demo
 *
 **************************************************************************/

uniform mat4 ciModelView;
uniform mat4 ciModelViewProjection;
in vec3 ciPosition;
in vec3 ciVelocity;
out vec3 vColor;

vec3 speedColors[9] =
    vec3[](vec3(0.0, 0.0, 0.2), vec3(0.0, 0.0, 0.4), vec3(0.0, 0.0, 0.8),
           vec3(0.0, 0.4, 0.4), vec3(0.0, 0.8, 0.8), vec3(0.0, 0.8, 0.4),
           vec3(0.4, 0.8, 0.0), vec3(0.8, 0.6, 0.0), vec3(0.8, 0.2, 0.0));

void main() {
  vec4 viewSpacePos = ciModelView * vec4(ciPosition, 1.0);
  // The further away a point is, the smaller its sprite
  float scale = log2(length(viewSpacePos));
  gl_PointSize = 10.0 / clamp(scale, 0.1, 1);
  gl_Position = ciModelViewProjection * vec4(ciPosition / 10.0, 1.0);
  float len = length(ciVelocity);
  int speed = int(ceil(len));
  vColor =
      mix(speedColors[max(0, speed)], speedColors[min(8, speed)], speed - len);
}
