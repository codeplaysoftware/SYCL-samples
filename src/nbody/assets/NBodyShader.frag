uniform sampler2D star_tex;
in vec3 vColor;
out vec4 oColor;

void main() {
  vec2 uv = vec2(gl_PointCoord.x, gl_PointCoord.y);
  vec4 tex = texture2D(star_tex, uv);
  oColor = vec4(vColor, tex.a);
}
