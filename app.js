/* =====================================================================
 * AR Beauty Filter
 * MediaPipe FaceMesh (face landmarks) + WebGL (real-time beauty shader)
 *
 * Features:
 *  - Skin smoothing / blemish removal (edge-preserving bilateral blur,
 *    limited to the face region so background & eyes stay sharp)
 *  - Skin brightening / whitening
 *  - Brightness boost
 *  - Face slimming (geometric warp on cheeks)
 *  - Eye enlarging (geometric bulge around each eye)
 *  - Lip tint
 * ===================================================================== */

(() => {
  "use strict";

  // ---- DOM ----
  const video = document.getElementById("video");
  const canvas = document.getElementById("output");
  const statusEl = document.getElementById("statusBadge") || { textContent: "" };
  const fpsEl = { textContent: "" }; // fps hidden in booth UI

  // All beauty/simulation strengths the engine reads (0..100).
  const params = { smooth: 0, whiten: 0, bright: 0, slim: 0, eye: 0, lips: 0, acne: 0, wrinkle: 0, dull: 0, glow: 0 };

  // Each skin mode is a preset of the params above.
  const MODES = {
    // ---- ผิวสุขภาพดี (healthy) — strong, "wow" beauty ----
    glow:    { smooth: 86, whiten: 55, bright: 30, slim: 24, eye: 22, lips: 24, glow: 55 },
    aura:    { smooth: 78, whiten: 46, bright: 40, slim: 20, eye: 18, lips: 28, glow: 78 },
    radiant: { smooth: 90, whiten: 62, bright: 32, slim: 28, eye: 26, lips: 22, glow: 66 },
    dewy:    { smooth: 92, whiten: 36, bright: 24, slim: 18, eye: 18, lips: 32, glow: 48 },
    // ---- ผิวมีปัญหา (problem) ----
    tone:     { acne: 32, dull: 22 },
    dull:     { dull: 62 },
    pigment:  { acne: 48 },
    vascular: { acne: 34 },
    tired:    { dull: 46, wrinkle: 26 },
    damaged:  { dull: 52, wrinkle: 46 },
    wrinkle:  { wrinkle: 72 },
  };

  let compare = false;
  let facingMode = "user";
  let faceMesh = null;
  let latestLandmarks = null;
  let fxVal = 1;         // effect intensity: 0 = ก่อน (original), 1 = หลัง (full)

  function applyMode(name) {
    const p = MODES[name] || {};
    for (const key in params) params[key] = p[key] || 0;
  }

  // ---------------------------------------------------------------
  // WebGL setup
  // ---------------------------------------------------------------
  const gl = canvas.getContext("webgl", { premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) {
    statusEl.textContent = "เบราว์เซอร์ไม่รองรับ WebGL";
    return;
  }

  const VERT = `
    attribute vec2 a_pos;
    varying vec2 v_uv;
    void main() {
      // Map clip-space quad to image-space uv (top-left origin).
      v_uv = vec2(a_pos.x * 0.5 + 0.5, 1.0 - (a_pos.y * 0.5 + 0.5));
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }
  `;

  const FRAG = `
    precision highp float;
    varying vec2 v_uv;

    uniform sampler2D u_tex;
    uniform vec2 u_texSize;

    uniform float u_smooth;   // 0..1 skin smoothing strength
    uniform float u_whiten;   // 0..1
    uniform float u_bright;   // 0..1
    uniform float u_slim;     // 0..1
    uniform float u_eye;      // 0..1
    uniform float u_lips;     // 0..1
    uniform float u_dull;     // 0..1 make skin dull/tired (desaturate+darken)
    uniform float u_glow;     // 0..1 radiant glow / aura bloom on skin
    uniform float u_split;    // 0..1 before/after wipe: x < split shows original

    uniform float u_hasFace;
    uniform vec2  u_faceCenter; // uv
    uniform vec2  u_faceRadius; // uv (rx, ry)

    uniform vec2 u_eyeL;   // uv
    uniform vec2 u_eyeR;   // uv
    uniform float u_eyeRad; // uv

    uniform vec2 u_cheekL; // uv
    uniform vec2 u_cheekR; // uv
    uniform float u_slimRad; // uv

    uniform vec2 u_mouth;   // uv center of mouth
    uniform float u_mouthRad;

    // --- skin problem simulation ---
    uniform float u_acne;     // 0..1 simulate pimples
    uniform float u_wrinkle;  // 0..1 simulate wrinkles
    // face-local coordinate frame (pixel space) used to anchor the
    // simulated features so they stick to the face as it moves.
    uniform vec2  u_faceO;     // origin (uv)
    uniform vec2  u_faceEx;    // unit x axis (across face, pixel space)
    uniform vec2  u_faceEy;    // unit y axis (forehead->chin, pixel space)
    uniform float u_faceHalfW; // px
    uniform float u_faceHalfH; // px

    // --- geometric warp: returns source uv to sample ---
    vec2 warp(vec2 uv) {
      vec2 src = uv;
      vec2 px = src * u_texSize;

      // Face slim: pinch cheeks toward face center line.
      if (u_slim > 0.001 && u_hasFace > 0.5) {
        float rad = u_slimRad * u_texSize.x;
        // left cheek -> push source outward (left) so face compresses inward
        vec2 cl = u_cheekL * u_texSize;
        float dl = distance(px, cl);
        if (dl < rad) {
          float infl = (1.0 - dl / rad);
          infl = infl * infl;
          px.x -= u_slim * infl * rad * 0.26;
        }
        vec2 cr = u_cheekR * u_texSize;
        float dr = distance(px, cr);
        if (dr < rad) {
          float infl = (1.0 - dr / rad);
          infl = infl * infl;
          px.x += u_slim * infl * rad * 0.26;
        }
      }

      // Eye enlarge: bulge (sample closer to center -> magnify).
      if (u_eye > 0.001 && u_hasFace > 0.5) {
        float rad = u_eyeRad * u_texSize.x;
        vec2 el = u_eyeL * u_texSize;
        float de = distance(px, el);
        if (de < rad) {
          float t = de / rad;
          float scale = 1.0 - u_eye * 0.5 * (1.0 - t);
          px = el + (px - el) * scale;
        }
        vec2 er = u_eyeR * u_texSize;
        float de2 = distance(px, er);
        if (de2 < rad) {
          float t = de2 / rad;
          float scale = 1.0 - u_eye * 0.5 * (1.0 - t);
          px = er + (px - er) * scale;
        }
      }

      return px / u_texSize;
    }

    // Elliptical face mask (1 inside face, 0 outside) with a WIDE soft edge
    // so the beauty region fades out gradually (no visible bright oval/ring).
    float faceMask(vec2 uv) {
      if (u_hasFace < 0.5) return 0.0;
      vec2 d = (uv - u_faceCenter) / u_faceRadius;
      float r = length(d);
      return 1.0 - smoothstep(0.45, 1.05, r);
    }

    float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

    // hash helpers for procedural placement
    float hash21(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    // map a screen uv into face-local coordinates (~[-1,1] across the face)
    vec2 faceLocal(vec2 uv) {
      vec2 relPx = (uv - u_faceO) * u_texSize;
      float lx = dot(relPx, u_faceEx) / max(u_faceHalfW, 1.0);
      float ly = dot(relPx, u_faceEy) / max(u_faceHalfH, 1.0);
      return vec2(lx, ly);
    }

    // skin-only weight: inside face, excluding eyes & mouth
    float skinFactor(vec2 uv) {
      float m = faceMask(uv);
      float de = min(distance(uv, u_eyeL), distance(uv, u_eyeR));
      m *= smoothstep(u_eyeRad * 0.45, u_eyeRad * 0.85, de);
      float dm = distance(uv, u_mouth);
      m *= smoothstep(u_mouthRad * 0.6, u_mouthRad * 1.0, dm);
      return m;
    }

    // simulated pimples: reddish soft spots scattered over the skin
    float acneField(vec2 lp) {
      float density = mix(5.0, 12.0, u_acne);
      vec2 g = lp * density;
      vec2 cell = floor(g);
      float spot = 0.0;
      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          vec2 c = cell + vec2(float(dx), float(dy));
          float h = hash21(c);
          if (h < (0.18 + 0.55 * u_acne)) {
            vec2 center = c + vec2(hash21(c + 1.7), hash21(c + 4.3));
            float d = length(g - center);
            float r = mix(0.12, 0.30, hash21(c + 7.1));
            spot = max(spot, smoothstep(r, 0.0, d) * (0.6 + 0.4 * hash21(c + 9.9)));
          }
        }
      }
      return spot;
    }

    // thin bright ridge generator for wrinkle lines
    float ridge(float t) { return pow(max(sin(t), 0.0), 16.0); }

    // simulated wrinkles: forehead creases, nasolabial folds, fine lines
    float wrinkleField(vec2 lp) {
      float w = 0.0;
      // forehead horizontal creases (upper-center)
      float fz = step(-0.85, lp.y) * step(lp.y, -0.32) * smoothstep(0.62, 0.22, abs(lp.x));
      w += fz * ridge(lp.y * 22.0 + sin(lp.x * 7.0) * 0.7) * 0.9;
      // nasolabial folds (left & right, mid face)
      float nzL = step(0.02, lp.y) * step(lp.y, 0.58) * smoothstep(0.14, 0.34, -lp.x) * smoothstep(0.64, 0.40, -lp.x);
      w += nzL * ridge((lp.x + lp.y) * 15.0) * 0.7;
      float nzR = step(0.02, lp.y) * step(lp.y, 0.58) * smoothstep(0.14, 0.34, lp.x) * smoothstep(0.64, 0.40, lp.x);
      w += nzR * ridge((lp.x - lp.y) * 15.0) * 0.7;
      // overall fine aging texture
      float fine = ridge((lp.x * 9.0 + lp.y * 31.0) + hash21(floor(lp * 38.0)) * 6.28) * 0.22;
      w += fine;
      return clamp(w, 0.0, 1.0);
    }

    void main() {
      // before/after wipe: left of the split shows the untouched camera
      if (u_split > 0.001 && v_uv.x < u_split) {
        gl_FragColor = vec4(texture2D(u_tex, v_uv).rgb, 1.0);
        return;
      }
      vec2 uv = warp(v_uv);
      vec3 orig = texture2D(u_tex, uv).rgb;
      vec3 color = orig;

      float mask = faceMask(v_uv);

      // ---- Skin smoothing (edge-preserving bilateral) ----
      if (u_smooth > 0.001 && mask > 0.001) {
        vec2 texel = 1.0 / u_texSize;
        float radius = mix(1.5, 6.5, u_smooth);
        // sigma for color similarity -> preserves edges (eyes, lips, brows)
        float sigmaC = 0.09 + 0.06 * u_smooth;
        float invC = 1.0 / (2.0 * sigmaC * sigmaC);

        vec3 sum = orig;
        float wsum = 1.0;
        // 16-tap ring sampling (two rings)
        for (int i = 0; i < 16; i++) {
          float a = float(i) * 0.3926990817; // 2*pi/16
          float ring = (i < 8) ? radius : radius * 0.55;
          vec2 off = vec2(cos(a), sin(a)) * ring * texel;
          vec3 s = texture2D(u_tex, uv + off).rgb;
          float cd = luma(s) - luma(orig);
          float w = exp(-cd * cd * invC);
          sum += s * w;
          wsum += w;
        }
        vec3 smoothed = sum / wsum;
        // blend by mask & strength
        float amt = u_smooth * mask;
        color = mix(color, smoothed, amt);
      }

      // ---- Brightening ----
      if (u_bright > 0.001) {
        color += u_bright * 0.11 * mask;
      }

      // ---- Whitening (lift toward warm white) ----
      if (u_whiten > 0.001) {
        vec3 target = mix(color, vec3(1.0, 0.98, 0.97), 0.42);
        color = mix(color, target, u_whiten * 0.34 * mask);
      }

      // ---- Radiant glow / aura (soft bloom on skin highlights) ----
      if (u_glow > 0.001 && mask > 0.001) {
        float hi = smoothstep(0.5, 0.92, luma(color));
        color += hi * u_glow * 0.40 * mask;              // glossy highlight bloom
        color = mix(color, color * 1.06 + 0.015, u_glow * 0.25 * mask); // overall radiance
      }

      // ---- Lip tint ----
      if (u_lips > 0.001 && u_hasFace > 0.5) {
        float dm = distance(v_uv, u_mouth) / max(u_mouthRad, 0.0001);
        float lipMask = 1.0 - smoothstep(0.5, 1.0, dm);
        // detect reddish/lip pixels
        float redness = clamp((color.r - max(color.g, color.b)) * 3.0, 0.0, 1.0);
        float lm = lipMask * redness;
        vec3 lipColor = vec3(0.85, 0.18, 0.32);
        color = mix(color, mix(color, lipColor, 0.6), u_lips * lm);
      }

      // ---- Dull / tired skin (desaturate + darken) ----
      if (u_dull > 0.001 && mask > 0.001) {
        float g = luma(color);
        vec3 dullc = mix(color, vec3(g), 0.6) * 0.86; // grayer & darker
        color = mix(color, dullc, u_dull * mask);
      }

      gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
    }
  `;

  function compileShader(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(sh));
    }
    return sh;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, VERT));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  // Fullscreen quad
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, 1,
  ]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, "a_pos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  // Texture for the video frame
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  // Uniform locations
  const U = {};
  [
    "u_texSize", "u_smooth", "u_whiten", "u_bright", "u_slim", "u_eye", "u_lips",
    "u_dull", "u_glow", "u_split",
    "u_hasFace", "u_faceCenter", "u_faceRadius", "u_eyeL", "u_eyeR", "u_eyeRad",
    "u_cheekL", "u_cheekR", "u_slimRad", "u_mouth", "u_mouthRad",
  ].forEach((n) => { U[n] = gl.getUniformLocation(program, n); });
  gl.uniform1i(gl.getUniformLocation(program, "u_tex"), 0);

  // ===============================================================
  // Mesh overlay program — draws acne/wrinkles on the 468-point face
  // mesh (canonical UVs) so they stick to the face & deform with it,
  // like TikTok/IG filters.
  // ===============================================================
  const MESH_VERT = `
    attribute vec2 a_pos;  // clip space (from landmarks)
    attribute vec2 a_uv;   // canonical face uv (static)
    varying vec2 v_uv;
    varying float v_sx;    // screen x (0..1) for the before/after wipe
    void main() {
      v_uv = a_uv;
      v_sx = a_pos.x * 0.5 + 0.5;
      gl_Position = vec4(a_pos, 0.0, 1.0);
    }
  `;
  const MESH_FRAG = `
    precision highp float;
    varying vec2 v_uv;
    varying float v_sx;
    uniform float u_acne;
    uniform float u_wrinkle;
    uniform float u_split;

    float hash21(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }
    float ridge(float t) { return pow(max(sin(t), 0.0), 16.0); }

    // interior skin weight in canonical uv (exclude eyes, mouth, boundary)
    float featureMask(vec2 uv) {
      float m = 1.0;
      m *= smoothstep(0.03, 0.13, uv.x) * smoothstep(0.03, 0.13, 1.0 - uv.x);
      m *= smoothstep(0.05, 0.13, uv.y) * smoothstep(0.03, 0.11, 1.0 - uv.y);
      m *= smoothstep(0.055, 0.10, distance(uv, vec2(0.30, 0.378)));  // left eye
      m *= smoothstep(0.055, 0.10, distance(uv, vec2(0.70, 0.378)));  // right eye
      m *= smoothstep(0.075, 0.12, distance(uv, vec2(0.50, 0.70)));   // mouth
      return clamp(m, 0.0, 1.0);
    }

    float acneField(vec2 uv) {
      float density = mix(7.0, 15.0, u_acne);
      vec2 g = uv * density;
      vec2 cell = floor(g);
      float spot = 0.0;
      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          vec2 c = cell + vec2(float(dx), float(dy));
          float h = hash21(c);
          if (h < (0.16 + 0.5 * u_acne)) {
            vec2 ctr = c + vec2(hash21(c + 1.7), hash21(c + 4.3));
            float d = length(g - ctr);
            float r = mix(0.10, 0.26, hash21(c + 7.1));
            spot = max(spot, smoothstep(r, 0.0, d) * (0.6 + 0.4 * hash21(c + 9.9)));
          }
        }
      }
      return spot;
    }

    float wrinkleField(vec2 uv) {
      float w = 0.0;
      // forehead horizontal creases (v ~0.08..0.30)
      float fz = smoothstep(0.07, 0.13, uv.y) * smoothstep(0.32, 0.24, uv.y)
               * smoothstep(0.18, 0.32, uv.x) * smoothstep(0.82, 0.68, uv.x);
      w += fz * ridge(uv.y * 110.0 + sin(uv.x * 18.0) * 0.7) * 0.9;
      // nasolabial folds (nose wing -> mouth corner)
      float nl = smoothstep(0.50, 0.55, uv.y) * smoothstep(0.78, 0.70, uv.y)
               * smoothstep(0.30, 0.37, uv.x) * smoothstep(0.48, 0.40, uv.x);
      w += nl * ridge((uv.x * 2.0 + uv.y) * 38.0) * 0.7;
      float nr = smoothstep(0.50, 0.55, uv.y) * smoothstep(0.78, 0.70, uv.y)
               * smoothstep(0.63, 0.56, uv.x) * smoothstep(0.70, 0.63, uv.x);
      w += nr * ridge((-uv.x * 2.0 + uv.y) * 38.0) * 0.7;
      // overall fine aging texture
      w += ridge((uv.x * 28.0 + uv.y * 85.0) + hash21(floor(uv * 110.0)) * 6.28) * 0.16;
      return clamp(w, 0.0, 1.0);
    }

    void main() {
      // before/after wipe: no simulation on the original (left) side
      if (u_split > 0.001 && v_sx < u_split) { gl_FragColor = vec4(1.0); return; }
      float m = featureMask(v_uv);
      if (m <= 0.001) { gl_FragColor = vec4(1.0); return; }  // multiply identity

      float spot = u_acne > 0.001 ? acneField(v_uv) * u_acne * m : 0.0;
      float wr = u_wrinkle > 0.001 ? wrinkleField(v_uv) * u_wrinkle * m : 0.0;

      // MULTIPLY tints (1.0 = no change) so the effect rides the skin's own
      // light & shadow instead of sitting on top as flat colour.
      vec3 acneTint = mix(vec3(1.0), vec3(0.78, 0.40, 0.36), clamp(spot * 0.95, 0.0, 1.0));
      vec3 wrTint   = mix(vec3(1.0), vec3(0.72, 0.64, 0.60), clamp(wr * 0.85, 0.0, 1.0));
      gl_FragColor = vec4(acneTint * wrTint, 1.0);
    }
  `;

  const meshProgram = gl.createProgram();
  gl.attachShader(meshProgram, compileShader(gl.VERTEX_SHADER, MESH_VERT));
  gl.attachShader(meshProgram, compileShader(gl.FRAGMENT_SHADER, MESH_FRAG));
  gl.linkProgram(meshProgram);
  if (!gl.getProgramParameter(meshProgram, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(meshProgram));
  }
  const meshLoc = {
    aPos: gl.getAttribLocation(meshProgram, "a_pos"),
    aUv: gl.getAttribLocation(meshProgram, "a_uv"),
    uAcne: gl.getUniformLocation(meshProgram, "u_acne"),
    uWrinkle: gl.getUniformLocation(meshProgram, "u_wrinkle"),
    uSplit: gl.getUniformLocation(meshProgram, "u_split"),
  };
  const haveMeshData = (typeof window.FACE_UVS !== "undefined" && typeof window.FACE_TRIS !== "undefined");
  const uvBuf = gl.createBuffer();
  const posBuf = gl.createBuffer();
  const idxBuf = gl.createBuffer();
  let meshPosArr = null;     // Float32Array clip positions (468*2)
  let smLm = null;           // EMA-smoothed normalized landmarks
  if (haveMeshData) {
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, window.FACE_UVS, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, window.FACE_TRIS, gl.STATIC_DRAW);
    meshPosArr = new Float32Array((window.FACE_UVS.length / 2) * 2);
  }

  // draw the simulated acne/wrinkle mesh over the current framebuffer
  function drawMesh(lm, acne, wrinkle) {
    if (!haveMeshData || !lm || (acne <= 0.001 && wrinkle <= 0.001)) return;
    const n = meshPosArr.length / 2;
    if (lm.length < n) return;

    // Adaptive smoothing (One-Euro style): filter hard when the face is
    // still (kills jitter) but follow almost instantly when it moves
    // (no lag -> the overlay no longer slips off during movement).
    if (!smLm) { smLm = new Float32Array(n * 2); for (let i = 0; i < n; i++) { smLm[i*2]=lm[i].x; smLm[i*2+1]=lm[i].y; } }
    for (let i = 0; i < n; i++) {
      const px = smLm[i*2], py = smLm[i*2+1];
      const dx = lm[i].x - px, dy = lm[i].y - py;
      const speed = Math.hypot(dx, dy);
      const a = Math.min(1, 0.30 + speed * 60); // still -> 0.30, moving -> ~1
      const sx = px + dx * a;
      const sy = py + dy * a;
      smLm[i*2] = sx; smLm[i*2+1] = sy;
      meshPosArr[i*2]   = sx * 2.0 - 1.0;
      meshPosArr[i*2+1] = 1.0 - sy * 2.0;
    }

    gl.useProgram(meshProgram);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ZERO, gl.SRC_COLOR); // multiply: result = skin * tint

    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, meshPosArr, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(meshLoc.aPos);
    gl.vertexAttribPointer(meshLoc.aPos, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
    gl.enableVertexAttribArray(meshLoc.aUv);
    gl.vertexAttribPointer(meshLoc.aUv, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1f(meshLoc.uAcne, acne);
    gl.uniform1f(meshLoc.uWrinkle, wrinkle);
    gl.uniform1f(meshLoc.uSplit, 0.0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
    gl.drawElements(gl.TRIANGLES, window.FACE_TRIS.length, gl.UNSIGNED_SHORT, 0);

    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(meshLoc.aUv);
  }

  // ---------------------------------------------------------------
  // Landmark helpers (MediaPipe FaceMesh indices)
  // ---------------------------------------------------------------
  // left eye corners: 33 (outer), 133 (inner); right eye: 362, 263
  // left cheek: 234 ; right cheek: 454 ; chin: 152 ; mouth: 13/14
  // EMA-smoothed face values to kill landmark jitter (the main cause of
  // the "floaty sticker" look). Higher alpha = more responsive, less smooth.
  let sm = null;
  const ALPHA = 0.45;
  function ema(prev, next) { return prev == null ? next : prev + (next - prev) * ALPHA; }

  function computeFaceUniforms(lm) {
    if (!lm) {
      gl.uniform1f(U.u_hasFace, 0.0);
      sm = null;
      return;
    }
    // bounding box over all landmarks
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of lm) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const rx = (maxX - minX) / 2 * 1.15;
    const ry = (maxY - minY) / 2 * 1.15;

    const mid = (a, b) => ({ x: (lm[a].x + lm[b].x) / 2, y: (lm[a].y + lm[b].y) / 2 });
    const eyeL = mid(33, 133);
    const eyeR = mid(362, 263);
    const eyeW = Math.hypot(lm[133].x - lm[33].x, lm[133].y - lm[33].y);
    const eyeRad = Math.max(eyeW * 1.6, 0.04);

    const cheekL = { x: lm[234].x, y: lm[234].y };
    const cheekR = { x: lm[454].x, y: lm[454].y };
    const slimRad = rx * 0.9;

    const mouth = mid(13, 14);
    const mouthW = Math.hypot(lm[291].x - lm[61].x, lm[291].y - lm[61].y);
    const mouthRad = Math.max(mouthW * 0.75, 0.03);

    // face-local frame (in pixel space) to anchor simulated acne/wrinkles
    const W = canvas.width || 1, H = canvas.height || 1;
    const forehead = lm[10], chin = lm[152];
    let exX = (cheekR.x - cheekL.x) * W, exY = (cheekR.y - cheekL.y) * H;
    let exLen = Math.hypot(exX, exY) || 1;
    exX /= exLen; exY /= exLen;
    let eyX = (chin.x - forehead.x) * W, eyY = (chin.y - forehead.y) * H;
    let eyLen = Math.hypot(eyX, eyY) || 1;
    eyX /= eyLen; eyY /= eyLen;
    const halfW = (exLen / 2) || 1;
    const halfH = (eyLen / 2) || 1;

    const raw = {
      cx, cy, rx, ry,
      eLx: eyeL.x, eLy: eyeL.y, eRx: eyeR.x, eRy: eyeR.y, eyeRad,
      cLx: cheekL.x, cLy: cheekL.y, cRx: cheekR.x, cRy: cheekR.y, slimRad,
      mx: mouth.x, my: mouth.y, mouthRad,
      exX, exY, eyX, eyY, halfW, halfH,
    };
    if (!sm) sm = Object.assign({}, raw);
    else for (const key in raw) sm[key] = ema(sm[key], raw[key]);

    gl.uniform1f(U.u_hasFace, 1.0);
    gl.uniform2f(U.u_faceCenter, sm.cx, sm.cy);
    gl.uniform2f(U.u_faceRadius, sm.rx, sm.ry);
    gl.uniform2f(U.u_eyeL, sm.eLx, sm.eLy);
    gl.uniform2f(U.u_eyeR, sm.eRx, sm.eRy);
    gl.uniform1f(U.u_eyeRad, sm.eyeRad);
    gl.uniform2f(U.u_cheekL, sm.cLx, sm.cLy);
    gl.uniform2f(U.u_cheekR, sm.cRx, sm.cRy);
    gl.uniform1f(U.u_slimRad, sm.slimRad);
    gl.uniform2f(U.u_mouth, sm.mx, sm.my);
    gl.uniform1f(U.u_mouthRad, sm.mouthRad);
  }

  // ---------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------
  let lastTime = performance.now();
  let frames = 0;
  let fpsAccum = 0;

  function render() {
    if (!video.videoWidth) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    }

    // --- base beauty pass ---
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    gl.uniform2f(U.u_texSize, canvas.width, canvas.height);

    // fx = before/after intensity (0 = original, 1 = full effect)
    const k = fxVal;
    gl.uniform1f(U.u_smooth, (params.smooth / 100) * k);
    gl.uniform1f(U.u_whiten, (params.whiten / 100) * k);
    gl.uniform1f(U.u_bright, (params.bright / 100) * k);
    gl.uniform1f(U.u_slim, (params.slim / 100) * k);
    gl.uniform1f(U.u_eye, (params.eye / 100) * k);
    gl.uniform1f(U.u_lips, (params.lips / 100) * k);
    gl.uniform1f(U.u_dull, (params.dull / 100) * k);
    gl.uniform1f(U.u_glow, (params.glow / 100) * k);
    gl.uniform1f(U.u_split, 0.0); // spatial wipe disabled; using intensity instead

    computeFaceUniforms(latestLandmarks);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // --- simulated acne/wrinkle overlay on the face mesh ---
    drawMesh(latestLandmarks, (params.acne / 100) * k, (params.wrinkle / 100) * k);

    // FPS
    const now = performance.now();
    const dt = now - lastTime;
    lastTime = now;
    fpsAccum += dt;
    frames++;
    if (fpsAccum >= 500) {
      fpsEl.textContent = Math.round(1000 / (fpsAccum / frames)) + " FPS";
      fpsAccum = 0;
      frames = 0;
    }
  }

  // ---------------------------------------------------------------
  // MediaPipe FaceMesh
  // ---------------------------------------------------------------
  function initFaceMesh() {
    faceMesh = new FaceMesh({
      locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    faceMesh.onResults((results) => {
      latestLandmarks =
        results.multiFaceLandmarks && results.multiFaceLandmarks.length
          ? results.multiFaceLandmarks[0]
          : null;
      render();
      statusEl.textContent = latestLandmarks ? "พบใบหน้า ✓" : "ไม่พบใบหน้า";
    });
  }

  let rafId = null;
  let currentStream = null;
  let sending = false;

  async function startCamera() {
    statusEl.textContent = "กำลังเปิดกล้อง…";
    try {
      if (!faceMesh) initFaceMesh();

      // stop any previous stream / loop
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      if (currentStream) { currentStream.getTracks().forEach((t) => t.stop()); }

      // Request the highest resolution the device can give us so the
      // preview stays sharp instead of being upscaled from a tiny frame.
      currentStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      });
      video.srcObject = currentStream;
      await video.play();

      // back camera should not be mirrored
      canvas.classList.toggle("no-mirror", facingMode === "environment");

      const track = currentStream.getVideoTracks()[0];
      const s = track.getSettings ? track.getSettings() : {};
      statusEl.textContent = "พร้อมใช้งาน" + (s.width ? ` (${s.width}×${s.height})` : "");

      // Own frame loop — full control over resolution & pacing.
      const loop = async () => {
        if (video.readyState >= 2 && !sending) {
          sending = true;
          try { await faceMesh.send({ image: video }); }
          catch (e) { /* ignore transient send errors */ }
          sending = false;
        }
        rafId = requestAnimationFrame(loop);
      };
      loop();
    } catch (err) {
      console.error(err);
      statusEl.textContent = "เปิดกล้องไม่สำเร็จ: " + err.message;
    }
  }

  function captureImage() {
    render();
    const out = document.createElement("canvas");
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext("2d");
    if (facingMode === "user") { ctx.translate(out.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(canvas, 0, 0);
    return out.toDataURL("image/png");
  }

  // ---------------------------------------------------------------
  // Screen navigation (booth flow)
  // ---------------------------------------------------------------
  const screens = {
    home: document.getElementById("screen-home"),
    camera: document.getElementById("screen-camera"),
    result: document.getElementById("screen-result"),
  };
  function showScreen(name) {
    for (const k in screens) {
      if (screens[k]) screens[k].classList.toggle("active", k === name);
    }
  }

  // ---- Home screen ----
  const emailInput = document.getElementById("emailInput");
  const startBtn = document.getElementById("startBtn");
  if (startBtn) {
    startBtn.addEventListener("click", () => {
      const email = (emailInput && emailInput.value || "").trim();
      if (emailInput && email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        emailInput.classList.add("invalid");
        emailInput.focus();
        return;
      }
      if (emailInput) emailInput.classList.remove("invalid");
      try { window.localStorage.setItem("pan_email", email); } catch (e) {}
      showScreen("camera");
      startCamera();
    });
  }

  // ---- Mode selection ----
  document.querySelectorAll(".mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyMode(btn.dataset.mode);
    });
  });

  // ---- Before/after intensity slider (0 = original, 100 = full effect) ----
  const splitSlider = document.getElementById("split");
  if (splitSlider) {
    fxVal = +splitSlider.value / 100;
    splitSlider.addEventListener("input", (e) => { fxVal = +e.target.value / 100; });
  }

  // ---- Switch camera ----
  const switchBtn = document.getElementById("switchBtn");
  if (switchBtn) {
    switchBtn.addEventListener("click", () => {
      facingMode = facingMode === "user" ? "environment" : "user";
      startCamera();
    });
  }

  // ---- Capture -> result ----
  let lastShot = null;
  const resultImg = document.getElementById("resultImg");
  const captureBtn = document.getElementById("captureBtn");
  if (captureBtn) {
    captureBtn.addEventListener("click", () => {
      lastShot = captureImage(); // capture exactly what's shown (current intensity)
      if (resultImg) resultImg.src = lastShot;
      showScreen("result");
    });
  }

  // ---- Result buttons ----
  const backBtn = document.getElementById("backBtn");
  if (backBtn) backBtn.addEventListener("click", () => showScreen("camera"));

  const downloadBtn = document.getElementById("downloadBtn");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      if (!lastShot) return;
      const a = document.createElement("a");
      a.download = "decode-your-skin-" + Date.now() + ".png";
      a.href = lastShot;
      a.click();
    });
  }

  const shareBtn = document.getElementById("shareBtn");
  if (shareBtn) {
    shareBtn.addEventListener("click", async () => {
      if (!lastShot) return;
      try {
        const blob = await (await fetch(lastShot)).blob();
        const file = new File([blob], "decode-your-skin.png", { type: "image/png" });
        if (navigator.canShare && navigator.canShare({ files: [file] })) {
          await navigator.share({ files: [file], title: "DECODE YOUR SKIN", text: "AI Photo Booth by PAN Clinic" });
        } else {
          // fallback: download
          const a = document.createElement("a");
          a.download = "decode-your-skin.png";
          a.href = lastShot;
          a.click();
        }
      } catch (e) { /* user cancelled */ }
    });
  }

  if (typeof FaceMesh === "undefined") {
    statusEl.textContent = "โหลดไลบรารีไม่สำเร็จ (ตรวจสอบอินเทอร์เน็ต)";
  }
})();
