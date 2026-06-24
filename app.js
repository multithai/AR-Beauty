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
  const overlay = document.getElementById("overlay");
  const startBtn = document.getElementById("startBtn");
  const statusEl = document.getElementById("status");
  const fpsEl = document.getElementById("fps");

  // ---- Controls ----
  const controls = {
    smooth: document.getElementById("smooth"),
    whiten: document.getElementById("whiten"),
    bright: document.getElementById("bright"),
    slim: document.getElementById("slim"),
    eye: document.getElementById("eye"),
    lips: document.getElementById("lips"),
  };
  const valLabels = {
    smooth: document.getElementById("v-smooth"),
    whiten: document.getElementById("v-whiten"),
    bright: document.getElementById("v-bright"),
    slim: document.getElementById("v-slim"),
    eye: document.getElementById("v-eye"),
    lips: document.getElementById("v-lips"),
  };

  const params = { smooth: 45, whiten: 25, bright: 15, slim: 20, eye: 15, lips: 0 };

  const PRESETS = {
    off:     { smooth: 0,  whiten: 0,  bright: 0,  slim: 0,  eye: 0,  lips: 0 },
    natural: { smooth: 45, whiten: 18, bright: 12, slim: 18, eye: 12, lips: 0 },
    smooth:  { smooth: 70, whiten: 30, bright: 18, slim: 25, eye: 18, lips: 10 },
    glam:    { smooth: 85, whiten: 40, bright: 25, slim: 40, eye: 35, lips: 45 },
  };

  let compare = false;   // show original (before) while held
  let facingMode = "user";
  let faceMesh = null;
  let latestLandmarks = null;

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
          px.x -= u_slim * infl * rad * 0.18;
        }
        vec2 cr = u_cheekR * u_texSize;
        float dr = distance(px, cr);
        if (dr < rad) {
          float infl = (1.0 - dr / rad);
          infl = infl * infl;
          px.x += u_slim * infl * rad * 0.18;
        }
      }

      // Eye enlarge: bulge (sample closer to center -> magnify).
      if (u_eye > 0.001 && u_hasFace > 0.5) {
        float rad = u_eyeRad * u_texSize.x;
        vec2 el = u_eyeL * u_texSize;
        float de = distance(px, el);
        if (de < rad) {
          float t = de / rad;
          float scale = 1.0 - u_eye * 0.35 * (1.0 - t);
          px = el + (px - el) * scale;
        }
        vec2 er = u_eyeR * u_texSize;
        float de2 = distance(px, er);
        if (de2 < rad) {
          float t = de2 / rad;
          float scale = 1.0 - u_eye * 0.35 * (1.0 - t);
          px = er + (px - er) * scale;
        }
      }

      return px / u_texSize;
    }

    // Elliptical face mask (1 inside face, 0 outside) with soft edge.
    float faceMask(vec2 uv) {
      if (u_hasFace < 0.5) return 0.0;
      vec2 d = (uv - u_faceCenter) / u_faceRadius;
      float r = length(d);
      return 1.0 - smoothstep(0.85, 1.05, r);
    }

    float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

    void main() {
      vec2 uv = warp(v_uv);
      vec3 orig = texture2D(u_tex, uv).rgb;
      vec3 color = orig;

      float mask = faceMask(v_uv);

      // ---- Skin smoothing (edge-preserving bilateral) ----
      if (u_smooth > 0.001 && mask > 0.001) {
        vec2 texel = 1.0 / u_texSize;
        float radius = mix(1.0, 4.0, u_smooth);
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
        color += u_bright * 0.18 * mask;
      }

      // ---- Whitening (lift + slight desaturation toward warm white) ----
      if (u_whiten > 0.001) {
        vec3 target = mix(color, vec3(1.0, 0.98, 0.97), 0.5);
        color = mix(color, target, u_whiten * 0.4 * mask);
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
    "u_hasFace", "u_faceCenter", "u_faceRadius", "u_eyeL", "u_eyeR", "u_eyeRad",
    "u_cheekL", "u_cheekR", "u_slimRad", "u_mouth", "u_mouthRad",
  ].forEach((n) => { U[n] = gl.getUniformLocation(program, n); });
  gl.uniform1i(gl.getUniformLocation(program, "u_tex"), 0);

  // ---------------------------------------------------------------
  // Landmark helpers (MediaPipe FaceMesh indices)
  // ---------------------------------------------------------------
  // left eye corners: 33 (outer), 133 (inner); right eye: 362, 263
  // left cheek: 234 ; right cheek: 454 ; chin: 152 ; mouth: 13/14
  function computeFaceUniforms(lm) {
    if (!lm) {
      gl.uniform1f(U.u_hasFace, 0.0);
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

    gl.uniform1f(U.u_hasFace, 1.0);
    gl.uniform2f(U.u_faceCenter, cx, cy);
    gl.uniform2f(U.u_faceRadius, rx, ry);
    gl.uniform2f(U.u_eyeL, eyeL.x, eyeL.y);
    gl.uniform2f(U.u_eyeR, eyeR.x, eyeR.y);
    gl.uniform1f(U.u_eyeRad, eyeRad);
    gl.uniform2f(U.u_cheekL, cheekL.x, cheekL.y);
    gl.uniform2f(U.u_cheekR, cheekR.x, cheekR.y);
    gl.uniform1f(U.u_slimRad, slimRad);
    gl.uniform2f(U.u_mouth, mouth.x, mouth.y);
    gl.uniform1f(U.u_mouthRad, mouthRad);
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

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    gl.uniform2f(U.u_texSize, canvas.width, canvas.height);

    const k = compare ? 0 : 1; // hold compare -> show original
    gl.uniform1f(U.u_smooth, (params.smooth / 100) * k);
    gl.uniform1f(U.u_whiten, (params.whiten / 100) * k);
    gl.uniform1f(U.u_bright, (params.bright / 100) * k);
    gl.uniform1f(U.u_slim, (params.slim / 100) * k);
    gl.uniform1f(U.u_eye, (params.eye / 100) * k);
    gl.uniform1f(U.u_lips, (params.lips / 100) * k);

    computeFaceUniforms(compare ? null : latestLandmarks);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

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

      overlay.classList.add("hidden");
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
      overlay.classList.remove("hidden");
    }
  }

  // ---------------------------------------------------------------
  // UI wiring
  // ---------------------------------------------------------------
  Object.keys(controls).forEach((key) => {
    controls[key].addEventListener("input", (e) => {
      params[key] = +e.target.value;
      valLabels[key].textContent = e.target.value;
      clearActivePreset();
    });
  });

  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    Object.keys(p).forEach((key) => {
      params[key] = p[key];
      controls[key].value = p[key];
      valLabels[key].textContent = p[key];
    });
  }
  function clearActivePreset() {
    document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
  }
  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      clearActivePreset();
      btn.classList.add("active");
      applyPreset(btn.dataset.preset);
    });
  });

  // Compare (press & hold)
  const compareBtn = document.getElementById("compareBtn");
  const setCompare = (v) => { compare = v; };
  compareBtn.addEventListener("mousedown", () => setCompare(true));
  compareBtn.addEventListener("mouseup", () => setCompare(false));
  compareBtn.addEventListener("mouseleave", () => setCompare(false));
  compareBtn.addEventListener("touchstart", (e) => { e.preventDefault(); setCompare(true); }, { passive: false });
  compareBtn.addEventListener("touchend", (e) => { e.preventDefault(); setCompare(false); });

  // Snapshot
  document.getElementById("snapBtn").addEventListener("click", () => {
    // re-render to be sure the framebuffer is current
    render();
    const out = document.createElement("canvas");
    out.width = canvas.width;
    out.height = canvas.height;
    const ctx = out.getContext("2d");
    if (facingMode === "user") {
      ctx.translate(out.width, 0);
      ctx.scale(-1, 1); // mirror to match preview
    }
    ctx.drawImage(canvas, 0, 0);
    const link = document.createElement("a");
    link.download = "ar-beauty-" + Date.now() + ".png";
    link.href = out.toDataURL("image/png");
    link.click();
  });

  // Switch camera
  document.getElementById("switchBtn").addEventListener("click", () => {
    facingMode = facingMode === "user" ? "environment" : "user";
    startCamera();
  });

  startBtn.addEventListener("click", startCamera);

  if (typeof FaceMesh === "undefined") {
    statusEl.textContent = "โหลดไลบรารีไม่สำเร็จ (ตรวจสอบอินเทอร์เน็ต)";
  } else {
    statusEl.textContent = "พร้อมเปิดกล้อง";
  }
})();
