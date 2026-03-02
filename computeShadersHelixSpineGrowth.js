// computeShadersHelixSpineGrowth.js
// "Further amalgamated" version:
// - Helix is not a fixed analytic spline.
// - New helix segments are *predicted from the current growth direction* (from the live backbone).
// - Spines are mechanically coupled back into the helix (reciprocal springs).
// Result: spine tip capture can bend the helix AND steer future growth.

export function createPosTargetShaderHelixSpineGrowth(consts) {
  const {
    HELIX_SEGS,
    SPINE_COUNT,
    NECK_SEGS,
    HEAD_PTS,
  } = consts;

  const TOTAL_HELIX = 2 * HELIX_SEGS;
  const SPINE_STRIDE = NECK_SEGS + HEAD_PTS;
  const TOTAL_SPINE = SPINE_COUNT * SPINE_STRIDE;
  const INPUT_START = TOTAL_HELIX + TOTAL_SPINE;

  return /* glsl */`
    precision highp float;

    uniform float time;
    uniform float dt;

    // Geometry params (scene units)
    uniform float helixRadius;
    uniform float helixPitch;        // distance per full 2π turn
    uniform float axialShift;        // axial offset between strands (DNA-like asymmetry)
    uniform float ds;                // nominal step per segment
    uniform float helixGrowRate;     // segments/sec
    uniform float initialHelixSegs;  // visible at t=0

    uniform float spineLenMax;
    uniform float spineDelay;
    uniform float spineGrowDur;
    uniform float headGrowDur;

    uniform float inputRingR;
    uniform vec3  inputRingCenter;

    // NOTE: sampler2D pos is injected by GPUComputationRenderer when posVar is set as a dependency.

    float PI = 3.141592653589793;

    float hash12(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    vec2 uvFromIndexF(float idx) {
      float x = mod(idx, resolution.x);
      float y = floor(idx / resolution.x);
      return (vec2(x + 0.5, y + 0.5) / resolution.xy);
    }

    vec4 readPos(int idx) {
      return texture2D(pos, uvFromIndexF(float(idx)));
    }

    float usedW(float w) {
      return step(0.5, floor(w + 1e-4));
    }

    // Fallback analytic seed along +Z (only used before the live chain exists)
    vec3 fallbackCenter(int j) {
      float z = float(j) * ds - 0.5 * axialShift;
      return vec3(0.0, 0.0, z);
    }

    vec3 centerAt(int j) {
      // midpoint of live strand A/B when available
      int aIdx = j;
      int bIdx = j + ${HELIX_SEGS};
      vec4 a4 = readPos(aIdx);
      vec4 b4 = readPos(bIdx);
      float ua = usedW(a4.w);
      float ub = usedW(b4.w);
      if (ua > 0.5 && ub > 0.5) return 0.5 * (a4.xyz + b4.xyz);
      return fallbackCenter(j);
    }

    vec3 safeNorm(vec3 v, vec3 fb) {
      float L = length(v);
      if (L < 1e-6) return fb;
      return v / L;
    }

    void frameFromT(vec3 t, out vec3 N, out vec3 B) {
      vec3 up = vec3(0.0, 1.0, 0.0);
      N = cross(up, t);
      float nl = length(N);
      if (nl < 1e-4) {
        up = vec3(1.0, 0.0, 0.0);
        N = cross(up, t);
        nl = length(N);
      }
      N /= max(nl, 1e-6);
      B = normalize(cross(t, N));
    }

    vec3 predictCenter(int j, int frontJ, vec3 cFront, vec3 tFront) {
      if (j <= frontJ) return centerAt(j);
      float k = float(j - frontJ);
      return cFront + tFront * (k * ds);
    }

    vec3 tangentPred(int j, int frontJ, vec3 cFront, vec3 tFront) {
      if (j <= 0) return vec3(0.0, 0.0, 1.0);
      vec3 c0 = predictCenter(j - 1, frontJ, cFront, tFront);
      vec3 c1 = predictCenter(j,     frontJ, cFront, tFront);
      return safeNorm(c1 - c0, tFront);
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / resolution.xy;
      vec2 frag = gl_FragCoord.xy - vec2(0.5);
      float idx = frag.x + frag.y * resolution.x;
      int i = int(idx + 0.5);

      vec3 posOut = vec3(0.0);
      float member = 0.0;
      float metaByte = 0.0;

      float growthFrontF = min(float(${HELIX_SEGS - 1}), initialHelixSegs + time * helixGrowRate);
      int frontJ = int(floor(growthFrontF + 1e-4));

      vec3 cFront = centerAt(frontJ);
      vec3 cPrev  = centerAt(max(frontJ - 1, 0));
      vec3 tFront = safeNorm(cFront - cPrev, vec3(0.0, 0.0, 1.0));

      // Helix strands (indices 0 .. 2*HELIX_SEGS-1)
      if (i < ${TOTAL_HELIX}) {
        int strand = i / ${HELIX_SEGS};
        int j = i - strand * ${HELIX_SEGS};

        member = step(float(j), growthFrontF);

        // Build predicted centerline at segment j
        vec3 c = predictCenter(j, frontJ, cFront, tFront);
        vec3 t = tangentPred(j, frontJ, cFront, tFront);
        vec3 N, B;
        frameFromT(t, N, B);

        float q = 2.0 * PI / max(helixPitch, 1e-6);
        float s = float(j) * ds;

        if (strand == 0) {
          float theta = q * s;
          posOut = c + helixRadius * (cos(theta) * N + sin(theta) * B);
          metaByte = 0.0;

          // lock the root seed a bit (gives a stable reference)
          if (j == 0) {
            posOut = vec3(helixRadius, 0.0, 0.0);
          }
        } else {
          // DNA-like: strand 2 uses an axial shift + π phase, per θ2(s)=q(s-Δ)+π
          vec3 cB = c - t * axialShift;
          float theta = q * (s - axialShift) + PI;
          posOut = cB + helixRadius * (cos(theta) * N + sin(theta) * B);
          metaByte = 1.0;

          if (j == 0) {
            posOut = vec3(-helixRadius, 0.0, -axialShift);
          }
        }
      }

      // Spine chains (after helix)
      else if (i < ${INPUT_START}) {
        int local = i - ${TOTAL_HELIX};
        int spineId = local / ${SPINE_STRIDE};
        int k = local - spineId * ${SPINE_STRIDE};

        float fbase = float(spineId) * float(${HELIX_SEGS - 1}) / max(float(${SPINE_COUNT - 1}), 1.0);
        int baseJ = int(floor(fbase + 0.5));

        float baseActive = step(float(baseJ) + 1.0, growthFrontF);

        // growth schedule
        float t0 = float(baseJ) * spineDelay;
        float tRel = max(0.0, time - t0);
        float neckProg = clamp(tRel / max(spineGrowDur, 1e-6), 0.0, 1.0);
        float headProg = clamp((tRel - spineGrowDur) / max(headGrowDur, 1e-6), 0.0, 1.0);

        // base point & local frame
        vec3 cBase = predictCenter(baseJ, frontJ, cFront, tFront);
        vec3 tBase = tangentPred(baseJ, frontJ, cFront, tFront);
        vec3 N, B;
        frameFromT(tBase, N, B);

        float phi = hash12(vec2(float(spineId), 7.13)) * 2.0 * PI;
        vec3 radial = normalize(cos(phi) * N + sin(phi) * B);

        vec3 base = cBase;
        vec3 tip  = base + radial * (spineLenMax * neckProg);

        if (k < ${NECK_SEGS}) {
          float front = neckProg * float(${NECK_SEGS - 1});
          float segActive = step(float(k), front + 1e-3);
          member = baseActive * segActive;

          float u = float(k) / max(float(${NECK_SEGS - 1}), 1.0);
          posOut = base + radial * (spineLenMax * neckProg * u);
          metaByte = 2.0;
        } else {
          int hk = k - ${NECK_SEGS};
          float front = headProg * float(${HEAD_PTS - 1});
          float segActive = step(float(hk), front + 1e-3);
          member = baseActive * segActive;

          float n1 = hash12(vec2(float(spineId), float(hk) + 19.1));
          float n2 = hash12(vec2(float(spineId) + 31.7, float(hk)));
          float n3 = hash12(vec2(float(spineId) + 9.2, float(hk) + 5.3));
          vec3 jitter = vec3(n1 - 0.5, n2 - 0.5, n3 - 0.5);

          posOut = tip + jitter * 0.6;
          metaByte = 3.0;
        }
      }

      // Inputs (one per spine) — "synaptic partners". These follow the moving helix base.
      else {
        int inputId = i - ${INPUT_START};

        float fbase = float(inputId) * float(${HELIX_SEGS - 1}) / max(float(${SPINE_COUNT - 1}), 1.0);
        int baseJ = int(floor(fbase + 0.5));
        float baseActive = step(float(baseJ) + 1.0, growthFrontF);

        vec3 cBase = predictCenter(baseJ, frontJ, cFront, tFront);
        vec3 tBase = tangentPred(baseJ, frontJ, cFront, tFront);
        vec3 N, B;
        frameFromT(tBase, N, B);

        float phi = hash12(vec2(float(inputId), 7.13)) * 2.0 * PI;
        vec3 radial = normalize(cos(phi) * N + sin(phi) * B);

        posOut = cBase + inputRingCenter + radial * inputRingR;
        member = baseActive;
        metaByte = 4.0;
      }

      gl_FragColor = vec4(posOut, member + metaByte / 256.0);
    }
  `;
}

export function createAccShaderHelixSpineGrowth(consts) {
  const {
    HELIX_SEGS,
    SPINE_COUNT,
    NECK_SEGS,
    HEAD_PTS,
  } = consts;

  const TOTAL_HELIX = 2 * HELIX_SEGS;
  const SPINE_STRIDE = NECK_SEGS + HEAD_PTS;
  const TOTAL_SPINE = SPINE_COUNT * SPINE_STRIDE;
  const INPUT_START = TOTAL_HELIX + TOTAL_SPINE;

  return /* glsl */`
    precision highp float;

    uniform float time;
    uniform float dt;

    // Helix mechanics
    uniform float kHelixLink;
    uniform float kHelixRung;
    uniform float kHelixBend;
    uniform float kAnchorRoot;

    // Spine mechanics
    uniform float kSpineLink;
    uniform float kSpineBend;
    uniform float kSpineBase;
    uniform float kHeadToTip;

    // Coupling: HELIX <-> SPINE base
    uniform float kHelixSpineCouple;

    // Tip capture
    uniform float kTipToInput;
    uniform float tipMaxForce;

    // Input pin
    uniform float kInputPin;

    // geometry
    uniform float ds;
    uniform float helixRadius;
    uniform float helixPitch;

    // Injected by GPUComputationRenderer via dependencies:
    // sampler2D pos, vel, posTarget

    float PI = 3.141592653589793;

    vec2 uvFromIndexF(float idx) {
      float x = mod(idx, resolution.x);
      float y = floor(idx / resolution.x);
      return (vec2(x + 0.5, y + 0.5) / resolution.xy);
    }

    vec4 readPos(int idx) {
      return texture2D(pos, uvFromIndexF(float(idx)));
    }

    float usedW(float w) {
      return step(0.5, floor(w + 1e-4));
    }

    vec3 spring(vec3 p, vec3 q, float restLen, float k) {
      vec3 d = q - p;
      float L = length(d);
      if (L < 1e-6) return vec3(0.0);
      return k * (L - restLen) * (d / L);
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / resolution.xy;
      vec2 frag = gl_FragCoord.xy - vec2(0.5);
      float idx = frag.x + frag.y * resolution.x;
      int i = int(idx + 0.5);

      vec4 p4 = texture2D(pos, uv);
      vec4 t4 = texture2D(posTarget, uv);

      float memInt = floor(p4.w + 1e-4);
      if (memInt < 0.5) {
        gl_FragColor = vec4(0.0);
        return;
      }

      vec3 p = p4.xyz;
      vec3 a = vec3(0.0);

      // Helix segment length based on pitch+radius (approx)
      float q = 2.0 * PI / max(helixPitch, 1e-6);
      float dtheta = q * ds;
      float chord = 2.0 * helixRadius * sin(0.5 * abs(dtheta));
      float helixSegLen = sqrt(chord*chord + ds*ds);

      if (i < ${TOTAL_HELIX}) {
        int strand = i / ${HELIX_SEGS};
        int j = i - strand * ${HELIX_SEGS};

        // Neighbor springs along the strand
        if (j > 0) {
          vec4 q4 = readPos(i - 1);
          if (usedW(q4.w) > 0.5) a += spring(p, q4.xyz, helixSegLen, kHelixLink);
        }
        if (j < ${HELIX_SEGS - 1}) {
          vec4 q4 = readPos(i + 1);
          if (usedW(q4.w) > 0.5) a += spring(p, q4.xyz, helixSegLen, kHelixLink);
        }

        // Rung spring between strands at the same j
        int other = (strand == 0) ? (i + ${HELIX_SEGS}) : (i - ${HELIX_SEGS});
        vec4 o4 = readPos(other);
        if (usedW(o4.w) > 0.5) a += spring(p, o4.xyz, 2.0 * helixRadius, kHelixRung);

        // Bending regularizer
        if (j > 0 && j < ${HELIX_SEGS - 1}) {
          vec4 pm = readPos(i - 1);
          vec4 pp = readPos(i + 1);
          if (usedW(pm.w) > 0.5 && usedW(pp.w) > 0.5) {
            vec3 curv = pm.xyz - 2.0 * p + pp.xyz;
            a += -kHelixBend * curv;
          }
        }

        // Anchor the very first segment (gives the whole structure a reference)
        if (j == 0) {
          a += (t4.xyz - p) * kAnchorRoot;
        }

        // --- Reciprocal coupling: helix is pulled toward the spine root ---
        // This is the missing piece that makes spines part of *helix mechanics*.
        // Each helix segment j gathers forces from any spines anchored at j.
        vec3 base = p;
        // use midpoint base to apply equal force to both strands
        if (usedW(o4.w) > 0.5) base = 0.5 * (p + o4.xyz);

        for (int sid = 0; sid < ${SPINE_COUNT}; sid++) {
          float fbase = float(sid) * float(${HELIX_SEGS - 1}) / max(float(${SPINE_COUNT - 1}), 1.0);
          int baseJ = int(floor(fbase + 0.5));
          if (baseJ != j) continue;

          int neck0Idx = ${TOTAL_HELIX} + sid * ${SPINE_STRIDE} + 0;
          vec4 n0 = readPos(neck0Idx);
          if (usedW(n0.w) < 0.5) continue;

          vec3 diff = (n0.xyz - base);
          // split the coupling across both strands by applying half on each strand fragment
          a += diff * (0.5 * kHelixSpineCouple);
        }
      }
      else if (i < ${INPUT_START}) {
        // Spine mechanics
        int local = i - ${TOTAL_HELIX};
        int spineId = local / ${SPINE_STRIDE};
        int k = local - spineId * ${SPINE_STRIDE};

        float fbase = float(spineId) * float(${HELIX_SEGS - 1}) / max(float(${SPINE_COUNT - 1}), 1.0);
        int baseJ = int(floor(fbase + 0.5));

        int aIdx = baseJ;
        int bIdx = baseJ + ${HELIX_SEGS};
        vec4 a4 = readPos(aIdx);
        vec4 b4 = readPos(bIdx);
        if (usedW(a4.w) < 0.5 || usedW(b4.w) < 0.5) {
          gl_FragColor = vec4(0.0);
          return;
        }

        vec3 base = 0.5 * (a4.xyz + b4.xyz);
        float spineSegLen = ds * 0.8;

        if (k < ${NECK_SEGS}) {
          if (k == 0) {
            // spine base tether to helix (one side of the reciprocal pair)
            a += spring(p, base, 0.0, kSpineBase);
          } else {
            vec4 pr = readPos(i - 1);
            if (usedW(pr.w) > 0.5) a += spring(p, pr.xyz, spineSegLen, kSpineLink);
          }

          if (k < ${NECK_SEGS - 1}) {
            vec4 nx = readPos(i + 1);
            if (usedW(nx.w) > 0.5) a += spring(p, nx.xyz, spineSegLen, kSpineLink);
          }

          if (k >= 1 && k < ${NECK_SEGS - 1}) {
            vec4 pm = readPos(i - 1);
            vec4 pp = readPos(i + 1);
            if (usedW(pm.w) > 0.5 && usedW(pp.w) > 0.5) {
              vec3 curv = pm.xyz - 2.0 * p + pp.xyz;
              a += -kSpineBend * curv;
            }
          }

          // Tip attracted to its input
          if (k == ${NECK_SEGS - 1}) {
            int inputIdx = ${INPUT_START} + spineId;
            vec4 in4 = readPos(inputIdx);
            if (usedW(in4.w) > 0.5) {
              vec3 f = (in4.xyz - p) * kTipToInput;
              float fm = length(f);
              if (fm > tipMaxForce) f *= (tipMaxForce / max(fm, 1e-6));
              a += f;
            }
          }
        } else {
          // Head particles tethered to tip
          int tipIdx = ${TOTAL_HELIX} + spineId * ${SPINE_STRIDE} + (${NECK_SEGS - 1});
          vec4 tip4 = readPos(tipIdx);
          if (usedW(tip4.w) > 0.5) a += spring(p, tip4.xyz, 0.6, kHeadToTip);
        }
      }
      else {
        // Inputs pinned to their moving targets
        a += (t4.xyz - p) * kInputPin;
      }

      gl_FragColor = vec4(a, 0.0);
    }
  `;
}
