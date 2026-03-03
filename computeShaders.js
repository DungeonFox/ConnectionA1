// computeShaders.js
// (FULL FILE) — updated from your last working computeShaders.js with ZERO divergence in existing shader logic.
// Changes are ONLY:
//   1) Added createExtPackShader() (packs incoming connected-card particles into THIS card's local sim space)
//   2) Upgraded createAccShader() to include robust proximity “tunneling” attraction toward closest incoming particle
//   3) Exported createExtPackShader in the module export list
//
// Everything else is preserved verbatim from your uploaded computeShaders.js (posTarget, vel, pos, tags, insideMask).

import { hash12, cnoise4 } from './glslNoise.js';
import { PI } from './glslUtils.js';

function createPosTargetShader() {
    return `
        ${PI}
        ${hash12}
        uniform float time;
                uniform float dt;
        uniform float baseR;
        uniform float pulseAmp;
        uniform float pulseHz;

uniform int shapeMode;
        uniform sampler2D niftiTex;
        uniform sampler2D asqTex; // Asset Sequencer positions (float4 per particle)
        uniform float activeCount;

        // PNG atlas (Preset B8, row-major layout)
        uniform sampler2D atlasTex;
        // Optional derived atlas for offscreen per-frame modulation
        uniform sampler2D atlasTexDerived;
	        // Per-particle tag byte (computed for all particles)
	        // NOTE: uniform sampler2D tags; is injected by GPUComputationRenderer when tagsVar is a dependency.
        // 0 = use atlas alpha as meta, 1 = use computed tags texture
        uniform float atlasMetaFromTags;

        // Optional virtual->physical frame mapping (1xN RGBA8: physical index in R byte)
        uniform sampler2D atlasSeqTex;
        uniform float atlasSeqEnabled;
        uniform float atlasSeqCount;

        // Select derived atlas per sampled frame (0 or 1)
        uniform float atlasUseDerived0;
        uniform float atlasUseDerived1;

        // Optional per-frame rect lookup (1×N RGBA32F: x,y,w,h in pixels).
        // Enables variable frame dimensions inside a single spritesheet.
        uniform sampler2D atlasRectTex;
        uniform float atlasRectEnabled;
        uniform float atlasRectCount;
        uniform vec2 atlasSize;
        uniform vec3 atlasPosMin;
        uniform vec3 atlasPosMax;
        uniform float atlasCount;
        uniform bool atlasFlipY;
        uniform int atlasSourceMode; // 0=full image base frame, 1=spritesheet frames

        // Atlas frame animation (spritesheet)
        // - If atlasAnimEnabled==0, frame params are ignored and frame 0 is sampled.
        // - atlasFrame is allowed to be fractional; atlasInterpolate controls lerp.
        // - atlasFrameGrid = (cols, rows)
        // - atlasFrameSize = (frameWidthPx, frameHeightPx)
        uniform float atlasAnimEnabled;
        uniform float atlasFrame;
        uniform float atlasFrameCount;
        uniform vec2 atlasFrameGrid;
        uniform vec2 atlasFrameSize;
        uniform float atlasLoop;
        uniform float atlasInterpolate;
        uniform float atlasMetaFromFrame0;

        // Per-frame packing transform (applied AFTER interpretation)
        uniform vec3 atlasFrameTranslate;
        uniform vec4 atlasFrameQuat; // (x,y,z,w)
        uniform vec3 atlasFrameScale;

        // Expression modulation of sampled RGB prior to interpretation.
        // All are optional; atlasExprEnabled==0 is identity.
        uniform float atlasExprEnabled;
        uniform vec3 atlasExprMul;
        uniform vec3 atlasExprAdd;
        uniform vec3 atlasExprGamma;
        uniform vec3 atlasExprClampMin;
        uniform vec3 atlasExprClampMax;
        uniform float atlasExprSinAmp;
        uniform float atlasExprSinFreq;
        uniform float atlasExprPhase;
        uniform float atlasExprTimeScale;
        uniform float atlasExprIdScale;
        uniform float atlasExprFrameScale;

        // Convert particle index i into atlas UVs for a given *physical* frame index.
        // Frames are packed as a spritesheet into the atlas PNG.
                vec4 _getAtlasRect(float pIndex) {
            float N = max(atlasRectCount, 1.0);
            float fi = clamp(floor(pIndex + 0.5), 0.0, N - 1.0);
            float u = (fi + 0.5) / N;
            return texture2D(atlasRectTex, vec2(u, 0.5));
        }

        // Convert particle index i into atlas UVs for a given *physical* frame index.
        // Supports:
        //   1) Uniform grid tiles (atlasFrameGrid + atlasFrameSize)
        //   2) Variable rect tiles (atlasRectTex) when atlasRectEnabled=1
        vec4 sampleAtlasFramePhysical(sampler2D tex, float i, float frameIndex) {
            vec2 img = max(atlasSize, vec2(1.0));

            // ---- Rect mode (variable frame dimensions) ----
            if (atlasRectEnabled > 0.5) {
                vec4 r = _getAtlasRect(frameIndex);
                float rx = floor(r.x + 0.5);
                float ry = floor(r.y + 0.5);
                float rw = max(1.0, floor(r.z + 0.5));
                float rh = max(1.0, floor(r.w + 0.5));

                float ay = floor(i / rw);
                float ax = mod(i, rw);

                if (ay >= rh) return vec4(0.0);

                float ly = atlasFlipY ? (rh - 1.0 - ay) : ay;

                float gx = rx + ax;
                float gy = ry + ly;

                if (gx < 0.0 || gy < 0.0 || gx >= img.x || gy >= img.y) return vec4(0.0);

                float u = (gx + 0.5) / img.x;
                // PNG row 0 is TOP; texture.flipY is false, so v=1 maps to TOP.
                float v = 1.0 - ((gy + 0.5) / img.y);
                return texture2D(tex, vec2(u, v));
            }

            // ---- Grid mode (uniform tiles) ----
            vec2 grid = max(atlasFrameGrid, vec2(1.0));
            vec2 fsz = max(atlasFrameSize, vec2(1.0));

            float ay = floor(i / fsz.x);
            float ax = mod(i, fsz.x);

            if (ay >= fsz.y) return vec4(0.0);

            float fx = mod(frameIndex, grid.x);
            float fy = floor(frameIndex / grid.x);

            float ly = atlasFlipY ? (fsz.y - 1.0 - ay) : ay;

            float gx = fx * fsz.x + ax;
            float gy = fy * fsz.y + ly;

            if (gx < 0.0 || gy < 0.0 || gx >= img.x || gy >= img.y) return vec4(0.0);

            float u = (gx + 0.5) / img.x;
            float v = 1.0 - ((gy + 0.5) / img.y);
            return texture2D(tex, vec2(u, v));
        }

        // Map a *virtual* timeline index to a *physical* spritesheet frame index.
        // atlasSeqTex is a 1xN RGBA8 texture storing the physical index in the R byte.
        float mapVirtualToPhysical(float vIndex) {
            if (atlasSeqEnabled < 0.5) return vIndex;
            float N = max(atlasSeqCount, 1.0);
            float vi = clamp(vIndex, 0.0, N - 1.0);
            float u = (vi + 0.5) / N;
            float r = texture2D(atlasSeqTex, vec2(u, 0.5)).r;
            return floor(r * 255.0 + 0.5);
        }

        // Sample an atlas frame with optional derived atlas selection.
        // Signature includes a reserved param (pOverride) for compatibility with earlier call sites.
        vec4 sampleAtlasFrame(float i, float vIndex, float pOverride, float useDerived) {
            float pIndex = mapVirtualToPhysical(vIndex);
            // If caller provides a positive override, use it (keeps forward-compat).
            if (pOverride > 0.5) pIndex = pOverride;

            vec4 a = sampleAtlasFramePhysical(atlasTex, i, pIndex);
            vec4 b = sampleAtlasFramePhysical(atlasTexDerived, i, pIndex);
            return mix(a, b, step(0.5, useDerived));
        }

        // Back-compat: 2-arg signature (no derived, no mapping)
        vec4 sampleAtlasFrame(float i, float frameIndex) {
            return sampleAtlasFrame(i, frameIndex, 0.0, 0.0);
        }

        vec3 applyAtlasExpr(vec3 c, float i, float frameF) {
            if (atlasExprEnabled < 0.5) return c;

            c = c * atlasExprMul + atlasExprAdd;
            c = max(c, vec3(0.0));
            c = pow(c, max(atlasExprGamma, vec3(0.0001)));

            if (atlasExprSinAmp > 0.000001) {
                float tt = time * atlasExprTimeScale;
                float idt = i * atlasExprIdScale;
                float ft = frameF * atlasExprFrameScale;
                vec3 arg = c * atlasExprSinFreq + vec3(tt + idt + ft + atlasExprPhase);
                c += atlasExprSinAmp * sin(arg);
            }

            c = clamp(c, atlasExprClampMin, atlasExprClampMax);
            return c;
        }

        vec3 quatRotate(vec4 q, vec3 v) {
            // q assumed normalized
            vec3 t = 2.0 * cross(q.xyz, v);
            return v + q.w * t + cross(q.xyz, t);
        }

        vec3 applyFrameXform(vec3 p) {
            vec3 ps = p * atlasFrameScale;
            vec3 pr = quatRotate(atlasFrameQuat, ps);
            return pr + atlasFrameTranslate;
        }

        // Returns xyz = decoded position, w = membership flag (0 = unused, 1 = used, 3 = dual-tagged)
        // with packed metaByte in the fractional part of w.
        // NOTE: Preset B8's alpha is packed meta. A value of 0 is valid for used particles.
        vec4 getAtlasB8(float i, vec2 uvP) {
            float used = 0.0;
            // Presence is by COUNT (atlasCount), not by texel value.
            if (atlasCount > 0.5 && i < atlasCount) used = 1.0;
            if (used < 0.5) {
                // Skip atlas sampling for unused slots, but still allow a tag byte.
                float metaTags = floor(texture2D(tags, uvP).r * 255.0 + 0.5);
                float metaByte = metaTags * step(0.5, atlasMetaFromTags);
                float packed = 0.0 + (metaByte / 256.0);
                return vec4(0.0, 0.0, 0.0, packed);
            }

            // Frame selection
            float fc = max(atlasFrameCount, 1.0);
            float frameF = (atlasAnimEnabled > 0.5) ? atlasFrame : 0.0;

            vec4 tex0;
            vec4 tex1;
            vec4 texM;
            float t = 0.0;

            if (atlasSourceMode == 0) {
                // Base-frame mode: the PNG is a single full frame.
                // Virtual frames are produced via expression modulation (frameF affects applyAtlasExpr only).
                tex0 = sampleAtlasFrame(i, 0.0, 0.0, atlasUseDerived0);
                tex1 = tex0;
                texM = tex0;
                t = 0.0;
            } else {
                // Spritesheet mode: sample frame f0 (and optionally f1 for interpolation)
                float f0 = floor(frameF);
                if (atlasLoop > 0.5) f0 = mod(f0, fc); else f0 = clamp(f0, 0.0, fc - 1.0);
                float f1 = f0 + 1.0;
                if (atlasLoop > 0.5) f1 = mod(f1, fc); else f1 = min(f1, fc - 1.0);
                t = (atlasInterpolate > 0.5) ? fract(frameF) : 0.0;

                tex0 = sampleAtlasFrame(i, f0, 0.0, atlasUseDerived0);
                tex1 = sampleAtlasFrame(i, f1, 0.0, atlasUseDerived1);
                // Meta is usually stable per particle; default is to read it from frame 0.
                texM = (atlasMetaFromFrame0 > 0.5) ? sampleAtlasFrame(i, 0.0, 0.0, 0.0) : tex0;
            }

            vec3 c = mix(tex0.rgb, tex1.rgb, t);
            c = applyAtlasExpr(c, i, frameF);

            vec3 pos = atlasPosMin + c * (atlasPosMax - atlasPosMin);
            pos = applyFrameXform(pos);

            float metaAtlas = floor(texM.a * 255.0 + 0.5) * used;
            float metaTags = floor(texture2D(tags, uvP).r * 255.0 + 0.5);
            float metaByte = mix(metaAtlas, metaTags, step(0.5, atlasMetaFromTags));
            float speedBucket = floor(metaByte / 64.0);
            float dual = step(2.5, speedBucket);
            float membership = used * (1.0 + 2.0 * dual);

            // Pack meta byte into fractional part, preserving 0/1/3 membership comparisons.
            float packed = membership + (metaByte / 256.0);
            return vec4(pos, packed);
        }

        vec3 getRingPosition(float i, float time) {
    float denom = max(activeCount, 1.0);
    float angle = (i / denom) * PI * 2.0;

    float rn = hash12(vec2(i * 0.123, i * 3.453));
    float tt = time * 0.001; // seconds
    float pulse = sin(tt * (pulseHz * 2.0 * PI)) * (baseR * pulseAmp);

    float rad = baseR * (1.0 + rn) + pulse;

    vec3 pos;
    pos.x = cos(angle) * rad;
    pos.y = sin(angle) * rad;
    pos.z = 0.0;
    return pos;
}


        vec3 getSphericalPosition(float i, float time) {
    float denom = max(activeCount, 1.0);
    float rn = hash12(vec2(i * 0.731, i * 1.337));
    float zN = hash12(vec2(i * 2.117, i * 0.411));

    float theta = acos(2.0 * rn - 1.0);
    float phi = (i / denom) * PI * 2.0;

    float tt = time * 0.001; // seconds
    float pulse = sin(tt * (pulseHz * 2.0 * PI)) * (baseR * pulseAmp);

    float radius = baseR * (0.75 + 0.75 * zN) + pulse;

    vec3 pos;
    pos.x = radius * sin(theta) * cos(phi);
    pos.y = radius * sin(theta) * sin(phi);
    pos.z = radius * cos(theta);
    return pos;
}


        void main() {
            // Use integer-ish indexing (gl_FragCoord is at texel centers, starting at 0.5)
            vec2 frag = gl_FragCoord.xy - vec2(0.5);
            float i = (frag.y * resolution.x) + frag.x;

            // Inactive slots (beyond activeCount) are forced unused.
            if (activeCount > 0.5 && i >= activeCount) {
                gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
                return;
            }
            vec2 uv = gl_FragCoord.xy / resolution.xy;

            vec3 pos = vec3(0.0);
            float member = 1.0;

            if (shapeMode == 0) {
                pos = getRingPosition(i, time);
                member = 1.0;
            } else if (shapeMode == 1) {
                pos = getSphericalPosition(i, time);
                member = 1.0;
            } else if (shapeMode == 2) {
                vec4 n4 = texture2D(niftiTex, uv);
                pos = n4.xyz;
                member = n4.w; // 0/1 present from loader
                // Ensure unused particles have a deterministic target.
                // Without this, unused slots can retain stale positions from a prior mode.
                if (member < 0.5) {
                    pos = vec3(0.0);
                }
            } else if (shapeMode == 4) {
                vec4 a4 = texture2D(asqTex, uv);
                pos = a4.xyz;
                member = a4.w; // 0/1 present from ASQ ingestion
                if (member < 0.5) {
                    pos = vec3(0.0);
                }
            } else if (shapeMode == 3) {
                vec4 p4 = getAtlasB8(i, uv);
                pos = p4.xyz;
                member = p4.w; // 0/1/3
            }

            // Pack computed tag byte into the fractional part of w for ALL shape modes.


            // IMPORTANT: membership is kept in the integer part (0/1/3) so used/unused checks must use floor(w).


            if (shapeMode != 3) {


                float metaByte = floor(texture2D(tags, uv).r * 255.0 + 0.5);


                float memInt = step(0.5, member);


                member = memInt + (metaByte / 256.0);


            }


            gl_FragColor = vec4(pos, member);
        }
    `;
}

/* ============================================================================
   NEW: createExtPackShader()
   Packs incoming connected card particles into THIS card's sim-local space.

   Contract:
     - srcPosN: source card's pos texture (xyz position in source sim-local space, w contains membership+meta)
     - srcResN: vec2(width,height) of that pos texture
     - srcActiveN: number of active particles in that source
     - srcMatN: mat4 that maps (source sim-local) -> (this card sim-local)
                i.e. inv(SB)*rel2*SA, where rel2 is your FLOW_CONV’d A->B transform.
     - Output extPos texture: xyz in THIS card sim-local space, w=1 for valid, else 0.

   Notes:
     - We use floor(p4.w) membership semantics, preserving your "membership int + metaByte/256" packing.
     - We do NOT attempt to sample beyond srcActiveN.
============================================================================ */
function createExtPackShader(MAX_SOURCES = 4) {
    return `
        ${PI}
        ${hash12}

        uniform float time;
        uniform float extEnabled;
        uniform float extCount;

        uniform sampler2D srcPos0;
        uniform sampler2D srcPos1;
        uniform sampler2D srcPos2;
        uniform sampler2D srcPos3;

        uniform vec2 srcRes0;
        uniform vec2 srcRes1;
        uniform vec2 srcRes2;
        uniform vec2 srcRes3;

        uniform float srcActive0;
        uniform float srcActive1;
        uniform float srcActive2;
        uniform float srcActive3;

        uniform mat4 srcMat0;
        uniform mat4 srcMat1;
        uniform mat4 srcMat2;
        uniform mat4 srcMat3;

        vec2 uvFromIndex(float idx, vec2 res) {
            float x = mod(idx, res.x);
            float y = floor(idx / res.x);
            return (vec2(x, y) + 0.5) / res;
        }

        float activeSrc(int s) {
            if (s == 0) return srcActive0;
            if (s == 1) return srcActive1;
            if (s == 2) return srcActive2;
            return srcActive3;
        }

        vec2 resSrc(int s) {
            if (s == 0) return srcRes0;
            if (s == 1) return srcRes1;
            if (s == 2) return srcRes2;
            return srcRes3;
        }

        vec4 sampleSrc(int s, float j) {
            vec2 r = resSrc(s);
            vec2 uv = uvFromIndex(j, r);
            if (s == 0) return texture2D(srcPos0, uv);
            if (s == 1) return texture2D(srcPos1, uv);
            if (s == 2) return texture2D(srcPos2, uv);
            return texture2D(srcPos3, uv);
        }

        mat4 matSrc(int s) {
            if (s == 0) return srcMat0;
            if (s == 1) return srcMat1;
            if (s == 2) return srcMat2;
            return srcMat3;
        }

        void main() {
            if (extEnabled < 0.5 || extCount < 0.5) {
                gl_FragColor = vec4(0.0);
                return;
            }

            // global external index
            vec2 frag = gl_FragCoord.xy - vec2(0.5);
            float g = frag.x + frag.y * resolution.x;

            float a0 = activeSrc(0);
            float a1 = activeSrc(1);
            float a2 = activeSrc(2);
            float a3 = activeSrc(3);

            int s = -1;
            float j = 0.0;

            if (g < a0) { s = 0; j = g; }
            else if (g < a0 + a1) { s = 1; j = g - a0; }
            else if (g < a0 + a1 + a2) { s = 2; j = g - (a0 + a1); }
            else if (g < a0 + a1 + a2 + a3) { s = 3; j = g - (a0 + a1 + a2); }

            if (s < 0) {
                gl_FragColor = vec4(0.0);
                return;
            }

            vec4 p4 = sampleSrc(s, j);

            // membership is integer part of w in your pipeline
            float membershipInt = floor(p4.w + 0.0001);
            float used = step(0.5, membershipInt);
            if (used < 0.5) {
                gl_FragColor = vec4(0.0);
                return;
            }

            vec3 pLocalB = (matSrc(s) * vec4(p4.xyz, 1.0)).xyz;
            gl_FragColor = vec4(pLocalB, 1.0);
        }
    `;
}

/* ============================================================================
   UPDATED: createAccShader()
   Adds “tunneling” attraction between connected cards, using packed extPos.

   Contract:
     - GPUComputationRenderer injects:
         sampler2D pos, posTarget, vel, extPos (extPos only if extPosVar is a dependency)
     - extPos is ALREADY in THIS card's sim-local space (created by createExtPackShader)
============================================================================ */
function createAccShader() {
    return /* glsl */`
        ${PI}
        ${hash12}

        uniform float time;
        uniform float dt;

        uniform float unusedEnabled;
        uniform float unusedTargetK;
        uniform float unusedTargetC;

        // Safety clamp: acceleration magnitude capped
        uniform float maxAcc;

        // Target spring
        uniform float targetK;
        uniform float targetC;

        // Local repulsion (soft collision)
        uniform float repelRadius;
        uniform float repelK;
        uniform float repelC;
        uniform float repelPower;

        // External tunneling (source card)
        uniform float extEnabled;       // 0/1
        uniform float extMode;          // 1 = tunnel, 2 = helix-tunnel
        uniform sampler2D extPos;       // src positions in src sim-local
        uniform mat4 extMat;            // maps src sim-local -> this sim-local
        uniform float extRadius;        // capture radius
        uniform float extK;             // tunnel stiffness
        uniform float extC;             // tunnel damping (along dir)
        uniform float extSamples;       // up to 32
        uniform float extPower;         // falloff shaping exponent

        // Optional swirl to approximate a "double helix" drain
        uniform float extTwist;         // 0 disables
        uniform float extTwistHz;       // cycles / second

        // Alpha-calibrated coupling (runtime-solved alpha0 can directly modulate acceleration)
        uniform float alpha0;
        uniform float alphaCoupling;

        // NOTE: sampler2D pos, posTarget, vel are injected by GPUComputationRenderer via dependencies.

        float smooth01(float x) { return clamp(x, 0.0, 1.0); }

        float inferAlphaHatFromForceDir(vec3 forceDir, vec3 tangent){
            float fParallel = abs(dot(forceDir, tangent));
            float fPerp = length(forceDir - dot(forceDir, tangent) * tangent);
            return atan(fParallel, max(fPerp, 1e-6));
        }

        // Deterministic-ish random vec2 per particle and sample index (no extra textures)
        vec2 rand2(float seed) {
            float a = hash12(vec2(seed, 13.37));
            float b = hash12(vec2(seed, 91.17));
            return vec2(a, b);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;

            vec4 p4 = texture2D(pos, uv);
            vec4 t4 = texture2D(posTarget, uv);

            float memInt = floor(p4.w + 0.0001);
            float used = step(0.5, memInt);
            
            // Allow "unused" particles to move (e.g., to travel along a scaffold path) when enabled.
            // When disabled, unused particles are frozen (legacy behavior).
            if (used < 0.5) {
                if (unusedEnabled > 0.5) {
                    vec3 toT = t4.xyz - p4.xyz;
                    vec3 a = toT * unusedTargetK - texture2D(vel, uv).xyz * unusedTargetC;
                    gl_FragColor = vec4(a, 1.0);
                } else {
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                }
                return;
            }

            vec3 p = p4.xyz;
            vec3 v = texture2D(vel, uv).xyz;

            vec3 a = vec3(0.0);

            // ------------------------------------------------------------------
            // Local repulsion against nearby texel neighbors (cheap approximation)
            // ------------------------------------------------------------------
            float rr = max(repelRadius, 1e-6);
            float rr2 = rr * rr;

            vec3 repelA = vec3(0.0);
            vec2 px = vec2(1.0 / resolution.x, 1.0 / resolution.y);

            // 8-neighborhood (plus a couple diagonals) keeps it stable and cheap
            vec2 offs[8];
            offs[0] = vec2( 1.0,  0.0);
            offs[1] = vec2(-1.0,  0.0);
            offs[2] = vec2( 0.0,  1.0);
            offs[3] = vec2( 0.0, -1.0);
            offs[4] = vec2( 1.0,  1.0);
            offs[5] = vec2(-1.0,  1.0);
            offs[6] = vec2( 1.0, -1.0);
            offs[7] = vec2(-1.0, -1.0);

            for (int i = 0; i < 8; i++) {
                vec2 uvn = uv + offs[i] * px;
                vec4 q4 = texture2D(pos, uvn);
                float qUsed = step(0.5, floor(q4.w + 0.0001));
                if (qUsed < 0.5) continue;

                vec3 d = p - q4.xyz;
                float d2 = dot(d, d);
                if (d2 < rr2 && d2 > 1e-10) {
                    float dist = sqrt(d2);
                    vec3 dir = d / dist;
                    float s = 1.0 - (dist / rr);
                    float shaped = pow(smooth01(s), max(repelPower, 0.5));
                    repelA += dir * (repelK * shaped);

                    // Damp only along the push direction (cheap; uses own v)
                    float vn = dot(v, dir);
                    repelA += (-dir) * (vn * repelC * shaped);
                }
            }
            a += repelA;

            // ------------------------------------------------------------------
            // Spring to target + isotropic damping
            // ------------------------------------------------------------------
            vec3 back = (t4.xyz - p);
            a += back * targetK;
            a += (-v) * targetC;

            // ------------------------------------------------------------------
            // External tunneling (nearest-of-random-samples)
            // ------------------------------------------------------------------
            if (extEnabled > 0.5 && extSamples > 0.5) {
                float r = max(extRadius, 1e-6);
                float r2 = r * r;

                // stable per-frag id
                float idx = (gl_FragCoord.x - 0.5) + (gl_FragCoord.y - 0.5) * resolution.x;

                float bestD2 = 1e20;
                vec3 bestQ = vec3(0.0);

                for (int k = 0; k < 32; k++) {
                    if (float(k) >= extSamples) break;

                    vec2 ruv = rand2(idx + float(k) * 19.13 + time * 0.0007);
                    vec4 q4 = texture2D(extPos, ruv);
                    float qUsed = step(0.5, q4.w);
                    if (qUsed < 0.5) continue;

                    vec3 q = (extMat * vec4(q4.xyz, 1.0)).xyz;

                    vec3 d = q - p;
                    float d2 = dot(d, d);
                    if (d2 < bestD2) { bestD2 = d2; bestQ = q; }
                }

                if (bestD2 < r2) {
                    float dist = sqrt(bestD2);
                    vec3 dir = (bestQ - p) / max(dist, 1e-6);

                    float s = 1.0 - (dist / r);
                    float shaped = pow(smooth01(s), max(extPower, 0.5));

                    // Spring toward closest incoming particle (tunnel)
                    a += (bestQ - p) * (extK * shaped);

                    // Damping along tunnel direction using current velocity
                    float vn = dot(v, dir);
                    a += (-dir) * (vn * extC * shaped);

                    // Alpha-calibrated acceleration coupling (direct particle-motion effect)
                    vec3 tangent = normalize(v + back + vec3(1e-6, 0.0, 0.0));
                    float alphaHat = inferAlphaHatFromForceDir(dir, tangent);
                    float alphaDeficit = max(alpha0 - alphaHat, 0.0);
                    float alphaPressure = smoothstep(0.01, 0.35, alphaDeficit);
                    float alphaGain = max(alphaCoupling, 0.0) * alphaPressure * shaped;
                    // Increase tunnel pull and add a mild transverse push to make the effect observable.
                    a += (bestQ - p) * (0.45 * alphaGain);
                    vec3 side = normalize(cross(tangent, dir) + vec3(1e-6, 0.0, 0.0));
                    a += side * (0.20 * alphaGain);

                    // Optional swirl: a stable helical component around dir
                    if (extMode > 1.5 && extTwist > 0.00001) {
                        vec3 axis = vec3(0.0, 1.0, 0.0);
                        vec3 ortho = cross(axis, dir);
                        float ol = length(ortho);
                        if (ol < 1e-4) {
                            axis = vec3(1.0, 0.0, 0.0);
                            ortho = cross(axis, dir);
                            ol = length(ortho);
                        }
                        ortho /= max(ol, 1e-6);
                        vec3 binorm = normalize(cross(dir, ortho));

                        float tt = time * 0.001;
                        float phase = hash12(vec2(idx, 7.7)) * (2.0 * PI) + tt * (extTwistHz * 2.0 * PI);
                        vec3 helixDir = ortho * sin(phase) + binorm * cos(phase);

                        a += helixDir * (extTwist * shaped);
                    }
                }
            }

            // NaN guard + clamp
            if (!(a.x == a.x) || !(a.y == a.y) || !(a.z == a.z)) {
                a = vec3(0.0);
            }
            float aLen = length(a);
            if (aLen > maxAcc) {
                a *= (maxAcc / max(aLen, 1e-6));
            }

            gl_FragColor = vec4(a, 1.0);
        }
    `;
}

function createVelShader() {
    return /* glsl */`
        ${PI}
        uniform float time;
        uniform float dt;
        uniform float maxVel;
        uniform float velDrag;
        uniform int resetCountdown;

        // NOTE: sampler2D acc, vel are injected by GPUComputationRenderer via dependencies.

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            vec4 a4 = texture2D(acc, uv);
            vec4 v4 = texture2D(vel, uv);

            // Optional hard reset (used when "preserve distribution on switch" is OFF)
            if (resetCountdown > 0) {
                gl_FragColor = vec4(0.0, 0.0, 0.0, v4.w);
                return;
            }

            vec3 v = v4.xyz;
            v += a4.xyz * dt;

            // Exponential-ish drag for stability across dt
            float drag = exp(-velDrag * dt);
            v *= drag;

            // NaN guard
            if (!(v.x == v.x) || !(v.y == v.y) || !(v.z == v.z)) {
                v = vec3(0.0);
            }

            float sp = length(v);
            if (sp > maxVel) v *= (maxVel / max(sp, 1e-6));

            gl_FragColor = vec4(v, v4.w);
        }
    `;
}

function createPosShader() {
    return /* glsl */`
        ${PI}
        uniform float time;
        uniform float dt;
        uniform int frame;
        uniform float maxPos;
        uniform int resetCountdown;

        // NOTE: sampler2D posTarget, pos, vel are injected by GPUComputationRenderer via dependencies.

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            vec4 tarNow = texture2D(posTarget, uv);
            vec4 p;

            if (frame < 2 || resetCountdown > 0) {
                // Startup seed OR explicit "snap" reset on mode switch.
                p = tarNow;
            } else {
                p = texture2D(pos, uv);
                p.xyz += texture2D(vel, uv).xyz * dt;

                // Refresh membership every frame so mode switches take effect immediately.
                float wasPresent = step(0.5, floor(p.w + 0.0001));
                float nowPresent = step(0.5, floor(tarNow.w + 0.0001));

                // If a particle toggled state, snap to target (avoids stale geometry)
                if (wasPresent < 0.5 && nowPresent > 0.5) p.xyz = tarNow.xyz;
                if (wasPresent > 0.5 && nowPresent < 0.5) p.xyz = tarNow.xyz;

                p.w = tarNow.w;
            }

            // Reset if NaN
            if (!(p.x == p.x) || !(p.y == p.y) || !(p.z == p.z)) {
                p = tarNow;
            }

            // Clamp radius
            float r = length(p.xyz);
            if (r > maxPos) p.xyz *= (maxPos / max(r, 1e-6));

            gl_FragColor = vec4(p.xyz, p.w);
        }
    `;
}

function createTagsShader() {
    return `
        ${PI}
        ${hash12}
	        // NOTE: uniform sampler2D pos; is injected by GPUComputationRenderer when posVar is a dependency.
        uniform float time;
        uniform int tagMode;   // 0=off, 1=angle, 2=height, 3=radius, 4=index, 5=noise
        uniform float tagScale;
        uniform float tagBias;

        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            vec3 p = texture2D(pos, uv).xyz;

            float tagByte = 0.0;

            if (tagMode == 1) {
                float ang = atan(p.y, p.x);
                float u = (ang / (PI * 2.0)) + 0.5;
                tagByte = floor(clamp(u, 0.0, 1.0) * 255.0 + 0.5);
            } else if (tagMode == 2) {
                float v = (p.y * tagScale) + tagBias;
                tagByte = floor(clamp(v, 0.0, 255.0));
            } else if (tagMode == 3) {
                float r = length(p) * tagScale + tagBias;
                tagByte = mod(floor(r), 256.0);
            } else if (tagMode == 4) {
                vec2 frag = gl_FragCoord.xy - vec2(0.5);
                float i = (frag.y * resolution.x) + frag.x;
                tagByte = mod(i, 256.0);
            } else if (tagMode == 5) {
                float n = hash12(p.xy * 0.01 + vec2(time * 0.0001, 13.37));
                tagByte = floor(n * 255.0 + 0.5);
            }

            float tagNorm = tagByte / 255.0;
            gl_FragColor = vec4(tagNorm, 0.0, 0.0, 1.0);
        }
    `;
}

// Computes per-particle inside flags (in world-space) for voxel/target AABB.
// Output (RGBA):
//  R = inside voxel AABB (1/0)
//  G = inside target AABB (1/0)
//  B = inside ANY enabled AABB (1/0, or 1 if both ops disabled)
//  A = membership "used" flag (>=1)
function createInsideMaskShader() {
    return /* glsl */`
        uniform float voxelBoundaryOps;
        uniform float targetVoxelBoundaryOps;

        uniform vec3 voxMin;
        uniform vec3 voxMax;
        uniform vec3 tgtMin;
        uniform vec3 tgtMax;

        // group transforms (match materials.js simulation point transform)
        uniform vec3 instanceOffset;
        uniform vec3 unusedOffset;
        uniform vec3 instanceGroupEuler;
        uniform vec3 unusedGroupEuler;
        uniform float enableDualGroupOverlap;

        // matrixWorld of the Points object that renders particles
        uniform mat4 pointsMatrixWorld;

        // NOTE: uniform sampler2D pos; is injected by GPUComputationRenderer when posVar is a dependency.

        mat3 rotX(float a){ float s=sin(a), c=cos(a); return mat3(1.,0.,0., 0.,c,-s, 0.,s,c); }
        mat3 rotY(float a){ float s=sin(a), c=cos(a); return mat3(c,0.,s, 0.,1.,0., -s,0.,c); }
        mat3 rotZ(float a){ float s=sin(a), c=cos(a); return mat3(c,-s,0., s,c,0., 0.,0.,1.); }
        mat3 rotXYZ(vec3 r){ return rotZ(r.z)*rotY(r.y)*rotX(r.x); }

        float insideAABB(vec3 p, vec3 mn, vec3 mx){
            float ix = step(mn.x, p.x) * step(p.x, mx.x);
            float iy = step(mn.y, p.y) * step(p.y, mx.y);
            float iz = step(mn.z, p.z) * step(p.z, mx.z);
            return ix * iy * iz;
        }

        void main() {
            // consistent with other compute shaders in this project
            vec2 uv = gl_FragCoord.xy / resolution.xy;

            vec4 pTex = texture2D(pos, uv);
            vec3 p0 = pTex.xyz;

            float membershipInt = floor(pTex.w + 0.0001);
            float used = step(0.5, membershipInt);
            float dualTagged = step(2.0, membershipInt);
            float dualEnabled = step(0.5, enableDualGroupOverlap);
            float doDual = dualTagged * dualEnabled;

            mat3 RI = rotXYZ(instanceGroupEuler);
            mat3 RU = rotXYZ(unusedGroupEuler);

            vec3 pInst = RI * p0 + instanceOffset;
            vec3 pUn   = RU * p0 + unusedOffset;
            vec3 pDual = RU * (RI * p0) + instanceOffset + unusedOffset;

            // follow the same mix logic as the points vertex shader
            vec3 pLocal = mix(pUn, pInst, used);
            pLocal = mix(pLocal, pDual, doDual);

            vec3 pW = (pointsMatrixWorld * vec4(pLocal, 1.0)).xyz;

            float voxOn = step(0.5, voxelBoundaryOps);
            float tgtOn = step(0.5, targetVoxelBoundaryOps);
            float hasOps = max(voxOn, tgtOn);

            float inV = insideAABB(pW, voxMin, voxMax) * voxOn;
            float inT = insideAABB(pW, tgtMin, tgtMax) * tgtOn;
            float inAny = step(0.5, inV + inT);
            // if neither boundary op is enabled, treat everything as inside (so UI can still show all particles)
            float insideAny = mix(1.0, inAny, hasOps);

            gl_FragColor = vec4(inV, inT, insideAny, used);
        }
    `;
}

export {
createPosTargetShader,
  createExtPackShader,
  createAccShader,
  createVelShader,
  createPosShader,
  createTagsShader,
  createInsideMaskShader
};