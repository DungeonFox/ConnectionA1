// materials.js — group membership driven by pos.w (0 unused, 1 used, 3 dual-tagged).
// Dual-tagged particles only receive BOTH group transforms when enableDualGroupOverlap is true.

import * as THREE from 'three';
import { cnoise3, hash12 } from './glslNoise.js';
import { uvFromIndex } from './glslUtils.js';
import { loadNiftiRGBA } from './niftiCLoader.js';

let t = THREE;

function createPointsMaterial(NUM_X, NUM_Y, params, renderer) {
    const {
        albedoThreshold = 0.1,
        colorLerpThreshold = 0.5,
        useNiftiColors = true,
        maxInstances = 1
    } = params;

    if (!t || !t.ShaderMaterial) throw new Error('THREE.js not properly loaded');
    if (!renderer) throw new Error('Renderer is required');

    const pointsVertexShader = `
        ${cnoise3}
        ${hash12}
        ${uvFromIndex}

        uniform float time;
        uniform float albedoThreshold;
        uniform float colorLerpThreshold;
        uniform bool useNiftiColors;
        uniform float maxValue;

        // Depth coloring (linearized view-space depth)
        uniform float cameraNear;
        uniform float cameraFar;
        // Absolute depth range used to map to 0..1 (in world units along view ray).
        // If <= 0, falls back to (cameraFar - cameraNear).
        uniform float depthRange;
        // Absolute start for mapping the depth metric to 0..1.
        // Example: set this near your cloud (e.g., ~400–600) for a camera at z=500.
        uniform float depthStart;
        // If true, use radial distance to camera instead of view-space Z.
        uniform int depthMetricMode;
        uniform bool depthInvert;
        uniform bool depthColorInstance;
        uniform bool depthColorUnused;

        // Atlas-driven per-particle colors (from sidecar + packed meta byte in pos.w fractional part)
        uniform bool atlasColorEnabled;
        // 0=off, 1=used/unused, 2=palette(0..7), 3=fixed (always atlasColorUsed)
        uniform int atlasColorMode;
        // Used for loadAccuracy normalization when depthMetricMode==4
        uniform float atlasLayerMax;
        uniform vec3 atlasColorUsed;
        uniform vec3 atlasColorUnused;
        uniform vec3 atlasPalette0;
        uniform vec3 atlasPalette1;
        uniform vec3 atlasPalette2;
        uniform vec3 atlasPalette3;
        uniform vec3 atlasPalette4;
        uniform vec3 atlasPalette5;
        uniform vec3 atlasPalette6;
        uniform vec3 atlasPalette7;


        uniform sampler2D niftiRGBA;
        uniform float niftiWeight;

        // Per-subset offsets (set from JS)
        uniform vec3 instanceOffset;
        uniform vec3 unusedOffset;

        // Legacy, kept for compatibility (not used for membership anymore)
        uniform float niftiPointCount;

        // NEW: gate for allowing a particle to belong to both groups.
        uniform bool enableDualGroupOverlap;

        // GPGPU fields
        uniform sampler2D posTarget;
        uniform sampler2D acc;
        uniform sampler2D vel;
        uniform sampler2D pos;

        // Shader-space group rotations (radians)
        uniform vec3 instanceGroupEuler;
        uniform vec3 unusedGroupEuler;

        // Euler XYZ rotation to mat3
        mat3 rotXYZ(vec3 r){
            float cx = cos(r.x), sx = sin(r.x);
            float cy = cos(r.y), sy = sin(r.y);
            float cz = cos(r.z), sz = sin(r.z);
            mat3 Rx = mat3( 1., 0., 0.,   0., cx,-sx,   0., sx, cx );
            mat3 Ry = mat3( cy, 0., sy,   0., 1., 0.,  -sy, 0., cy );
            mat3 Rz = mat3( cz,-sz, 0.,   sz, cz, 0.,   0., 0., 1. );
            return Rz * Ry * Rx;
        }

        attribute float aIndex;

        varying float alpha;
        varying vec3 col;
        varying float shouldDiscard;

        void main() {
            int i = int(aIndex + 0.5);
            ivec2 size = ivec2(${NUM_X}, ${NUM_Y});
            vec2 uv = uvFromIndex(i, size);

            vec4 pTex = texture2D(pos, uv);
            vec4 vTex = texture2D(vel, uv);

            float n = hash12(vec2(float(i), 0.0));
            float ps = 3.0 + pow(n, 2.0) * 5.0;
            alpha = 0.2 + pow(n, 20.0) * 0.3;
            shouldDiscard = 0.0;
            if (n > 0.999) { ps *= 1.1; alpha *= 2.0; }

            // Color from niftiRGBA if enabled
            vec3 niftiColor = vec3(0.0);
            float albedo = 0.0;
            if (useNiftiColors && niftiWeight > 0.0) {
                vec4 data = texture2D(niftiRGBA, uv);
                niftiColor = data.rgb;
                albedo = data.a;
            }

	            // NOTE: Many RTT-to-card pipelines end up vertically flipped (UV or blit path).
	            // If your particles appear upside-down, flip Y here.
	            vec3 p0 = pTex.xyz;
	            p0.y *= -1.0;

            // Membership from pos.w:
            //   0 => unused only
            //   1 => used (instance only)
            //   3 => dual-tagged (instance + unused) when enableDualGroupOverlap
            float membershipInt = floor(pTex.w + 1e-4);
            float used = step(0.5, membershipInt);
            float dualTagged = step(2.0, membershipInt);
            float dualEnabled = enableDualGroupOverlap ? 1.0 : 0.0;
            float doDual = dualTagged * dualEnabled;

            // Atlas meta byte packed into the fractional part of pTex.w:
            //   pTex.w = membership(0/1/3) + metaByte/256.
            float metaFrac = pTex.w - membershipInt;
            float metaByte = floor(metaFrac * 256.0 + 0.5);
            float layerLow = mod(metaByte, 64.0);
            float high2 = floor(metaByte / 64.0);
            // Reconstruct a stable monotonic “layer-ish” index from the full byte.
            float layerApprox = layerLow + high2 * 64.0;
            float colorIndex = mod(layerLow, 8.0);
            float denom = max(atlasLayerMax, 1.0);
            float accuracy01 = clamp(layerApprox / denom, 0.0, 1.0);

            mat3 RI = rotXYZ(instanceGroupEuler);
            mat3 RU = rotXYZ(unusedGroupEuler);

            vec3 pInst = RI * p0 + instanceOffset;
            vec3 pUn   = RU * p0 + unusedOffset;
            vec3 pDual = RU * (RI * p0) + instanceOffset + unusedOffset;

            // Exclusive by default, dual overrides when enabled
            vec3 p = mix(pUn, pInst, used);
            p = mix(p, pDual, doDual);

            // Original color ramp by velocity
            vec3 originalColor1 = vec3(1.0, 0.0, 0.0);
            vec3 originalColor2 = vec3(1.0, 0.4, 0.2);
            float velocityFactor = pow(length(vTex.xyz) / 1.5, 5.0) * 1.5;
            velocityFactor = clamp(velocityFactor, 0.0, 1.0);
            vec3 originalColor = mix(originalColor1, originalColor2, velocityFactor);

            if (useNiftiColors && albedo > albedoThreshold && albedo != 0.0) {
                float lerpFactor = albedo <= colorLerpThreshold ? albedo / colorLerpThreshold : 1.0;
                col = mix(originalColor * maxValue, niftiColor, lerpFactor);
                alpha *= (0.7 + (albedo / maxValue) * 0.3);
            } else {
                col = originalColor * maxValue;
            }

            // Override particle color from atlas sidecar (if enabled).
            if (atlasColorEnabled && atlasColorMode > 0) {
                vec3 c = mix(atlasColorUnused, atlasColorUsed, used);
                if (atlasColorMode == 3) {
                    c = atlasColorUsed;
                } else if (atlasColorMode == 2) {
                    int ci = int(colorIndex + 0.5);
                    if (ci == 0) c = atlasPalette0;
                    else if (ci == 1) c = atlasPalette1;
                    else if (ci == 2) c = atlasPalette2;
                    else if (ci == 3) c = atlasPalette3;
                    else if (ci == 4) c = atlasPalette4;
                    else if (ci == 5) c = atlasPalette5;
                    else if (ci == 6) c = atlasPalette6;
                    else c = atlasPalette7;
                }
                col = c;
            }


            // Optional depth coloring (similar spirit to three.js depth texture examples,
            // but computed directly from view-space Z for particles).
            vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
            float viewZ = -mvPosition.z;

            // Depth metric (view-independent options are useful for “height/accuracy” fades):
            // 0=viewZ, 1=distance, 2=worldY, 3=worldZ, 4=atlasAccuracy01 (from packed meta)
            vec3 worldP = (modelMatrix * vec4(p, 1.0)).xyz;
            float metric = viewZ;
            if (depthMetricMode == 1) {
                metric = length(worldP - cameraPosition);
            } else if (depthMetricMode == 2) {
                metric = worldP.y;
            } else if (depthMetricMode == 3) {
                metric = worldP.z;
            } else if (depthMetricMode == 4) {
                metric = accuracy01;
            }

            float span = depthRange;
            if (span <= 0.0) {
                if (depthMetricMode <= 1) span = (cameraFar - cameraNear);
                else span = 1.0;
            }
            span = max(span, 1e-6);
            float start = depthStart;
            // Default start only makes sense for camera-based metrics.
            // Do not clamp/override depthStart; negative and zero starts are valid.
            float depth01 = clamp((metric - start) / span, 0.0, 1.0);
            if (depthInvert) depth01 = 1.0 - depth01;
            bool doDepth = (used > 0.5 && depthColorInstance) || (used < 0.5 && depthColorUnused);
            if (doDepth) {
                float k = depth01;
                col *= k;
                alpha *= k;
            }

            gl_PointSize = ps;
            gl_Position = projectionMatrix * mvPosition;
        }
    `;

    const pointsFragmentShader = `
        uniform float maxValue;
        varying float alpha;
        varying vec3 col;
        varying float shouldDiscard;

        void main() {
            if (shouldDiscard > 0.0) discard;
            vec3 normalizedCol = col / maxValue;
            gl_FragColor = vec4(normalizedCol, alpha);
        }
    `;

    let material = new t.ShaderMaterial({
        uniforms: {
            time:               { value: 0.0 },
            albedoThreshold:    { value: albedoThreshold },
            colorLerpThreshold: { value: colorLerpThreshold },
            useNiftiColors:     { value: useNiftiColors },
            maxValue:           { value: 255.0 },

            // Depth coloring
            cameraNear:         { value: 0.1 },
            cameraFar:          { value: 2000.0 },
            depthRange:         { value: 2000.0 },
            depthStart:         { value: 0.1 },
            depthMetricMode:    { value: 0 },
            depthInvert:        { value: false },
            depthColorInstance: { value: false },
            depthColorUnused:   { value: false },
            atlasColorEnabled: { value: false },
            atlasColorMode:    { value: 0 },
            atlasLayerMax:     { value: 255.0 },
            atlasColorUsed:    { value: new t.Vector3(255, 255, 255) },
            atlasColorUnused:  { value: new t.Vector3(32, 32, 32) },
            atlasPalette0:     { value: new t.Vector3(255, 0, 0) },
            atlasPalette1:     { value: new t.Vector3(0, 255, 0) },
            atlasPalette2:     { value: new t.Vector3(0, 0, 255) },
            atlasPalette3:     { value: new t.Vector3(255, 255, 0) },
            atlasPalette4:     { value: new t.Vector3(255, 0, 255) },
            atlasPalette5:     { value: new t.Vector3(0, 255, 255) },
            atlasPalette6:     { value: new t.Vector3(255, 255, 255) },
            atlasPalette7:     { value: new t.Vector3(128, 128, 128) },

            niftiRGBA:          { value: null },
            niftiWeight:        { value: 0.0 },

            instanceOffset:     { value: new t.Vector3(0, 0, 0) },
            unusedOffset:       { value: new t.Vector3(0, 0, 0) },

            niftiPointCount:    { value: 0.0 },

            enableDualGroupOverlap: { value: false },

            posTarget:          { value: null },
            acc:                { value: null },
            vel:                { value: null },
            pos:                { value: null },

            instanceGroupEuler: { value: new t.Vector3(0, 0, 0) },
            unusedGroupEuler:   { value: new t.Vector3(0, 0, 0) }
        },
        vertexShader: pointsVertexShader,
        fragmentShader: pointsFragmentShader,
        transparent: true,
        depthWrite: true,
        depthTest: false,
        blending: t.AdditiveBlending
    });

    return material;
}

function createNiftiTexture(rgbaData, nX, nY) {
    const textureSize = nX * nY;
    const textureData = new Float32Array(textureSize * 4);

    let dataIndex = 0;
    for (let i = 0; i < textureSize; i++) {
        const texIndex = i * 4;
        if (dataIndex < rgbaData.length) {
            textureData[texIndex + 0] = rgbaData[dataIndex + 3];
            textureData[texIndex + 1] = rgbaData[dataIndex + 4];
            textureData[texIndex + 2] = rgbaData[dataIndex + 5];
            textureData[texIndex + 3] = rgbaData[dataIndex + 6];
            dataIndex += 7;
        } else {
            textureData[texIndex + 0] = 0.0;
            textureData[texIndex + 1] = 0.0;
            textureData[texIndex + 2] = 0.0;
            textureData[texIndex + 3] = 0.0;
        }
    }

    const texture = new t.DataTexture(
        textureData,
        nX, nY,
        t.RGBAFormat,
        t.FloatType
    );
    texture.needsUpdate = true;
    return texture;
}

async function integrateNiftiColors(particleSystem, niftiFile, coordinates, options = {}) {
    const {
        albedoThreshold = 0.1,
        colorLerpThreshold = 0.5,
        scale = 1.0
    } = options;

    try {
        const rgbaData = await loadNiftiRGBA(niftiFile, coordinates, { scale });

        const nX = Math.sqrt(particleSystem.geometry.attributes.position.count);
        const nY = nX;
        const niftiTexture = createNiftiTexture(rgbaData, nX, nY);

        if (particleSystem.material.uniforms.niftiRGBA) {
            particleSystem.material.uniforms.niftiRGBA.value = niftiTexture;
            particleSystem.material.uniforms.niftiWeight.value = 1.0;
            particleSystem.material.uniforms.albedoThreshold.value = albedoThreshold;
            particleSystem.material.uniforms.colorLerpThreshold.value = colorLerpThreshold;
            particleSystem.material.uniforms.useNiftiColors.value = true;
        }

        return {
            texture: niftiTexture,
            originalData: rgbaData,
            pointCount: rgbaData.length / 7
        };
    } catch (error) {
        console.error('Error integrating NIfTI colors:', error);
    }
}

export { createPointsMaterial, createNiftiTexture, integrateNiftiColors };

// Compose material: renders multiple guided frames at once (instanced), using the atlas directly.
function createComposeMaterial(NUM_X, NUM_Y, params = {}, renderer) {
    const {
        albedoThreshold = 0.1,
        colorLerpThreshold = 0.5,
        useNiftiColors = true,
        maxInstances = 1
    } = params;

    const composeVertexShader = `
        precision highp float;
        precision highp int;

        uniform float time;
        uniform float maxValue;
        uniform bool useNiftiColors;
        uniform float albedoThreshold;
        uniform float colorLerpThreshold;
        uniform sampler2D niftiRGBA;
        uniform float niftiWeight;

        // Depth coloring
        uniform float cameraNear;
        uniform float cameraFar;
        uniform float depthRange;
        uniform float depthStart;
        uniform int depthMetricMode;
        uniform bool depthInvert;
        uniform bool depthColorInstance;
        uniform bool depthColorUnused;

        // Atlas colors
        uniform bool atlasColorEnabled;
        uniform int atlasColorMode;
        uniform float atlasLayerMax;
        uniform vec3 atlasColorUsed;
        uniform vec3 atlasColorUnused;
        uniform vec3 atlasPalette0;
        uniform vec3 atlasPalette1;
        uniform vec3 atlasPalette2;
        uniform vec3 atlasPalette3;
        uniform vec3 atlasPalette4;
        uniform vec3 atlasPalette5;
        uniform vec3 atlasPalette6;
        uniform vec3 atlasPalette7;

        // Group offsets
        uniform vec3 instanceOffset;
        uniform vec3 unusedOffset;
        uniform float niftiPointCount;
        uniform bool enableDualGroupOverlap;

        // Shader-space group rotations (radians)
        uniform vec3 instanceGroupEuler;
        uniform vec3 unusedGroupEuler;

        // Atlas sampling
        uniform sampler2D atlasTex;
        uniform vec2 atlasSize;
        uniform vec3 atlasPosMin;
        uniform vec3 atlasPosMax;
        uniform float atlasCount;
        uniform bool atlasFlipY;
        uniform int atlasSourceMode; // 0 base-frame, 1 spritesheet
        uniform float atlasFrameCount;
        uniform vec2 atlasFrameGrid;
        uniform vec2 atlasFrameSize;
        uniform float atlasLoop;
        uniform float atlasInterpolate;
        uniform float atlasMetaFromFrame0;

        // Expression modulation (pre-interpret)
        uniform float atlasExprEnabled;
        uniform vec3  atlasExprMul;
        uniform vec3  atlasExprAdd;
        uniform vec3  atlasExprGamma;
        uniform vec3  atlasExprClampMin;
        uniform vec3  atlasExprClampMax;
        uniform float atlasExprSinAmp;
        uniform float atlasExprSinFreq;
        uniform float atlasExprPhase;
        uniform float atlasExprTimeScale;
        uniform float atlasExprIdScale;
        uniform float atlasExprFrameScale;

        uniform float activeCount;

	        // NOTE: three.js injects the built-in attribute 'position' automatically.
	        // Do not redeclare it here (it causes a GLSL compile error: redefinition).
        attribute vec3 iTranslate;
        attribute vec4 iQuat;
        attribute vec3 iScale;
        attribute float iFrame;

        mat3 rotXYZ(vec3 r){
            float cx = cos(r.x), sx = sin(r.x);
            float cy = cos(r.y), sy = sin(r.y);
            float cz = cos(r.z), sz = sin(r.z);
            mat3 Rx = mat3( 1., 0., 0.,   0., cx,-sx,   0., sx, cx );
            mat3 Ry = mat3( cy, 0., sy,   0., 1., 0.,  -sy, 0., cy );
            mat3 Rz = mat3( cz,-sz, 0.,   sz, cz, 0.,   0., 0., 1. );
            return Rz * Ry * Rx;
        }

        float hash12(vec2 p) {
            vec3 p3  = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        vec2 uvFromIndex(int i, ivec2 size) {
            int x = i % size.x;
            int y = i / size.x;
            return vec2((float(x) + 0.5) / float(size.x), (float(y) + 0.5) / float(size.y));
        }

        vec3 quatRotate(vec4 q, vec3 v) {
            vec3 t = 2.0 * cross(q.xyz, v);
            return v + q.w * t + cross(q.xyz, t);
        }

        vec2 pixelToUV(vec2 p, vec2 texSize) {
            return vec2((p.x + 0.5) / texSize.x, 1.0 - (p.y + 0.5) / texSize.y);
        }

        vec4 sampleAtlasFrame(int i, float frameIndex) {
            float W = atlasSize.x;
            float H = atlasSize.y;
            float ax = mod(float(i), W);
            float ay = floor(float(i) / W);
            if (ay >= H) return vec4(0.0);
            float ly = atlasFlipY ? (H - 1.0 - ay) : ay;

            if (atlasSourceMode == 0) {
                // Base-frame mode
                return texture2D(atlasTex, pixelToUV(vec2(ax, ly), atlasSize));
            }

            // Spritesheet tile selection
            vec2 grid = max(atlasFrameGrid, vec2(1.0));
            vec2 fsz  = max(atlasFrameSize, vec2(1.0));
            float fc = max(atlasFrameCount, 1.0);
            float f = frameIndex;
            if (atlasLoop > 0.5) f = mod(f, fc); else f = clamp(f, 0.0, fc - 1.0);
            float fx = mod(f, grid.x);
            float fy = floor(f / grid.x);

            float localY = atlasFlipY ? (fsz.y - 1.0 - ay) : ay;
            vec2 p = vec2(fx * fsz.x + ax, fy * fsz.y + localY);
            return texture2D(atlasTex, pixelToUV(p, atlasSize));
        }

        vec3 applyAtlasExpr(vec3 c, float id, float frameF) {
            if (atlasExprEnabled < 0.5) return c;
            vec3 outc = c;
            outc = outc * atlasExprMul + atlasExprAdd;
            outc = pow(max(outc, vec3(0.0)), atlasExprGamma);
            outc = clamp(outc, atlasExprClampMin, atlasExprClampMax);
            float phase = atlasExprPhase;
            phase += atlasExprTimeScale * (time * 0.001);
            phase += atlasExprIdScale * id;
            phase += atlasExprFrameScale * frameF;
            float s = sin(phase * atlasExprSinFreq);
            outc += atlasExprSinAmp * s;
            return clamp(outc, 0.0, 1.0);
        }

        varying float alpha;
        varying vec3 col;
        varying float shouldDiscard;

        void main() {
            int i = gl_VertexID;
            shouldDiscard = 0.0;
            if (float(i) >= activeCount) {
                shouldDiscard = 1.0;
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                alpha = 0.0;
                col = vec3(0.0);
                return;
            }

	            // Respect atlasCount (valid pixels) when provided.
	            if (atlasCount > 0.5 && float(i) >= atlasCount) {
	                shouldDiscard = 1.0;
	                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
	                alpha = 0.0;
	                col = vec3(0.0);
	                return;
	            }

            ivec2 size = ivec2(${NUM_X}, ${NUM_Y});
            vec2 uv = uvFromIndex(i, size);

            // Sample atlas (one physical frame) and then modulate to form the virtual frame.
            vec4 tex0 = sampleAtlasFrame(i, (atlasSourceMode == 0) ? 0.0 : iFrame);
            vec4 texM = (atlasMetaFromFrame0 > 0.5) ? sampleAtlasFrame(i, 0.0) : tex0;

            vec3 c = applyAtlasExpr(tex0.rgb, float(i), iFrame);
            vec3 p0 = atlasPosMin + c * (atlasPosMax - atlasPosMin);

            // Apply per-frame packing transform (instance attributes)
            vec3 ps = p0 * iScale;
            vec3 pr = quatRotate(iQuat, ps);
            vec3 pPacked = pr + iTranslate;

            // Used/unus... (based on i < atlasCount)
            float used = step(float(i) + 0.5, atlasCount);
            float metaByte = floor(texM.a * 255.0 + 0.5);
            float speedBucket = floor(metaByte / 32.0);
            float dualTagged = step(4.0, speedBucket);
            float doDual = (enableDualGroupOverlap ? 1.0 : 0.0) * dualTagged * used;

            float layerLow = mod(metaByte, 64.0);
            float high2 = floor(metaByte / 64.0);
            float layerApprox = layerLow + high2 * 64.0;
            float colorIndex = mod(layerLow, 8.0);
            float denom = max(atlasLayerMax, 1.0);
            float accuracy01 = clamp(layerApprox / denom, 0.0, 1.0);

            mat3 RI = rotXYZ(instanceGroupEuler);
            mat3 RU = rotXYZ(unusedGroupEuler);

            vec3 pInst = RI * pPacked + instanceOffset;
            vec3 pUn   = RU * pPacked + unusedOffset;
            vec3 pDual = RU * (RI * pPacked) + instanceOffset + unusedOffset;

            vec3 p = mix(pUn, pInst, used);
            p = mix(p, pDual, doDual);

            // Random point size/alpha
            float n = hash12(vec2(float(i), 0.0));
            float psiz = pow(n, 2.0) * 2.0;
            alpha = 0.2 + pow(n, 20.0) * 0.3;
            if (n > 0.999) { psiz *= 1.1; alpha *= 2.0; }

            // NIfTI-based color
            vec3 niftiColor = vec3(0.0);
            float albedo = 0.0;
            if (useNiftiColors && niftiWeight > 0.0) {
                vec4 data = texture2D(niftiRGBA, uv);
                niftiColor = data.rgb;
                albedo = data.a;
            }

            // Original velocity ramp not available here (no vel texture); fallback to grayscale by membership
            vec3 originalColor = mix(vec3(0.2), vec3(1.0), used);

            if (useNiftiColors && albedo > albedoThreshold && albedo != 0.0) {
                float lerpFactor = albedo <= colorLerpThreshold ? albedo / colorLerpThreshold : 1.0;
                col = mix(originalColor * maxValue, niftiColor, lerpFactor);
                alpha *= (0.7 + (albedo / maxValue) * 0.3);
            } else {
                col = originalColor * maxValue;
            }

            // Override particle color from atlas sidecar
            if (atlasColorEnabled && atlasColorMode > 0) {
                vec3 cc = mix(atlasColorUnused, atlasColorUsed, used);
                if (atlasColorMode == 3) {
                    cc = atlasColorUsed;
                } else if (atlasColorMode == 2) {
                    int ci = int(colorIndex + 0.5);
                    if (ci == 0) cc = atlasPalette0;
                    else if (ci == 1) cc = atlasPalette1;
                    else if (ci == 2) cc = atlasPalette2;
                    else if (ci == 3) cc = atlasPalette3;
                    else if (ci == 4) cc = atlasPalette4;
                    else if (ci == 5) cc = atlasPalette5;
                    else if (ci == 6) cc = atlasPalette6;
                    else cc = atlasPalette7;
                }
                col = cc;
            }

            // Depth coloring (supports view- and world-space metrics, including loadAccuracy)
            vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
            float viewZ = -mvPosition.z;

            vec3 worldP = (modelMatrix * vec4(p, 1.0)).xyz;
            float metric = viewZ;
            if (depthMetricMode == 1) {
                metric = length(worldP - cameraPosition);
            } else if (depthMetricMode == 2) {
                metric = worldP.y;
            } else if (depthMetricMode == 3) {
                metric = worldP.z;
            } else if (depthMetricMode == 4) {
                metric = accuracy01;
            }

            float span = depthRange;
            if (span <= 0.0) {
                if (depthMetricMode <= 1) span = (cameraFar - cameraNear);
                else span = 1.0;
            }
            span = max(span, 1e-6);
            float depth01 = clamp((metric - depthStart) / span, 0.0, 1.0);
            if (depthInvert) depth01 = 1.0 - depth01;

            bool doDepth = (used > 0.5 && depthColorInstance) || (used < 0.5 && depthColorUnused);
            if (doDepth) {
                float k = depth01;
                col *= k;
                alpha *= k;
            }

            gl_PointSize = psiz;
            gl_Position = projectionMatrix * mvPosition;
        }
    `;

    const composeFragmentShader = `
        uniform float maxValue;
        varying float alpha;
        varying vec3 col;
        varying float shouldDiscard;
        void main() {
            if (shouldDiscard > 0.0) discard;
            vec3 normalizedCol = col / maxValue;
            gl_FragColor = vec4(normalizedCol, alpha);
        }
    `;

    const material = new t.ShaderMaterial({
        uniforms: {
            time:               { value: 0.0 },
            albedoThreshold:    { value: albedoThreshold },
            colorLerpThreshold: { value: colorLerpThreshold },
            useNiftiColors:     { value: useNiftiColors },
            maxValue:           { value: 255.0 },

            cameraNear:         { value: 0.1 },
            cameraFar:          { value: 2000.0 },
            depthRange:         { value: 2000.0 },
            depthStart:         { value: 0.1 },
            depthMetricMode:    { value: 0 },
            depthInvert:        { value: false },
            depthColorInstance: { value: false },
            depthColorUnused:   { value: false },

            atlasColorEnabled:  { value: false },
            atlasColorMode:     { value: 0 },
            atlasLayerMax:      { value: 255.0 },
            atlasColorUsed:     { value: new t.Vector3(255, 255, 255) },
            atlasColorUnused:   { value: new t.Vector3(32, 32, 32) },
            atlasPalette0:      { value: new t.Vector3(255, 0, 0) },
            atlasPalette1:      { value: new t.Vector3(0, 255, 0) },
            atlasPalette2:      { value: new t.Vector3(0, 0, 255) },
            atlasPalette3:      { value: new t.Vector3(255, 255, 0) },
            atlasPalette4:      { value: new t.Vector3(255, 0, 255) },
            atlasPalette5:      { value: new t.Vector3(0, 255, 255) },
            atlasPalette6:      { value: new t.Vector3(255, 255, 255) },
            atlasPalette7:      { value: new t.Vector3(128, 128, 128) },

            niftiRGBA:          { value: null },
            niftiWeight:        { value: 0.0 },

            instanceOffset:     { value: new t.Vector3(0, 0, 0) },
            unusedOffset:       { value: new t.Vector3(0, 0, 0) },
            niftiPointCount:    { value: 0.0 },
            enableDualGroupOverlap: { value: false },

            instanceGroupEuler: { value: new t.Vector3(0, 0, 0) },
            unusedGroupEuler:   { value: new t.Vector3(0, 0, 0) },

            // Atlas
            atlasTex:           { value: null },
            atlasSize:          { value: new t.Vector2(1, 1) },
            atlasPosMin:        { value: new t.Vector3(0, 0, 0) },
            atlasPosMax:        { value: new t.Vector3(1, 1, 1) },
            atlasCount:         { value: 0.0 },
            atlasFlipY:         { value: false },
            atlasSourceMode:    { value: 0 },
            atlasFrameCount:    { value: 1.0 },
            atlasFrameGrid:     { value: new t.Vector2(1, 1) },
            atlasFrameSize:     { value: new t.Vector2(1, 1) },
            atlasLoop:          { value: 1.0 },
            atlasInterpolate:   { value: 0.0 },
            atlasMetaFromFrame0:{ value: 1.0 },

            atlasExprEnabled:   { value: 0.0 },
            atlasExprMul:       { value: new t.Vector3(1, 1, 1) },
            atlasExprAdd:       { value: new t.Vector3(0, 0, 0) },
            atlasExprGamma:     { value: new t.Vector3(1, 1, 1) },
            atlasExprClampMin:  { value: new t.Vector3(0, 0, 0) },
            atlasExprClampMax:  { value: new t.Vector3(1, 1, 1) },
            atlasExprSinAmp:    { value: 0.0 },
            atlasExprSinFreq:   { value: Math.PI * 2 },
            atlasExprPhase:     { value: 0.0 },
            atlasExprTimeScale: { value: 0.0 },
            atlasExprIdScale:   { value: 0.0 },
            atlasExprFrameScale:{ value: 0.0 },

            activeCount:        { value: (NUM_X * NUM_Y) }
        },
        vertexShader: composeVertexShader,
        fragmentShader: composeFragmentShader,
        transparent: true,
        depthTest: true,
        depthWrite: false
    });

    return material;
}

export { createComposeMaterial };