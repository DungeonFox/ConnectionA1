// coupledShadersContract_dna_math.js
// COMPLETE ARCHITECTURE IMPLEMENTATION
// MDPI Photonics 2020 helix math + Dendritic spine calcium dynamics

import { hash12 } from './glslNoise.js';
import { PI } from './glslUtils.js';

// ==================== ARCHITECTURE: SF_SpineDynamics ====================
// Calcium compartment ODE with influx/pump/diffusion
// Bell-shaped Ca-dependent growth/zip pressure

export function createChemShader() {
  return /* glsl */`
    ${PI}
    ${hash12}

    uniform float time;
    uniform float dt;

    uniform float nodeCount;
    uniform float neckSeg;
    uniform float headCount;
    uniform float secondEnabled;

    uniform float ds;
    uniform float pitch;
    uniform float qPitch;
    uniform float zipMode;

    uniform sampler2D extPos;
    uniform float extSamples;
    uniform float extRadius;

    // MDPI Helix Parameters
    uniform float helixR;
    uniform float cotAlpha;
    uniform float alphaExp;
    uniform float u_s;
    uniform float alpha0;

    vec2 uvFromIndex(float idx){
      float x = mod(idx, resolution.x);
      float y = floor(idx / resolution.x);
      return (vec2(x, y) + 0.5) / resolution.xy;
    }

    vec4 readPos(float idx){ return texture2D(pos, uvFromIndex(idx)); }
    vec4 readChem(float idx){ return texture2D(chem, uvFromIndex(idx)); }

    // Find nearest external particle for spine-dendrite coupling
    vec4 nearestExt(vec3 p, float seed){
      float dMin = 1e9;
      vec3 dir = vec3(0.0, 0.0, 1.0);
      for (int k = 0; k < 32; k++){
        if (float(k) >= extSamples) break;
        float a = hash12(vec2(seed + float(k) * 19.13, 13.37 + time * 0.0007));
        float b = hash12(vec2(seed + float(k) * 91.17, 42.42 + time * 0.0009));
        vec2 ruv = vec2(a, b);
        vec4 q4 = texture2D(extPos, ruv);
        float qUsed = step(0.5, floor(q4.w + 1e-4));
        if (qUsed < 0.5) continue;
        vec3 dq = q4.xyz - p;
        float d = length(dq);
        if (d < dMin){
          dMin = d;
          dir = dq / max(d, 1e-6);
        }
      }
      return vec4(dMin, dir);
    }

    // Bell-shaped function for Ca-dependent growth
    // Smooth rise between [a0,a1], plateau, fall between [b0,b1]
    float bell(float x, float a0, float a1, float b0, float b1){
      return smoothstep(a0, a1, x) * (1.0 - smoothstep(b0, b1, x));
    }

    // ==================== ARCHITECTURE: SD_Compartment ====================
    // Ca ODE: dCa/dt = influx - Ca/τ_pump - k_diff*(Ca - Ca_dend)
    float updateCalcium(float Ca, float CaDend, float inflow, float dt){
      float tauPump = 0.35;     // seconds-ish
      float kPump = 1.0 / max(tauPump, 1e-3);
      float kDiff = 0.75;       // neck conductance proxy
      
      Ca += dt * (2.0 * inflow);      // Influx from external proximity
      Ca -= dt * (kPump * Ca);        // Pump/decay
      Ca -= dt * (kDiff * (Ca - CaDend)); // Diffusion to dendrite
      
      return clamp(Ca, 0.0, 1.2);
    }

    // ==================== ARCHITECTURE: SD_Bimodal ====================
    // Map Ca to growth/zip pressure using bell-shaped rule
    // Moderate Ca (0.12-0.55) promotes growth/stability
    // High Ca (>0.70) promotes shrink/collapse (unzip pressure)
    vec2 mapCaToGrowthZip(float Ca){
      float CaGrow = bell(Ca, 0.12, 0.30, 0.55, 0.85);
      float CaShrink = smoothstep(0.70, 0.95, Ca);
      return vec2(CaGrow, CaShrink);
    }

    void main(){
      vec2 frag = floor(gl_FragCoord.xy);
      float i = frag.y * resolution.x + frag.x;

      float N = nodeCount;
      float neck = neckSeg;
      float head = headCount;
      float perSpine = neck + head;

      float idxStrandA0 = N;
      float idxStrandB0 = N + N;
      float idxSpineP0  = N + 2.0*N;
      float spinePCount = N * perSpine;
      float idxSpineS0  = idxSpineP0 + spinePCount;
      float spineSCount = N * perSpine * secondEnabled;
      float activeEnd   = idxSpineS0 + spineSCount;

      vec4 c = texture2D(chem, gl_FragCoord.xy / resolution.xy);

      // Inactive particles decay
      if (i >= activeEnd){
        c *= exp(-dt * 2.0);
        gl_FragColor = c;
        return;
      }

      // ==================== GREEN NODES (Backbone) ====================
      // chem.x = g (growth/activation 0..1)
      // chem.y = φ (helix phase radians)
      // chem.z = gGap (opening/separation scalar)
      // chem.w = gRot (extra twist increment)
      if (i < N){
        float k = i;

        if (k < 0.5){
          // Seed node
          c.x = 1.0;
          c.y = 0.0;
          c.z = 0.0;
          c.w = 0.0;
          gl_FragColor = c;
          return;
        }

        // Gate growth from previous node
        vec4 cPrev = readChem(k - 1.0);
        float gate = smoothstep(0.25, 0.85, cPrev.x);

        // Read local spine calcium (hub is at tipIdxP when neck=1)
        float tipIdxP = idxSpineP0 + k * perSpine + (neck - 1.0);
        float CaLocal = readChem(tipIdxP).x;

        // SD_Bimodal: Map Ca to growth/zip pressure
        vec2 caMap = mapCaToGrowthZip(CaLocal);
        float CaGrow = caMap.x;
        float CaShrink = caMap.y;

        // SD_GrowthGreen: Update activation
        float gTarget = gate;
        float gSpeed = 0.55 + 0.75 * CaGrow;
        c.x += dt * (gTarget - c.x) * gSpeed;
        c.x -= dt * (0.35 * CaShrink) * c.x;
        c.x = clamp(c.x, 0.0, 1.0);

        // SD_UpdateGapRot: Gap target from zipMode + Ca pressure
        // Zipped -> gap≈0, Unzipped -> gap≈1.2
        // High Ca adds unzip pressure
        float gapClosed = 0.0;
        float gapOpen = 1.2;
        float baseGapTarget = mix(gapOpen, gapClosed, zipMode);
        float gapTarget = mix(baseGapTarget, gapOpen, CaShrink);
        c.z += dt * gate * (gapTarget - c.z) * 1.25;
        c.z = clamp(c.z, 0.0, 1.6);

        // Extra twist increment from external torque
        float torque = 0.0;
        if (extSamples > 0.5){
          vec3 p0 = readPos(k).xyz;
          vec3 pm = readPos(k - 1.0).xyz;
          vec3 pp = readPos(min(k + 1.0, N - 1.0)).xyz;
          vec3 t = normalize(pp - pm + vec3(1e-6));
          vec4 ne = nearestExt(p0, k * 37.7 + 11.0);
          vec3 dir = ne.yzw;
          torque = dot(cross(t, dir), t);
        }
        float wTarget = 0.15 * torque;
        c.w += dt * gate * (wTarget - c.w) * 0.9;
        c.w *= exp(-dt * 0.35);
        c.w = clamp(c.w, -0.25, 0.25);

        // Helix phase propagation: φ_k = φ_{k-1} + q*ds + gRot
        float dphi = qPitch * ds + c.w;
        c.y = mod(cPrev.y + dphi, 2.0 * PI);

        gl_FragColor = c;
        return;
      }

      // ==================== SPINE/HUB CHEM (Calcium Compartment) ====================
      float kNode = -1.0;
      float local = -1.0;
      float isTip = 0.0;

      if (i >= idxSpineP0 && i < idxSpineP0 + spinePCount){
        float rel = i - idxSpineP0;
        kNode = floor(rel / perSpine);
        local = rel - kNode * perSpine;
        isTip = step(neck - 1.5, local) * step(local, neck - 0.5);
      }

      if (isTip > 0.5){
        vec3 p = texture2D(pos, gl_FragCoord.xy / resolution.xy).xyz;

        // SD_Compartment: Calcium influx from external proximity
        vec4 ne = nearestExt(p, i * 13.1 + 7.0);
        float d = ne.x;
        float inflow = 1.0 - smoothstep(extRadius * 0.35, extRadius, d);
        inflow *= step(0.5, extSamples);

        // Couple to dendrite (green node activation)
        float CaDend = readChem(kNode).x;

        // Update calcium ODE
        float Ca = updateCalcium(c.x, CaDend, inflow, dt);
        
        gl_FragColor = vec4(Ca, c.y, c.z, c.w);
        return;
      }

      c *= exp(-dt * 1.0);
      gl_FragColor = c;
    }
  `;
}

// ==================== ARCHITECTURE: SF_HelixGenerator + SF_FrameTransport ====================
// MDPI-compliant helix generation with parallel transport frame

export function createCoupledPosTargetShader() {
  return /* glsl */`
    ${PI}
    ${hash12}

    uniform float time;
    uniform float dt;

    uniform float nodeCount;
    uniform float neckSeg;
    uniform float headCount;
    uniform float secondEnabled;

    uniform float ds;
    uniform float helixR;
    uniform float pitch;
    uniform float qPitch;
    uniform float axialShift;

    uniform sampler2D extPos;
    uniform float extSamples;
    uniform float extRadius;

    uniform vec3 wellOrigin;
    uniform vec3 unusedOffset;

    uniform float flowEnabled;
    uniform float flowSpeed;
    uniform float flowRad;
    uniform float electricityJitter;
    uniform float pulseFrequency;
    uniform float pulseSpeed;
    uniform float zipMode;

    // MDPI Parameters
    uniform float cotAlpha;
    uniform float alphaExp;
    uniform float u_s;
    uniform float alpha0;

    vec2 uvFromIndex(float idx){
      float x = mod(idx, resolution.x);
      float y = floor(idx / resolution.x);
      return (vec2(x, y) + 0.5) / resolution.xy;
    }

    vec4 readPos(float idx){ return texture2D(pos, uvFromIndex(idx)); }
    vec4 readChem(float idx){ return texture2D(chem, uvFromIndex(idx)); }

    // ==================== ARCHITECTURE: FT_Transport ====================
    // Parallel transport frame (rotation-minimizing Bishop frame)
    void makeFrame(vec3 t, out vec3 N, out vec3 B){
      // Deterministic initial normal
      vec3 up = vec3(0.0, 1.0, 0.0);
      if (abs(dot(up, t)) > 0.92) up = vec3(1.0, 0.0, 0.0);
      N = normalize(cross(up, t));
      B = normalize(cross(t, N));
    }

    // FT_ReOrtho: Re-orthonormalize frame
    void orthonormalize(inout vec3 N, inout vec3 B, vec3 T){
      N = normalize(N - dot(N, T) * T);
      B = normalize(cross(T, N));
    }

    vec3 getWellPosition(float i){
      float seed = i * 17.17 + 13.37;
      float r = hash12(vec2(seed, 1.1)) * flowRad * 0.5;
      float theta = hash12(vec2(seed, 2.2)) * 2.0 * PI;
      float phi = hash12(vec2(seed, 3.3)) * PI;
      vec3 offset;
      offset.x = r * sin(phi) * cos(theta);
      offset.y = r * sin(phi) * sin(theta);
      offset.z = r * cos(phi);
      return wellOrigin + offset;
    }

    // ==================== ARCHITECTURE: RT_Waypts + RT_Advance ====================
    // Yellow highway routing: origin -> base -> hub -> dest
    vec3 routeViaYellow(float i, float kNode, vec3 dest, float queue01){
      float seed = i * 17.17 + 13.37;
      float t = time * flowSpeed;
      float phase = fract(t * 0.08 + hash12(vec2(seed, 5.1)));

      vec3 origin = getWellPosition(i);
      vec3 base = readPos(kNode).xyz;

      float Nn = nodeCount;
      float neck = neckSeg;
      float head = headCount;
      float perSpine = neck + head;
      float idxSpineP0 = Nn + 2.0 * Nn;
      float tipIdxP = idxSpineP0 + kNode * perSpine + (neck - 1.0);
      vec3 hub = readPos(tipIdxP).xyz;

      vec3 p = origin;
      if (phase < 0.35){
        float u = smoothstep(0.0, 0.35, phase);
        p = mix(origin, base, u);
      } else if (phase < 0.7){
        float u = smoothstep(0.35, 0.7, phase);
        p = mix(base, hub, u);
      } else {
        float u = smoothstep(0.7, 1.0, phase);
        p = mix(hub, dest, u);
      }
      return mix(p, dest, clamp(queue01, 0.0, 1.0));
    }

    void main(){
      vec2 frag = floor(gl_FragCoord.xy);
      float i = frag.y * resolution.x + frag.x;

      float Nn = nodeCount;
      float neck = neckSeg;
      float head = headCount;
      float perSpine = neck + head;

      float idxStrandA0 = Nn;
      float idxStrandB0 = Nn + Nn;
      float idxSpineP0  = Nn + 2.0*Nn;
      float spinePCount = Nn * perSpine;
      float idxSpineS0  = idxSpineP0 + spinePCount;
      float spineSCount = Nn * perSpine * secondEnabled;
      float activeEnd   = idxSpineS0 + spineSCount;

      // Grey rungs
      float idxRung0 = activeEnd;
      float segPerConn = 2.0;
      float connPerNode = 2.0;
      float rungCount = Nn * connPerNode * segPerConn;

      vec3 outPos = getWellPosition(i);
      float membership = 0.0;
      float meta = 7.0;

      // ==================== ARCHITECTURE: TR_Rungs ====================
      if (i >= idxRung0 && i < idxRung0 + rungCount) {
        float j = i - idxRung0;
        float perNode = connPerNode * segPerConn;
        float k = floor(j / perNode);
        float r = j - k * perNode;
        float side = floor(r / segPerConn);
        float seg  = r - side * segPerConn;

        vec4 ck = readChem(k);
        float g = ck.x;
        float gGap = ck.z;

        // ZS_Zippedness: Compute local zip from global mode and gap
        float localZip = zipMode * (1.0 - smoothstep(0.30, 1.20, gGap));
        localZip = clamp(localZip, 0.0, 1.0);

        // Get backbone frame
        vec3 p0 = readPos(k).xyz;
        vec3 pm = readPos(max(k - 1.0, 0.0)).xyz;
        vec3 t = normalize(p0 - pm + vec3(0.0, 0.0, 1e-6));
        if (k < 0.5) t = vec3(0.0, 0.0, 1.0);
        vec3 N, B;
        makeFrame(t, N, B);
        orthonormalize(N, B, t);

        // MDPI Helix geometry with gap-modulated radius
        float qP = qPitch;
        float phi = ck.y;
        float gapPhase = 0.35 * PI * clamp(gGap, 0.0, 1.6);
        float R = helixR + gGap;

        // Axial shift implies phase shift: Δθ = q*Δx
        float dThetaAxial = qP * axialShift;

        // MDPI Eq. (12): Strand A and B with phase offset
        float phiA = phi - 0.5 * gapPhase;
        float phiB = phi + PI + dThetaAxial + 0.5 * gapPhase;

        vec3 baseA = p0 - 0.5 * t * axialShift;
        vec3 baseB = p0 + 0.5 * t * axialShift;

        vec3 endA = baseA + R * (cos(phiA) * N + sin(phiA) * B);
        vec3 endB = baseB + R * (cos(phiB) * N + sin(phiB) * B);

        float tipIdxP = idxSpineP0 + k * perSpine + (neck - 1.0);
        vec3 hub = readPos(tipIdxP).xyz;

        vec3 endP = mix(endA, endB, side);
        float frac = (seg + 1.0) / (segPerConn + 1.0);

        outPos = mix(hub, endP, frac);
        membership = step(0.05, g) * smoothstep(0.05, 0.25, localZip);
        meta = 7.0;
        gl_FragColor = vec4(outPos, membership + meta / 256.0);
        return;
      }

      if (i >= idxRung0 + rungCount){
        gl_FragColor = vec4(outPos, membership + meta/256.0);
        return;
      }

      // ==================== GREEN NODES ====================
      if (i < Nn){
        float k = i;
        vec4 ck = readChem(k);
        float g = ck.x;

        membership = step(0.05, g);
        meta = 1.0;

        if (k < 0.5){
          outPos = vec3(0.0);
          gl_FragColor = vec4(outPos, 1.0 + meta/256.0);
          return;
        }

        if (membership < 0.5){
          gl_FragColor = vec4(outPos, membership + meta/256.0);
          return;
        }

        // Forward growth along tangent
        vec3 pPrev = readPos(k - 1.0).xyz;
        vec3 pPrev2 = readPos(max(k - 2.0, 0.0)).xyz;
        vec3 t = normalize(pPrev - pPrev2 + vec3(0.0, 0.0, 1e-4));
        outPos = pPrev + t * ds;
        gl_FragColor = vec4(outPos, membership + meta/256.0);
        return;
      }

      // ==================== CYAN STRAND A ====================
      if (i >= idxStrandA0 && i < idxStrandA0 + Nn){
        float k = i - idxStrandA0;
        vec4 ck = readChem(k);
        float g = ck.x;
        float gGap = ck.z;
        float phi = ck.y;

        float localZip = zipMode * (1.0 - smoothstep(0.30, 1.20, gGap));
        localZip = clamp(localZip, 0.0, 1.0);

        membership = step(0.05, g);
        meta = 5.0;

        vec3 p0 = readPos(k).xyz;
        vec3 pm = readPos(max(k - 1.0, 0.0)).xyz;
        vec3 t = normalize(p0 - pm + vec3(0.0, 0.0, 1e-6));
        if (k < 0.5) t = vec3(0.0, 0.0, 1.0);
        vec3 N, B;
        makeFrame(t, N, B);
        orthonormalize(N, B, t);

        // MDPI: Strand A with gap phase
        float gapPhase = 0.35 * PI * clamp(gGap, 0.0, 1.6);
        float R = helixR + gGap;
        float phiA = phi - 0.5 * gapPhase;
        vec3 baseA = p0 - 0.5 * t * axialShift;
        vec3 dest = baseA + R * (cos(phiA) * N + sin(phiA) * B);

        if (membership < 0.5){
          if (flowEnabled > 0.5){
            float prevG = readChem(max(k - 1.0, 0.0)).x;
            float queue01 = smoothstep(0.25, 0.85, prevG);
            outPos = routeViaYellow(i, k, dest, queue01);
          } else {
            outPos = getWellPosition(i);
          }
          gl_FragColor = vec4(outPos, membership + meta/256.0);
          return;
        }

        outPos = dest;
        gl_FragColor = vec4(outPos, membership + meta/256.0);
        return;
      }

      // ==================== MAGENTA STRAND B ====================
      if (i >= idxStrandB0 && i < idxStrandB0 + Nn){
        float k = i - idxStrandB0;
        vec4 ck = readChem(k);
        float g = ck.x;
        float gGap = ck.z;
        float phi = ck.y;

        membership = step(0.05, g);
        meta = 4.0;

        vec3 p0 = readPos(k).xyz;
        vec3 pm = readPos(max(k - 1.0, 0.0)).xyz;
        vec3 t = normalize(p0 - pm + vec3(0.0, 0.0, 1e-6));
        if (k < 0.5) t = vec3(0.0, 0.0, 1.0);
        vec3 N, B;
        makeFrame(t, N, B);
        orthonormalize(N, B, t);

        // MDPI Eq. (12): Strand B with axial shift phase offset
        float qP = qPitch;
        float gapPhase = 0.35 * PI * clamp(gGap, 0.0, 1.6);
        float R = helixR + gGap;
        float dThetaAxial = qP * axialShift;

        float phiB = phi + PI + dThetaAxial + 0.5 * gapPhase;
        vec3 baseB = p0 + 0.5 * t * axialShift;
        vec3 dest = baseB + R * (cos(phiB) * N + sin(phiB) * B);

        if (membership < 0.5){
          if (flowEnabled > 0.5){
            float prevG = readChem(max(k - 1.0, 0.0)).x;
            float queue01 = smoothstep(0.25, 0.85, prevG);
            outPos = routeViaYellow(i, k, dest, queue01);
          } else {
            outPos = getWellPosition(i);
          }
          gl_FragColor = vec4(outPos, membership + meta/256.0);
          return;
        }

        outPos = dest;
        gl_FragColor = vec4(outPos, membership + meta/256.0);
        return;
      }

      // ==================== YELLOW HUB ====================
      if (i >= idxSpineP0 && i < idxSpineP0 + spinePCount){
        float rel = i - idxSpineP0;
        float kNode = floor(rel / perSpine);
        float local = rel - kNode * perSpine;

        vec4 cNode = readChem(kNode);
        float g = cNode.x;
        float gGap = cNode.z;
        float phi = cNode.y;

        membership = step(0.05, g);
        meta = 3.0;

        if (membership < 0.5){
          gl_FragColor = vec4(outPos, membership + meta/256.0);
          return;
        }

        vec3 p0 = readPos(kNode).xyz;
        vec3 pm = readPos(max(kNode - 1.0, 0.0)).xyz;
        vec3 t = normalize(p0 - pm + vec3(0.0, 0.0, 1e-6));
        if (kNode < 0.5) t = vec3(0.0, 0.0, 1.0);
        vec3 N, B;
        makeFrame(t, N, B);
        orthonormalize(N, B, t);

        // MDPI: Hub at midpoint of strands
        float qP = qPitch;
        float gapPhase = 0.35 * PI * clamp(gGap, 0.0, 1.6);
        float R = helixR + gGap;
        float dThetaAxial = qP * axialShift;

        float phiA = phi - 0.5 * gapPhase;
        float phiB = phi + PI + dThetaAxial + 0.5 * gapPhase;

        vec3 baseA = p0 - 0.5 * t * axialShift;
        vec3 baseB = p0 + 0.5 * t * axialShift;
        vec3 endA = baseA + R * (cos(phiA) * N + sin(phiA) * B);
        vec3 endB = baseB + R * (cos(phiB) * N + sin(phiB) * B);

        outPos = 0.5 * (endA + endB);
        gl_FragColor = vec4(outPos, membership + meta/256.0);
        return;
      }

      gl_FragColor = vec4(outPos, membership + meta/256.0);
    }
  `;
}