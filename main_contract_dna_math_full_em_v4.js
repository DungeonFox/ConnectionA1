import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import GPUComputationRenderer from './GPUComputationRenderer.js';
import { createPosTargetShader, createAccShader, createVelShader, createPosShader, createTagsShader } from './computeShaders.js';
import { createPointsMaterial } from './materials.js';
import { 
  createChemShader, 
  createFlowStateShader,
  createCoupledPosTargetShader,
  normalizeVec3,
  inferAlphaHatFromForceComponents
} from './coupledShadersContract_dna_math_full_em_v4.js';
import { createAcceptanceValidationRunner } from './validation/acceptance.js';
import { applyMovementUniforms, stepMovementCompute, bindMovementTextures } from './particleMovement.js';

// ==================== ARCHITECTURE: SF_StateSchema ====================
const TEX_SIZE = 64;
const COUNT = TEX_SIZE * TEX_SIZE;

// Layout knobs
const NODE_COUNT = 128;
const NECK_SEG = 1;
const HEAD_COUNT = 0;
const SECOND_ENABLED = 0.0;

// Derived indices
const PER_SPINE = NECK_SEG + HEAD_COUNT;
const SPINE_P_COUNT = NODE_COUNT * PER_SPINE;
const SPINE_S_COUNT = (SECOND_ENABLED > 0.5) ? (NODE_COUNT * PER_SPINE) : 0;
const IDX_RUNG0 = (NODE_COUNT + 2 * NODE_COUNT) + SPINE_P_COUNT + SPINE_S_COUNT;
const SEG_PER_CONN = 2;
const CONN_PER_NODE = 2;
const RUNG_COUNT = NODE_COUNT * CONN_PER_NODE * SEG_PER_CONN;
const RENDER_COUNT = IDX_RUNG0 + RUNG_COUNT;

// ==================== ARCHITECTURE: SF_HelixGenerator ====================
// MDPI Helix Parameters
const DS = 1.0;
const HELIX_R = 2.2;
const PITCH = 7.5;
const AXIAL_SHIFT = 0.5 * HELIX_R;

// Derived MDPI parameters
const Q_PITCH = 2.0 * Math.PI / PITCH;
const COT_ALPHA = (2.0 * Math.PI * HELIX_R) / PITCH;
const ALPHA_EXP = Math.atan(PITCH / (2.0 * Math.PI * HELIX_R));
const U_S = 0.5 * COT_ALPHA;
const ALPHA_0 = 38.0 * Math.PI / 180.0;

const HELIX_CONVENTION = Object.freeze({
  handednessSign: 1.0,
  strandAPhaseOffset: 0.5 * Math.PI,
  strandBPhaseOffset: 1.5 * Math.PI,
  angleUnitScale: 1.0 // radians
});

// ==================== ARCHITECTURE: EM FIELD CONTROLS ====================
// EM (Electromagnetic) field state
const EM_STATE = {
  enabled: false,           // extEnabled
  radius: 15.0,            // extRadius - EM field capture radius
  k: 50.0,                // extK - EM force stiffness (repulsion strength)
  c: 10.0,                // extC - EM damping
  mode: 2.0,              // extMode - 2=helix-tunnel for DNA
  twist: 0.5,             // extTwist - helical EM component
  twistHz: 1.0,           // extTwistHz - EM oscillation frequency
  samples: 16.0           // extSamples - EM field samples
};

const SEEK_MODE_PRESETS = Object.freeze({
  manual: { seekStrength: 1.0, captureRadius: 15.0, torqueBias: 0.5, phaseLock: 0.25, damping: 1.0 },
  assist: { seekStrength: 0.8, captureRadius: 12.0, torqueBias: 0.35, phaseLock: 0.35, damping: 0.9 },
  aggressive: { seekStrength: 1.8, captureRadius: 20.0, torqueBias: 0.7, phaseLock: 0.2, damping: 0.55 },
  precision: { seekStrength: 0.6, captureRadius: 9.0, torqueBias: 0.2, phaseLock: 0.75, damping: 1.3 }
});

const SEEK_MODE_STATE = {
  mode: 'manual',
  ...SEEK_MODE_PRESETS.manual
};


const SEEK_POLICY_STATE = {
  seekWDistance: 1.0,
  seekWTangent: 0.6,
  seekWRadial: 0.4,
  seekWTorque: 0.5,
  seekMaxInfluence: 0.7
};


const MAX_SEEK_TARGETS = 8;
const SEEK_TARGET_STATE = {
  mode: 'off',
  armed: false,
  targetRadius: 20.0,
  targetWeight: 1.0,
  targets: []
};

let zipMode = 1.0;
let targetZipMode = 1.0;

const FLOW_TRANSPORT_STATE = {
  baseSpeed: 0.22,
  speedJitter: 0.12,
  flowScale: 1.0,
  transportGain: 0.35
};

const DEBUG_STATE = {
  mode: 0,
  showZipField: false,
  showCalcium: false,
  showEMField: false  // NEW: Visualize EM field
};

const RESIDUAL_STATE = {
  sampleEveryNFrames: 12,
  lastFrame: -1,
  sampleCount: 0,
  meanDeltaAlpha: 0,
  meanAbsDeltaAlpha: 0,
  rmsDeltaAlpha: 0,
  meanDeltaQ: 0,
  meanAbsDeltaQ: 0,
  rmsDeltaQ: 0,
  sampledNodes: 0
};

const chemReadbackBuffer = new Float32Array(TEX_SIZE * TEX_SIZE * 4);

const CALIBRATION_STATE = {
  alpha0Solved: ALPHA_0,
  alpha0Applied: ALPHA_0,
  alpha0ManualOffset: 0.0,
  calibrationEnabled: true,
  residualAtAlpha0: Number.NaN,
  converged: false,
  sampleEveryNFrames: 8,
  lastFrame: -1,
  samples: 0,
  iterations: 0,
  candidateCount: 0,
  minAlpha: 0,
  maxAlpha: 0
};

const posReadbackBuffer = new Float32Array(TEX_SIZE * TEX_SIZE * 4);
const extReadbackBuffer = new Float32Array(TEX_SIZE * TEX_SIZE * 4);

function applyAlpha0ToRuntime() {
  CALIBRATION_STATE.alpha0Applied = CALIBRATION_STATE.alpha0Solved + CALIBRATION_STATE.alpha0ManualOffset;
  sysA.chemVar.material.uniforms.alpha0.value = CALIBRATION_STATE.alpha0Applied;
  sysA.posTargetVar.material.uniforms.alpha0.value = CALIBRATION_STATE.alpha0Applied;
  sysA.accVar.material.uniforms.alpha0.value = CALIBRATION_STATE.alpha0Applied;
  sysA.architecture.alpha0Solved = CALIBRATION_STATE.alpha0Applied;
}

function solveAlpha0FromRadialEquilibrium() {
  const posRT = sysA.gpu.getCurrentRenderTarget(sysA.posVar);
  const extRT = sysB.gpu.getCurrentRenderTarget(sysB.posVar);
  renderer.readRenderTargetPixels(posRT, 0, 0, TEX_SIZE, TEX_SIZE, posReadbackBuffer);
  renderer.readRenderTargetPixels(extRT, 0, 0, TEX_SIZE, TEX_SIZE, extReadbackBuffer);

  const extActive = [];
  for (let i = 0; i < COUNT; i++) {
    const i4 = i * 4;
    const tag = Math.floor(extReadbackBuffer[i4 + 3] + 1e-4);
    if (tag < 1) continue;
    extActive.push([extReadbackBuffer[i4], extReadbackBuffer[i4 + 1], extReadbackBuffer[i4 + 2]]);
  }

  const idxSpineP0 = NODE_COUNT + 2 * NODE_COUNT;
  const alphaMin = 12.0 * Math.PI / 180.0;
  const alphaMax = 80.0 * Math.PI / 180.0;
  const candidateCount = 69;
  const candidateStep = (alphaMax - alphaMin) / (candidateCount - 1);
  const alphaCandidates = Array.from({ length: candidateCount }, (_, i) => alphaMin + i * candidateStep);
  const residualSums = new Float64Array(candidateCount);

  let sampleCount = 0;
  for (let k = 1; k < NODE_COUNT - 1; k++) {
    const k4 = k * 4;
    const p0 = [posReadbackBuffer[k4], posReadbackBuffer[k4 + 1], posReadbackBuffer[k4 + 2]];
    const km4 = (k - 1) * 4;
    const kp4 = (k + 1) * 4;
    const t = normalizeVec3([
      posReadbackBuffer[kp4] - posReadbackBuffer[km4],
      posReadbackBuffer[kp4 + 1] - posReadbackBuffer[km4 + 1],
      posReadbackBuffer[kp4 + 2] - posReadbackBuffer[km4 + 2]
    ]);
    if (!t) continue;

    const hubIdx = idxSpineP0 + k * PER_SPINE + (NECK_SEG - 1);
    const hub4 = hubIdx * 4;
    const radial = normalizeVec3([
      p0[0] - posReadbackBuffer[hub4],
      p0[1] - posReadbackBuffer[hub4 + 1],
      p0[2] - posReadbackBuffer[hub4 + 2]
    ]);
    if (!radial) continue;

    let nearest = null;
    let d2Min = Number.POSITIVE_INFINITY;
    for (const q of extActive) {
      const dx = q[0] - p0[0];
      const dy = q[1] - p0[1];
      const dz = q[2] - p0[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < d2Min) {
        d2Min = d2;
        nearest = [dx, dy, dz];
      }
    }
    if (!nearest) continue;

    const forceDir = normalizeVec3(nearest);
    if (!forceDir) continue;

    const components = inferAlphaHatFromForceComponents(forceDir, t);
    const fRadial = Math.abs(
      forceDir[0] * radial[0] +
      forceDir[1] * radial[1] +
      forceDir[2] * radial[2]
    );

    for (let c = 0; c < candidateCount; c++) {
      const alpha = alphaCandidates[c];
      const radialProxy = fRadial * Math.cos(alpha) - components.fParallel * Math.sin(alpha);
      residualSums[c] += radialProxy;
    }
    sampleCount += 1;
  }

  const residuals = alphaCandidates.map((_, i) => sampleCount > 0 ? residualSums[i] / sampleCount : Number.NaN);
  let bestIdx = 0;
  let bestAbs = Number.POSITIVE_INFINITY;
  for (let i = 0; i < residuals.length; i++) {
    const absVal = Math.abs(residuals[i]);
    if (absVal < bestAbs) {
      bestAbs = absVal;
      bestIdx = i;
    }
  }

  let converged = false;
  for (let i = 1; i < residuals.length; i++) {
    const a = residuals[i - 1];
    const b = residuals[i];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    if (a === 0.0 || b === 0.0 || (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)) {
      converged = true;
      break;
    }
  }

  CALIBRATION_STATE.alpha0Solved = alphaCandidates[bestIdx] ?? ALPHA_0;
  CALIBRATION_STATE.residualAtAlpha0 = residuals[bestIdx] ?? Number.NaN;
  CALIBRATION_STATE.converged = converged && sampleCount > 0;
  CALIBRATION_STATE.samples = sampleCount;
  CALIBRATION_STATE.iterations = candidateCount;
  CALIBRATION_STATE.candidateCount = candidateCount;
  CALIBRATION_STATE.minAlpha = alphaMin;
  CALIBRATION_STATE.maxAlpha = alphaMax;

  applyAlpha0ToRuntime();
}

function updateAlpha0Calibration() {
  if (!CALIBRATION_STATE.calibrationEnabled) {
    applyAlpha0ToRuntime();
    return;
  }
  if ((frame % CALIBRATION_STATE.sampleEveryNFrames) !== 0) return;
  solveAlpha0FromRadialEquilibrium();
  CALIBRATION_STATE.lastFrame = frame;
}

function updateResidualMetrics() {
  if ((frame % RESIDUAL_STATE.sampleEveryNFrames) !== 0) return;
  const rt = sysA.gpu.getCurrentRenderTarget(sysA.chemVar);
  renderer.readRenderTargetPixels(rt, 0, 0, TEX_SIZE, TEX_SIZE, chemReadbackBuffer);

  const idxSpineP0 = NODE_COUNT + 2 * NODE_COUNT;
  let count = 0;
  let sumAlpha = 0;
  let sumAbsAlpha = 0;
  let sumSqAlpha = 0;
  let sumQ = 0;
  let sumAbsQ = 0;
  let sumSqQ = 0;

  for (let k = 0; k < NODE_COUNT; k++) {
    const i = idxSpineP0 + k * PER_SPINE + (NECK_SEG - 1);
    const i4 = i * 4;
    const alphaHat = chemReadbackBuffer[i4 + 1];
    const qHat = chemReadbackBuffer[i4 + 2];
    if (!Number.isFinite(alphaHat) || !Number.isFinite(qHat)) continue;
    const deltaAlpha = alphaHat - ALPHA_EXP;
    const deltaQ = qHat - Q_PITCH;
    sumAlpha += deltaAlpha;
    sumAbsAlpha += Math.abs(deltaAlpha);
    sumSqAlpha += deltaAlpha * deltaAlpha;
    sumQ += deltaQ;
    sumAbsQ += Math.abs(deltaQ);
    sumSqQ += deltaQ * deltaQ;
    count += 1;
  }

  if (count === 0) return;
  RESIDUAL_STATE.lastFrame = frame;
  RESIDUAL_STATE.sampleCount += 1;
  RESIDUAL_STATE.sampledNodes = count;
  RESIDUAL_STATE.meanDeltaAlpha = sumAlpha / count;
  RESIDUAL_STATE.meanAbsDeltaAlpha = sumAbsAlpha / count;
  RESIDUAL_STATE.rmsDeltaAlpha = Math.sqrt(sumSqAlpha / count);
  RESIDUAL_STATE.meanDeltaQ = sumQ / count;
  RESIDUAL_STATE.meanAbsDeltaQ = sumAbsQ / count;
  RESIDUAL_STATE.rmsDeltaQ = Math.sqrt(sumSqQ / count);
}

const hud = document.getElementById('hud');
const seekPanel = document.getElementById('seek-panel');
const seekModeSelect = document.getElementById('seek-mode');
const seekModeReadout = document.getElementById('seek-mode-readout');
const seekResetButton = document.getElementById('seek-reset');
const seekTargetModeSelect = document.getElementById('seek-target-mode');
const seekArmPickButton = document.getElementById('seek-arm-pick');
const seekClearTargetsButton = document.getElementById('seek-clear-targets');
const seekTargetReadout = document.getElementById('seek-target-readout');
const seekTargetXInput = document.getElementById('seek-targetX');
const seekTargetYInput = document.getElementById('seek-targetY');
const seekTargetZInput = document.getElementById('seek-targetZ');
const seekApplyXYZButton = document.getElementById('seek-apply-xyz');

const seekControls = {
  seekStrength: document.getElementById('seek-seekStrength'),
  captureRadius: document.getElementById('seek-captureRadius'),
  torqueBias: document.getElementById('seek-torqueBias'),
  phaseLock: document.getElementById('seek-phaseLock'),
  damping: document.getElementById('seek-damping'),
  targetRadius: document.getElementById('seek-targetRadius'),
  targetWeight: document.getElementById('seek-targetWeight')
};

const seekValueLabels = {
  seekStrength: document.getElementById('seek-val-seekStrength'),
  captureRadius: document.getElementById('seek-val-captureRadius'),
  torqueBias: document.getElementById('seek-val-torqueBias'),
  phaseLock: document.getElementById('seek-val-phaseLock'),
  damping: document.getElementById('seek-val-damping'),
  targetRadius: document.getElementById('seek-val-targetRadius'),
  targetWeight: document.getElementById('seek-val-targetWeight')
};

function syncSeekPanelFromState() {
  if (seekModeSelect) seekModeSelect.value = SEEK_MODE_STATE.mode;
  if (seekTargetModeSelect) seekTargetModeSelect.value = SEEK_TARGET_STATE.mode;
  for (const key of Object.keys(seekControls)) {
    const control = seekControls[key];
    if (!control) continue;
    const sourceState = (key === 'targetRadius' || key === 'targetWeight') ? SEEK_TARGET_STATE : SEEK_MODE_STATE;
    control.value = String(sourceState[key]);
    if (seekValueLabels[key]) seekValueLabels[key].textContent = Number(sourceState[key]).toFixed(2);
  }
  if (seekModeReadout) seekModeReadout.textContent = `mode: ${SEEK_MODE_STATE.mode}`;
  if (seekTargetReadout) {
    const armedText = SEEK_TARGET_STATE.armed ? ' | ARMED' : '';
    seekTargetReadout.textContent = `targets: ${SEEK_TARGET_STATE.targets.length}${armedText}`;
  }
  if (SEEK_TARGET_STATE.targets.length > 0) {
    const t0 = SEEK_TARGET_STATE.targets[SEEK_TARGET_STATE.targets.length - 1];
    if (seekTargetXInput) seekTargetXInput.value = t0.x.toFixed(3);
    if (seekTargetYInput) seekTargetYInput.value = (-t0.y).toFixed(3);
    if (seekTargetZInput) seekTargetZInput.value = t0.z.toFixed(3);
  }
}

function applySeekPreset(modeName) {
  const preset = SEEK_MODE_PRESETS[modeName] ?? SEEK_MODE_PRESETS.manual;
  SEEK_MODE_STATE.mode = modeName in SEEK_MODE_PRESETS ? modeName : 'manual';
  Object.assign(SEEK_MODE_STATE, preset);
  syncSeekPanelFromState();
}

function deriveSeekPolicyUniforms() {
  const strength = Math.max(0.0, SEEK_MODE_STATE.seekStrength);
  const phaseLock = Math.max(0.0, Math.min(1.0, SEEK_MODE_STATE.phaseLock));
  const damping = Math.max(0.0, SEEK_MODE_STATE.damping);

  SEEK_POLICY_STATE.seekWDistance = 0.35 + 0.90 * strength;
  SEEK_POLICY_STATE.seekWTangent = 0.20 + 1.20 * phaseLock;
  SEEK_POLICY_STATE.seekWRadial = 0.15 + 0.75 * damping;
  SEEK_POLICY_STATE.seekWTorque = SEEK_MODE_STATE.torqueBias;
  SEEK_POLICY_STATE.seekMaxInfluence = Math.max(0.05, Math.min(2.0, 0.35 + 0.03 * SEEK_MODE_STATE.captureRadius));
}


function projectPointToScreen(point, cameraRef, canvas) {
  const v = new THREE.Vector3(point[0], point[1], point[2]).project(cameraRef);
  const rect = canvas.getBoundingClientRect();
  return {
    x: (v.x * 0.5 + 0.5) * rect.width,
    // Vertical-axis correction: use non-inverted NDC-y mapping to match pointer picking orientation.
    y: (v.y * 0.5 + 0.5) * rect.height,
    z: v.z
  };
}

function addOrReplaceSeekTarget(target) {
  const t = {
    x: Number(target.x),
    y: Number(target.y),
    z: Number(target.z),
    radius: Number(target.radius ?? SEEK_TARGET_STATE.targetRadius),
    weight: Number(target.weight ?? SEEK_TARGET_STATE.targetWeight)
  };

  if (!Number.isFinite(t.x) || !Number.isFinite(t.y) || !Number.isFinite(t.z)) return false;

  if (SEEK_TARGET_STATE.mode === 'off') SEEK_TARGET_STATE.mode = 'single';

  if (SEEK_TARGET_STATE.mode === 'single') {
    SEEK_TARGET_STATE.targets = [t];
  } else {
    SEEK_TARGET_STATE.targets.push(t);
    if (SEEK_TARGET_STATE.targets.length > MAX_SEEK_TARGETS) SEEK_TARGET_STATE.targets.shift();
  }

  SEEK_TARGET_STATE.armed = false;
  syncSeekPanelFromState();
  updateSeekTargetUniforms();
  return true;
}

function applyXYZTargetFromInputs() {
  const x = Number(seekTargetXInput?.value);
  const y = -Number(seekTargetYInput?.value);
  const z = Number(seekTargetZInput?.value);
  const ok = addOrReplaceSeekTarget({ x, y, z });
  if (!ok) console.warn('[SEEK TARGET] Invalid XYZ input; expected numeric x, y, z.');
}

function updateSeekTargetUniforms() {
  if (!sysA?.accVar?.material?.uniforms) return;
  const uniforms = sysA.accVar.material.uniforms;
  const modeMap = { off: 0.0, single: 1.0, multi: 2.0 };
  if (uniforms.seekTargetMode) uniforms.seekTargetMode.value = modeMap[SEEK_TARGET_STATE.mode] ?? 0.0;
  if (uniforms.seekTargetCount) uniforms.seekTargetCount.value = Math.min(SEEK_TARGET_STATE.targets.length, MAX_SEEK_TARGETS);

  if (uniforms.seekTargetPositions) {
    for (let i = 0; i < MAX_SEEK_TARGETS; i++) {
      const t = SEEK_TARGET_STATE.targets[i];
      if (t) {
        uniforms.seekTargetPositions.value[i].set(t.x, t.y, t.z, t.weight);
        uniforms.seekTargetRadii.value[i] = t.radius;
      } else {
        uniforms.seekTargetPositions.value[i].set(0, 0, 0, 0);
        uniforms.seekTargetRadii.value[i] = 0;
      }
    }
    uniforms.seekTargetRadii.needsUpdate = true;
  }
}

function pickSeekTargetFromClick(event) {
  if (!SEEK_TARGET_STATE.armed || SEEK_TARGET_STATE.mode === 'off') return;
  if (!sysB?.gpu?.getCurrentRenderTarget || !renderer?.domElement) return;

  const rect = renderer.domElement.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;

  const extRT = sysB.gpu.getCurrentRenderTarget(sysB.posVar);
  renderer.readRenderTargetPixels(extRT, 0, 0, TEX_SIZE, TEX_SIZE, extReadbackBuffer);

  let best = null;
  let bestD2 = Number.POSITIVE_INFINITY;
  for (let i = 0; i < COUNT; i++) {
    const i4 = i * 4;
    const tag = Math.floor(extReadbackBuffer[i4 + 3] + 1e-4);
    if (tag < 1) continue;
    const p = [extReadbackBuffer[i4], extReadbackBuffer[i4 + 1], extReadbackBuffer[i4 + 2]];
    const screen = projectPointToScreen(p, camera, renderer.domElement);
    if (screen.z < -1 || screen.z > 1) continue;
    const dx = screen.x - mouseX;
    const dy = screen.y - mouseY;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = p;
    }
  }

  if (!best) return;
  addOrReplaceSeekTarget({
    x: best[0],
    y: best[1],
    z: best[2],
    radius: SEEK_TARGET_STATE.targetRadius,
    weight: SEEK_TARGET_STATE.targetWeight
  });
}

function applySeekStateToUniforms() {
  EM_STATE.k = 50.0 * SEEK_MODE_STATE.seekStrength;
  EM_STATE.radius = SEEK_MODE_STATE.captureRadius;
  EM_STATE.twist = SEEK_MODE_STATE.torqueBias;
  EM_STATE.twistHz = 0.25 + SEEK_MODE_STATE.phaseLock * 3.75;
  EM_STATE.c = 10.0 * SEEK_MODE_STATE.damping;
  deriveSeekPolicyUniforms();

  const applySystem = (sys) => {
    const uniforms = sys.accVar.material.uniforms;
    uniforms.extK.value = EM_STATE.k;
    uniforms.extRadius.value = EM_STATE.radius;
    uniforms.extTwist.value = EM_STATE.twist;
    uniforms.extTwistHz.value = EM_STATE.twistHz;
    uniforms.extC.value = EM_STATE.c;
  };

  if (sysA && sysB) {
    applySystem(sysA);
    applySystem(sysB);
    if (sysA.chemVar?.material?.uniforms?.extRadius) sysA.chemVar.material.uniforms.extRadius.value = EM_STATE.radius;
    if (sysA.posTargetVar?.material?.uniforms?.extRadius) sysA.posTargetVar.material.uniforms.extRadius.value = EM_STATE.radius;

    const chemUniforms = sysA.chemVar?.material?.uniforms;
    const posUniforms = sysA.posTargetVar?.material?.uniforms;
    for (const uniforms of [chemUniforms, posUniforms]) {
      if (!uniforms) continue;
      if (uniforms.seekWDistance) uniforms.seekWDistance.value = SEEK_POLICY_STATE.seekWDistance;
      if (uniforms.seekWTangent) uniforms.seekWTangent.value = SEEK_POLICY_STATE.seekWTangent;
      if (uniforms.seekWRadial) uniforms.seekWRadial.value = SEEK_POLICY_STATE.seekWRadial;
      if (uniforms.seekWTorque) uniforms.seekWTorque.value = SEEK_POLICY_STATE.seekWTorque;
      if (uniforms.seekMaxInfluence) uniforms.seekMaxInfluence.value = SEEK_POLICY_STATE.seekMaxInfluence;
    }
    updateSeekTargetUniforms();
  }
}

function setupSeekPanelBindings() {
  if (!seekPanel) return;

  if (seekModeSelect) {
    seekModeSelect.addEventListener('change', () => {
      applySeekPreset(seekModeSelect.value);
    });
  }

  for (const [key, control] of Object.entries(seekControls)) {
    if (!control) continue;
    control.addEventListener('input', () => {
      if (key === 'targetRadius' || key === 'targetWeight') {
        SEEK_TARGET_STATE[key] = Number(control.value);
      } else {
        SEEK_MODE_STATE.mode = 'manual';
        SEEK_MODE_STATE[key] = Number(control.value);
      }
      syncSeekPanelFromState();
    });
  }

  if (seekTargetModeSelect) {
    seekTargetModeSelect.addEventListener('change', () => {
      SEEK_TARGET_STATE.mode = seekTargetModeSelect.value;
      if (SEEK_TARGET_STATE.mode === 'off') {
        SEEK_TARGET_STATE.armed = false;
        SEEK_TARGET_STATE.targets = [];
      }
      syncSeekPanelFromState();
      updateSeekTargetUniforms();
    });
  }

  if (seekArmPickButton) {
    seekArmPickButton.addEventListener('click', () => {
      if (SEEK_TARGET_STATE.mode === 'off') SEEK_TARGET_STATE.mode = 'single';
      SEEK_TARGET_STATE.armed = !SEEK_TARGET_STATE.armed;
      syncSeekPanelFromState();
    });
  }

  if (seekClearTargetsButton) {
    seekClearTargetsButton.addEventListener('click', () => {
      SEEK_TARGET_STATE.targets = [];
      SEEK_TARGET_STATE.armed = false;
      syncSeekPanelFromState();
      updateSeekTargetUniforms();
    });
  }

  if (seekApplyXYZButton) {
    seekApplyXYZButton.addEventListener('click', applyXYZTargetFromInputs);
  }

  if (seekResetButton) {
    seekResetButton.addEventListener('click', () => {
      applySeekPreset('manual');
    });
  }

  syncSeekPanelFromState();
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);
document.body.appendChild(renderer.domElement);
renderer.domElement.addEventListener('pointerdown', pickSeekTargetFromClick);

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(0, 18, 75);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function makeIndexGeometry(count) {
  const geom = new THREE.BufferGeometry();
  const aIndex = new Float32Array(count);
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) aIndex[i] = i;
  geom.setAttribute('aIndex', new THREE.BufferAttribute(aIndex, 1));
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setDrawRange(0, count);
  geom.computeBoundingSphere();
  geom.boundingSphere.radius = 1e9;
  return geom;
}

// ==================== SYSTEM B: External EM Field Source ====================
function makeSystemB() {
  const gpu = new GPUComputationRenderer(TEX_SIZE, TEX_SIZE, renderer);
  
  const tex0 = gpu.createTexture();
  for (let i = 0; i < tex0.image.data.length; i += 4) {
    tex0.image.data[i + 0] = 0;
    tex0.image.data[i + 1] = 0;
    tex0.image.data[i + 2] = 0;
    tex0.image.data[i + 3] = 1;
  }

  const posTargetVar = gpu.addVariable('posTarget', createPosTargetShader(), tex0);
  const accVar       = gpu.addVariable('acc',       createAccShader(),      tex0);
  const velVar       = gpu.addVariable('vel',       createVelShader(),      tex0);
  const posVar       = gpu.addVariable('pos',       createPosShader(),      tex0);

  const tags0 = gpu.createTexture();
  for (let i = 0; i < tags0.image.data.length; i += 4) {
    tags0.image.data[i + 0] = 0;
    tags0.image.data[i + 1] = 0;
    tags0.image.data[i + 2] = 0;
    tags0.image.data[i + 3] = 1;
  }
  const tagsVar = gpu.addVariable('tags', createTagsShader(), tags0);

  gpu.setVariableDependencies(posTargetVar, [posTargetVar, tagsVar]);
  gpu.setVariableDependencies(accVar,       [accVar, posTargetVar, posVar, velVar]);
  gpu.setVariableDependencies(velVar,       [velVar, accVar]);
  gpu.setVariableDependencies(posVar,       [posVar, posTargetVar, velVar]);
  gpu.setVariableDependencies(tagsVar,      [tagsVar, posVar]);

  // ==================== EM FIELD UNIFORMS ====================
  posTargetVar.material.uniforms.time = { value: 0 };
  posTargetVar.material.uniforms.dt   = { value: 0.016 };
  posTargetVar.material.uniforms.baseR = { value: 20.0 };
  posTargetVar.material.uniforms.pulseAmp = { value: 0.0 };
  posTargetVar.material.uniforms.pulseHz  = { value: 0.0 };
  posTargetVar.material.uniforms.shapeMode = { value: 0 };
  posTargetVar.material.uniforms.activeCount = { value: COUNT };

  Object.assign(tagsVar.material.uniforms, {
    time: { value: 0 },
    tagMode: { value: 4 },
    tagScale: { value: 1.0 },
    tagBias: { value: 0.0 },
  });

  // ==================== EM FIELD CONFIGURATION ====================
  Object.assign(accVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    maxAcc: { value: 300.0 },
    targetK: { value: 90.0 },
    targetC: { value: 18.0 },
    repelRadius: { value: 0.0 },
    repelK: { value: 0.0 },
    repelC: { value: 0.0 },
    repelPower: { value: 2.0 },
    
    // EM FIELD PARAMETERS (ext*)
    extEnabled: { value: EM_STATE.enabled },
    extMode: { value: EM_STATE.mode },        // 2 = helix-tunnel for DNA
    extPos: { value: null },                 // Will be set to sysA.pos
    extMat: { value: new THREE.Matrix4() },
    extRadius: { value: EM_STATE.radius },    // EM capture radius
    extK: { value: EM_STATE.k },              // EM force stiffness
    extC: { value: EM_STATE.c },              // EM damping
    extSamples: { value: EM_STATE.samples },  // EM samples
    extPower: { value: 2.0 },
    extTwist: { value: EM_STATE.twist },      // Helical EM component
    extTwistHz: { value: EM_STATE.twistHz }, // EM oscillation
    alpha0: { value: ALPHA_0 },
    alphaCoupling: { value: 0.0 },
  });

  Object.assign(velVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    maxVel: { value: 120.0 },
    velDrag: { value: 0.08 },
    resetCountdown: { value: 0 },
  });
  
  Object.assign(posVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    frame: { value: 0 },
    maxPos: { value: 500.0 },
    resetCountdown: { value: 0 },
  });

  const err = gpu.init();
  if (err) console.error('System B init error:', err);

  const geom = makeIndexGeometry(COUNT);
  const mat = createPointsMaterial(TEX_SIZE, TEX_SIZE, { useNiftiColors: false }, renderer);
  mat.uniforms.atlasColorEnabled.value = true;
  mat.uniforms.atlasColorMode.value = 2;
  mat.uniforms.instanceOffset.value.set(18, 0, 0);
  mat.uniforms.unusedOffset.value.set(0, 0, 0);

  const pts = new THREE.Points(geom, mat);
  pts.frustumCulled = false;
  pts.visible = false; // System B is invisible (external field source)
  scene.add(pts);

  return { gpu, posTargetVar, accVar, velVar, posVar, tagsVar, mat, points: pts };
}

// ==================== SYSTEM A: DNA with EM Response ====================
function makeSystemA(getExtTexture) {
  const gpu = new GPUComputationRenderer(TEX_SIZE, TEX_SIZE, renderer);

  const tex0 = gpu.createTexture();
  for (let i = 0; i < tex0.image.data.length; i += 4) {
    const idx = (i / 4) | 0;
    const active = (idx < RENDER_COUNT) ? 1.0 : 0.0;
    let pal = 7;
    if (idx < NODE_COUNT) pal = 1;
    else if (idx < 2 * NODE_COUNT) pal = 5;
    else if (idx < 3 * NODE_COUNT) pal = 4;
    else if (idx < 4 * NODE_COUNT) pal = 3;
    else pal = 7;

    tex0.image.data[i + 0] = 0;
    tex0.image.data[i + 1] = 0;
    tex0.image.data[i + 2] = 0;
    tex0.image.data[i + 3] = active + pal / 256.0;
  }

  const flow0 = gpu.createTexture();
  for (let i = 0; i < flow0.image.data.length; i += 4) {
    const idx = (i / 4) | 0;
    const active = idx < RENDER_COUNT;
    const u = (idx + 0.5) / COUNT;
    const progress = active ? (0.5 + 0.5 * Math.sin(12.9898 * u + 78.233)) % 1 : 0.0;
    const speedNoise = 0.5 + 0.5 * Math.sin(93.9898 * u + 11.113);
    const speed = active ? Math.max(0, FLOW_TRANSPORT_STATE.baseSpeed + (speedNoise * 2.0 - 1.0) * FLOW_TRANSPORT_STATE.speedJitter) : 0.0;
    flow0.image.data[i + 0] = progress;
    flow0.image.data[i + 1] = speed;
    flow0.image.data[i + 2] = 0.0;
    flow0.image.data[i + 3] = active ? 1.0 : 0.0;
  }

  const chemVar      = gpu.addVariable('chem',      createChemShader(),             tex0);
  const flowStateVar = gpu.addVariable('flowState', createFlowStateShader(),        flow0);
  const posTargetVar = gpu.addVariable('posTarget', createCoupledPosTargetShader(), tex0);
  const accVar       = gpu.addVariable('acc',       createAccShader(),              tex0);
  const velVar       = gpu.addVariable('vel',       createVelShader(),              tex0);
  const posVar       = gpu.addVariable('pos',       createPosShader(),              tex0);

  gpu.setVariableDependencies(chemVar,      [chemVar, posVar]);
  gpu.setVariableDependencies(flowStateVar, [flowStateVar]);
  gpu.setVariableDependencies(posTargetVar, [posTargetVar, chemVar, posVar, flowStateVar]);
  gpu.setVariableDependencies(accVar,       [accVar, posTargetVar, posVar, velVar]);
  gpu.setVariableDependencies(velVar,       [velVar, accVar]);
  gpu.setVariableDependencies(posVar,       [posVar, posTargetVar, velVar]);

  // Chem uniforms
  Object.assign(chemVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    nodeCount: { value: NODE_COUNT },
    neckSeg: { value: NECK_SEG },
    headCount: { value: HEAD_COUNT },
    secondEnabled: { value: SECOND_ENABLED },
    ds: { value: DS },
    pitch: { value: PITCH },
    qPitch: { value: Q_PITCH },
    zipMode: { value: zipMode },
    flowEnabled: { value: 1.0 },
    extPos: { value: getExtTexture() },
    extSamples: { value: EM_STATE.enabled ? EM_STATE.samples : 0.0 },
    extRadius: { value: EM_STATE.radius },
    seekWDistance: { value: SEEK_POLICY_STATE.seekWDistance },
    seekWTangent: { value: SEEK_POLICY_STATE.seekWTangent },
    seekWRadial: { value: SEEK_POLICY_STATE.seekWRadial },
    seekWTorque: { value: SEEK_POLICY_STATE.seekWTorque },
    seekMaxInfluence: { value: SEEK_POLICY_STATE.seekMaxInfluence },
    helixR: { value: HELIX_R },
    cotAlpha: { value: COT_ALPHA },
    alphaExp: { value: ALPHA_EXP },
    u_s: { value: U_S },
    alpha0: { value: ALPHA_0 },
    helixHandednessSign: { value: HELIX_CONVENTION.handednessSign },
    strandAPhaseOffset: { value: HELIX_CONVENTION.strandAPhaseOffset },
    strandBPhaseOffset: { value: HELIX_CONVENTION.strandBPhaseOffset },
    angleUnitScale: { value: HELIX_CONVENTION.angleUnitScale }
  });

  // Flow-state uniforms
  Object.assign(flowStateVar.material.uniforms, {
    dt: { value: 0.016 },
    nodeCount: { value: NODE_COUNT },
    neckSeg: { value: NECK_SEG },
    headCount: { value: HEAD_COUNT },
    secondEnabled: { value: SECOND_ENABLED },
    flowEnabled: { value: 1.0 },
    flowScale: { value: FLOW_TRANSPORT_STATE.flowScale },
    baseSpeed: { value: FLOW_TRANSPORT_STATE.baseSpeed },
    speedJitter: { value: FLOW_TRANSPORT_STATE.speedJitter }
  });

  // PosTarget uniforms
  Object.assign(posTargetVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    nodeCount: { value: NODE_COUNT },
    neckSeg: { value: NECK_SEG },
    headCount: { value: HEAD_COUNT },
    secondEnabled: { value: SECOND_ENABLED },
    flowState: { value: null },
    ds: { value: DS },
    helixR: { value: HELIX_R },
    pitch: { value: PITCH },
    qPitch: { value: Q_PITCH },
    axialShift: { value: AXIAL_SHIFT },
    extPos: { value: getExtTexture() },
    extSamples: { value: EM_STATE.enabled ? EM_STATE.samples : 0.0 },
    extRadius: { value: EM_STATE.radius },
    seekWDistance: { value: SEEK_POLICY_STATE.seekWDistance },
    seekWTangent: { value: SEEK_POLICY_STATE.seekWTangent },
    seekWRadial: { value: SEEK_POLICY_STATE.seekWRadial },
    seekWTorque: { value: SEEK_POLICY_STATE.seekWTorque },
    seekMaxInfluence: { value: SEEK_POLICY_STATE.seekMaxInfluence },
    wellOrigin: { value: new THREE.Vector3(0, 0, 0) },
    unusedOffset: { value: new THREE.Vector3(0, 0, 0) },
    flowEnabled: { value: 1.0 },
    flowSpeed: { value: 1.4 },
    flowRad: { value: 2.8 },
    electricityJitter: { value: 0.8 },
    pulseFrequency: { value: 6.0 },
    pulseSpeed: { value: 2.0 },
    zipMode: { value: zipMode },
    transportGain: { value: FLOW_TRANSPORT_STATE.transportGain },
    cotAlpha: { value: COT_ALPHA },
    alphaExp: { value: ALPHA_EXP },
    u_s: { value: U_S },
    alpha0: { value: ALPHA_0 },
    helixHandednessSign: { value: HELIX_CONVENTION.handednessSign },
    strandAPhaseOffset: { value: HELIX_CONVENTION.strandAPhaseOffset },
    strandBPhaseOffset: { value: HELIX_CONVENTION.strandBPhaseOffset },
    angleUnitScale: { value: HELIX_CONVENTION.angleUnitScale }
  });

  // ==================== EM FIELD IN ACC SHADER ====================
  Object.assign(accVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    maxAcc: { value: 300.0 },
    targetK: { value: 90.0 },
    targetC: { value: 18.0 },
    repelRadius: { value: 0.0 },
    repelK: { value: 0.0 },
    repelC: { value: 0.0 },
    repelPower: { value: 2.0 },
    
    // EM FIELD - System A responds to System B's field
    extEnabled: { value: EM_STATE.enabled },
    extMode: { value: EM_STATE.mode },
    extPos: { value: null }, // Will be sysB.pos
    extMat: { value: new THREE.Matrix4() },
    extRadius: { value: EM_STATE.radius },
    extK: { value: EM_STATE.k },
    extC: { value: EM_STATE.c },
    extSamples: { value: EM_STATE.enabled ? EM_STATE.samples : 0.0 },
    extPower: { value: 2.0 },
    extTwist: { value: EM_STATE.twist },
    extTwistHz: { value: EM_STATE.twistHz },
    alpha0: { value: ALPHA_0 },
    alphaCoupling: { value: 12.0 },
    seekTargetMode: { value: 0.0 },
    seekTargetCount: { value: 0.0 },
    seekTargetPositions: { value: Array.from({ length: MAX_SEEK_TARGETS }, () => new THREE.Vector4()) },
    seekTargetRadii: { value: new Float32Array(MAX_SEEK_TARGETS) },
  });

  Object.assign(velVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    maxVel: { value: 120.0 },
    velDrag: { value: 0.08 },
    resetCountdown: { value: 0 },
  });
  
  Object.assign(posVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    frame: { value: 0 },
    maxPos: { value: 500.0 },
    resetCountdown: { value: 0 },
  });

  const err = gpu.init();
  if (err) console.error('System A init error:', err);

  const geom = makeIndexGeometry(RENDER_COUNT);
  const mat = createPointsMaterial(TEX_SIZE, TEX_SIZE, { useNiftiColors: false }, renderer);
  mat.uniforms.atlasColorEnabled.value = true;
  mat.uniforms.atlasColorMode.value = 2;
  mat.uniforms.instanceOffset.value.set(0, 0, 0);
  mat.uniforms.unusedOffset.value.set(0, 0, 0);
  mat.uniforms.chem = { value: null };

  const pts = new THREE.Points(geom, mat);
  pts.frustumCulled = false;
  scene.add(pts);

  return { 
    gpu, 
    chemVar, 
    flowStateVar, 
    posTargetVar, 
    accVar, 
    velVar, 
    posVar, 
    mat, 
    points: pts,
    architecture: {
      NODE_COUNT, NECK_SEG, HEAD_COUNT, 
      IDX_RUNG0, RUNG_COUNT, RENDER_COUNT,
      HELIX_R, PITCH, AXIAL_SHIFT, Q_PITCH,
      COT_ALPHA, ALPHA_EXP, U_S, ALPHA_0,
      alpha0Solved: CALIBRATION_STATE.alpha0Solved,
      HELIX_CONVENTION
    }
  };
}

const sysB = makeSystemB();
const sysA = makeSystemA(() => sysB.posVar.material.uniforms.pos.value);

// Bind System B as EM field source for System A
sysA.accVar.material.uniforms.extPos.value = sysB.gpu.getCurrentRenderTarget(sysB.posVar).texture;

// Warm-up compute once before calibration so force/proxy uses initialized textures.
stepMovementCompute({ sysA, sysB });
solveAlpha0FromRadialEquilibrium();
setupSeekPanelBindings();
applySeekStateToUniforms();

// ==================== CONTROLS ====================
window.addEventListener('keydown', (e) => {
  // Zip toggle
  if (e.key === 'z' || e.key === 'Z') {
    targetZipMode = (targetZipMode > 0.5) ? 0.0 : 1.0;
    console.log(`[SF_ZipSolver] Zip mode: ${zipMode.toFixed(2)} -> ${targetZipMode.toFixed(2)}`);
  }
  
  // Debug modes
  if (e.key === 'd' || e.key === 'D') {
    DEBUG_STATE.showZipField = !DEBUG_STATE.showZipField;
    DEBUG_STATE.mode = DEBUG_STATE.showZipField ? 1 : 0;
    if (sysA.mat.uniforms.showZipField) sysA.mat.uniforms.showZipField.value = DEBUG_STATE.showZipField;
    if (sysA.mat.uniforms.debugMode) sysA.mat.uniforms.debugMode.value = DEBUG_STATE.mode;
    console.log(`[SF_DebugValidate] Zip field: ${DEBUG_STATE.showZipField}`);
  }
  
  if (e.key === 'c' || e.key === 'C') {
    DEBUG_STATE.showCalcium = !DEBUG_STATE.showCalcium;
    DEBUG_STATE.mode = DEBUG_STATE.showCalcium ? 2 : 0;
    if (sysA.mat.uniforms.showCalcium) sysA.mat.uniforms.showCalcium.value = DEBUG_STATE.showCalcium;
    if (sysA.mat.uniforms.debugMode) sysA.mat.uniforms.debugMode.value = DEBUG_STATE.mode;
    console.log(`[SF_DebugValidate] Calcium: ${DEBUG_STATE.showCalcium}`);
  }
  
  // ==================== EM FIELD CONTROLS ====================
  if (e.key === 'e' || e.key === 'E') {
    // Toggle EM field
    EM_STATE.enabled = !EM_STATE.enabled;
    const emSamples = EM_STATE.enabled ? EM_STATE.samples : 0.0;
    
    // Update System A (DNA response)
    sysA.accVar.material.uniforms.extEnabled.value = EM_STATE.enabled;
    sysA.accVar.material.uniforms.extSamples.value = emSamples;
    sysA.chemVar.material.uniforms.extSamples.value = emSamples;
    sysA.posTargetVar.material.uniforms.extSamples.value = emSamples;
    
    // Update System B (EM source)
    sysB.accVar.material.uniforms.extEnabled.value = EM_STATE.enabled;
    sysB.accVar.material.uniforms.extSamples.value = emSamples;
    
    console.log(`[EM FIELD] ${EM_STATE.enabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`[EM FIELD] Radius: ${EM_STATE.radius}, K: ${EM_STATE.k}, Mode: ${EM_STATE.mode}`);
  }
  
  if (e.key === 'r' || e.key === 'R') {
    // Increase EM radius
    EM_STATE.radius = Math.min(EM_STATE.radius + 5.0, 50.0);
    sysA.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
    sysB.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
    sysA.chemVar.material.uniforms.extRadius.value = EM_STATE.radius;
    console.log(`[EM FIELD] Radius: ${EM_STATE.radius.toFixed(1)}`);
  }
  
  if (e.key === 'f' || e.key === 'F') {
    // Decrease EM radius
    EM_STATE.radius = Math.max(EM_STATE.radius - 5.0, 5.0);
    sysA.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
    sysB.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
    sysA.chemVar.material.uniforms.extRadius.value = EM_STATE.radius;
    console.log(`[EM FIELD] Radius: ${EM_STATE.radius.toFixed(1)}`);
  }
  
  if (e.key === 't' || e.key === 'T') {
    // Cycle EM mode
    EM_STATE.mode = EM_STATE.mode === 1.0 ? 2.0 : 1.0;
    sysA.accVar.material.uniforms.extMode.value = EM_STATE.mode;
    sysB.accVar.material.uniforms.extMode.value = EM_STATE.mode;
    const modeName = EM_STATE.mode === 1.0 ? 'TUNNEL' : 'HELIX-TUNNEL';
    console.log(`[EM FIELD] Mode: ${modeName} (${EM_STATE.mode})`);
  }

  if (e.key === '[') {
    sysA.accVar.material.uniforms.alphaCoupling.value = Math.max(0.0, sysA.accVar.material.uniforms.alphaCoupling.value - 1.0);
    console.log(`[ALPHA] acc coupling: ${sysA.accVar.material.uniforms.alphaCoupling.value.toFixed(2)}`);
  }

  if (e.key === ']') {
    sysA.accVar.material.uniforms.alphaCoupling.value = Math.min(80.0, sysA.accVar.material.uniforms.alphaCoupling.value + 1.0);
    console.log(`[ALPHA] acc coupling: ${sysA.accVar.material.uniforms.alphaCoupling.value.toFixed(2)}`);
  }

  if (e.key === '-' || e.key === '_') {
    CALIBRATION_STATE.alpha0ManualOffset -= Math.PI / 180.0;
    applyAlpha0ToRuntime();
    console.log(`[ALPHA] alpha0 offset: ${(CALIBRATION_STATE.alpha0ManualOffset * 180 / Math.PI).toFixed(2)}°`);
  }

  if (e.key === '=' || e.key === '+') {
    CALIBRATION_STATE.alpha0ManualOffset += Math.PI / 180.0;
    applyAlpha0ToRuntime();
    console.log(`[ALPHA] alpha0 offset: ${(CALIBRATION_STATE.alpha0ManualOffset * 180 / Math.PI).toFixed(2)}°`);
  }

  if (e.key === 'p' || e.key === 'P') {
    CALIBRATION_STATE.calibrationEnabled = !CALIBRATION_STATE.calibrationEnabled;
    console.log(`[ALPHA] calibration ${CALIBRATION_STATE.calibrationEnabled ? 'LIVE' : 'PAUSED'}`);
  }
});

function setAlphaCoupling(value) {
  const coupling = Math.max(0.0, Math.min(60.0, value));
  sysA.accVar.material.uniforms.alphaCoupling.value = coupling;
  return coupling;
}

function toggleAlphaSource() {
  ALPHA_CONTROL_STATE.useSolvedAlpha0 = !ALPHA_CONTROL_STATE.useSolvedAlpha0;
  const alphaUniform = ALPHA_CONTROL_STATE.useSolvedAlpha0 ? CALIBRATION_STATE.alpha0Solved : ALPHA_0;
  applyAlpha0Uniforms(alphaUniform);
  console.log(`[α CONTROL] alpha0 source: ${ALPHA_CONTROL_STATE.useSolvedAlpha0 ? 'solved' : 'fixed ALPHA_0'} (${(alphaUniform * 180 / Math.PI).toFixed(2)}°)`);
}

function triggerAlphaRecalibration() {
  solveAlpha0FromRadialEquilibrium();
  console.log(`[α CONTROL] recalibrated alpha0Solved=${(CALIBRATION_STATE.alpha0Solved * 180 / Math.PI).toFixed(2)}° residual=${CALIBRATION_STATE.residualAtAlpha0.toExponential(2)}`);
}

function setCalibrationSampleEveryNFrames(value) {
  CALIBRATION_STATE.sampleEveryNFrames = Math.max(1, Math.min(120, Math.round(value)));
  console.log(`[α CONTROL] calibration sampleEveryNFrames=${CALIBRATION_STATE.sampleEveryNFrames}`);
}

function setEMEnabled(enabled) {
  EM_STATE.enabled = !!enabled;
  const emSamples = EM_STATE.enabled ? EM_STATE.samples : 0.0;
  sysA.accVar.material.uniforms.extEnabled.value = EM_STATE.enabled;
  sysB.accVar.material.uniforms.extEnabled.value = EM_STATE.enabled;
  sysA.chemVar.material.uniforms.extSamples.value = emSamples;
  sysA.accVar.material.uniforms.extSamples.value = emSamples;
  sysB.accVar.material.uniforms.extSamples.value = emSamples;
}

function setEMRadius(radius) {
  EM_STATE.radius = radius;
  sysA.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
  sysB.accVar.material.uniforms.extRadius.value = EM_STATE.radius;
  sysA.chemVar.material.uniforms.extRadius.value = EM_STATE.radius;
}

function setEMTwist(twist) {
  EM_STATE.twist = twist;
  sysA.accVar.material.uniforms.extTwist.value = EM_STATE.twist;
  sysB.accVar.material.uniforms.extTwist.value = EM_STATE.twist;
}

function setFlowEnabled(enabled) {
  sysA.chemVar.material.uniforms.flowEnabled.value = enabled;
  sysA.flowStateVar.material.uniforms.flowEnabled.value = enabled;
  sysA.posTargetVar.material.uniforms.flowEnabled.value = enabled;
}

const validationRunner = createAcceptanceValidationRunner({
  renderer,
  sysA,
  sysB,
  constants: window.DNASpineArchitecture?.constants || {
    TEX_SIZE, NODE_COUNT, NECK_SEG,
    HELIX_R, PITCH, AXIAL_SHIFT, Q_PITCH,
    COT_ALPHA, ALPHA_EXP, U_S,
    HELIX_CONVENTION,
    DS, IDX_RUNG0
  },
  emState: EM_STATE,
  setTargetZipMode: (value) => { targetZipMode = value; },
  setFlowEnabled,
  setEMEnabled,
  setEMRadius,
  setEMTwist,
  getZipMode: () => zipMode,
  getAlphaCalibration: () => ({ ...CALIBRATION_STATE }),
  getAccCalibration: () => ({
    alpha0: sysA.accVar.material.uniforms.alpha0.value,
    alphaCoupling: sysA.accVar.material.uniforms.alphaCoupling.value
  }),
  texSize: TEX_SIZE
});

let lastT = performance.now();
let frame = 0;

function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = Math.min((now - lastT) / 1000, 0.033);
  lastT = now;

  zipMode += (targetZipMode - zipMode) * (1.0 - Math.exp(-dt * 3.0));
  applySeekStateToUniforms();

  const t = now / 1000;

  applyMovementUniforms({ sysA, sysB, t, dt, frame, zipMode });
  stepMovementCompute({ sysA, sysB });
  bindMovementTextures({ sysA, sysB });

  updateAlpha0Calibration();

  validationRunner.update();
  updateResidualMetrics();

  // Debug uniforms
  if (sysA.mat.uniforms.showZipField) sysA.mat.uniforms.showZipField.value = DEBUG_STATE.showZipField;
  if (sysA.mat.uniforms.showCalcium) sysA.mat.uniforms.showCalcium.value = DEBUG_STATE.showCalcium;
  if (sysA.mat.uniforms.debugMode) sysA.mat.uniforms.debugMode.value = DEBUG_STATE.mode;

  // HUD
  if (hud) {
    const arch = sysA.architecture;
    const emStatus = EM_STATE.enabled ? 'ON' : 'OFF';
    const emMode = EM_STATE.mode === 1.0 ? 'TUNNEL' : 'HELIX';
    const debugStatus = DEBUG_STATE.mode === 0 ? 'NORMAL' : 
                       DEBUG_STATE.mode === 1 ? 'ZIP FIELD' : 
                       DEBUG_STATE.mode === 2 ? 'CALCIUM' : 'OTHER';
    
    hud.textContent = [
      `=== DNA-Spine with EM Field ===`,
      `Z: zip | D: debug zip | C: debug Ca | E: EM toggle`,
      `R/F: EM radius +/- | T: EM mode`,
      `[ / ]: α-coupling -/+ | - / =: α₀ offset -/+ 1° | P: pause/live α calibration`,
      ``,
      `[SF_HelixGenerator - MDPI Parameters]`,
      `  r=${arch.HELIX_R.toFixed(2)} h=${arch.PITCH.toFixed(2)} α_exp=${(arch.ALPHA_EXP * 180 / Math.PI).toFixed(1)}°`,
      `  α_0 solved(raw)=${(CALIBRATION_STATE.alpha0Solved * 180 / Math.PI).toFixed(2)}° | α_0 applied=${(arch.alpha0Solved * 180 / Math.PI).toFixed(2)}°`,
      `  residual=${CALIBRATION_STATE.residualAtAlpha0.toExponential(2)} | offset=${(CALIBRATION_STATE.alpha0ManualOffset * 180 / Math.PI).toFixed(2)}° | mode=${CALIBRATION_STATE.calibrationEnabled ? 'LIVE' : 'PAUSED'}`,
      `  solve=${CALIBRATION_STATE.converged ? 'converged' : 'no-root'} samples=${CALIBRATION_STATE.samples} scan=${CALIBRATION_STATE.candidateCount} (frame ${CALIBRATION_STATE.lastFrame})`,
      ``,
      `[EM FIELD - ${emStatus}]`,
      `  Radius: ${EM_STATE.radius.toFixed(1)} | K: ${EM_STATE.k} | Mode: ${emMode}`,
      `  Twist: ${EM_STATE.twist} | TwistHz: ${EM_STATE.twistHz}`,
      `  Physics: ${EM_STATE.enabled ? 'Radial REPULSION (α < α₀)' : 'No EM force'}`,
      `  α-coupling(acc): ${sysA.accVar.material.uniforms.alphaCoupling.value.toFixed(2)}`,
      ``,
      `[SF_ZipSolver]`,
      `  zipMode=${zipMode.toFixed(3)} (target: ${targetZipMode.toFixed(1)})`,
      ``,
      `[Seek Targets]`,
      `  mode=${SEEK_TARGET_STATE.mode} count=${SEEK_TARGET_STATE.targets.length} armed=${SEEK_TARGET_STATE.armed ? 'yes' : 'no'}`,
      `  radius=${SEEK_TARGET_STATE.targetRadius.toFixed(1)} weight=${SEEK_TARGET_STATE.targetWeight.toFixed(2)}`,
      ``,
      `[SF_DebugValidate]`,
      `  Mode: ${debugStatus}`,
      ``,
      `[Residual Inference]`,
      `  nodes=${RESIDUAL_STATE.sampledNodes} sample#=${RESIDUAL_STATE.sampleCount} (frame ${RESIDUAL_STATE.lastFrame})`,
      `  Δα mean=${(RESIDUAL_STATE.meanDeltaAlpha * 180 / Math.PI).toFixed(3)}° | |Δα| mean=${(RESIDUAL_STATE.meanAbsDeltaAlpha * 180 / Math.PI).toFixed(3)}° | rms=${(RESIDUAL_STATE.rmsDeltaAlpha * 180 / Math.PI).toFixed(3)}°`,
      `  Δq mean=${RESIDUAL_STATE.meanDeltaQ.toFixed(5)} | |Δq| mean=${RESIDUAL_STATE.meanAbsDeltaQ.toFixed(5)} | rms=${RESIDUAL_STATE.rmsDeltaQ.toFixed(5)}`,
      ``,
      `[Validation]`,
      `  status=${validationRunner.getHudSummary().status} ` +
      `(${validationRunner.getHudSummary().passed}/${validationRunner.getHudSummary().total} pass, ` +
      `${validationRunner.getHudSummary().failed} fail)`,
      `  scenario=${validationRunner.getHudSummary().currentScenario}`,
      `  checks/scenario=${validationRunner.getHudSummary().metricCount}`,
      ``,
      `Frame: ${frame} | dt: ${(dt*1000).toFixed(2)}ms`
    ].join('\n');
  }

  controls.update();
  renderer.render(scene, camera);
  frame++;
}

animate();

window.DNASpineArchitecture = {
  sysA, sysB, DEBUG_STATE, EM_STATE, SEEK_MODE_STATE, SEEK_TARGET_STATE, RESIDUAL_STATE, validationRunner,
  getResidualMetrics: () => ({ ...RESIDUAL_STATE }),
  getAlphaCalibration: () => ({ ...CALIBRATION_STATE }),
  constants: {
    TEX_SIZE, COUNT, NODE_COUNT, NECK_SEG, HEAD_COUNT,
    RENDER_COUNT, IDX_RUNG0, RUNG_COUNT,
    HELIX_R, PITCH, AXIAL_SHIFT, DS,
    Q_PITCH, COT_ALPHA, ALPHA_EXP, U_S, ALPHA_0,
    alpha0Solved: CALIBRATION_STATE.alpha0Solved,
    HELIX_CONVENTION
  }
};

console.log('[EM FIELD] Controls: E=toggle, R/F=radius, T=mode');
console.log('[EM FIELD] DNA α=' + (ALPHA_EXP * 180 / Math.PI).toFixed(1) + '° | solved α₀=' + (CALIBRATION_STATE.alpha0Solved * 180 / Math.PI).toFixed(2) + '° | residual=' + CALIBRATION_STATE.residualAtAlpha0.toExponential(2));
