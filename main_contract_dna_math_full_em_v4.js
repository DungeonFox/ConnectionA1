import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import GPUComputationRenderer from './GPUComputationRenderer.js';
import { createPosTargetShader, createAccShader, createVelShader, createPosShader, createTagsShader } from './computeShaders.js';
import { createPointsMaterial } from './materials.js';
import { 
  createChemShader, 
  createCoupledPosTargetShader
} from './coupledShadersContract_dna_math_full_em_v4.js';
import { createAcceptanceValidationRunner } from './validation/acceptance.js';

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

const HELIX_CONVENTION = {
  handednessSign: 1.0,
  phaseOffsetA: 0.5 * Math.PI,
  phaseOffsetB: 1.5 * Math.PI,
  angleUnitScale: 1.0,
  angleUnitName: 'radians'
};

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

let zipMode = 1.0;
let targetZipMode = 1.0;

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

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 1);
document.body.appendChild(renderer.domElement);

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

  const chemVar      = gpu.addVariable('chem',      createChemShader(),             tex0);
  const posTargetVar = gpu.addVariable('posTarget', createCoupledPosTargetShader(), tex0);
  const accVar       = gpu.addVariable('acc',       createAccShader(),              tex0);
  const velVar       = gpu.addVariable('vel',       createVelShader(),              tex0);
  const posVar       = gpu.addVariable('pos',       createPosShader(),              tex0);

  gpu.setVariableDependencies(chemVar,      [chemVar, posVar]);
  gpu.setVariableDependencies(posTargetVar, [posTargetVar, chemVar, posVar]);
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
    helixR: { value: HELIX_R },
    cotAlpha: { value: COT_ALPHA },
    alphaExp: { value: ALPHA_EXP },
    u_s: { value: U_S },
    alpha0: { value: ALPHA_0 },
    helixHandednessSign: { value: HELIX_CONVENTION.handednessSign },
    phaseOffsetA: { value: HELIX_CONVENTION.phaseOffsetA },
    phaseOffsetB: { value: HELIX_CONVENTION.phaseOffsetB },
    angleUnitScale: { value: HELIX_CONVENTION.angleUnitScale }
  });

  // PosTarget uniforms
  Object.assign(posTargetVar.material.uniforms, {
    time: { value: 0 },
    dt: { value: 0.016 },
    nodeCount: { value: NODE_COUNT },
    neckSeg: { value: NECK_SEG },
    headCount: { value: HEAD_COUNT },
    secondEnabled: { value: SECOND_ENABLED },
    ds: { value: DS },
    helixR: { value: HELIX_R },
    pitch: { value: PITCH },
    qPitch: { value: Q_PITCH },
    axialShift: { value: AXIAL_SHIFT },
    extPos: { value: getExtTexture() },
    extSamples: { value: EM_STATE.enabled ? EM_STATE.samples : 0.0 },
    extRadius: { value: EM_STATE.radius },
    wellOrigin: { value: new THREE.Vector3(0, 0, 0) },
    unusedOffset: { value: new THREE.Vector3(0, 0, 0) },
    flowEnabled: { value: 1.0 },
    flowSpeed: { value: 1.4 },
    flowRad: { value: 2.8 },
    electricityJitter: { value: 0.8 },
    pulseFrequency: { value: 6.0 },
    pulseSpeed: { value: 2.0 },
    zipMode: { value: zipMode },
    cotAlpha: { value: COT_ALPHA },
    alphaExp: { value: ALPHA_EXP },
    u_s: { value: U_S },
    alpha0: { value: ALPHA_0 },
    helixHandednessSign: { value: HELIX_CONVENTION.handednessSign },
    phaseOffsetA: { value: HELIX_CONVENTION.phaseOffsetA },
    phaseOffsetB: { value: HELIX_CONVENTION.phaseOffsetB },
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
      COT_ALPHA, ALPHA_EXP, U_S, ALPHA_0
    }
  };
}

const sysB = makeSystemB();
const sysA = makeSystemA(() => sysB.posVar.material.uniforms.pos.value);

// Bind System B as EM field source for System A
sysA.accVar.material.uniforms.extPos.value = sysB.gpu.getCurrentRenderTarget(sysB.posVar).texture;

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
});

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
    DS, IDX_RUNG0,
    HELIX_CONVENTION
  },
  emState: EM_STATE,
  setTargetZipMode: (value) => { targetZipMode = value; },
  setFlowEnabled,
  setEMEnabled,
  setEMRadius,
  setEMTwist,
  getZipMode: () => zipMode,
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

  const t = now / 1000;

  // Update all uniforms
  sysA.chemVar.material.uniforms.time.value = t;
  sysA.chemVar.material.uniforms.dt.value = dt;
  sysA.chemVar.material.uniforms.zipMode.value = zipMode;
  sysA.chemVar.material.uniforms.flowEnabled.value = sysA.posTargetVar.material.uniforms.flowEnabled.value;
  
  sysA.posTargetVar.material.uniforms.time.value = t;
  sysA.posTargetVar.material.uniforms.dt.value = dt;
  sysA.posTargetVar.material.uniforms.zipMode.value = zipMode;

  sysA.accVar.material.uniforms.time.value = t;
  sysA.accVar.material.uniforms.dt.value = dt;
  sysA.velVar.material.uniforms.time.value = t;
  sysA.velVar.material.uniforms.dt.value = dt;
  sysA.posVar.material.uniforms.time.value = t;
  sysA.posVar.material.uniforms.dt.value = dt;
  sysA.posVar.material.uniforms.frame.value = frame;

  // System B updates
  sysB.posTargetVar.material.uniforms.time.value = t;
  sysB.accVar.material.uniforms.time.value = t;
  sysB.velVar.material.uniforms.time.value = t;
  sysB.posVar.material.uniforms.time.value = t;

  // Compute
  sysB.gpu.compute();
  sysA.gpu.compute();

  // Bind textures
  sysB.mat.uniforms.posTarget.value = sysB.gpu.getCurrentRenderTarget(sysB.posTargetVar).texture;
  sysB.mat.uniforms.acc.value       = sysB.gpu.getCurrentRenderTarget(sysB.accVar).texture;
  sysB.mat.uniforms.vel.value       = sysB.gpu.getCurrentRenderTarget(sysB.velVar).texture;
  sysB.mat.uniforms.pos.value       = sysB.gpu.getCurrentRenderTarget(sysB.posVar).texture;

  sysA.mat.uniforms.posTarget.value = sysA.gpu.getCurrentRenderTarget(sysA.posTargetVar).texture;
  sysA.mat.uniforms.acc.value       = sysA.gpu.getCurrentRenderTarget(sysA.accVar).texture;
  sysA.mat.uniforms.vel.value       = sysA.gpu.getCurrentRenderTarget(sysA.velVar).texture;
  sysA.mat.uniforms.pos.value       = sysA.gpu.getCurrentRenderTarget(sysA.posVar).texture;
  sysA.mat.uniforms.chem.value      = sysA.gpu.getCurrentRenderTarget(sysA.chemVar).texture;

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
      ``,
      `[SF_HelixGenerator - MDPI Parameters]`,
      `  r=${arch.HELIX_R.toFixed(2)} h=${arch.PITCH.toFixed(2)} α_exp=${(arch.ALPHA_EXP * 180 / Math.PI).toFixed(1)}°`,
      `  α_0=${(arch.ALPHA_0 * 180 / Math.PI).toFixed(1)}° (critical angle)`,
      ``,
      `[EM FIELD - ${emStatus}]`,
      `  Radius: ${EM_STATE.radius.toFixed(1)} | K: ${EM_STATE.k} | Mode: ${emMode}`,
      `  Twist: ${EM_STATE.twist} | TwistHz: ${EM_STATE.twistHz}`,
      `  Physics: ${EM_STATE.enabled ? 'Radial REPULSION (α < α₀)' : 'No EM force'}`,
      ``,
      `[SF_ZipSolver]`,
      `  zipMode=${zipMode.toFixed(3)} (target: ${targetZipMode.toFixed(1)})`,
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
  sysA, sysB, DEBUG_STATE, EM_STATE, RESIDUAL_STATE, validationRunner,
  getResidualMetrics: () => ({ ...RESIDUAL_STATE }),
  constants: {
    TEX_SIZE, COUNT, NODE_COUNT, NECK_SEG, HEAD_COUNT,
    RENDER_COUNT, IDX_RUNG0, RUNG_COUNT,
    HELIX_R, PITCH, AXIAL_SHIFT, DS,
    Q_PITCH, COT_ALPHA, ALPHA_EXP, U_S, ALPHA_0,
    HELIX_CONVENTION
  }
};

console.log('[EM FIELD] Controls: E=toggle, R/F=radius, T=mode');
console.log('[EM FIELD] DNA α=' + (ALPHA_EXP * 180 / Math.PI).toFixed(1) + '° < α₀=' + (ALPHA_0 * 180 / Math.PI).toFixed(1) + '° → EM causes REPULSION');
