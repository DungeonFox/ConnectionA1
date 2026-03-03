const TWO_PI = Math.PI * 2.0;

function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function readTexturePixels(renderer, renderTarget, texSize) {
  const out = new Float32Array(texSize * texSize * 4);
  renderer.readRenderTargetPixels(renderTarget, 0, 0, texSize, texSize, out);
  return out;
}

function vec3At(data, idx) {
  const i4 = idx * 4;
  return [data[i4], data[i4 + 1], data[i4 + 2]];
}

function dist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.hypot(dx, dy, dz);
}

function wrapAnglePi(a) {
  let v = a;
  while (v > Math.PI) v -= TWO_PI;
  while (v < -Math.PI) v += TWO_PI;
  return v;
}

function round3(v) {
  return Math.round(v * 1000) / 1000;
}

function makeScenarioPlan(seedBase) {
  const rand = mulberry32(seedBase);
  const sample = () => round3(rand());
  return [
    { name: 'grow', frames: 180, seed: seedBase + 1, zipTarget: 1.0, flowEnabled: 1.0, emEnabled: false, emRadius: 15.0, emTwist: 0.45 + 0.1 * sample() },
    { name: 'zip', frames: 140, seed: seedBase + 2, zipTarget: 1.0, flowEnabled: 1.0, emEnabled: true, emRadius: 14.0 + 2.0 * sample(), emTwist: 0.50 + 0.1 * sample() },
    { name: 'unzip', frames: 160, seed: seedBase + 3, zipTarget: 0.0, flowEnabled: 1.0, emEnabled: true, emRadius: 16.0 + 2.0 * sample(), emTwist: 0.55 + 0.1 * sample() },
    { name: 'reroute', frames: 140, seed: seedBase + 4, zipTarget: 0.0, flowEnabled: 0.0, emEnabled: true, emRadius: 18.0 + 2.0 * sample(), emTwist: 0.60 + 0.1 * sample() },
    { name: 'rezip', frames: 180, seed: seedBase + 5, zipTarget: 1.0, flowEnabled: 1.0, emEnabled: true, emRadius: 15.0 + 2.0 * sample(), emTwist: 0.50 + 0.1 * sample() }
  ];
}

function evaluateInvariants({ posPixels, chemPixels, constants, zipMode, scenarioName }) {
  const {
    NODE_COUNT,
    NECK_SEG,
    HELIX_R,
    Q_PITCH,
    DS,
    IDX_RUNG0
  } = constants;

  const perSpine = NECK_SEG;
  const idxStrandA0 = NODE_COUNT;
  const idxStrandB0 = NODE_COUNT * 2;
  const idxSpineP0 = NODE_COUNT * 3;

  let activeNodes = 0;
  let radiusChecks = 0;
  let radiusPass = 0;
  let phaseChecks = 0;
  let phasePass = 0;
  let hubChecks = 0;
  let hubPass = 0;
  let rungChecks = 0;
  let rungPass = 0;
  let gapSum = 0;

  for (let k = 0; k < NODE_COUNT; k++) {
    const chem = vec3At(chemPixels, k);
    const g = chem[0];
    const phi = chem[1];
    const gGap = chem[2];

    if (g <= 0.05) continue;
    activeNodes++;
    gapSum += gGap;

    const a = vec3At(posPixels, idxStrandA0 + k);
    const b = vec3At(posPixels, idxStrandB0 + k);
    const hub = vec3At(posPixels, idxSpineP0 + k * perSpine + (NECK_SEG - 1));

    const expectedR = HELIX_R + gGap;
    const errA = Math.abs(dist(a, hub) - expectedR);
    const errB = Math.abs(dist(b, hub) - expectedR);
    radiusChecks += 2;
    if (errA <= 0.45) radiusPass++;
    if (errB <= 0.45) radiusPass++;

    const mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    hubChecks++;
    if (dist(hub, mid) <= 0.25) hubPass++;

    const sideOffset = k * 4;
    const rung0 = vec3At(posPixels, IDX_RUNG0 + sideOffset);
    const rung1 = vec3At(posPixels, IDX_RUNG0 + sideOffset + 1);
    const rung2 = vec3At(posPixels, IDX_RUNG0 + sideOffset + 2);
    const rung3 = vec3At(posPixels, IDX_RUNG0 + sideOffset + 3);

    const d0 = dist(hub, rung0);
    const d1 = dist(hub, rung1);
    const d2 = dist(hub, rung2);
    const d3 = dist(hub, rung3);
    const da = dist(hub, a);
    const db = dist(hub, b);

    rungChecks += 2;
    if (d0 < d1 && d1 < da + 1e-3) rungPass++;
    if (d2 < d3 && d3 < db + 1e-3) rungPass++;

    if (k > 0) {
      const prevPhi = chemPixels[(k - 1) * 4 + 1];
      const w = chemPixels[k * 4 + 3];
      const observed = wrapAnglePi(phi - prevPhi);
      const expected = wrapAnglePi(Q_PITCH * DS + w);
      phaseChecks++;
      if (Math.abs(observed - expected) <= 0.12) phasePass++;
    }
  }

  const meanGap = activeNodes > 0 ? gapSum / activeNodes : 0;
  const zipBoundPass = (
    (scenarioName === 'zip' || scenarioName === 'rezip') ? (meanGap <= 0.35) :
    (scenarioName === 'unzip' || scenarioName === 'reroute') ? (meanGap >= 0.75) :
    (meanGap >= 0.0 && meanGap <= 1.25)
  );

  const metrics = {
    helixRadiusTolerance: { pass: radiusPass, total: radiusChecks, ratio: radiusChecks ? radiusPass / radiusChecks : 0 },
    pitchPhaseConsistency: { pass: phasePass, total: phaseChecks, ratio: phaseChecks ? phasePass / phaseChecks : 0 },
    hubMidpointRelation: { pass: hubPass, total: hubChecks, ratio: hubChecks ? hubPass / hubChecks : 0 },
    rungOrdering: { pass: rungPass, total: rungChecks, ratio: rungChecks ? rungPass / rungChecks : 0 },
    zipBoundBehavior: { pass: zipBoundPass ? 1 : 0, total: 1, ratio: zipBoundPass ? 1 : 0, meanGap: round3(meanGap), zipMode: round3(zipMode) }
  };

  const allPass = Object.values(metrics).every((m) => m.ratio >= 0.9 || (m.total === 1 && m.pass === 1));
  return { allPass, metrics, activeNodes };
}

function triggerJsonDownload(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function createAcceptanceValidationRunner(config) {
  const {
    renderer,
    sysA,
    constants,
    emState,
    setTargetZipMode,
    setFlowEnabled,
    setEMEnabled,
    setEMRadius,
    setEMTwist,
    getZipMode,
    texSize
  } = config;

  const seed = 1337;
  const scenarios = makeScenarioPlan(seed);
  const results = {
    suite: 'dna-acceptance-live-texture',
    version: 1,
    deterministicSeed: seed,
    createdAt: new Date().toISOString(),
    scenarios: [],
    summary: { passed: 0, failed: 0, total: scenarios.length, status: 'running' }
  };

  window.DNAValidationResults = results;
  window.downloadDNAValidationResults = () => triggerJsonDownload(results, `dna-validation-${Date.now()}.json`);

  let scenarioIndex = 0;
  let frameInScenario = 0;
  let finalized = false;

  function applyScenario(s) {
    setTargetZipMode(s.zipTarget);
    setFlowEnabled(s.flowEnabled);
    setEMEnabled(s.emEnabled);
    setEMRadius(s.emRadius);
    setEMTwist(s.emTwist);
  }

  function completeScenario(s) {
    const posPixels = readTexturePixels(renderer, sysA.gpu.getCurrentRenderTarget(sysA.posVar), texSize);
    const chemPixels = readTexturePixels(renderer, sysA.gpu.getCurrentRenderTarget(sysA.chemVar), texSize);
    const invariantReport = evaluateInvariants({
      posPixels,
      chemPixels,
      constants,
      zipMode: getZipMode(),
      scenarioName: s.name
    });

    const record = {
      name: s.name,
      seed: s.seed,
      frames: s.frames,
      parameterSnapshot: {
        zipTarget: s.zipTarget,
        flowEnabled: s.flowEnabled,
        emEnabled: s.emEnabled,
        emRadius: round3(s.emRadius),
        emTwist: round3(s.emTwist),
        emMode: emState.mode
      },
      result: invariantReport
    };
    results.scenarios.push(record);
    if (invariantReport.allPass) results.summary.passed += 1;
    else results.summary.failed += 1;
    window.DNAValidationResults = results;
  }

  applyScenario(scenarios[0]);

  return {
    update() {
      if (finalized) return;
      const s = scenarios[scenarioIndex];
      frameInScenario += 1;
      if (frameInScenario < s.frames) return;

      completeScenario(s);
      scenarioIndex += 1;
      frameInScenario = 0;

      if (scenarioIndex >= scenarios.length) {
        finalized = true;
        results.summary.status = results.summary.failed === 0 ? 'pass' : 'fail';
        console.log('[Validation] Completed deterministic scenario suite.', results);
        return;
      }

      applyScenario(scenarios[scenarioIndex]);
    },

    getHudSummary() {
      const current = scenarios[Math.min(scenarioIndex, scenarios.length - 1)];
      return {
        status: results.summary.status,
        passed: results.summary.passed,
        failed: results.summary.failed,
        total: results.summary.total,
        currentScenario: finalized ? 'complete' : `${current.name} (${frameInScenario}/${current.frames})`
      };
    }
  };
}
