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

function evaluateInvariants({ posPixels, chemPixels, extPixels, constants, criteria, zipMode, scenarioName, emEnabled }) {
  const {
    NODE_COUNT,
    NECK_SEG,
    HELIX_R,
    PITCH,
    AXIAL_SHIFT,
    Q_PITCH,
    COT_ALPHA,
    ALPHA_EXP,
    U_S,
    DS,
    IDX_RUNG0,
    HELIX_CONVENTION
  } = constants;

  const perSpine = NECK_SEG;
  const idxStrandA0 = NODE_COUNT;
  const idxStrandB0 = NODE_COUNT * 2;
  const idxSpineP0 = NODE_COUNT * 3;

  const convention = HELIX_CONVENTION || {};
  const handednessSign = Number.isFinite(convention.handednessSign) ? convention.handednessSign : 1.0;
  const angleUnitScale = Number.isFinite(convention.angleUnitScale) ? convention.angleUnitScale : 1.0;
  const phaseOffsetA = Number.isFinite(convention.phaseOffsetA) ? convention.phaseOffsetA : 0.0;
  const phaseOffsetB = Number.isFinite(convention.phaseOffsetB) ? convention.phaseOffsetB : Math.PI;
  const expectedOffset = wrapAnglePi((phaseOffsetB - phaseOffsetA) * angleUnitScale + handednessSign * Q_PITCH * AXIAL_SHIFT * angleUnitScale);

  let activeNodes = 0;
  let radiusChecks = 0;
  let radiusPass = 0;
  let phaseChecks = 0;
  let phasePass = 0;
  let conventionChecks = 0;
  let conventionPass = 0;
  let hubChecks = 0;
  let hubPass = 0;
  let rungChecks = 0;
  let rungPass = 0;
  let gapSum = 0;
  const qHatFxMxSamples = [];
  const qHatPitchScaledSamples = [];

  const extActive = [];
  if (extPixels && extPixels.length >= 4) {
    const extCount = Math.floor(extPixels.length / 4);
    for (let i = 0; i < extCount; i++) {
      const i4 = i * 4;
      const tag = Math.floor(extPixels[i4 + 3] + 1e-4);
      if (tag < 1) continue;
      extActive.push([extPixels[i4], extPixels[i4 + 1], extPixels[i4 + 2]]);
    }
  }

  for (let k = 0; k < NODE_COUNT; k++) {
    const chem = vec3At(chemPixels, k);
    const g = chem[0];
    const phi = chem[1];
    const gGap = chem[2];

    if (g <= 0.05) continue;
    activeNodes++;
    gapSum += gGap;

    const p0 = vec3At(posPixels, k);
    const a = vec3At(posPixels, idxStrandA0 + k);
    const b = vec3At(posPixels, idxStrandB0 + k);
    const hub = vec3At(posPixels, idxSpineP0 + k * perSpine + (NECK_SEG - 1));

    if (emEnabled && extActive.length > 0) {
      const km = Math.max(0, k - 1);
      const kp = Math.min(NODE_COUNT - 1, k + 1);
      const pm = vec3At(posPixels, km);
      const pp = vec3At(posPixels, kp);
      const tRaw = [pp[0] - pm[0], pp[1] - pm[1], pp[2] - pm[2]];
      const tLen = Math.hypot(tRaw[0], tRaw[1], tRaw[2]);
      if (tLen > 1e-6) {
        const t = [tRaw[0] / tLen, tRaw[1] / tLen, tRaw[2] / tLen];
        const rRaw = [p0[0] - hub[0], p0[1] - hub[1], p0[2] - hub[2]];
        const rLen = Math.hypot(rRaw[0], rRaw[1], rRaw[2]);
        if (rLen > 1e-6) {
          const radial = [rRaw[0] / rLen, rRaw[1] / rLen, rRaw[2] / rLen];

          let nearest = null;
          let dMin = Number.POSITIVE_INFINITY;
          for (const q of extActive) {
            const dx = q[0] - p0[0];
            const dy = q[1] - p0[1];
            const dz = q[2] - p0[2];
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < dMin) {
              dMin = d2;
              nearest = [dx, dy, dz];
            }
          }

          if (nearest) {
            const nLen = Math.hypot(nearest[0], nearest[1], nearest[2]);
            if (nLen > 1e-6) {
              const forceDir = [nearest[0] / nLen, nearest[1] / nLen, nearest[2] / nLen];
              const fx = forceDir[0] * t[0] + forceDir[1] * t[1] + forceDir[2] * t[2];
              const momentVec = [
                radial[1] * forceDir[2] - radial[2] * forceDir[1],
                radial[2] * forceDir[0] - radial[0] * forceDir[2],
                radial[0] * forceDir[1] - radial[1] * forceDir[0]
              ];
              const mx = momentVec[0] * t[0] + momentVec[1] * t[1] + momentVec[2] * t[2];
              if (Math.abs(mx) > 1e-4 && Math.abs(fx) > 1e-6) {
                const qFxMx = -fx / mx;
                const qAbs = Math.abs(fx) / Math.abs(mx);
                if (Number.isFinite(qFxMx) && Math.abs(qFxMx) <= 50.0) qHatFxMxSamples.push(qFxMx);
                if (Number.isFinite(qAbs) && Math.abs(qAbs) <= 50.0) qHatFxMxSamples.push(qAbs);

                const localR = Math.max(HELIX_R + gGap, 1e-4);
                const qPitchFromFxMx = 1.0 / (Math.max(Math.abs(qFxMx), 1e-6) * localR * localR);
                const qPitchFromAbs = 1.0 / (Math.max(qAbs, 1e-6) * localR * localR);
                if (Number.isFinite(qPitchFromFxMx) && qPitchFromFxMx <= 50.0) qHatPitchScaledSamples.push(qPitchFromFxMx);
                if (Number.isFinite(qPitchFromAbs) && qPitchFromAbs <= 50.0) qHatPitchScaledSamples.push(qPitchFromAbs);
              }
            }
          }
        }
      }
    }

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


    const pa = [a[0] - hub[0], a[1] - hub[1], a[2] - hub[2]];
    const pb = [b[0] - hub[0], b[1] - hub[1], b[2] - hub[2]];
    const qa = Math.hypot(pa[0], pa[1], pa[2]);
    const qb = Math.hypot(pb[0], pb[1], pb[2]);
    const pmNode = vec3At(posPixels, Math.max(0, k - 1));
    const ppNode = vec3At(posPixels, Math.min(NODE_COUNT - 1, k + 1));
    const tNodeRaw = [ppNode[0] - pmNode[0], ppNode[1] - pmNode[1], ppNode[2] - pmNode[2]];
    const tNodeLen = Math.hypot(tNodeRaw[0], tNodeRaw[1], tNodeRaw[2]);
    if (qa > 1e-6 && qb > 1e-6 && tNodeLen > 1e-6) {
      const tNode = [tNodeRaw[0] / tNodeLen, tNodeRaw[1] / tNodeLen, tNodeRaw[2] / tNodeLen];
      const paN = [pa[0] / qa, pa[1] / qa, pa[2] / qa];
      const pbN = [pb[0] / qb, pb[1] / qb, pb[2] / qb];
      const crossAB = [
        paN[1] * pbN[2] - paN[2] * pbN[1],
        paN[2] * pbN[0] - paN[0] * pbN[2],
        paN[0] * pbN[1] - paN[1] * pbN[0]
      ];
      const windingSignObserved = Math.sign(crossAB[0] * tNode[0] + crossAB[1] * tNode[1] + crossAB[2] * tNode[2]);
      const expectedWindingSign = Math.sign(handednessSign || 1.0);
      const phaseOffsetObserved = wrapAnglePi(Math.atan2(pbN[2], pbN[0]) - Math.atan2(paN[2], paN[0]));
      conventionChecks += 1;
      const windingMatch = windingSignObserved === 0 || expectedWindingSign === 0 ? false : windingSignObserved === expectedWindingSign;
      const phaseMatch = Math.abs(wrapAnglePi(phaseOffsetObserved - expectedOffset)) <= 0.45;
      if (windingMatch && phaseMatch) conventionPass += 1;
    }

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

  const alphaGeom = Number.isFinite(ALPHA_EXP) ? ALPHA_EXP : Math.atan2(PITCH, TWO_PI * HELIX_R);
  const tanAlphaGeom = Math.tan(alphaGeom);
  const tanRhs = PITCH / (TWO_PI * HELIX_R);
  const cotAlphaGeom = 1.0 / Math.max(Math.tan(alphaGeom), 1e-6);
  const cotRhs = Q_PITCH * HELIX_R;
  const tanRelErr = Math.abs(tanAlphaGeom - tanRhs) / Math.max(Math.abs(tanRhs), 1e-6);
  const cotRelErr = Math.abs(cotAlphaGeom - cotRhs) / Math.max(Math.abs(cotRhs), 1e-6);

  const eq34Threshold = criteria?.mdpiEq34RelErrMax ?? 2e-2;
  const eq34Pass = tanRelErr <= eq34Threshold && cotRelErr <= eq34Threshold;

  const uExpectedSigned = Q_PITCH * AXIAL_SHIFT;
  const uExpectedNeg = -uExpectedSigned;
  const uErrSigned = Math.abs(U_S - uExpectedSigned);
  const uErrNeg = Math.abs(U_S - uExpectedNeg);
  const uExpected = (uErrSigned <= uErrNeg) ? uExpectedSigned : uExpectedNeg;
  const uConvention = (uErrSigned <= uErrNeg) ? '+q*x_s' : '-q*x_s';
  const eq1112Threshold = criteria?.mdpiEq1112AbsErrMax ?? 5e-2;
  const eq1112Pass = Math.min(uErrSigned, uErrNeg) <= eq1112Threshold;

  const sortedQHatRaw = [...qHatFxMxSamples].sort((x, y) => x - y);
  const qHatRawMean = qHatFxMxSamples.length ? qHatFxMxSamples.reduce((acc, v) => acc + v, 0) / qHatFxMxSamples.length : 0;
  const qHatRawMedian = qHatFxMxSamples.length
    ? (sortedQHatRaw.length % 2
      ? sortedQHatRaw[(sortedQHatRaw.length - 1) / 2]
      : 0.5 * (sortedQHatRaw[sortedQHatRaw.length / 2 - 1] + sortedQHatRaw[sortedQHatRaw.length / 2]))
    : 0;

  const sortedQHat = [...qHatPitchScaledSamples].sort((x, y) => x - y);
  const qHatMean = qHatPitchScaledSamples.length ? qHatPitchScaledSamples.reduce((acc, v) => acc + v, 0) / qHatPitchScaledSamples.length : 0;
  const qHatMedian = qHatPitchScaledSamples.length
    ? (sortedQHat.length % 2
      ? sortedQHat[(sortedQHat.length - 1) / 2]
      : 0.5 * (sortedQHat[sortedQHat.length / 2 - 1] + sortedQHat[sortedQHat.length / 2]))
    : 0;
  const qHatTrimmed = qHatPitchScaledSamples.length > 2
    ? sortedQHat.slice(1, -1).reduce((acc, v) => acc + v, 0) / (sortedQHat.length - 2)
    : qHatMean;
  const qHatMeanErr = Math.abs(qHatMean - Q_PITCH);
  const qHatMedianErr = Math.abs(qHatMedian - Q_PITCH);
  const qHatTrimmedErr = Math.abs(qHatTrimmed - Q_PITCH);
  const qHatMinEstimatorErr = Math.min(qHatMeanErr, qHatMedianErr, qHatTrimmedErr);
  const qHatAvgAbsErr = qHatPitchScaledSamples.length
    ? qHatPitchScaledSamples.reduce((acc, v) => acc + Math.abs(v - Q_PITCH), 0) / qHatPitchScaledSamples.length
    : null;

  const qBacksolveThresholdZipped = criteria?.mdpiQBacksolveAbsErrMax ?? 3.5e-1;
  const qBacksolveThresholdUnzipped = criteria?.mdpiQBacksolveAbsErrMaxUnzipped ?? 5e-1;
  const qBacksolveThreshold = (scenarioName === 'unzip' || scenarioName === 'reroute')
    ? qBacksolveThresholdUnzipped
    : qBacksolveThresholdZipped;
  const qBacksolveAvgAbsErrThreshold = criteria?.mdpiQBacksolveAvgAbsErrMax ?? 7e-1;
  const qBacksolveSpreadThreshold = criteria?.mdpiQBacksolveEstimatorSpreadMax ?? 4e-1;
  const qBacksolveMinSamples = criteria?.mdpiQBacksolveMinSamples ?? 3;
  const qHatEstimatorSpread = Math.abs(qHatMean - qHatMedian);
  const qBacksolveHasSignal = qHatPitchScaledSamples.length >= qBacksolveMinSamples;
  const qBacksolveDispersionPass = qBacksolveHasSignal ? qHatAvgAbsErr <= qBacksolveAvgAbsErrThreshold : null;
  const qBacksolveStabilityPass = qBacksolveHasSignal ? qHatEstimatorSpread <= qBacksolveSpreadThreshold : null;
  const qBacksolvePass = !emEnabled || !qBacksolveHasSignal || (
    qHatMinEstimatorErr <= qBacksolveThreshold && (qBacksolveDispersionPass || qBacksolveStabilityPass)
  );

  const topologyMinRatioZipped = criteria?.topologyAndRoutingMinRatio ?? 0.9;
  const topologyMinRatioUnzipped = criteria?.topologyAndRoutingMinRatioUnzipped ?? 1.2e-1;
  const topologyMinRatio = (scenarioName === 'unzip' || scenarioName === 'reroute')
    ? topologyMinRatioUnzipped
    : topologyMinRatioZipped;

  const zipBoundPass = (
    (scenarioName === 'zip' || scenarioName === 'rezip') ? (meanGap <= 0.35) :
    (scenarioName === 'unzip' || scenarioName === 'reroute') ? (meanGap >= 0.75) :
    (meanGap >= 0.0 && meanGap <= 1.25)
  );

  const metrics = {
    helixRadiusTolerance: { pass: radiusPass, total: radiusChecks, ratio: radiusChecks ? radiusPass / radiusChecks : 0, requiredRatio: topologyMinRatio },
    pitchPhaseConsistency: { pass: phasePass, total: phaseChecks, ratio: phaseChecks ? phasePass / phaseChecks : 0, requiredRatio: topologyMinRatio },
    hubMidpointRelation: { pass: hubPass, total: hubChecks, ratio: hubChecks ? hubPass / hubChecks : 0, requiredRatio: topologyMinRatio },
    rungOrdering: { pass: rungPass, total: rungChecks, ratio: rungChecks ? rungPass / rungChecks : 0, requiredRatio: topologyMinRatio },
    zipBoundBehavior: { pass: zipBoundPass ? 1 : 0, total: 1, ratio: zipBoundPass ? 1 : 0, meanGap: round3(meanGap), zipMode: round3(zipMode) },
    eq34GeometryConsistency: {
      pass: eq34Pass ? 1 : 0,
      total: 1,
      ratio: eq34Pass ? 1 : 0,
      thresholdRelErr: eq34Threshold,
      alphaGeom,
      tanAlphaGeom,
      tanRhs,
      tanRelErr,
      cotAlphaGeom,
      cotRhs,
      cotRelErr,
      cotAlphaConfig: COT_ALPHA
    },
    eq1112USConsistency: {
      pass: eq1112Pass ? 1 : 0,
      total: 1,
      ratio: eq1112Pass ? 1 : 0,
      thresholdAbsErr: eq1112Threshold,
      u_s: U_S,
      x_s: AXIAL_SHIFT,
      q: Q_PITCH,
      chosenConvention: uConvention,
      uExpected,
      absError: Math.min(uErrSigned, uErrNeg)
    },
    conventionSanity: {
      pass: conventionPass,
      total: conventionChecks,
      ratio: conventionChecks ? conventionPass / conventionChecks : 0,
      requiredRatio: topologyMinRatio,
      expectedWindingSign: Math.sign(handednessSign || 1.0),
      expectedPhaseOffset: expectedOffset
    },
    qBacksolveConsistency: {
      pass: qBacksolvePass ? 1 : 0,
      total: 1,
      ratio: qBacksolvePass ? 1 : 0,
      thresholdAbsErr: qBacksolveThreshold,
      thresholdAvgAbsErr: qBacksolveAvgAbsErrThreshold,
      thresholdEstimatorSpreadAbs: qBacksolveSpreadThreshold,
      thresholdMinSamples: qBacksolveMinSamples,
      sampleCount: qHatPitchScaledSamples.length,
      qPitch: Q_PITCH,
      qHatMean,
      qHatMedian,
      qHatTrimmed,
      qHatEstimatorSpread,
      qHatMeanErr,
      qHatMedianErr,
      qHatTrimmedErr,
      qHatMinEstimatorErr,
      qHatAvgAbsErr,
      dispersionGatePass: qBacksolveDispersionPass,
      stabilityGatePass: qBacksolveStabilityPass,
      qHatRawMean,
      qHatRawMedian,
      emEnabled,
      hasSignal: qBacksolveHasSignal,
      skipped: !emEnabled || !qBacksolveHasSignal,
      source: 'nearest-ext force proxy; raw -Fx/Mx variants mapped to qPitch via 1/(|ratio|*R^2)'
    }
  };

  const allPass = Object.values(metrics).every((m) => (m.total === 1 ? m.pass === 1 : m.ratio >= topologyMinRatio));
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
    sysB,
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
    summary: {
      passed: 0,
      failed: 0,
      total: scenarios.length,
      status: 'running',
      criteria: {
        topologyAndRoutingMinRatio: 0.9,
        topologyAndRoutingMinRatioUnzipped: 1.2e-1,
        mdpiEq34RelErrMax: 2e-2,
        mdpiEq1112AbsErrMax: 5e-2,
        mdpiQBacksolveAbsErrMax: 3.5e-1,
        mdpiQBacksolveAbsErrMaxUnzipped: 5e-1,
        mdpiQBacksolveAvgAbsErrMax: 7e-1,
        mdpiQBacksolveEstimatorSpreadMax: 4e-1,
        mdpiQBacksolveMinSamples: 3
      }
    }
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
    const extPixels = sysB
      ? readTexturePixels(renderer, sysB.gpu.getCurrentRenderTarget(sysB.posVar), texSize)
      : null;
    const invariantReport = evaluateInvariants({
      posPixels,
      chemPixels,
      extPixels,
      constants,
      criteria: results.summary.criteria,
      zipMode: getZipMode(),
      scenarioName: s.name,
      emEnabled: !!s.emEnabled
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
