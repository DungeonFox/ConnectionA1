export function applyMovementUniforms({ sysA, sysB, t, dt, frame, zipMode }) {
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

  sysB.posTargetVar.material.uniforms.time.value = t;
  sysB.accVar.material.uniforms.time.value = t;
  sysB.velVar.material.uniforms.time.value = t;
  sysB.posVar.material.uniforms.time.value = t;
}

export function stepMovementCompute({ sysA, sysB }) {
  sysB.gpu.compute();
  sysA.gpu.compute();
}

export function bindMovementTextures({ sysA, sysB }) {
  sysB.mat.uniforms.posTarget.value = sysB.gpu.getCurrentRenderTarget(sysB.posTargetVar).texture;
  sysB.mat.uniforms.acc.value = sysB.gpu.getCurrentRenderTarget(sysB.accVar).texture;
  sysB.mat.uniforms.vel.value = sysB.gpu.getCurrentRenderTarget(sysB.velVar).texture;
  sysB.mat.uniforms.pos.value = sysB.gpu.getCurrentRenderTarget(sysB.posVar).texture;

  sysA.mat.uniforms.posTarget.value = sysA.gpu.getCurrentRenderTarget(sysA.posTargetVar).texture;
  sysA.mat.uniforms.acc.value = sysA.gpu.getCurrentRenderTarget(sysA.accVar).texture;
  sysA.mat.uniforms.vel.value = sysA.gpu.getCurrentRenderTarget(sysA.velVar).texture;
  sysA.mat.uniforms.pos.value = sysA.gpu.getCurrentRenderTarget(sysA.posVar).texture;
  sysA.mat.uniforms.chem.value = sysA.gpu.getCurrentRenderTarget(sysA.chemVar).texture;
}
