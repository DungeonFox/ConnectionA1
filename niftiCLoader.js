// niftiCLoader.js (stub)
// This demo doesn't require NIFTI, but materials.js imports this module.
// We provide a tiny placeholder so the project runs without external assets.

import * as THREE from 'three';

export async function loadNiftiRGBA(/* url */) {
  const data = new Uint8Array([0, 0, 0, 255]);
  const tex = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat);
  tex.needsUpdate = true;
  return { texture: tex, maxValue: 1.0 };
}
