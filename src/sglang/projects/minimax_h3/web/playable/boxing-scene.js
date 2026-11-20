import * as THREE from './vendor/three-0.180.0/three.module.min.js';
import {CAPSULES, createMatch, fighterPose, startAttack, stepMatch} from './boxing-logic.mjs';

export function createBoxingScene(width, height, onStatus) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(46, width / height, .1, 80);
  camera.position.set(0, 2.8, 5.4);
  camera.lookAt(0, 1, 0);
  const gray = new THREE.ShaderMaterial({
    vertexShader: `varying float opticalZ;
      void main() { vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        opticalZ = -viewPosition.z; gl_Position = projectionMatrix * viewPosition; }`,
    fragmentShader: `varying float opticalZ;
      void main() { float v = floor(clamp(230.0 - 36.0 * log(1.0 + max(opticalZ, 0.0)), 45.0, 230.0)) / 255.0;
        gl_FragColor = vec4(v, v, v, 1.0); }`,
  });
  const human = new THREE.ShaderMaterial({
    vertexShader: `void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `void main() { gl_FragColor = vec4(60.0/255.0, 169.0/255.0, 239.0/255.0, 1.0); }`,
  });
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 8);
  const sphere = new THREE.SphereGeometry(1, 8, 6);
  const box = new THREE.BoxGeometry(1, 1, 1);
  const geometries = [cylinder, sphere, box];
  const between = (a, b, radius, material = gray) => {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    const mesh = new THREE.Mesh(cylinder, material);
    mesh.position.copy(start).add(end).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.sub(start).normalize());
    mesh.scale.set(radius, start.distanceTo(new THREE.Vector3(...b)), radius);
    scene.add(mesh);
  };
  const platform = new THREE.Mesh(box, gray);
  platform.position.set(0, -.17, 0);
  platform.scale.set(5.3, .3, 3.9);
  scene.add(platform);
  const ground = new THREE.Mesh(box, gray);
  ground.position.set(0, -.37, 0);
  ground.scale.set(200, .1, 200);
  scene.add(ground);
  for (const x of [-2.52, 2.52]) for (const z of [-1.82, 1.82]) {
    between([x, 0, z], [x, 1.08, z], .055);
  }
  for (const y of [.43, .78, 1.05]) {
    for (const z of [-1.82, 1.82]) between([-2.52, y, z], [2.52, y, z], .022);
    for (const x of [-2.52, 2.52]) between([x, y, -1.82], [x, y, 1.82], .022);
  }
  const bodies = [0, 1].map(() => CAPSULES.map(([, , , radius]) => {
    const shaft = new THREE.Mesh(cylinder, human);
    const capA = new THREE.Mesh(sphere, human);
    const capB = new THREE.Mesh(sphere, human);
    capA.scale.setScalar(radius);
    capB.scale.setScalar(radius);
    scene.add(shaft, capA, capB);
    return {shaft, capA, capB, radius};
  }));
  let match = createMatch(), lastStatus = '';
  const updatePose = () => {
    for (let i = 0; i < 2; i++) {
      const points = fighterPose(match.fighters[i], match.fighters[1 - i]);
      for (let j = 0; j < CAPSULES.length; j++) {
        const [, a, b] = CAPSULES[j];
        const {shaft, capA, capB, radius} = bodies[i][j];
        const start = new THREE.Vector3(...points[a]), end = new THREE.Vector3(...points[b]);
        const delta = end.clone().sub(start), length = delta.length();
        capA.position.copy(start);
        capB.position.copy(end);
        shaft.visible = length > .0001;
        if (shaft.visible) {
          shaft.position.copy(start).add(end).multiplyScalar(.5);
          shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), delta.normalize());
          shaft.scale.set(radius, length, radius);
        }
      }
    }
    const status = `${match.phase}:${match.fighters[0].hp}:${match.fighters[1].hp}:${match.winner}`;
    if (status !== lastStatus) { lastStatus = status; onStatus?.({phase: match.phase,
      playerHp: match.fighters[0].hp, opponentHp: match.fighters[1].hp, winner: match.winner}); }
  };
  const reset = () => { match = createMatch(); updatePose(); };
  const press = code => {
    if (code === 'KeyR') { reset(); return; }
    if (match.phase === 'result') return;
    if (code === 'KeyJ' || code === 'KeyK') {
      if (startAttack(match.fighters[0], code === 'KeyJ' ? 'jab' : 'heavy')) match.phase = 'playing';
      updatePose();
    }
  };
  const render = (dt, keys) => {
    stepMatch(match, dt, {up: keys.has('KeyW'), down: keys.has('KeyS'), left: keys.has('KeyA'), right: keys.has('KeyD')});
    updatePose();
  };
  reset();
  return {scene, camera, render, reset, getSimulationState: () => structuredClone(match), press,
    dispose() { for (const geometry of geometries) geometry.dispose(); gray.dispose(); human.dispose(); }};
}
