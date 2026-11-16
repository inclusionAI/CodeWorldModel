import * as THREE from './vendor/three-0.180.0/three.module.min.js';
import {createRace, ROAD_HALF_WIDTH, stepRace, trackPose, TRACK_LENGTH} from './racing-logic.mjs';

export function createRacingScene(width, height, onStatus) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(62, width / height, .1, 350);
  const geometries = [], materials = [];
  const gray = shade();
  const red = color([209, 58, 48]), blue = color([60, 169, 239]), green = color([79, 149, 80]);
  const cube = geometry(new THREE.BoxGeometry(1, 1, 1));
  const cylinder = geometry(new THREE.CylinderGeometry(1, 1, 1, 7));
  const cone = geometry(new THREE.ConeGeometry(1, 1, 7));
  const sphere = geometry(new THREE.SphereGeometry(1, 8, 6));
  let race = createRace(), lastStatus = '';

  function geometry(value) { geometries.push(value); return value; }
  function color(rgb) {
    const value = new THREE.ShaderMaterial({uniforms: {semantic: {value: new THREE.Vector3(...rgb.map(c => c / 255))}},
      vertexShader: `void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform vec3 semantic; void main() { gl_FragColor = vec4(semantic, 1.0); }`});
    materials.push(value); return value;
  }
  function shade() {
    const value = new THREE.ShaderMaterial({
      vertexShader: `varying float opticalZ;
        void main() { vec4 p = modelViewMatrix * vec4(position, 1.0);
          opticalZ = -p.z; gl_Position = projectionMatrix * p; }`,
      fragmentShader: `varying float opticalZ;
        void main() { float v = floor(clamp(230.0 - 36.0 * log(1.0 + max(opticalZ, 0.0)), 45.0, 230.0)) / 255.0;
          gl_FragColor = vec4(vec3(v), 1.0); }`});
    materials.push(value); return value;
  }
  function box(parent, material, x, y, z, sx, sy, sz) {
    const mesh = new THREE.Mesh(cube, material);
    mesh.position.set(x, y, z); mesh.scale.set(sx, sy, sz); parent.add(mesh);
    return mesh;
  }
  function rod(parent, material, a, b, radius, shape = cube) {
    const start = new THREE.Vector3(...a), end = new THREE.Vector3(...b);
    const mesh = new THREE.Mesh(shape, material);
    mesh.position.copy(start).add(end).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(start).normalize());
    mesh.scale.set(radius, start.distanceTo(end), radius); parent.add(mesh);
  }
  function ribbon(inner, outer, y, material) {
    const positions = [], indices = [];
    for (let d = 0; d <= TRACK_LENGTH; d += 5) {
      for (const lateral of [inner, outer]) {
        const p = trackPose(d, lateral);
        positions.push(p.x, y, p.z);
      }
      if (d) { const i = positions.length / 3 - 4; indices.push(i, i + 1, i + 2, i + 1, i + 3, i + 2); }
    }
    const meshGeometry = geometry(new THREE.BufferGeometry());
    meshGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    meshGeometry.setIndex(indices);
    meshGeometry.computeVertexNormals();
    const mesh = new THREE.Mesh(meshGeometry, material);
    mesh.material.side = THREE.DoubleSide;
    scene.add(mesh);
  }

  box(scene, gray, 82, -.5, -120, 520, .8, 520);
  ribbon(-ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, .08, gray);
  for (const side of [-1, 1]) {
    ribbon(side * 5, side * 5.6, .15, gray);
    ribbon(side * 5.6, side * 8, .025, gray);
    for (let d = 0; d < TRACK_LENGTH; d += 25) {
      const p = trackPose(d, side * 8);
      box(scene, gray, p.x, .45, p.z, .13, .9, .13);
    }
    for (let d = 0; d < TRACK_LENGTH; d += 5) {
      const a = trackPose(d, side * 8), b = trackPose(d + 5, side * 8);
      rod(scene, gray, [a.x, .68, a.z], [b.x, .68, b.z], .07);
    }
    for (let d = 25; d < TRACK_LENGTH; d += 65) {
      const p = trackPose(d, side * (20 + d % 3));
      box(scene, gray, p.x, 1.3, p.z, .35, 2.6, .35);
      const crown = new THREE.Mesh(cone, green);
      crown.position.set(p.x, 3.5, p.z); crown.scale.set(1.7, 4, 1.7); scene.add(crown);
    }
  }
  for (let d = 90; d < TRACK_LENGTH; d += 125) {
    const p = trackPose(d, d % 2 ? -38 : 38);
    box(scene, gray, p.x, 4, p.z, 10, 8, 10);
  }
  const start = trackPose(0);
  box(scene, gray, start.x, .1, start.z, 10, .04, .25);

  function makeCar() {
    const group = new THREE.Group(); scene.add(group);
    const corners = [-1, 1];
    for (const x of corners) for (const z of corners) rod(group, red, [x * .95, .35, z * 2.1], [x * .95, 1.85, z * 2.1], .085);
    for (const y of [.35, 1.85]) {
      for (const z of corners) rod(group, red, [-.95, y, z * 2.1], [.95, y, z * 2.1], .085);
      for (const x of corners) rod(group, red, [x * .95, y, -2.1], [x * .95, y, 2.1], .085);
    }
    const head = new THREE.Mesh(sphere, blue);
    head.position.set(0, 1.57, .4); head.scale.setScalar(.24); group.add(head);
    rod(group, blue, [0, .75, .45], [0, 1.36, .4], .22, cylinder);
    for (const side of [-1, 1]) {
      rod(group, blue, [side * .15, 1.28, .4], [side * .52, 1.04, -.18], .1, cylinder);
      rod(group, blue, [side * .14, .75, .45], [side * .35, .52, -.18], .13, cylinder);
      rod(group, blue, [side * .35, .52, -.18], [side * .35, .38, -.65], .11, cylinder);
    }
    return group;
  }
  const cars = [makeCar(), makeCar()];
  function update() {
    for (const [index, car] of [race.player, race.opponent].entries()) {
      const pose = trackPose(car.distance, car.lateral);
      cars[index].position.set(pose.x, .08, pose.z);
      cars[index].rotation.y = -(pose.heading + car.headingOffset);
    }
    const player = trackPose(race.player.distance, race.player.lateral);
    camera.position.set(player.x - Math.sin(player.heading) * 8, 5.2, player.z + Math.cos(player.heading) * 8);
    camera.lookAt(player.x + Math.sin(player.heading) * 13, 1.1, player.z - Math.cos(player.heading) * 13);
    const status = `${race.phase}:${Math.ceil(race.countdown)}:${Math.floor(race.player.distance / TRACK_LENGTH)}:${Math.round(race.player.speed * 3.6)}:${Math.abs(race.player.lateral) > ROAD_HALF_WIDTH}`;
    if (status !== lastStatus) {
      lastStatus = status;
      onStatus?.({phase: race.phase, countdown: Math.ceil(race.countdown), laps: Math.floor(race.player.distance / TRACK_LENGTH),
        speed: Math.round(race.player.speed * 3.6), offroad: Math.abs(race.player.lateral) > ROAD_HALF_WIDTH});
    }
  }
  const reset = () => { race = createRace(); lastStatus = ''; update(); };
  const render = (dt, keys) => {
    stepRace(race, dt, {throttle: keys.has('KeyW') || keys.has('ArrowUp'), brake: keys.has('KeyS') || keys.has('ArrowDown'),
      left: keys.has('KeyA') || keys.has('ArrowLeft'), right: keys.has('KeyD') || keys.has('ArrowRight')});
    update();
  };
  reset();
  return {scene, camera, render, reset, getSimulationState: () => structuredClone(race), press() {}, dispose() {
    for (const item of geometries) item.dispose();
    for (const item of materials) item.dispose();
  }};
}
