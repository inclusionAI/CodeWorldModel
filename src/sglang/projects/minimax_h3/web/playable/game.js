import * as THREE from './vendor/three-0.180.0/three.module.min.js';
import {createBoxingScene} from './boxing-scene.js';
import {createRacingScene} from './racing-scene.js';

const CONTROLS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
export const SCENES = {
  boxing: {style: 'synthetic-boxing-depth-semantic-rgb/v1', help: '点击画面后按 WASD 移动，J 轻拳，K 重拳。R 重开，串流期间禁用。'},
  racing: {style: 'synthetic-racing-depth-semantic-rgb/v1', help: '点击画面后按 W/↑ 加速，S/↓ 刹车，A/D 或 ←/→ 转向。首次加速开始倒计时。R 重开，串流期间禁用。'},
  street: {style: 'synthetic-street-depth-semantic-rgb/v1', help: '点击画面后使用 WASD 移动，方向键转向，拖动调整视角。'},
};

export function createGame(canvas, { width = 672, height = 384, scene: sceneId = 'street', onInput = () => {}, onStatus = () => {}, canReset = () => true } = {}) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 4096 || height > 4096) {
    throw new RangeError('Game dimensions must be integers between 1 and 4096.');
  }
  if (!SCENES[sceneId]) throw new RangeError('Unknown game scene.');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  renderer.setClearColor(0x698fe0, 1);
  const scene = new THREE.Scene();
  const managedScene = sceneId === 'boxing' ? createBoxingScene(width, height, onStatus)
    : sceneId === 'racing' ? createRacingScene(width, height, onStatus) : null;
  const camera = new THREE.PerspectiveCamera(70, width / height, 0.1, 220);
  camera.rotation.order = 'YXZ';
  const keys = new Set();
  const materials = [];
  const geometries = new Set();
  const obstacles = [];
  const cars = [];
  const people = [];
  let elapsed = 0;
  let yaw = 0;
  let drag = null;
  let disposed = false;
  let controlsSuspended = false;

  // shortcut: sample palette with approximate distance shading, replace when the fusion shader is available.
  function material(rgb) {
    const result = new THREE.ShaderMaterial({
      uniforms: { semantic: { value: new THREE.Vector3(...rgb.map(channel => channel / 255)) } },
      vertexShader: `varying float distanceToCamera;
        void main() {
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          distanceToCamera = length(viewPosition.xyz);
          gl_Position = projectionMatrix * viewPosition;
        }`,
      fragmentShader: `uniform vec3 semantic;
        varying float distanceToCamera;
        void main() {
          float depthShade = mix(1.0, 0.48, smoothstep(2.0, 170.0, distanceToCamera));
          gl_FragColor = vec4(semantic * depthShade, 1.0);
        }`,
    });
    materials.push(result);
    return result;
  }
  if (!managedScene) {
    const road = material([58, 60, 76]);
    const pavement = material([72, 74, 87]);
    const building = material([10, 11, 15]);
    const vegetation = material([79, 149, 80]);
    const vehicle = material([209, 58, 48]);
    const pedestrian = material([60, 169, 239]);
    const pole = material([25, 26, 34]);

    function mesh(geometry, appearance, x, y, z) {
      geometries.add(geometry);
      const result = new THREE.Mesh(geometry, appearance);
      result.position.set(x, y, z);
      scene.add(result);
      return result;
    }
    const cube = new THREE.BoxGeometry(1, 1, 1);
    function box(x, z, sx, sy, sz, appearance, collides = false) {
      const result = mesh(cube, appearance, x, sy / 2, z);
      result.scale.set(sx, sy, sz);
      if (collides) obstacles.push({ x, z, halfX: sx / 2 + 0.4, halfZ: sz / 2 + 0.4 });
      return result;
    }

    box(0, -65, 240, 0.08, 240, road);
    for (const x of [-60, -20, 20, 60]) {
      for (let row = 0; row < 4; row++) {
        const z = -10 - row * 40;
        box(x, z, 28, 0.16, 28, pavement);
        const h = 9 + ((Math.abs(x) + row * 7) % 19);
        box(x - 6.5, z, 11, h, 23, building, true);
        box(x + 6.5, z + 1.5, 11, h * 0.72, 20, building, true);
        for (const side of [-1, 1]) {
          const treeX = x + side * 13;
          mesh(new THREE.CylinderGeometry(0.18, 0.25, 4, 5), vegetation, treeX, 2, z - 10);
          mesh(new THREE.ConeGeometry(1.5, 4, 5), vegetation, treeX, 5, z - 10);
          box(treeX, z + 10, 0.12, 5, 0.12, pole);
        }
      }
    }
    for (let i = 0; i < 10; i++) {
      const lane = i % 2 === 0 ? -2.6 : 2.6;
      const car = box(lane, -20 - i * 14, 1.85, 1.5, 4.2, vehicle);
      cars.push({ mesh: car, start: -20 - i * 14, direction: i % 2 === 0 ? 1 : -1 });
    }
    const capsule = new THREE.CapsuleGeometry(0.28, 1.15, 3, 6);
    for (let i = 0; i < 18; i++) {
      const x = i % 2 === 0 ? -6.5 : 6.5;
      const z = -9 - i * 8;
      const person = mesh(capsule, pedestrian, x, 1, z);
      people.push({ mesh: person, start: z, phase: i * 0.8 });
    }
  }

  const controls = sceneId === 'boxing' ? new Set([...CONTROLS, 'KeyJ', 'KeyK', 'KeyR'])
    : sceneId === 'racing' ? new Set([...CONTROLS, 'KeyR']) : CONTROLS;
  const previousTabIndex = canvas.getAttribute('tabindex');
  const previousLabel = canvas.getAttribute('aria-label');
  const previousTouchAction = canvas.style.touchAction;
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', sceneId === 'boxing' ? 'Playable proxy boxing. W A S D move, J jab, K heavy punch, R reset.'
    : sceneId === 'racing' ? 'Playable proxy racing. W or up accelerate, S or down brake, A D or arrows steer, R reset.'
    : 'Playable proxy street. W A S D move, arrow keys move and turn. Drag to look.');
  canvas.style.touchAction = 'none';

  function keyDown(event) {
    if (!controls.has(event.code)) return;
    event.preventDefault();
    if (ignoreInput({kind: 'keydown', key: event.code, repeat: event.repeat === true})) return;
    if (event.code === 'KeyR') {
      if (canReset()) { reset(); onInput({kind: 'keydown', key: event.code}); }
      return;
    }
    if (!keys.has(event.code)) {
      keys.add(event.code);
      managedScene?.press(event.code);
      onInput({ kind: 'keydown', key: event.code });
    }
  }
  function keyUp(event) {
    if (!controls.has(event.code)) return;
    event.preventDefault();
    if (ignoreInput({kind: 'keyup', key: event.code})) return;
    if (keys.delete(event.code)) onInput({ kind: 'keyup', key: event.code });
  }
  function ignoreInput(event) {
    if (!controlsSuspended) return false;
    onInput({...event, disposition: 'ignored_during_startup_pause'});
    return true;
  }
  function releaseKeys(reason) {
    for (const key of keys) onInput({kind: 'keyup', key,
      ...(reason === 'startup_pause' ? {disposition: 'released_on_startup_pause'} : {})});
    keys.clear();
    const pointer = drag?.id;
    drag = null;
    if (reason === 'startup_pause' && pointer !== undefined && canvas.hasPointerCapture(pointer)) canvas.releasePointerCapture(pointer);
  }
  function setControlsSuspended(value) {
    if (disposed || controlsSuspended === value) return;
    if (value) releaseKeys('startup_pause');
    controlsSuspended = value;
  }
  function getSimulationState() {
    return {scene: sceneId, ...(managedScene ? {state: managedScene.getSimulationState()} : {}),
      elapsed_seconds: elapsed, yaw, x: camera.position.x, z: camera.position.z,
      controls_suspended: controlsSuspended, held_keys: [...keys]};
  }
  function pointerDown(event) {
    if (event.button !== 0) return;
    if (ignoreInput({kind: 'pointerdown', pointer_id: event.pointerId, client_x: event.clientX})) return;
    canvas.focus({ preventScroll: true });
    if (managedScene) return;
    canvas.setPointerCapture(event.pointerId);
    drag = { id: event.pointerId, x: event.clientX };
  }
  function pointerMove(event) {
    if (!drag || event.pointerId !== drag.id) return;
    if (ignoreInput({kind: 'pointermove', pointer_id: event.pointerId, client_x: event.clientX, movement_x: event.movementX})) return;
    const dx = event.clientX - drag.x;
    drag.x = event.clientX;
    if (dx !== 0) {
      yaw -= dx * 0.004;
      onInput({ kind: 'pointer', key: 'yaw', dx });
    }
  }
  function pointerUp(event) {
    if (ignoreInput({kind: event.type || 'pointerup', pointer_id: event.pointerId})) return;
    if (drag?.id !== event.pointerId) return;
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  }
  const handlers = { keydown: keyDown, keyup: keyUp, blur: releaseKeys, pointerdown: pointerDown,
    pointermove: pointerMove, pointerup: pointerUp, pointercancel: pointerUp, lostpointercapture: pointerUp };
  for (const [name, handler] of Object.entries(handlers)) canvas.addEventListener(name, handler);

  function canMove(x, z) {
    return x > -85 && x < 85 && z > -168 && z < 30
      && !obstacles.some(obstacle => Math.abs(x - obstacle.x) < obstacle.halfX && Math.abs(z - obstacle.z) < obstacle.halfZ);
  }
  function render(dtSeconds = 0) {
    if (disposed || controlsSuspended) return;
    const dt = Number.isFinite(dtSeconds) ? Math.max(0, Math.min(dtSeconds, 0.1)) : 0;
    elapsed += dt;
    if (managedScene) {
      managedScene.render(dt, keys);
      renderer.render(managedScene.scene, managedScene.camera);
      return;
    }
    const forward = Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown'));
    const strafe = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
    yaw += (Number(keys.has('ArrowLeft')) - Number(keys.has('ArrowRight'))) * dt * 1.7;
    const distance = dt * 5 / (Math.hypot(forward, strafe) || 1);
    const dx = (-Math.sin(yaw) * forward + Math.cos(yaw) * strafe) * distance;
    const dz = (-Math.cos(yaw) * forward - Math.sin(yaw) * strafe) * distance;
    if (canMove(camera.position.x + dx, camera.position.z)) camera.position.x += dx;
    if (canMove(camera.position.x, camera.position.z + dz)) camera.position.z += dz;
    camera.rotation.set(-0.035, yaw, 0);
    for (const car of cars) car.mesh.position.z = -155 + ((car.start + 155 + elapsed * 3 * car.direction) % 170 + 170) % 170;
    for (const person of people) person.mesh.position.z = person.start + Math.sin(elapsed * 0.4 + person.phase) * 2;
    renderer.render(scene, camera);
  }

  function reset() {
    if (disposed) return;
    controlsSuspended = false;
    releaseKeys();
    elapsed = 0;
    if (managedScene) { managedScene.reset(); render(0); return; }
    yaw = 0;
    camera.position.set(0, 1.7, 9);
    render(0);
  }
  function capture() {
    if (disposed) throw new Error('The game has been disposed.');
    const gl = renderer.getContext();
    if (gl.isContextLost()) throw new Error('The proxy WebGL context was lost.');
    const rgba = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
    const rgb = new Uint8Array(width * height * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const source = ((height - 1 - y) * width + x) * 4;
        const target = (y * width + x) * 3;
        rgb[target] = rgba[source];
        rgb[target + 1] = rgba[source + 1];
        rgb[target + 2] = rgba[source + 2];
      }
    }
    return rgb;
  }
  function dispose() {
    if (disposed) return;
    controlsSuspended = false;
    releaseKeys();
    disposed = true;
    for (const [name, handler] of Object.entries(handlers)) canvas.removeEventListener(name, handler);
    managedScene?.dispose();
    for (const geometry of geometries) geometry.dispose();
    for (const appearance of materials) appearance.dispose();
    renderer.dispose();
    if (previousTabIndex === null) canvas.removeAttribute('tabindex');
    else canvas.setAttribute('tabindex', previousTabIndex);
    if (previousLabel === null) canvas.removeAttribute('aria-label');
    else canvas.setAttribute('aria-label', previousLabel);
    canvas.style.touchAction = previousTouchAction;
  }

  reset();
  function prepareRun() { reset(); setControlsSuspended(true); }
  function stop() { setControlsSuspended(false); }
  return { width, height, render, reset, prepareRun, stop, capture, dispose, setControlsSuspended, getSimulationState };
}
