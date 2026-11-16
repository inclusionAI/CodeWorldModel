import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as THREE from '../web/playable/vendor/three-0.180.0/three.module.min.js';
import {createBoxingScene} from '../web/playable/boxing-scene.js';
import {createRacingScene} from '../web/playable/racing-scene.js';

let rendered = 0, disposed = 0;
class Renderer {
  setPixelRatio() {}
  setSize() {}
  setClearColor() {}
  render() { rendered++; }
  dispose() { disposed++; }
  getContext() { return {isContextLost: () => false, readPixels() {}, RGBA: 1, UNSIGNED_BYTE: 2}; }
}
const context = vm.createContext({THREE: {...THREE, WebGLRenderer: Renderer}, createBoxingScene, createRacingScene, structuredClone});
vm.runInContext(readFileSync(new URL('../web/playable/game.js', import.meta.url), 'utf8')
  .replace(/^import .+;\r?\n/gm, '').replace(/^export /gm, ''), context);
const createGame = vm.runInContext('createGame', context);
const plain = value => JSON.parse(JSON.stringify(value));
for (const scene of ['street', 'boxing', 'racing']) {
  const listeners = {}, attributes = new Map(), pointers = new Set(), inputs = [];
  const canvas = {style: {}, focus() {},
    getAttribute: name => attributes.get(name) ?? null,
    setAttribute: (name, value) => attributes.set(name, value), removeAttribute: name => attributes.delete(name),
    addEventListener: (name, handler) => { listeners[name] = handler; },
    removeEventListener: name => { delete listeners[name]; },
    setPointerCapture: id => pointers.add(id), hasPointerCapture: id => pointers.has(id),
    releasePointerCapture: id => pointers.delete(id)};
  let resetAllowed = true;
  const game = createGame(canvas, {width: 1, height: 1, scene, onInput: event => inputs.push(event), canReset: () => resetAllowed});
  const key = (code, type = 'keydown') => listeners[type]({code, preventDefault() {}});
  const initial = plain(game.getSimulationState());
  if (scene === 'street') listeners.pointerdown({button: 0, pointerId: 7, clientX: 0});
  key('KeyW');
  game.render(.05);
  assert.notDeepEqual(plain(game.getSimulationState()), initial, `${scene} responds before pausing`);
  game.setControlsSuspended(true);
  assert.equal(pointers.size, 0);
  const frozen = plain(game.getSimulationState());
  assert.deepEqual(frozen.held_keys, []);
  const renderCount = rendered;
  for (const code of ['KeyW', 'KeyJ', 'KeyK', 'KeyR']) key(code);
  key('KeyW', 'keyup');
  game.render(.1);
  assert.deepEqual(plain(game.getSimulationState()), frozen, `${scene} freezes simulation and keypress mutations`);
  assert.equal(rendered, renderCount, 'suspension skips rendering');
  assert.ok(inputs.some(event => event.disposition === 'released_on_startup_pause'));
  assert.ok(inputs.some(event => event.disposition === 'ignored_during_startup_pause'));
  game.setControlsSuspended(false);
  resetAllowed = false;
  const beforeResetKey = plain(game.getSimulationState());
  key('KeyR');
  assert.deepEqual(plain(game.getSimulationState()), beforeResetKey, 'session reset guard survives suspension');
  key(scene === 'boxing' ? 'KeyJ' : 'KeyW');
  game.render(.05);
  assert.equal(game.getSimulationState().elapsed_seconds, .1, 'managed and street clocks advance after resume');
  assert.notDeepEqual(plain(game.getSimulationState()), beforeResetKey);
  const copy = game.getSimulationState();
  if (scene === 'boxing') copy.state.fighters[0].hp = -1;
  if (scene === 'racing') copy.state.player.speed = -1;
  assert.notEqual(game.getSimulationState().state?.fighters?.[0].hp, -1);
  assert.notEqual(game.getSimulationState().state?.player?.speed, -1);
  game.reset();
  assert.deepEqual(plain(game.getSimulationState()), initial, 'reset restores pose and elapsed time');
  game.dispose();
  assert.equal(Object.keys(listeners).length, 0);
  assert.equal(attributes.size, 0);
}
assert.equal(disposed, 3);
console.log('PASS street/boxing/racing input suspension, J/K/R, pointer release, clocks, snapshots, reset and cleanup');
