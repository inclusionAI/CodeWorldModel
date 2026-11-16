import assert from 'node:assert/strict';
import {createBallpointSource} from '../web/playable/ballpoint-source.mjs';

class Target {
  constructor() { this.listeners = new Map(); this.events = []; }
  addEventListener(name, callback) {
    if (!this.listeners.has(name)) this.listeners.set(name, new Set());
    this.listeners.get(name).add(callback);
  }
  removeEventListener(name, callback) { this.listeners.get(name)?.delete(callback); }
  dispatchEvent(event) {
    this.events.push(event);
    for (const callback of this.listeners.get(event.type) ?? []) callback(event);
  }
  fire(type, fields = {}) { this.dispatchEvent({type, preventDefault() {}, ...fields}); }
}
class InputEvent {
  constructor(type, fields) { Object.assign(this, {type}, fields); }
}
const document = new Target(), window = new Target(), timers = new Map(), frames = [];
const draws = [], overlays = [], inputs = [], statuses = [], errors = [];
let nextTimer = 0, sourceGame, readyCount = 0, steps = 0, resets = 0, suspended = false;
const canvas = () => Object.assign(new Target(), {
  width: 0, height: 0, setAttribute() {},
  focus() { document.activeElement = this; },
  requestPointerLock() { return Promise.reject(Error('Denied by browser')); },
  getContext() { return {drawImage(...args) { draws.push(args); },
    getImageData() { return {data: new Uint8ClampedArray([1,2,3,255,4,5,6,255])}; }}; },
});
const game = () => ({
  canvas: {width: 1280, height: 720},
  step() { steps++; }, reset() { resets++; }, suspend(value) { suspended = value; },
  overlay(context, width, height) { overlays.push([width, height]); }, state() { return {score: 7}; },
});
document.createElement = name => {
  if (name === 'canvas') return canvas();
  assert.equal(name, 'iframe');
  const contentWindow = Object.assign(new Target(), {ballpoint: sourceGame,
    KeyboardEvent: InputEvent, MouseEvent: InputEvent, WheelEvent: InputEvent});
  const frame = {contentWindow, remove() { this.removed = true; }};
  frames.push(frame);
  return frame;
};
document.body = {append() {}};
document.exitPointerLock = () => { document.pointerLockElement = null; document.fire('pointerlockchange'); };
Object.assign(globalThis, {document, window, location: {href: 'http://localhost:8770/'},
  setTimeout(callback) { const id = ++nextTimer; timers.set(id, callback); return id; },
  clearTimeout(id) { timers.delete(id); }});
const create = output => createBallpointSource(output, {width: 2, height: 1,
  url: './ballpoint-breach/index.html?stream=1', onInput: event => inputs.push(event),
  onReady: () => readyCount++, onError: error => errors.push(error), onPointerStatus: status => statuses.push(status)});

sourceGame = game();
const output = canvas(), source = create(output), iframe = frames.at(-1);
assert.equal(source.getSimulationState().state, 'loading');
assert.throws(() => source.capture(), /尚未就绪/);
const preparation = source.prepareRun();
iframe.onload();
await preparation;
assert.equal(readyCount, 1);
assert.equal(resets, 1);
assert.equal(suspended, true);
assert.equal(source.getSimulationState().state, 'held');
source.render(.04);
assert.equal(steps, 0, 'preparation freezes simulation');
output.fire('keydown', {code: 'KeyW', key: 'w'});
assert.equal(inputs.at(-1).disposition, 'ignored_during_startup_pause');
assert.equal(iframe.contentWindow.events.length, 0, 'suspended keys never reach the game');
assert.deepEqual(Array.from(source.capture()), [1,2,3,4,5,6]);
assert.deepEqual(source.getSourceInfo().source, {width: 1280, height: 720});
assert.equal(source.getSourceInfo().proxy.trainingAligned, false);
assert.equal(source.getSourceInfo().capture.resize, 'full frame; no crop');
const reference = source.referenceCanvas(4, 2);
assert.deepEqual([reference.width, reference.height], [4, 2]);
assert.deepEqual(draws.at(-1), [sourceGame.canvas, 0, 0, 4, 2]);
assert.deepEqual(overlays.at(-1), [4, 2]);
source.setControlsSuspended(false);
source.render(.04);
assert.equal(steps, 1);
output.fire('keydown', {code: 'KeyW', key: 'w'});
assert.equal(iframe.contentWindow.events.at(-1).code, 'KeyW');
output.fire('mousedown', {target: output, button: 0});
await Promise.resolve();
assert.match(statuses.at(-1), /画布内控制/);
output.fire('mousedown', {target: output, button: 0});
assert.equal(iframe.contentWindow.events.at(-1).type, 'mousedown');
window.fire('mousemove', {target: output, movementX: 3, movementY: 1});
assert.equal(iframe.contentWindow.events.at(-1).movementX, 3);
const inputCount = iframe.contentWindow.events.length;
window.fire('mousemove', {target: {}, movementX: 30, movementY: 10});
assert.equal(iframe.contentWindow.events.length, inputCount, 'fallback does not steer outside the canvas');
output.fire('keydown', {code: 'Escape'});
window.fire('mousemove', {target: output, movementX: 3});
assert.equal(iframe.contentWindow.events.length, inputCount, 'Escape releases fallback input');
source.dispose();
assert.equal(iframe.removed, true);
assert.equal(timers.size, 0);
for (const target of [output, document, window]) {
  assert.ok([...target.listeners.values()].every(listeners => listeners.size === 0));
}
assert.throws(() => source.capture(), /尚未就绪/);
assert.deepEqual(errors, []);

sourceGame = {canvas: {width: 2, height: 1}};
const incompatible = create(canvas());
const incompatiblePreparation = incompatible.prepareRun();
frames.at(-1).onload();
await assert.rejects(incompatiblePreparation, /streaming 构建缺少/);
assert.equal(readyCount, 1, 'an incompatible external build never becomes ready');
assert.match(errors.at(-1).message, /streaming 构建缺少/);
assert.equal(timers.size, 0);
incompatible.dispose();

sourceGame = undefined;
const timedOut = create(canvas()), lateFrame = frames.at(-1);
const timedPreparation = timedOut.prepareRun();
lateFrame.onload();
[...timers.values()][0]();
await assert.rejects(timedPreparation, /加载超时/);
lateFrame.contentWindow.ballpoint = game();
lateFrame.contentWindow.fire('ballpoint-ready');
assert.equal(readyCount, 1, 'late assets cannot revive a failed preparation');
assert.equal(timedOut.getSimulationState().state, 'loading');
timedOut.dispose();
assert.equal(timers.size, 0);
console.log('PASS external Ballpoint API, same-state reference, readiness, suspension, pointer fallback, failure and disposal');
