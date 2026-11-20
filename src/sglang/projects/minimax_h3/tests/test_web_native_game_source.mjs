import assert from 'node:assert/strict';
import {pack} from '../web/playable/stream.mjs';
import {createNativeGameSource} from '../web/playable/native-game-source.mjs';

let now = 1000, closedBitmaps = 0, paints = 0, errors = [], decode;
const timers = new Map();
globalThis.performance = {now: () => now};
globalThis.setTimeout = (callback, delay) => { const id = Symbol(); timers.set(id, {callback, delay}); return id; };
globalThis.clearTimeout = id => timers.delete(id);
globalThis.location = {href: 'http://127.0.0.1:8765/'};
const events = () => ({listeners: {}, addEventListener(name, handler) { this.listeners[name] = handler; },
  removeEventListener(name) { delete this.listeners[name]; }});
globalThis.window = events();
globalThis.document = events();
globalThis.createImageBitmap = async () => decode ? decode.promise : bitmap();
function bitmap() { return {width: 2, height: 1, close() { closedBitmaps++; }}; }
function deferred() { let resolve; const promise = new Promise(done => { resolve = done; }); return {promise, resolve}; }
class Socket {
  static OPEN = 1;
  static all = [];
  constructor(url) { this.url = url; this.readyState = 0; this.messages = []; Socket.all.push(this); }
  send(message) { this.messages.push(JSON.parse(message)); }
  open() { this.readyState = 1; this.onopen(); }
  close() { this.readyState = 3; this.onclose?.(); }
  reply(op, {image = false, evidence = {}, blocked = false, unchanged = false, cancelled = false, waiting = false} = {}) {
    const request = this.messages.find(message => message.op === op && !message.replied);
    assert.ok(request, `pending ${op} request`);
    request.replied = true;
    this.onmessage({data: pack({op: 'result', id: request.id, unchanged, cancelled, waiting,
      ...(op === 'prepare' ? {binding: {protocol: 'native_game_commands/v1',
        capability: 'secret', instance_id: 'native', epoch: 1, frame_id: 2}} : {}),
      ...(op === 'capture' ? blocked ? {blocked: 'source_budget'} : {sample: {source_frame: request.source_frame,
        capture_ns: request.capture_ns, source: {instance_id: 'native', epoch: 1, frame_id: 2}}} : {}),
      evidence: image ? {atomic: true, epoch: 1, instance_id: 'native', frame_id: 2,
        paused: op === 'prepare' || op === 'pause',
        jpeg_sha256: 'a', seated_vehicle: false, ...evidence} : null},
    image ? new Uint8Array([1]) : undefined).buffer});
  }
}
globalThis.WebSocket = Socket;
const flush = () => new Promise(setImmediate);
function setup(scene = 'native_game') {
  const canvas = {...events(), style: {}, focus() {}, setPointerCapture() {},
    getContext: () => ({drawImage() { paints++; }, getImageData() { throw new Error('Native game must never read browser RGB'); }})};
  const inputs = [];
  const source = createNativeGameSource(canvas, {width: 2, height: 1, scene, onInput: event => inputs.push(event),
    onError: error => errors.push(error), canReset: () => false});
  const socket = Socket.all.at(-1);
  return {canvas, source, socket, inputs};
}
async function prepare(pair) {
  const prepared = pair.source.prepareRun();
  if (pair.socket.readyState !== Socket.OPEN) pair.socket.open();
  await flush(); pair.socket.reply('prepare', {image: true}); await prepared;
}
async function resume(pair) {
  const resumed = pair.source.setControlsSuspended(false);
  await flush(); pair.socket.reply('resume'); await flush(); pair.socket.reply('frame', {image: true});
  await resumed;
}

const first = setup();
await prepare(first);
assert.equal(first.source.takeBinding().capability, 'secret');
assert.equal(first.source.takeBinding(), null, 'binding consumed once and never part of evidence');
assert.doesNotMatch(JSON.stringify(first.source.getSimulationState()), /secret|capability/);
const captured = first.source.captureNative(0, 12345);
await flush(); first.socket.reply('capture');
const sample = await captured;
assert.equal(sample.source_frame, 0);
assert.equal(sample.capture_ns, 12345);
assert.equal(sample.browser_receipt_ns, now * 1e6);
assert.ok(sample.command_bytes > 0);
assert.equal(first.source.capture, undefined);
const budgetWait = first.source.captureNative(1, 12346);
await flush(); first.socket.reply('capture', {blocked: true});
assert.equal((await budgetWait).blocked, true);
const publicationWait = first.source.captureNative(1, 12346, 7);
await flush();
assert.equal(first.socket.messages.at(-1).segment, 7);
first.socket.reply('capture', {waiting: true});
assert.equal((await publicationWait).waiting, true, 'a bounded native wait consumes no media sample');
const resumed = first.source.setControlsSuspended(false);
assert.equal(first.source.captureReady, false, 'resume gates sampling until observed native state and fresh JPEG');
await flush(); first.socket.reply('resume'); await flush();
assert.equal(first.source.captureReady, false);
first.socket.reply('frame', {image: true}); await resumed;
await flush();
assert.equal(first.socket.messages.filter(message => message.op === 'frame' && !message.replied).length, 1);
assert.equal([...timers.values()].some(timer => timer.delay > 0 && timer.delay < 100), false, 'preview has no extra clock');
first.socket.reply('frame', {unchanged: true});
await flush();
assert.equal(first.socket.messages.filter(message => message.op === 'frame' && !message.replied).length, 1, 'timeout renews one bounded wait');
decode = deferred();
first.socket.reply('frame', {image: true, evidence: {frame_id: 3}});
await flush();
const duringDecode = first.source.captureNative(1, 12347);
await flush(); first.socket.reply('capture');
assert.equal((await duringDecode).source_frame, 1, 'slow preview decode cannot hold capture lane');
assert.equal(first.socket.messages.filter(message => message.op === 'frame' && !message.replied).length, 0);
decode.resolve(bitmap()); await flush(); decode = null;
const duringWait = first.source.captureNative(2, 12348);
await flush();
assert.equal(first.socket.messages.filter(message => message.id && !message.replied).length, 2, 'one request per lane');
first.socket.reply('capture');
assert.equal((await duringWait).source_frame, 2, 'preview HTTP wait cannot hold capture lane');
first.canvas.listeners.keydown({code: 'KeyW', type: 'keydown', preventDefault() {}});
first.canvas.listeners.keydown({code: 'KeyV', type: 'keydown', preventDefault() {}});
assert.equal(first.socket.messages.at(-1).value.move_ud, -1);
assert.equal(first.socket.messages.at(-1).value.buttons, 32);
window.listeners.blur();
assert.equal(first.socket.messages.at(-1).op, 'release');
assert.deepEqual(first.source.getSimulationState().held_keys, []);
const paused = first.source.setControlsSuspended(true);
assert.equal(first.source.captureReady, false, 'pause gates input and capture synchronously');
await flush();
const pendingFrame = first.socket.messages.find(message => message.op === 'frame' && !message.replied);
if (pendingFrame) { first.socket.reply('frame', {cancelled: true}); await flush(); }
first.socket.reply('pause', {image: true, evidence: {frame_id: 3}}); await paused;
assert.equal(first.source.captureReady, true);
now = 1000;
first.source.dispose();
assert.deepEqual(Object.keys(first.canvas.listeners), []);
assert.equal(first.socket.readyState, 3);

const cancelled = setup();
const preparing = cancelled.source.prepareRun();
cancelled.socket.open(); await flush();
const stopping = cancelled.source.stop();
cancelled.socket.reply('prepare', {image: true}); await preparing; await flush();
assert.equal(cancelled.source.captureReady, false, 'cancelled preparation cannot publish frame zero');
cancelled.socket.reply('pause', {image: true}); await stopping;
assert.equal(cancelled.socket.messages.some(message => message.op === 'resume'), false);
cancelled.source.dispose();

const stale = setup();
const stalePrepare = stale.source.prepareRun(); stale.socket.open(); await flush();
decode = deferred();
stale.socket.reply('prepare', {image: true}); await flush();
const beforePaint = paints;
stale.source.dispose();
decode.resolve(bitmap()); await stalePrepare;
decode = null;
assert.equal(paints, beforePaint, 'late bitmap from disposed scene never paints');
assert.equal(stale.source.captureReady, false);
assert.equal(errors.length, 0);
assert.ok(closedBitmaps >= 4);
assert.equal(timers.size, 0);
const legacy = setup();
const legacyPrepare = legacy.source.prepareRun();
legacy.socket.open(); await flush();
legacy.socket.reply('prepare', {image: true, evidence: {atomic: false}});
await assert.rejects(legacyPrepare, /Invalid native game frame/);
legacy.source.dispose();
const changedEpoch = setup();
await prepare(changedEpoch);
const epochResume = changedEpoch.source.setControlsSuspended(false);
await flush(); changedEpoch.socket.reply('resume'); await flush();
changedEpoch.socket.reply('frame', {image: true, evidence: {epoch: 2}});
await assert.rejects(epochResume, /Invalid native game frame/);
assert.equal(changedEpoch.source.captureReady, false);
changedEpoch.source.dispose();
assert.equal(timers.size, 0);
console.log('PASS native game metadata samples, separate binding, JPEG preview without RGB readback, input release, async gates and disposal');

const overlap = setup();
await prepare(overlap);
await resume(overlap);
const overlapPause = overlap.source.setControlsSuspended(true);
await flush();
overlap.socket.reply('frame', {cancelled: true}); await flush();
decode = deferred();
overlap.socket.reply('pause', {image: true});
await flush();
const overlapPrepare = overlap.source.prepareRun();
assert.equal(overlap.source.getSimulationState().state, 'preparing');
decode.resolve(bitmap());
await overlapPause; await flush();
decode = null;
assert.equal(overlap.source.getSimulationState().state, 'preparing');
overlap.socket.reply('prepare', {image: true});
await overlapPrepare;
assert.equal(overlap.source.getSimulationState().state, 'held');
assert.equal(overlap.source.captureReady, true);
overlap.source.dispose();
assert.equal(errors.length, 0);
assert.equal(timers.size, 0);

const completed = setup();
await prepare(completed);
await resume(completed);
completed.socket.reply('frame', {cancelled: true});
await flush();
assert.equal(completed.socket.messages.filter(message => message.id && !message.replied).length, 0,
  'automatic media pause ends preview polling without restarting it');
const nextRun = completed.source.prepareRun();
await flush(); completed.socket.reply('prepare', {image: true}); await nextRun;
assert.equal(completed.source.captureReady, true, 'same controller can prepare after EOS cancellation');
completed.source.dispose();
assert.equal(errors.length, 0);
assert.equal(timers.size, 0);
