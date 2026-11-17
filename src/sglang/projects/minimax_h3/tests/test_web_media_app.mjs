import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import * as stream from '../web/playable/stream.mjs';
import * as media from '../web/playable/media.mjs';
import * as waits from '../web/playable/waits.mjs';
import * as charts from '../web/playable/charts.mjs';
import {sourceScenes, sceneAvailable} from '../web/playable/sources.mjs';

const capability = {protocol: 'h264_annexb_v1', codec: 'avc1.42C028', fps: 24, frame_window: 4,
  width: 1, height: 1, bitrate_min_bps: 500000, bitrate_max_bps: 20000000, bitrate_default_bps: 8000000};
let now = 1000, draws = 0, flushes = 0, downloaded, disposals = 0, resets = 0;
const timers = new Map();
let timerId = 0;
let imageCloses = 0;
let sourcePixel = new Uint8Array([0, 0, 0, 255]);
const previewWrites = [];
const nodes = new Map();
function node(id) {
  if (nodes.has(id)) return nodes.get(id);
  const element = {value: ({seed: '2026', bitrate: '8', 'capture-policy': 'continuous',
    'startup-mode': 'baseline', 'source-budget': 'bounded51', 'playback-mode': 'fixed', scene: 'boxing'})[id] ?? '',
    style: {}, listeners: {}, dataset: {}, children: [], attributes: {}, hidden: id === 'reference-preview', classList: {toggle() {}}, focus() {}, click() {},
    addEventListener(event, callback) { this.listeners[event] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    cloneNode() { return {...this, listeners: {}}; },
    replaceWith(replacement) { nodes.set(id, replacement); },
    getContext() { return {clearRect() {}, putImageData(image) { if (id === 'reference-preview-image') previewWrites.push(image); },
      getImageData: () => ({data: sourcePixel}),
      drawImage(frame) { if ('closeCount' in frame) { assert.equal(frame.closeCount, 0); draws++; } }}; }};
  nodes.set(id, element);
  return element;
}
class FakeSocket {
  static OPEN = 1;
  static all = [];
  constructor(url) { this.url = url; this.readyState = 0; this.bufferedAmount = 0; this.sent = []; this.payloads = []; FakeSocket.all.push(this); }
  send(data) {
    if (typeof data === 'string') { this.sent.push(JSON.parse(data)); this.payloads.push(null); }
    else {
      const {metadata, pixels} = stream.unpack(data.buffer);
      this.sent.push(metadata); this.payloads.push(pixels.slice());
      if (metadata.op === 'start') this.startPixels = pixels.slice();
    }
  }
  open() { this.readyState = 1; this.onopen(); }
  message(metadata, payload = new Uint8Array()) { return this.onmessage({data: stream.pack(metadata, payload).buffer}); }
  text(metadata) { return this.onmessage({data: JSON.stringify(metadata)}); }
  close(event = {code: 1000, reason: '', wasClean: true}) { this.readyState = 3; this.onclose?.(event); }
}
class FakeDecoder {
  static supported = true;
  static async isConfigSupported() { return {supported: this.supported}; }
  constructor(callbacks) { this.callbacks = callbacks; this.frames = []; this.state = 'unconfigured'; }
  configure() { this.state = 'configured'; }
  decode(chunk) {
    const frame = {timestamp: chunk.timestamp, closeCount: 0, close() { this.closeCount++; }};
    this.frames.push(frame);
    this.callbacks.output(frame);
  }
  async flush() { flushes++; }
  close() { this.state = 'closed'; }
}
globalThis.VideoDecoder = FakeDecoder;
globalThis.EncodedVideoChunk = class { constructor(data) { Object.assign(this, data); } };
const fixture = {source_budget: {eligible: true, reason: 'fixture_h264'}, mode: 'fixture',
  encoder_recipe: {width: 1, height: 1, proxy_width: 1, proxy_height: 1},
  recipe: {protocol: 'raw'}, media: capability};
const sandbox = vm.createContext({...stream, ...media, ...waits, ...charts, console, ArrayBuffer, Uint8Array, Map, Set,
  sourceScenes, sceneAvailable,
  Option: class { constructor(text, value) { Object.assign(this, {textContent: text, value}); } },
  createTorcsSource() { throw Error('Unexpected TORCS source'); },
  createBallpointSource() { throw Error('Unexpected Ballpoint source'); },
  createNativeGameSource() { throw Error('Unexpected native source'); },
  performance: {now: () => now}, WebSocket: FakeSocket,
  document: {getElementById: node, createElement: () => node(`canvas-${nodes.size}`), addEventListener() {}},
  window: {addEventListener() {}}, location: {protocol: 'https:', host: 'gateway.example.com',
    href: 'https://gateway.example.com/proxy/instance_1:8765/'}, requestAnimationFrame() {},
  fetch: async url => {
    if (url === './config') return {ok: true, json: async () => fixture};
    throw new Error(`Unexpected fetch ${url}`);
  },
  createImageBitmap: async () => ({width: 1, height: 1, close() { imageCloses++; }}),
  createGame: (_canvas, options) => ({width: 1, height: 1, scene: options.scene, canReset: options.canReset, suspended: false, elapsed: 0, held: new Set(),
    capture() { return new Uint8Array([Math.round(this.elapsed * 100), 0, 0]); },
    reset() { resets++; this.suspended = false; this.elapsed = 0; this.held.clear(); },
    prepareRun() { this.reset(); this.setControlsSuspended(true); },
    stop() { this.setControlsSuspended(false); },
    press(code) { if (!this.suspended) this.held.add(code); },
    render(dt) { if (!this.suspended) this.elapsed += dt; }, dispose() { disposals++; },
    setControlsSuspended(value) { this.suspended = value; if (value) this.held.clear(); },
    getSimulationState() { return {scene: this.scene, elapsed_seconds: this.elapsed, controls_suspended: this.suspended, held_keys: [...this.held]}; }}),
  createTimeline: () => ({reset() {}, add() {}, addObservation() {}, waitCoverage: () => ({}), paint() {}, dispose() {}}),
  paintChart() {}, disposeCharts() {}, ImageData: class { constructor(data, width, height) { Object.assign(this, {data, width, height}); } },
  Blob, URL: class extends URL {
    static createObjectURL(blob) { downloaded = blob; return 'blob:receipt'; }
    static revokeObjectURL() {}
  }, setTimeout(callback, delay) { timers.set(++timerId, {callback, delay}); return timerId; },
  clearTimeout(id) { timers.delete(id); }});
const source = readFileSync(new URL('../web/playable/app.js', import.meta.url), 'utf8').replace(/^import .+;\r?\n/gm, '');
vm.runInContext(source, sandbox);
await new Promise(setImmediate);
const state = expression => vm.runInContext(expression, sandbox);
assert.equal(state('game.scene'), 'boxing');
assert.equal(state('referenceSource'), 'fixture-proxy', 'fixture seeds a matching initial proxy reference');
assert.equal(node('start').disabled, false);
assert.equal(node('reference-preview').hidden, false);
assert.equal(node('reference-preview-caption').textContent, 'Proxy 测试参考图');
assert.equal(node('reference-preview-image').width, 1);
assert.equal(node('reference-preview-image').height, 1);
assert.deepEqual(Array.from(previewWrites.at(-1).data), [0, 0, 0, 255]);
node('scene').value = 'street';
await node('scene').listeners.change();
assert.equal(state('game.scene'), 'street');
assert.equal(state('referenceScene'), 'street');
assert.equal(disposals, 1, 'switch disposes the previous renderer and listeners');
fixture.mode = 'model';
fixture.codec = {decoder: 'causal', backend: 'fast', compile: true, cuda_graph: true};
await state('initialize()');
assert.equal(node('mode').textContent, '真实模型 serving · causal decoder · fast + compile + CUDA Graph');
assert.equal(state('reference'), null, 'model street has no built-in reference');
assert.equal(node('start').disabled, true);
assert.match(node('reference-info').textContent, /街景请上传/);
sourcePixel = new Uint8Array([17, 34, 51, 0]);
const closesBeforeUpload = imageCloses;
node('reference').files = [new Blob(['upload'])];
await node('reference').listeners.change();
const uploadedReference = state('reference');
assert.deepEqual(Array.from(uploadedReference), [17, 34, 51]);
assert.deepEqual(Array.from(previewWrites.at(-1).data), [17, 34, 51, 255], 'preview shows submitted opaque RGB');
assert.equal(node('reference-preview-caption').textContent, '上传参考图');
assert.equal(imageCloses, closesBeforeUpload + 1, 'the uploaded bitmap is released');
node('scene').value = 'boxing';
await node('scene').listeners.change();
assert.equal(state('reference'), null, 'boxing rejects the street reference');
assert.equal(node('reference-preview').hidden, true);
assert.equal(node('start').disabled, true, 'model boxing waits for an upload');
assert.equal(state('referenceSource'), null);
const beforeBoxingStart = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, beforeBoxingStart, 'handler rejects missing boxing reference');
node('reference').files = [new Blob(['boxing reference'])];
await node('reference').listeners.change();
assert.equal(state('referenceSource'), 'upload');
assert.equal(node('start').disabled, false);
const beforeRejectedUpload = previewWrites.length;
node('reference').files = [new Blob([new Uint8Array(20 * 1024 * 1024 + 1)])];
await node('reference').listeners.change();
assert.equal(node('reference-preview').hidden, true, 'rejected upload clears the old thumbnail');
assert.equal(node('start').disabled, true);
assert.equal(previewWrites.length, beforeRejectedUpload, 'rejected upload paints nothing');
assert.match(node('reference-info').textContent, /处理失败/);
const decodeImage = sandbox.createImageBitmap;
sandbox.createImageBitmap = async () => { throw new Error('Image decode failed'); };
node('reference').files = [new Blob(['invalid image'])];
await node('reference').listeners.change();
assert.equal(node('reference-preview').hidden, true, 'decode failure leaves no stale thumbnail');
assert.match(node('reference-info').textContent, /处理失败/);
sandbox.createImageBitmap = decodeImage;
node('reference').files = [new Blob(['boxing reference'])];
await node('reference').listeners.change();
node('scene').value = 'racing';
await node('scene').listeners.change();
assert.equal(state('game.scene'), 'racing');
assert.equal(state('reference'), null, 'racing rejects the boxing reference');
assert.equal(node('reference-preview').hidden, true);
assert.equal(node('start').disabled, true, 'model racing needs a matched upload');
assert.equal(node('racing-hud').hidden, false);
assert.match(node('reference-info').textContent, /赛车/);
const beforeRacingStart = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, beforeRacingStart);
node('reference').files = [new Blob(['matched racing reference'])];
await node('reference').listeners.change();
assert.equal(state('referenceScene'), 'racing');
assert.equal(state('referenceSource'), 'upload');
assert.equal(node('start').disabled, false, 'matched racing upload enables the model session');
let resolveRacingUpload;
const oldCreateImageBitmap = sandbox.createImageBitmap;
const racingFile = new Blob(['racing reference']);
sandbox.createImageBitmap = async source => source === racingFile
  ? await new Promise(resolve => { resolveRacingUpload = resolve; }) : oldCreateImageBitmap(source);
node('reference').files = [racingFile];
const pendingRacingUpload = node('reference').listeners.change();
assert.equal(node('reference-preview').hidden, true, 'pending upload hides the previous reference');
assert.match(node('reference-info').textContent, /正在处理/);
node('scene').value = 'street';
await node('scene').listeners.change();
resolveRacingUpload({width: 1, height: 1, close() { imageCloses++; }});
await pendingRacingUpload;
sandbox.createImageBitmap = oldCreateImageBitmap;
assert.equal(state('reference'), null, 'late racing upload cannot seed street');
assert.equal(state('referenceSource'), null);
assert.equal(node('racing-hud').hidden, true);
fixture.mode = 'fixture';
await state('initialize()');
assert.equal(node('mode').textContent, '测试回显 · 无模型推理');
const resetsBeforeRacing = resets;
node('scene').value = 'racing';
await node('scene').listeners.change();
assert.equal(state('referenceScene'), 'racing');
assert.equal(state('referenceSource'), 'fixture-proxy');
assert.ok(resets > resetsBeforeRacing, 'fixture reference captures the reset pose');
assert.equal(node('racing-hud').hidden, false);
node('fixture-reference').listeners.click();
const frozenPreview = previewWrites.at(-1);
const previewCount = previewWrites.length;
await node('start').listeners.click();
assert.equal(node('reference-preview').hidden, false);
assert.equal(node('scene').disabled, true);
assert.equal(node('reset-game').disabled, true);
const lockedScene = state('game');
assert.equal(lockedScene.canReset(), false);
const lockedResets = resets;
node('reset-game').listeners.click();
assert.equal(resets, lockedResets, 'reset handler rejects active sessions');
node('scene').value = 'boxing';
await node('scene').listeners.change();
assert.equal(state('game'), lockedScene, 'handler rejects a switch while starting');
assert.equal(node('scene').value, 'racing');
const socket = FakeSocket.all.at(-1);
assert.equal(socket.url, 'wss://gateway.example.com/proxy/instance_1:8765/stream');
socket.open();
assert.equal(socket.sent[0].video.bitrate_bps, 8000000);
assert.equal(node('bitrate').disabled, true);
await socket.message({op: 'started', source_budget: fixture.source_budget,
  session_id: 'encoded', media: {...capability, bitrate_bps: 8000000}});
state(`run.capture = Array.from({length: 17}, (_, id) => ({id, capture_ns: 1000000000 + id, pixels: new Uint8Array(3)})); run.captured = 17; sendChunk();`);
assert.equal(socket.sent.at(-1).op, 'chunk');
await socket.message({op: 'batch', session_id: 'encoded', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 0, status: 'accepted', source_frames: []});
assert.equal(state('run.credit'), false, 'batch receipt does not grant upload credit');
await socket.message({op: 'admitted', session_id: 'encoded', chunk_id: 0, status: 'admitted', rpc_ms: 2});
assert.equal(state('run.credit'), true);
state(`run.capture = Array.from({length: 5}, (_, i) => ({id: 17 + i, capture_ns: 1000000017 + i, pixels: new Uint8Array(3)})); run.captured = 22;`);
node('finish').listeners.click();
assert.equal(state('phase'), 'draining');
assert.equal(socket.sent.at(-1).eos, true);
const sources = Array.from({length: 22}, (_, id) => ({source_frame: id, capture_ns: 1000000000 + id, chunk_id: id < 17 ? 0 : 1}));
await socket.message({op: 'batch', session_id: 'encoded', chunk_id: 1, output_id: 1, output_frame_start: 0,
  width: 1, height: 1, frame_count: 22, status: 'complete', source_frames: sources});
const payload = new Uint8Array([0, 0, 0, 1, 65]);
for (let ordinal = 0; ordinal < 22; ordinal++) {
  await socket.message({op: 'video_frame', session_id: 'encoded', output_id: 1, output_frame: ordinal,
    ...sources[ordinal], chunk_id: 1, timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667,
    key_frame: ordinal === 0, encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
  assert.equal(socket.sent.filter(m => m.op === 'frame_ack').length, ordinal, 'decode does not ACK');
  if (ordinal < 21) { now += 50; state('tick(performance.now())'); }
}
assert.equal(draws, 21);
await socket.message({op: 'complete', session_id: 'encoded', status: 'complete', total_frames: 22});
assert.equal(flushes, 1);
assert.equal(state('run.backend_complete'), true);
assert.equal(state('phase'), 'draining', 'complete still waits for paced playback');
node('scene').value = 'boxing';
await node('scene').listeners.change();
assert.equal(state('game'), lockedScene, 'handler rejects a switch while draining');
await socket.message({op: 'admitted', session_id: 'encoded', chunk_id: 1, status: 'admitted', rpc_ms: 3});
assert.equal(state('run.credit'), false, 'late EOS admission grants no new credit');
now += 50; state('tick(performance.now())');
assert.equal(draws, 22);
assert.equal(previewWrites.length, previewCount, 'playback never repaints the reference');
assert.equal(previewWrites.at(-1), frozenPreview);
assert.equal(state('phase'), 'idle');
node('scene').value = 'boxing';
await node('scene').listeners.change();
assert.equal(state('game.scene'), 'boxing');
assert.equal(state('referenceSource'), 'fixture-proxy');
assert.equal(node('start').disabled, false);
assert.equal(state('run.decoder.decoder.frames.every(frame => frame.closeCount === 1)'), true);
assert.equal(socket.sent.filter(m => m.op === 'frame_ack').length, 22);
assert.equal(state('run.events.filter(e => e.event === "chunk_receive").every(e => e.encoded && e.payload_bytes === 0)'), true);
node('download').listeners.click();
const receipt = JSON.parse(await downloaded.text());
assert.equal(receipt.media_target.bitrate_bps, 8000000);
assert.equal(receipt.media_effective.bitrate_bps, 8000000);
assert.equal(receipt.media_received.encoded_bytes, 110);
assert.equal(receipt.source_budget.selected, 'bounded51');
assert.equal(receipt.source_budget.effective, 'bounded51');
assert.equal(receipt.source_budget.lead_frames, 0, 'successful drain returns every source credit');
assert.match(receipt.source_budget.game_time_policy, /wall_time_render/);
assert.equal(receipt.proxy.scene, 'racing', 'receipt freezes the scene of the completed run');
assert.equal(receipt.startup.snapshot.simulation.scene, 'racing', 'terminal simulation survives switching to boxing');
assert.deepEqual(receipt.startup.snapshot.simulation, JSON.parse(state('JSON.stringify(run.terminalSimulation)')));
assert.equal(receipt.proxy.style, 'synthetic-racing-depth-semantic-rgb/v1');
assert.equal(receipt.proxy.reference_source, 'fixture-proxy');
assert.match(receipt.timing.rpc_series, /admission/);
state('updateMetrics(performance.now())');
const completedLatest = node('latest-draw-age').textContent;
const completedAge = Number.parseInt(node('visible-age').textContent);
const completedRolling = node('display-age').textContent;
assert.match(node('latency-window').textContent, /已结束 · 保留最后 10 秒/);
now += 11000;
state('tick(performance.now())');
assert.equal(node('latest-draw-age').textContent, completedLatest);
assert.equal(Number.parseInt(node('visible-age').textContent), completedAge + 11000);
assert.equal(node('display-age').textContent, completedRolling, 'completed rolling window stays frozen');
node('download').listeners.click();
const laterReceipt = JSON.parse(await downloaded.text());
assert.deepEqual(laterReceipt.timing.latency_snapshot.rolling, receipt.timing.latency_snapshot.rolling);
assert.equal(laterReceipt.timing.latency_snapshot.rolling.frozen, true);
assert.equal(laterReceipt.timing.latency_snapshot.latest_draw.capture_ns, sources.at(-1).capture_ns);
assert.equal(laterReceipt.timing.latency_snapshot.visible_source.age_ms,
  receipt.timing.latency_snapshot.visible_source.age_ms + 11000);
assert.deepEqual(laterReceipt.events, receipt.events, 'metric refresh preserves raw events and timestamps');
assert.match(laterReceipt.timing.percentile_population, /event time, not capture time/);
assert.equal(laterReceipt.timing.chart_max_samples_per_series, 4096);

FakeDecoder.supported = false;
const socketCount = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, socketCount, 'unsupported codec fails before opening WebSocket');
for (const id of ['visible-age', 'latest-draw-age', 'receive-age', 'display-age']) {
  assert.equal(node(id).textContent, '—', `a new run clears completed ${id} even if codec preparation fails`);
}
assert.match(node('status').textContent, /不支持/);
assert.equal(state('run.outcome'), 'error');
FakeDecoder.supported = true;
await node('start').listeners.click();
const cancelledSocket = FakeSocket.all.at(-1);
cancelledSocket.open();
await cancelledSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'cancelled', media: {...capability, bitrate_bps: 8000000}});
state(`run.capture = Array.from({length: 17}, (_, id) => ({id, capture_ns: 1000000000 + id, pixels: new Uint8Array(3)})); sendChunk();`);
await cancelledSocket.message({op: 'batch', session_id: 'cancelled', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 1, status: 'accepted', source_frames: [sources[0]]});
await cancelledSocket.message({op: 'video_frame', session_id: 'cancelled', chunk_id: 0, output_id: 0, output_frame: 0,
  ...sources[0], timestamp_us: 0, duration_us: 41667, key_frame: true,
  encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
node('cancel').listeners.click();
assert.equal(state('run.decoder.decoder.frames[0].closeCount'), 1);
assert.equal(state('run.decoder.closed'), true);
assert.equal(cancelledSocket.sent.filter(m => m.op === 'frame_ack').length, 0, 'cancel closes undrawn output without ACK');
assert.equal(cancelledSocket.sent.at(-1).op, 'cancel');
await cancelledSocket.message({op: 'cancelled'});
assert.equal(state('phase'), 'idle');
assert.equal(state('run.socketClosure.first.local_intent'), 'explicit_cancel');
assert.equal(state('run.socketClosure.media.code'), 1000);

node('observe-waits').checked = true;
const actualPerformance = globalThis.performance;
Object.defineProperty(globalThis, 'performance', {configurable: true, value: {now: () => now}});
await node('start').listeners.click();
const observedSocket = FakeSocket.all.at(-1);
observedSocket.open();
assert.equal(observedSocket.sent[0].observe_waits, true);
assert.equal(node('observe-waits').disabled, true);
await observedSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'observed', media: {...capability, bitrate_bps: 8000000},
  wait_observability: {schema: waits.WAIT_SCHEMA, enabled: true}});
state(`run.capture = Array.from({length: 34}, (_, id) => ({id, capture_ns: 1000000000 + id * 1000,
  pixels_ready_ns: 1000000100 + id * 1000, pixels: new Uint8Array(3)})); run.captured = 34; sendChunk();`);
assert.equal(state('run.capture.length'), 17);
now += 200;
await observedSocket.message({op: 'admitted', session_id: 'observed', chunk_id: 0, status: 'admitted', rpc_ms: 2,
  waits: {schema: waits.WAIT_SCHEMA, gateway: {ack_window: {start_ns: 3, end_ns: 1, outcome: 'complete'}}}});
assert.equal(state('phase'), 'running', 'malformed optional observation does not interrupt valid admission');
assert.equal(state('run.waitCoverage.malformed'), 1);
const sentPackets = state('run.events.filter(e => e.event === "chunk_send")');
assert.equal(sentPackets[0].waits.browser.packet_ready_ns, 1000016100);
assert.equal(sentPackets[1].waits.browser.packet_ready_ns, 1000033100);
assert.equal(sentPackets[1].waits.browser.credit_obtained_ns, now * 1e6);
state(`run.capture = Array.from({length: 5}, (_, i) => ({id: 34 + i, capture_ns: 1000100000 + i,
  pixels_ready_ns: 1000100100 + i, pixels: new Uint8Array(3)})); run.captured = 39;`);
node('finish').listeners.click();
const finishedAt = now * 1e6;
now += 200;
await observedSocket.message({op: 'admitted', session_id: 'observed', chunk_id: 1, status: 'admitted', rpc_ms: 2});
assert.equal(state('run.events.filter(e => e.event === "chunk_send").at(-1).waits.browser.packet_ready_ns'), finishedAt);
assert.equal(state('phase'), 'draining');
node('cancel').listeners.click();
await observedSocket.message({op: 'cancelled'});

await node('start').listeners.click();
const drawnSocket = FakeSocket.all.at(-1);
drawnSocket.open();
await drawnSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'drawn', media: {...capability, bitrate_bps: 8000000},
  wait_observability: {schema: waits.WAIT_SCHEMA, enabled: true}});
state(`run.capture = Array.from({length: 5}, (_, id) => ({id, capture_ns: 1000000000 + id,
  pixels_ready_ns: 1000000100 + id, pixels: new Uint8Array(3)})); run.captured = 5;`);
node('finish').listeners.click();
await drawnSocket.message({op: 'admitted', session_id: 'drawn', chunk_id: 0, status: 'admitted', rpc_ms: 2});
await drawnSocket.message({op: 'batch', session_id: 'drawn', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 5, status: 'complete', source_frames: sources.slice(0, 5),
  waits: {schema: waits.WAIT_SCHEMA, frontend: {rgb_enqueued_ns: 1, rgb_taken_ns: 2}}});
for (let ordinal = 0; ordinal < 5; ordinal++) {
  await drawnSocket.message({op: 'video_frame', session_id: 'drawn', output_id: 0, output_frame: ordinal,
    ...sources[ordinal], chunk_id: 0, timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667,
    key_frame: ordinal === 0, encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length,
    waits: {schema: waits.WAIT_SCHEMA, frontend: {publication: {start_ns: 2, end_ns: 3, outcome: 'complete'}, event_dequeued_ns: 4}}}, payload);
  now += 50; state('tick(performance.now())');
}
await drawnSocket.message({op: 'complete', session_id: 'drawn', status: 'complete', total_frames: 5,
  wait_coverage: {frontend_complete: true, gateway_complete: false, socket_write: 'not_collected'}});
now += 50; state('tick(performance.now())');
node('download').listeners.click();
const observedReceipt = JSON.parse(await downloaded.text());
assert.equal(observedReceipt.wait_observability.enabled, true);
assert.equal(observedReceipt.wait_observability.browser_complete, true);
assert.equal(observedReceipt.wait_observability.remote_complete, false);
assert.ok(observedReceipt.wait_observability.uncollected.includes('socket_write_completion'));
assert.equal(observedReceipt.events.filter(event => event.event === 'frame_draw').length, 5);
for (const event of observedReceipt.events.filter(event => event.event === 'frame_draw')) {
  const parsed = waits.parseWaitObservation(event, 'drawn');
  assert.equal(parsed.status, 'valid');
  assert.equal(parsed.rows[0].spans.find(span => span.name === 'playback_queue').end_ns - event.waits.browser.decoded_ns, 50e6);
  assert.ok(event.waits.browser.draw_end_ns <= event.waits.browser.ack_enqueue_ns);
}
assert.equal(drawnSocket.sent.filter(m => m.op === 'frame_ack').length, 5);
assert.equal(state('run.waitCoverage.malformed'), 0, 'new run resets old malformed state');
const oldDrawnEvents = state('run.events.length');
await observedSocket.message({op: 'admitted', session_id: 'observed', chunk_id: 2, status: 'admitted'});
assert.equal(state('run.events.length'), oldDrawnEvents, 'old socket cannot append new-run observations');
state('for (let i = 0; i < 10010; i++) record("retention", {chunk_id: i, waits: {schema: WAIT_SCHEMA, browser: {packet_ready_ns: 1, credit_obtained_ns: 3}}})');
assert.equal(state('run.events.length'), 10000);
assert.ok(state('run.waitCoverage.intervals_omitted') > 0);
await node('start').listeners.click();
const drainCancelSocket = FakeSocket.all.at(-1);
drainCancelSocket.open();
await drainCancelSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'drain-cancel', media: {...capability, bitrate_bps: 8000000},
  wait_observability: {schema: waits.WAIT_SCHEMA, enabled: true}});
state(`run.capture = Array.from({length: 5}, (_, id) => ({id, capture_ns: 1000000000 + id,
  pixels_ready_ns: 1000000100 + id, pixels: new Uint8Array(3)})); run.captured = 5;`);
node('finish').listeners.click();
await drainCancelSocket.message({op: 'admitted', session_id: 'drain-cancel', chunk_id: 0, status: 'admitted', rpc_ms: 2});
await drainCancelSocket.message({op: 'batch', session_id: 'drain-cancel', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 5, status: 'complete', source_frames: sources.slice(0, 5)});
for (let ordinal = 0; ordinal < 5; ordinal++) {
  await drainCancelSocket.message({op: 'video_frame', session_id: 'drain-cancel', output_id: 0, output_frame: ordinal,
    ...sources[ordinal], chunk_id: 0, timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667,
    key_frame: ordinal === 0, encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
  if (ordinal === 0) { now += 50; state('tick(performance.now())'); }
}
await drainCancelSocket.message({op: 'complete', session_id: 'drain-cancel', status: 'complete', total_frames: 5});
assert.equal(state('run.backend_complete'), true);
assert.equal(state('run.playback.length'), 4);
node('cancel').listeners.click();
await drainCancelSocket.message({op: 'cancelled'});
node('download').listeners.click();
const cancelledDrainReceipt = JSON.parse(await downloaded.text());
assert.equal(cancelledDrainReceipt.outcome, 'cancelled');
assert.equal(cancelledDrainReceipt.wait_observability.browser_discarded, 4);
assert.equal(cancelledDrainReceipt.wait_observability.browser_complete, false,
  'cancellation cannot report browser completion after discarding the playback tail');
Object.defineProperty(globalThis, 'performance', {configurable: true, value: actualPerformance});
fixture.media.control_protocol = 'media_control/v1';
const secret = 'a'.repeat(43), freshSecret = 'b'.repeat(43);
async function splitConnection(token = secret) {
  await node('start').listeners.click();
  const data = FakeSocket.all.at(-1);
  data.open();
  assert.equal(data.sent.length, 0, 'reference waits for successful control binding');
  await data.text({op: 'control_offer', protocol: 'media_control/v1', capability: token});
  const control = FakeSocket.all.at(-1);
  assert.equal(control.url, 'wss://gateway.example.com/proxy/instance_1:8765/control');
  control.open();
  assert.deepEqual(control.sent[0], {op: 'bind', capability: token});
  assert.equal(data.sent.length, 0);
  return {data, control};
}
async function boundConnection(token = secret, sid = 'split', sourceBudget = fixture.source_budget) {
  const pair = await splitConnection(token);
  await pair.control.text({op: 'bound', protocol: 'media_control/v1'});
  assert.equal(pair.data.sent[0].op, 'start');
  await pair.data.message({op: 'started', source_budget: sourceBudget, session_id: sid, media: {...capability, bitrate_bps: 8000000},
    wait_observability: {schema: waits.WAIT_SCHEMA, enabled: true}});
  return pair;
}

await node('start').listeners.click();
const beforeHello = FakeSocket.all.at(-1);
beforeHello.open();
node('cancel').listeners.click();
await beforeHello.text({op: 'control_offer', protocol: 'media_control/v1', capability: secret});
assert.equal(beforeHello.sent.length, 0, 'cancel before offer cannot upload a reference');
assert.equal(state('phase'), 'idle');

const binding = await splitConnection();
node('cancel').listeners.click();
await binding.control.text({op: 'bound', protocol: 'media_control/v1'});
assert.equal(binding.data.sent.length, 0, 'late bound after cancellation cannot start');
assert.equal(binding.control.readyState, 3);
assert.equal(state('phase'), 'idle');

const uploading = await splitConnection(freshSecret);
await uploading.control.text({op: 'bound', protocol: 'media_control/v1'});
node('cancel').listeners.click();
assert.equal(uploading.control.sent.at(-1).op, 'cancel');
assert.deepEqual(uploading.data.sent.map(m => m.op), ['start']);
await uploading.data.message({op: 'started', source_budget: fixture.source_budget, session_id: 'late', media: {...capability, bitrate_bps: 8000000}});
assert.equal(state('phase'), 'stopping', 'late started cannot overwrite stopping');
await uploading.control.text({op: 'cancelled'});
assert.equal(state('phase'), 'idle');

const split = await boundConnection();
state(`run.capture = Array.from({length: 5}, (_, id) => ({id, capture_ns: 1000000000 + id,
  pixels_ready_ns: 1000000100 + id, pixels: new Uint8Array(3)})); run.captured = 5;`);
node('finish').listeners.click();
await split.data.message({op: 'admitted', session_id: 'split', chunk_id: 0, status: 'admitted', rpc_ms: 2});
await split.data.message({op: 'batch', session_id: 'split', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 5, status: 'complete', source_frames: sources.slice(0, 5)});
for (let ordinal = 0; ordinal < 5; ordinal++) {
  await split.data.message({op: 'video_frame', session_id: 'split', output_id: 0, output_frame: ordinal,
    ...sources[ordinal], chunk_id: 0, timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667,
    key_frame: ordinal === 0, encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
  if (ordinal < 4) { now += 50; state('tick(performance.now())'); }
}
await split.data.message({op: 'complete', session_id: 'split', status: 'complete', total_frames: 5});
assert.equal(split.control.sent.filter(m => m.op === 'frame_ack').length, 4);
assert.equal(state('phase'), 'draining');
now += 50; state('tick(performance.now())');
assert.equal(split.control.sent.filter(m => m.op === 'frame_ack').length, 5);
assert.equal(split.data.sent.filter(m => m.op === 'frame_ack').length, 0);
assert.equal(split.data.readyState, 1, 'server closes data after consuming final control ACK');
assert.equal(state('phase'), 'draining');
split.data.close();
assert.equal(state('phase'), 'idle');
assert.equal(state('run.socketClosure.first.channel'), 'media');
assert.equal(state('run.socketClosure.first.local_intent'), 'normal_eos');
node('download').listeners.click();
const splitText = await downloaded.text();
assert.ok(!splitText.includes(secret));
assert.ok(!splitText.includes('capability'));
assert.equal(JSON.parse(splitText).control_protocol, 'media_control/v1');
assert.equal(JSON.parse(splitText).socket_closure.media.code, 1000);
assert.equal(JSON.parse(splitText).socket_closure.control.code, 1000);
assert.equal(state('run.events.some(event => event.event === "control_offer" || event.capability)'), false);

const lostControl = await boundConnection(freshSecret, 'lost-control');
const eventsBeforeOldCallback = state('run.events.length');
await split.control.text({op: 'bound', protocol: 'media_control/v1'});
await split.data.message({op: 'started', source_budget: fixture.source_budget, session_id: 'stale'});
split.control.onclose?.({code: 1006, reason: 'stale', wasClean: false});
assert.equal(state('run.events.length'), eventsBeforeOldCallback);
assert.equal(state('run.socketClosure.first'), null, 'old run close cannot alter new run receipt');
state(`run.capture = Array.from({length: 17}, (_, id) => ({id, capture_ns: 1000000000 + id, pixels: new Uint8Array(3)})); sendChunk();`);
await lostControl.data.message({op: 'batch', session_id: 'lost-control', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 1, status: 'accepted', source_frames: sources.slice(0, 1)});
await lostControl.data.message({op: 'video_frame', session_id: 'lost-control', output_id: 0, output_frame: 0,
  ...sources[0], chunk_id: 0, timestamp_us: 0, duration_us: 41667, key_frame: true,
  encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
lostControl.control.close({code: 1006, reason: 'lost control', wasClean: false});
assert.equal(lostControl.data.readyState, 3);
assert.equal(state('run.decoder.decoder.frames[0].closeCount'), 1);
assert.equal(lostControl.control.sent.filter(m => m.op === 'frame_ack').length, 0);
assert.equal(state('phase'), 'idle');
assert.equal(state('run.outcome'), 'disconnected');
assert.equal(state('run.socketClosure.first.channel'), 'control');
assert.equal(state('run.socketClosure.control.code'), 1006);
assert.equal(state('run.socketClosure.control.wasClean'), false);
assert.equal(state('run.socketClosure.media.code'), 1000);
node('download').listeners.click();
assert.equal(JSON.parse(await downloaded.text()).socket_closure.control.reason, 'lost control');

const lostDrain = await boundConnection(freshSecret, 'lost-drain');
state('run.backend_complete = true; run.completing = true; run.outcome = "complete"; run.playback.push({}); phase = "draining"');
lostDrain.control.close();
assert.equal(state('run.outcome'), 'disconnected', 'control loss during EOS playback cannot report a completed browser run');
assert.equal(state('phase'), 'idle');

const errorControl = await boundConnection();
await errorControl.control.text({op: 'error', error: 'Backend cleanup uncertain',
  wait_coverage: {gateway_complete: false}});
assert.equal(state('run.events.find(e => e.event === "remote_error").metadata.error'), 'Backend cleanup uncertain');
assert.equal(state('run.waitCoverage.remote.gateway_complete'), false);
assert.equal(errorControl.data.readyState, 3);

for (const action of ['cancel', 'error']) {
  const draining = await boundConnection(freshSecret, `finish-${action}`);
  node('finish').listeners.click();
  assert.equal(state('run.closeIntent'), 'normal_eos');
  if (action === 'cancel') node('cancel').listeners.click();
  else state("fail(new Error('decoder failed while draining'))");
  await draining.data.message({op: 'cancelled'});
  assert.equal(state('run.socketClosure.first.local_intent'), action === 'cancel' ? 'explicit_cancel' : 'error');
}

const socketError = await boundConnection(secret, 'socket-error');
socketError.control.onerror?.();
assert.equal(state('run.socketClosure.first.kind'), 'error');
assert.equal(state('run.socketClosure.first.code'), null);
assert.equal(state('run.socketClosure.control.kind'), 'close');
assert.equal(state('run.socketClosure.control.code'), 1000);

const timedOut = await splitConnection();
const timeout = [...timers.values()].find(timer => timer.delay === 5000);
assert.ok(timeout);
timeout.callback();
assert.equal(timedOut.data.readyState, 3);
assert.equal(timedOut.control.readyState, 3);
assert.equal(timedOut.data.sent.length, 0);
assert.equal(state('phase'), 'idle');

fixture.media.control_protocol = 'unsupported/v2';
const unsupportedSockets = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, unsupportedSockets, 'unsupported control protocol fails before reserving server owner');
assert.equal(state('phase'), 'idle');

fixture.media.control_protocol = 'media_control/v1';
fixture.media.proxy_protocol = 'unsupported/v2';
const proxySockets = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, proxySockets, 'unknown proxy protocol cannot upload a reference');
fixture.media.proxy_protocol = 'proxy_frames/v1';
const incremental = await boundConnection(secret, 'incremental');
state(`run.capture = [{id: 0, capture_ns: 1000000000, pixels_ready_ns: 1000000001,
  pixels: new Uint8Array([1,2,3]), input: {id: 99}}]; run.captured = 1; sendChunk();`);
assert.equal(incremental.data.sent.at(-1).op, 'proxy_frame', 'frame 1 leaves before frame 2 exists');
assert.equal(state('run.credit'), false);
assert.equal(state('run.capture.length'), 1, 'uploaded records remain in logical capacity');
assert.equal(state('run.capture[0].pixels'), null);
assert.equal(state('run.ledger.pending.size'), 0, 'prefix does not commit source ownership');
state(`currentPrompt = {revision: 2, text: 'commit prompt', emitted_ns: 1000000010};
  for (let id = 1; id < 17; id++) run.capture.push({id, capture_ns: 1000000000 + id,
    pixels_ready_ns: 1000000001 + id, pixels: new Uint8Array([id+1,id+2,id+3])});
  run.captured = 17; sendChunk();`);
assert.equal(incremental.data.sent.filter(m => m.op === 'proxy_frame').length, 17, 'reservation permits remaining 16 frames');
const firstCommit = incremental.data.sent.at(-1);
assert.equal(firstCommit.op, 'proxy_commit');
assert.equal(firstCommit.prompt.text, 'commit prompt');
assert.equal(incremental.data.payloads.at(-1).length, 0);
assert.deepEqual(Array.from(Buffer.concat(incremental.data.payloads.slice(1, 18))),
  Array.from({length: 17}, (_, id) => [id+1,id+2,id+3]).flat());
state(`currentPrompt = {revision: 3, text: 'next prompt', emitted_ns: 1000000100};
  run.capture = Array.from({length: 5}, (_, i) => ({id: 17+i, capture_ns: 1000000017+i,
    pixels_ready_ns: 1000000018+i, pixels: new Uint8Array([i,20,30])})); run.captured = 22; sendChunk();`);
assert.equal(incremental.data.sent.at(-1), firstCommit, 'next chunk waits for admitted');
node('finish').listeners.click();
await incremental.data.message({op: 'admitted', session_id: 'incremental', chunk_id: 0, status: 'admitted', rpc_ms: 2});
assert.equal(state('phase'), 'draining');
assert.equal(incremental.data.sent.at(-1).eos, true);
assert.equal(incremental.data.sent.at(-1).capture_ns.length, 5);
assert.equal(incremental.data.sent.at(-1).prompt.text, 'next prompt');
assert.equal(firstCommit.prompt.text, 'commit prompt', 'a later edit cannot change submitted prompt');
assert.deepEqual(Array.from(Buffer.concat(incremental.data.payloads.slice(19, 24))),
  Array.from({length: 5}, (_, i) => [i,20,30]).flat());
assert.equal(state('run.ledger.pending.get(0).input.id'), 99);
assert.equal(state('run.waitCoverage.malformed'), 0);
node('download').listeners.click();
const incrementalReceipt = JSON.parse(await downloaded.text());
assert.equal(incrementalReceipt.proxy_protocol, 'proxy_frames/v1');
assert.match(incrementalReceipt.wait_observability.gateway_request_boundary.chunk, /proxy_commit/);
assert.equal(incrementalReceipt.events.filter(e => e.event === 'proxy_frame_enqueue').length, 22);
assert.equal(incrementalReceipt.events.filter(e => e.event === 'chunk_send')[0].payload_bytes, 51);
assert.ok(incrementalReceipt.events.find(e => e.event === 'chunk_send').waits.browser.proxy_commit_enqueue);
node('cancel').listeners.click();
await incremental.control.text({op: 'cancelled'});

let stopCases = 0;
for (let pending = 0; pending <= 34; pending++) {
  for (let sent = 0; sent <= Math.min(pending, 17); sent++) {
    const pair = await boundConnection(secret, 'stop-prefix');
    state(`run.capture = Array.from({length: ${pending}}, (_, id) => ({id, capture_ns: 1000000000+id,
      pixels_ready_ns: 1000000001+id, pixels: id < ${sent} ? null : new Uint8Array(3)}));
      run.captured = ${pending}; run.credit = ${sent === 0}; run.chunkId = ${sent ? 1 : 0};
      run.upload = ${sent ? `{chunkId: 0, sent: ${sent}}` : 'null'}; currentPrompt = null;`);
    node('finish').listeners.click();
    for (let turns = 0; state('phase') !== 'draining' && turns < 60; turns++) {
      const admitted = state('run.admissionChunk');
      if (admitted !== undefined && admitted !== null) {
        await pair.data.message({op: 'admitted', session_id: 'stop-prefix', chunk_id: admitted, status: 'admitted', rpc_ms: 1});
      }
      now += 50;
      state('tick(performance.now())');
      assert.ok(state('run.capture.length') <= 34);
      assert.ok(state('(run.upload?.sent ?? 0)') <= 17);
    }
    assert.equal(state('phase'), 'draining', `stop pending=${pending} uploaded=${sent}`);
    const commits = pair.data.sent.filter(m => m.op === 'proxy_commit');
    const total = pending <= 5 ? 5 : pending <= 22 ? 22 : 39;
    assert.deepEqual(commits.map(m => m.capture_ns.length), [...Array((total - 5) / 17).fill(17), 5]);
    assert.deepEqual([...Array(sent).keys(), ...pair.data.sent.filter(m => m.op === 'proxy_frame').map(m => m.source_frame)],
      [...Array(total).keys()], 'each real source frame uploads exactly once');
    assert.equal(state('run.captured'), total);
    assert.equal(state('run.ledger.pending.size'), total);
    pair.control.close();
    assert.equal(state('run.upload'), null);
    stopCases++;
  }
}
assert.equal(stopCases, 477);

async function admitBudgetInput(pair, session) {
  const chunk = state('run.admissionChunk');
  if (chunk !== undefined && chunk !== null) {
    await pair.data.message({op: 'admitted', session_id: session, chunk_id: chunk, status: 'admitted', rpc_ms: 1});
  }
}

async function captureThrough(pair, session, total) {
  for (let ticks = 0; state('run.captured') < total && ticks < 100; ticks++) {
    await admitBudgetInput(pair, session);
    now += 42; state('tick(performance.now())');
  }
  await admitBudgetInput(pair, session);
  assert.equal(state('run.captured'), total);
}

async function drawBudgetBatch(pair, session, count, afterDraw = () => {}) {
  const outputId = state('run.ledger.outputId'), start = state('run.ledger.emitted'), chunk = state('run.chunkId - 1');
  const sources = Array.from(state(`Array.from(run.ledger.pending.values()).slice(0, ${count})
    .map(f => ({source_frame: f.id, capture_ns: f.capture_ns, chunk_id: f.chunk_id}))`));
  const captured = state('run.captured');
  await pair.data.message({op: 'batch', session_id: session, chunk_id: chunk, output_id: outputId,
    output_frame_start: start, width: 1, height: 1, frame_count: count,
    status: state('phase') === 'draining' ? 'complete' : 'accepted', source_frames: sources});
  assert.equal(state('run.captured'), captured, 'batch receipt cannot return capture credits');
  for (let index = 0; index < count; index++) {
    const ordinal = start + index;
    await pair.data.message({op: 'video_frame', session_id: session, chunk_id: chunk, output_id: outputId,
      output_frame: ordinal, source_frame: sources[index].source_frame, capture_ns: sources[index].capture_ns,
      timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667, key_frame: ordinal === 0,
      encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
    now += 42; state('tick(performance.now())');
    afterDraw(index + 1);
  }
}

const budgeted = await boundConnection(secret, 'budgeted');
assert.equal(state('run.sourceBudget.effective'), 'bounded51');
assert.equal(state('run.sourceBudget.peak_lead_frames'), 1, 'frame zero consumes one capture credit');
await captureThrough(budgeted, 'budgeted', 51);
const beforeBudgetWait = state('game.elapsed');
for (let ticks = 0; ticks < 10; ticks++) { now += 42; state('tick(performance.now())'); }
assert.equal(state('run.captured'), 51, 'admission without output cannot exceed the global budget');
assert.equal(state('run.capture.length'), 0, 'global lead includes downstream frames after local queue drains');
assert.equal(state('run.sourceBudget.wait_count'), 1);
assert.equal(state('run.pacer.waitCount'), 0, 'budget waits do not count as network or local backpressure');
assert.ok(state('game.elapsed') > beforeBudgetWait + .4, 'budget wait preserves wall-time game motion');
await drawBudgetBatch(budgeted, 'budgeted', 1, () => assert.equal(state('run.captured'), 51));
await captureThrough(budgeted, 'budgeted', 52);
assert.equal(state('run.upload.sent'), 1, 'one draw lets the next packet upload its first frame');
const missedBeforePartialWait = state('run.missed');
for (let ticks = 0; ticks < 10; ticks++) { now += 42; state('tick(performance.now())'); }
assert.equal(state('run.captured'), 52, 'a partial upload waits when all credits are consumed');
assert.equal(state('run.upload.sent'), 1);
await drawBudgetBatch(budgeted, 'budgeted', 16, count => assert.equal(state('run.captured'), 51 + count));
await captureThrough(budgeted, 'budgeted', 68);
assert.equal(state('run.missed'), missedBeforePartialWait, 'released credit resumes capture without catch-up slots');
assert.equal(state('run.upload'), null, 'draw credits let the interrupted packet finish');
assert.equal(state('run.sourceBudget.peak_lead_frames'), 51);
node('finish').listeners.click();
for (let ticks = 0; ticks < 4; ticks++) { now += 42; state('tick(performance.now())'); }
assert.equal(state('run.captured'), 68, 'finishing cannot bypass the cap to start EOS');
await drawBudgetBatch(budgeted, 'budgeted', 1, () => assert.equal(state('run.captured'), 68));
await captureThrough(budgeted, 'budgeted', 69);
assert.equal(state('run.upload.sent'), 1, 'EOS also begins with a single returned credit');
for (let ticks = 0; ticks < 4; ticks++) { now += 42; state('tick(performance.now())'); }
assert.equal(state('run.captured'), 69, 'partial EOS cannot bypass the cap');
await drawBudgetBatch(budgeted, 'budgeted', 4, count => assert.equal(state('run.captured'), 68 + count));
await captureThrough(budgeted, 'budgeted', 73);
assert.equal(state('phase'), 'draining');
assert.deepEqual(budgeted.data.sent.filter(m => m.op === 'proxy_commit').map(m => m.capture_ns.length), [17, 17, 17, 17, 5]);
await drawBudgetBatch(budgeted, 'budgeted', 51);
await budgeted.data.message({op: 'complete', session_id: 'budgeted', status: 'complete', total_frames: 73});
now += 42; state('tick(performance.now())');
budgeted.data.close();
node('download').listeners.click();
const budgetReceipt = JSON.parse(await downloaded.text());
assert.equal(budgetReceipt.source_budget.lead_frames, 0);
assert.equal(budgetReceipt.source_budget.peak_lead_frames, 51);
assert.equal(budgetReceipt.source_budget.wait_count, 4);
assert.ok(budgetReceipt.source_budget.wait_ms > 0);
assert.equal(budgetReceipt.counts.capture_backpressure_count, 0);
assert.equal(budgetReceipt.counts.display_dropped, 0);
assert.deepEqual(budgetReceipt.events.filter(e => e.event === 'frame_draw').map(e => e.source_frame),
  Array.from({length: 73}, (_, id) => id));
assert.ok(budgetReceipt.events.filter(e => e.event === 'frame_capture').every(e => e.lead_frames <= 51));

node('source-budget').value = 'off';
const unlimited = await boundConnection(secret, 'budget-off');
await captureThrough(unlimited, 'budget-off', 68);
assert.equal(state('run.sourceBudget.effective'), 'off');
assert.equal(state('run.sourceBudget.wait_count'), 0);
node('cancel').listeners.click();
await unlimited.control.text({op: 'cancelled'});
node('source-budget').value = 'bounded51';
const unsupported = await boundConnection(secret, 'budget-unsupported',
  {eligible: false, reason: 'unsupported_recipe_or_decoder'});
assert.equal(state('run.sourceBudget.effective'), 'off', 'started declaration supersedes config');
node('cancel').listeners.click();
await unsupported.control.text({op: 'cancelled'});
const restarted = await boundConnection(secret, 'budget-restart');
assert.equal(state('run.captured'), 1);
assert.equal(state('run.sourceBudget.peak_lead_frames'), 1);
assert.equal(state('run.sourceBudget.wait_count'), 0);
await captureThrough(restarted, 'budget-restart', 51);
now += 42; state('tick(performance.now())');
node('cancel').listeners.click();
await restarted.control.text({op: 'cancelled'});
node('download').listeners.click();
const cancelledBudget = JSON.parse(await downloaded.text()).source_budget;
now += 1000;
node('download').listeners.click();
assert.equal(JSON.parse(await downloaded.text()).source_budget.wait_ms, cancelledBudget.wait_ms,
  'cancel closes the budget wait clock');
console.log('PASS 51-frame global cap, per-frame draw credits, interrupted partial upload and EOS, wall-time motion, baseline and restart');

const blockedProxy = await boundConnection(secret, 'blocked-proxy');
blockedProxy.data.bufferedAmount = 128 * 1024 * 1024;
state(`run.capture = Array.from({length: 34}, (_, id) => ({id, capture_ns: 1000000000+id,
  pixels_ready_ns: 1000000001+id, pixels: new Uint8Array(3)})); run.captured = 34; sendChunk();`);
for (let i = 0; i < 40; i++) { now += 50; state('tick(performance.now())'); }
assert.equal(state('run.capture.length'), 34);
assert.equal(state('run.captured'), 34);
assert.equal(state('run.credit'), true, 'blocked first send cannot consume credit');
assert.equal(blockedProxy.data.sent.length, 1);
blockedProxy.data.bufferedAmount = 0;
state('sendChunk()');
assert.equal(state('run.capture.length'), 17);
for (let i = 0; i < 40; i++) { now += 50; state('tick(performance.now())'); }
assert.equal(state('run.capture.length'), 34, 'pending admission retains the same logical cap');
assert.equal(blockedProxy.data.sent.filter(m => m.op === 'proxy_frame').length, 17);
node('cancel').listeners.click();
assert.equal(blockedProxy.control.sent.at(-1).op, 'cancel');
assert.equal(state('run.capture.length'), 0);
await blockedProxy.control.text({op: 'cancelled'});

const noWaitsProxy = await boundConnection(secret, 'no-waits-proxy');
state(`run.observe_waits = false; run.capture = [{id: 0, capture_ns: 1000000000, pixels: new Uint8Array(3)}];
  run.captured = 1; sendChunk();`);
assert.equal(state('run.events.some(e => e.event === "proxy_frame_enqueue")'), false);
node('cancel').listeners.click();
assert.equal(state('run.upload'), null);
await noWaitsProxy.control.text({op: 'cancelled'});

node('startup-mode').value = 'startup_pause';
node('playback-mode').value = 'fixed';
const catchup = await boundConnection(secret, 'catchup');
state(`run.capture = Array.from({length: 17}, (_, id) => ({id, capture_ns: 1000000000 + id,
  pixels: new Uint8Array(3)})); run.captured = 17; sendChunk();`);
assert.equal(state('run.startup.state'), 'waiting_batch');
assert.equal(state('game.suspended'), true);
await catchup.data.message({op: 'admitted', session_id: 'catchup', chunk_id: 0, status: 'admitted', rpc_ms: 1});
await catchup.data.message({op: 'batch', session_id: 'catchup', chunk_id: 0, output_id: 0,
  output_frame_start: 0, width: 1, height: 1, frame_count: 0, status: 'accepted',
  accepted_frames: 17, total_frames: 0, source_frames: []});
assert.equal(state('run.startup.state'), 'collecting', 'zero output requires another complete ordinary packet');
state(`run.capture = Array.from({length: 17}, (_, index) => ({id: index + 17,
  capture_ns: 1000000017 + index, pixels: new Uint8Array(3)})); run.captured = 34; sendChunk();`);
assert.equal(state('run.startup.state'), 'waiting_batch');
const catchupSources = state('Array.from(run.ledger.pending.values()).slice(0, 29).map(f => ({source_frame: f.id, capture_ns: f.capture_ns, chunk_id: f.chunk_id}))');
await catchup.data.message({op: 'batch', session_id: 'catchup', chunk_id: 1, output_id: 1,
  output_frame_start: 0, width: 1, height: 1, frame_count: 29, status: 'accepted',
  accepted_frames: 34, total_frames: 29, source_frames: catchupSources});
assert.equal(state('run.startup.state'), 'draining_produced');
for (let ordinal = 0; ordinal < 29; ordinal++) {
  await catchup.data.message({op: 'video_frame', session_id: 'catchup', chunk_id: 1,
    output_id: 1, output_frame: ordinal, source_frame: catchupSources[ordinal].source_frame,
    capture_ns: catchupSources[ordinal].capture_ns,
    timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667,
    key_frame: ordinal === 0, encode_start_ns: 1, encode_end_ns: 2, encoded_bytes: payload.length}, payload);
  now += 50; state('tick(performance.now())');
}
assert.equal(state('run.startup.state'), 'resumed');
assert.equal(state('run.ledger.pending.size'), 5, 'retained decoder tail does not block resumption');
assert.equal(state('run.sourceBudget.effective'), 'bounded51');
assert.equal(state('run.sourceBudget.wait_count'), 0, 'zero-output startup can acquire its next ordinary packet');
assert.equal(catchup.control.sent.filter(m => m.op === 'frame_ack').length, 29);
assert.equal(state('run.events.findIndex(e => e.event === "startup_transition" && e.to === "resumed") > run.events.findLastIndex(e => e.event === "frame_ack")'), true, 'catch-up resumes after enqueueing the watermark ACK');
node('download').listeners.click();
const catchupReceipt = JSON.parse(await downloaded.text());
assert.equal(catchupReceipt.startup.snapshot.drawn_frames, 29);
assert.equal(catchupReceipt.playback.mode, 'fixed');
node('cancel').listeners.click();
await catchup.control.text({op: 'cancelled'});

const stoppedCatchup = await boundConnection(secret, 'catchup-stop');
state(`run.capture = Array.from({length: 17}, (_, id) => ({id, capture_ns: 1000000000 + id,
  pixels: new Uint8Array(3)})); run.captured = 17; sendChunk();`);
node('finish').listeners.click();
assert.equal(state('game.suspended'), false);
for (let i = 0; i < 5; i++) { now += 50; state('tick(performance.now())'); }
await stoppedCatchup.data.message({op: 'admitted', session_id: 'catchup-stop', chunk_id: 0, status: 'admitted', rpc_ms: 1});
assert.equal(stoppedCatchup.data.sent.filter(m => m.op === 'proxy_commit').at(-1).eos, true);
assert.equal(stoppedCatchup.data.sent.filter(m => m.op === 'proxy_commit').at(-1).capture_ns.length, 5);
node('cancel').listeners.click();
await stoppedCatchup.control.text({op: 'cancelled'});
node('startup-mode').value = 'baseline';

node('playback-mode').value = 'adaptive';
for (const target of [24, 11]) {
  const paced = await boundConnection(secret, `cadence-${target}`);
  state(`run.displayPacer.targetFps = ${target}; nextDraw = performance.now();`);
  const started = now;
  let queued = 0;
  for (let refresh = 0; refresh < 300; refresh++) {
    now = started + refresh * 1000 / 60;
    while (state('run.playback.length') < (target === 24 ? 4 : 2)) {
      state(`run.playback.push({id: ${queued++}, capture_ns: 1000000000, pixels: new Uint8Array(3)});`);
    }
    const displayed = state('run.displayed');
    state('tick(performance.now())');
    assert.ok(state('run.displayed') - displayed <= 1, 'one refresh draws at most one source frame');
  }
  assert.equal(state('run.displayed'), target * 5, `${target} FPS target draws ${target * 5} frames over five seconds at 60 Hz`);
  assert.deepEqual(Array.from(state('run.events.filter(e => e.event === "frame_draw").map(e => e.source_frame)')),
    Array.from({length: target * 5}, (_, id) => id), 'paced playback preserves every source frame in order');
  assert.equal(state('run.dropped'), 0);
  now += 1000;
  const beforeStall = state('run.displayed');
  for (let retry = 0; retry < 4; retry++) state('tick(performance.now())');
  assert.equal(state('run.displayed'), beforeStall + 1, 'missed deadlines do not create a catch-up burst');
  node('cancel').listeners.click();
  await paced.control.text({op: 'cancelled'});
}
node('playback-mode').value = 'fixed';
now = Math.ceil(now);
console.log('PASS adaptive 24 and 11 FPS cadence at 60 Hz, ordered frames, no drops or catch-up burst');

delete globalThis.VideoDecoder;
delete globalThis.EncodedVideoChunk;
delete fixture.media;
await node('start').listeners.click();
const rawSocket = FakeSocket.all.at(-1);
rawSocket.open();
assert.equal(Object.hasOwn(rawSocket.sent[0], 'video'), false, 'legacy config starts raw with no WebCodecs');
await rawSocket.message({op: 'started', source_budget: {eligible: false, reason: 'requires_h264'}, session_id: 'legacy'});
assert.equal(state('phase'), 'running');
assert.equal(state('run.transport_mode'), 'legacy_rgb24');
assert.equal(node('visible-age').textContent, '—', 'a new run clears the previous visible source');
assert.equal(node('latest-draw-age').textContent, '—');
assert.equal(node('receive-age').textContent, '—');
assert.equal(node('display-age').textContent, '—');
for (let i = 0; i < 17; i++) { now += 50; state('tick(performance.now())'); }
const rawInput = rawSocket.sent.at(-1);
assert.equal(rawInput.op, 'chunk');
assert.equal(rawInput.capture_ns.length, 17);
const rawSource = {source_frame: 0, capture_ns: rawInput.capture_ns[0], chunk_id: 0};
now += 12000;
await rawSocket.message({op: 'output', session_id: 'legacy', chunk_id: 0, output_id: 0, output_frame_start: 0,
  width: 1, height: 1, frame_count: 1, status: 'accepted', source_frames: [rawSource]}, new Uint8Array(3));
state('lastMetrics = performance.now() - 200; tick(performance.now() - 5)');
const startupLatency = now - rawSource.capture_ns / 1e6;
assert.equal(Number.parseInt(node('visible-age').textContent), startupLatency,
  'visible age samples the clock after drawing, not the earlier animation-frame callback timestamp');
assert.equal(node('receive-age').textContent, `${startupLatency.toFixed(0)} / ${startupLatency.toFixed(0)} ms`,
  'an old capture received now belongs to the current receive event population');
assert.equal(node('display-age').textContent, `${startupLatency.toFixed(0)} / ${startupLatency.toFixed(0)} ms`);
assert.equal(node('latest-draw-age').textContent, `${startupLatency.toFixed(0)} ms · 源帧 0`);
const latestWhileStalled = node('latest-draw-age').textContent;
const startupEvents = state('JSON.stringify(run.events.filter(event => ["frame_receive", "frame_draw"].includes(event.event)))');
now += 9999;
state('tick(performance.now()); updateMetrics(performance.now())');
assert.notEqual(node('receive-age').textContent, '—', 'events younger than 10 seconds remain in the population');
now += 2;
state('updateMetrics(performance.now())');
assert.equal(node('receive-age').textContent, '—', 'receive samples expire without a new arrival');
assert.equal(node('display-age').textContent, '—', 'old startup draws leave the rolling population');
assert.equal(node('latest-draw-age').textContent, latestWhileStalled);
assert.equal(node('visible-age').textContent, `${(startupLatency + 10001).toFixed(0)} ms · 源帧 0`);
assert.match(node('latency-window').textContent, /最近 10 秒.*接收 0 \/ 绘制 0 样本/);
node('download').listeners.click();
const stalledReceipt = JSON.parse(await downloaded.text());
const stalledLatency = stalledReceipt.timing.latency_snapshot;
assert.equal(stalledLatency.clock, 'browser performance.now');
assert.equal(stalledLatency.snapshot_ns, now * 1e6);
assert.equal(stalledLatency.latest_draw.capture_to_draw_ms, startupLatency);
assert.equal(stalledLatency.latest_draw.draw_ns, (now - 10001) * 1e6);
assert.equal(stalledLatency.visible_source.age_ms, startupLatency + 10001);
assert.deepEqual(stalledLatency.rolling.receive, {samples: 0, p50_ms: null, p95_ms: null});
assert.deepEqual(stalledLatency.rolling.draw, {samples: 0, p50_ms: null, p95_ms: null});
assert.equal(stalledLatency.rolling.frozen, false);
assert.equal(JSON.stringify(stalledReceipt.events.filter(event => ['frame_receive', 'frame_draw'].includes(event.event))), startupEvents);
node('cancel').listeners.click();
assert.equal(node('visible-age').textContent, '—', 'erasing the canvas immediately clears current age');
assert.equal(node('latest-draw-age').textContent, '—', 'erasing the canvas immediately clears latest visible draw');
node('download').listeners.click();
const clearedLatency = JSON.parse(await downloaded.text()).timing.latency_snapshot;
assert.equal(clearedLatency.visible_source, null);
assert.equal(clearedLatency.latest_draw, null);
await rawSocket.message({op: 'cancelled'});
console.log('PASS app media/control/legacy, incremental bytes and prompt, 477 stop-prefix states, bounded backpressure, optional waits, event-time latency expiry, visible age, frozen completion and downloads');
fixture.mode = 'model';
await state('initialize()');
node('scene').value = 'boxing';
await node('scene').listeners.change();
sourcePixel = new Uint8Array([61, 82, 103, 7]);
node('reference').files = [new Blob(['boxing upload'])];
await node('reference').listeners.change();
assert.equal(state('referenceSource'), 'upload');
await node('start').listeners.click();
const uploadSocket = FakeSocket.all.at(-1);
assert.equal(state('run.proxy.reference_source'), 'upload');
uploadSocket.open();
assert.deepEqual(Array.from(uploadSocket.startPixels), [61, 82, 103], 'model start sends uploaded reference RGB');
node('cancel').listeners.click();
await uploadSocket.message({op: 'cancelled'});
node('download').listeners.click();
assert.equal(JSON.parse(await downloaded.text()).proxy.reference_source, 'upload');
state('referenceSource = "fixture-proxy"');
const beforeInvalidStart = FakeSocket.all.length;
await node('start').listeners.click();
assert.equal(FakeSocket.all.length, beforeInvalidStart, 'fixture source cannot start model mode');
console.log('PASS app media lifecycle, uploaded reference RGB/start/receipt and model reference gate');


fixture.media = capability;
delete fixture.media.control_protocol;
delete fixture.media.proxy_protocol;
globalThis.VideoDecoder = FakeDecoder;
globalThis.EncodedVideoChunk = class { constructor(data) { Object.assign(this, data); } };
fixture.mode = 'fixture';
await state('initialize()');
const checkSupport = FakeDecoder.isConfigSupported;
let resolveSupport;
FakeDecoder.isConfigSupported = () => new Promise(resolve => { resolveSupport = resolve; });
const starting = node('start').listeners.click();
assert.equal(state('phase'), 'starting');
const beforeHandshake = state('JSON.stringify(game.getSimulationState())');
const heldReference = state('reference');
for (const code of ['KeyW', 'KeyJ', 'KeyK', 'KeyR']) state(`game.press("${code}")`);
now += 3000; state('tick(performance.now())');
assert.equal(state('JSON.stringify(game.getSimulationState())'), beforeHandshake, 'decoder preparation freezes simulation and input');
assert.equal(state('run.captured'), 0);
resolveSupport({supported: true});
await starting;
FakeDecoder.isConfigSupported = checkSupport;
const delayedSocket = FakeSocket.all.at(-1);
delayedSocket.open();
now += 3000; state('tick(performance.now())');
assert.equal(state('JSON.stringify(game.getSimulationState())'), beforeHandshake, 'model handshake freezes simulation');
await delayedSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'first-capture', media: {...capability, bitrate_bps: 8000000}});
assert.equal(state('run.captured'), 1, 'started synchronously captures frame zero');
assert.equal(state('run.capture[0].id'), 0);
assert.deepEqual(Array.from(state('run.capture[0].pixels')), [0, 0, 0], 'frame zero uses the reset pose');
assert.equal(state('run.capture[0].capture_ns'), now * 1e6);
assert.equal(state('reference'), heldReference, 'first capture does not replace the reference');
state('tick(performance.now())');
assert.equal(state('run.captured'), 1, 'same-time tick cannot duplicate frame zero');
now += 41; state('tick(performance.now())');
assert.equal(state('run.captured'), 1);
now += 1; state('tick(performance.now())');
assert.equal(state('run.captured'), 2, 'frame one follows the existing 24 FPS pacer');
node('cancel').listeners.click();
await delayedSocket.message({op: 'cancelled'});
node('download').listeners.click();
const terminalReceipt = JSON.parse(await downloaded.text());
node('reset-game').listeners.click();
state('game.render(.1)');
node('download').listeners.click();
assert.deepEqual(JSON.parse(await downloaded.text()).startup.snapshot.simulation, terminalReceipt.startup.snapshot.simulation,
  'idle reset and rendering cannot mutate terminal simulation');
console.log('PASS delayed media/model startup freezes reset pose, first capture, frame-one cadence and terminal receipts');

fixture.media.control_protocol = 'media_control/v1';
const delayedBinding = await splitConnection();
const bindingPose = state('JSON.stringify(game.getSimulationState())');
for (const code of ['KeyW', 'KeyJ', 'KeyK', 'KeyR']) state(`game.press("${code}")`);
now += 1000; state('tick(performance.now())');
assert.equal(state('JSON.stringify(game.getSimulationState())'), bindingPose, 'unbound control channel keeps the reset pose frozen');
assert.equal(state('run.captured'), 0);
await delayedBinding.control.text({op: 'bound', protocol: 'media_control/v1'});
now += 1000; state('tick(performance.now())');
assert.equal(state('JSON.stringify(game.getSimulationState())'), bindingPose, 'bound control still waits for started before simulation');
assert.equal(state('run.captured'), 0);
await delayedBinding.data.message({op: 'started', source_budget: fixture.source_budget, session_id: 'delayed-bind', media: {...capability, bitrate_bps: 8000000}});
assert.equal(state('run.captured'), 1);
assert.deepEqual(Array.from(state('run.capture[0].pixels')), [0, 0, 0]);
node('cancel').listeners.click();
await delayedBinding.control.text({op: 'cancelled'});
console.log('PASS delayed split-control binding preserves frame-zero pose');

fixture.native_game_source = {protocol: 'native_game_commands/v1'};
delete fixture.media.control_protocol;
let holdPrepare = false, holdResume = false, holdCapture = false, resolvePrepare, resolveResume, resolveCapture;
let referenceDraws = 0;
const preparedNative = {instance_id: 'native', epoch: 3, frame_id: 7};
const nativeStarted = {protocol: 'native_game_commands/v1', ...preparedNative, budget_frames: 51};
sandbox.createNativeGameSource = (canvas, options) => {
  const source = sandbox.createGame(canvas, options);
  let generation = 0;
  source.prepareRun = () => {
    source.reset(); source.setControlsSuspended(true);
    return holdPrepare ? new Promise(resolve => { resolvePrepare = resolve; }) : Promise.resolve();
  };
  source.setControlsSuspended = value => {
    const expected = ++generation;
    source.suspended = true; source.held.clear();
    if (!value && holdResume) return new Promise(resolve => { resolveResume = () => {
      if (generation === expected) source.suspended = false;
      resolve();
    }; });
    source.suspended = value;
    return Promise.resolve();
  };
  source.stop = () => { generation++; source.suspended = true; source.held.clear(); };
  source.capture = () => { throw new Error('Native game proxy RGB cannot be read in browser'); };
  source.captureNative = (source_frame, capture_ns) => {
    const sample = {source_frame, capture_ns, source: {...preparedNative, atomic: true}, command_bytes: 80};
    return holdCapture && source_frame > 0 ? new Promise(resolve => { resolveCapture = () => resolve(sample); }) : Promise.resolve(sample);
  };
  source.drawReference = () => { referenceDraws++; };
  source.takeBinding = () => ({protocol: 'native_game_commands/v1', capability: 'secret-native-capability', ...preparedNative});
  source.getSimulationState = () => ({scene: options.scene, source: preparedNative});
  return source;
};
sourcePixel = new Uint8Array([120, 0, 0, 255]);
const oldCanvas = node('proxy');
await state("selectScene('native_game')");
assert.notEqual(node('proxy'), oldCanvas, 'Native game receives a fresh canvas after Three.js');
holdPrepare = true;
const beforeNativeGameSocket = FakeSocket.all.length;
const pendingNativeGame = node('start').listeners.click();
assert.equal(state('phase'), 'starting');
assert.equal(node('cancel').disabled, false, 'preparation can be cancelled before inference socket creation');
node('cancel').listeners.click();
assert.equal(state('phase'), 'idle');
await state("selectScene('boxing')");
resolvePrepare(); await pendingNativeGame;
assert.equal(FakeSocket.all.length, beforeNativeGameSocket, 'late preparation creates no inference socket');
assert.equal(state('game.scene'), 'boxing');

holdPrepare = false;
await state("selectScene('native_game')");
await node('start').listeners.click();
const nativeGameSocket = FakeSocket.all.at(-1); nativeGameSocket.open();
holdResume = true;
const pendingStarted = nativeGameSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'native-game-delayed', source: nativeStarted, media: {...capability, bitrate_bps: 8000000}});
await new Promise(setImmediate);
assert.equal(state('run.captured'), 1, 'frozen native game frame zero precedes asynchronous resume');
assert.equal(nativeGameSocket.startPixels.length, 0, 'prepared native game reference sends no browser RGB');
assert.equal(state('run.capture[0].pixels'), undefined, 'browser holds sample metadata only');
now += 1000; state('tick(performance.now())');
assert.equal(state('run.captured'), 1, 'resume barrier blocks capture while receive/draw remain available');
node('cancel').listeners.click();
resolveResume(); await pendingStarted;
assert.notEqual(state('phase'), 'running', 'late source resume cannot revive a cancelled run');
assert.equal(state('game.suspended'), true);
await nativeGameSocket.message({op: 'cancelled'});

holdResume = false;
node('reference').files = [new Blob(['Native game user reference'])];
await node('reference').listeners.change();
const nativeGameUpload = state('reference');
await node('start').listeners.click();
const uploadedNativeGameSocket = FakeSocket.all.at(-1); uploadedNativeGameSocket.open();
await uploadedNativeGameSocket.message({op: 'started', source_budget: fixture.source_budget, session_id: 'native-game-upload', source: nativeStarted, media: {...capability, bitrate_bps: 8000000}});
assert.equal(state('reference'), nativeGameUpload, 'Native game preparation preserves uploaded reference');
assert.deepEqual(Array.from(uploadedNativeGameSocket.startPixels), Array.from(nativeGameUpload));
assert.equal(state("run.events.find(event => event.event === 'frame_capture').source.atomic"), true);
node('cancel').listeners.click(); await uploadedNativeGameSocket.message({op: 'cancelled'});
await state("selectScene('racing')");
node('download').listeners.click();
const nativeGameReceipt = JSON.parse(await downloaded.text());
assert.equal(nativeGameReceipt.proxy.scene, 'native_game', 'completed run source survives switching back to Three.js');
assert.equal(nativeGameReceipt.proxy.reference_source, 'upload');
assert.equal(nativeGameReceipt.source_transport.proxy_upload_bytes, 0);
assert.ok(nativeGameReceipt.source_transport.acknowledged_capture_command_bytes > 0);
assert.doesNotMatch(JSON.stringify(nativeGameReceipt), /secret-native-capability|capability/);
assert.equal(state('game.scene'), 'racing');
console.log('PASS native game async preparation/cancel/resume, fresh canvas, frozen reference pixels, uploads and immutable run evidence');

async function startNative(sid, startup = 'baseline') {
  node('startup-mode').value = startup;
  await state("selectScene('native_game')");
  await node('start').listeners.click();
  const ws = FakeSocket.all.at(-1); ws.open();
  await ws.message({op: 'started', source_budget: fixture.source_budget, session_id: sid, source: nativeStarted, media: {...capability, bitrate_bps: 8000000}});
  return ws;
}
async function nativeTick() {
  now += 50; state('tick(performance.now())');
  await new Promise(setImmediate);
}
holdCapture = true;
const nativeFinish = await startNative('native-finish');
assert.equal(state('run.capturePending'), true, 'resume immediately arms native capture without RAF');
for (let sample = 1; sample < 4; sample++) { resolveCapture(); await new Promise(setImmediate); }
assert.equal(state('run.captured'), 4);
const scheduledSample = state('run.pacer.next');
for (let tick = 0; tick < 10; tick++) await nativeTick();
assert.equal(state('run.captured'), 4, 'one pending request owns the next source index');
assert.equal(state('run.pacer.next'), scheduledSample, 'RAF never imposes a native game sampling clock');
node('finish').listeners.click();
assert.equal(nativeFinish.sent.filter(m => m.op === 'native_game_chunk').length, 0, 'finish waits for reserved capture');
resolveCapture(); await new Promise(setImmediate);
const nativeEos = nativeFinish.sent.find(m => m.op === 'native_game_chunk');
assert.equal(nativeEos.eos, true);
assert.equal(nativeEos.capture_ns.length, 5);
assert.equal(nativeFinish.payloads[nativeFinish.sent.indexOf(nativeEos)].length, 0);
assert.equal(state('run.captured'), 5, 'completion commits exact EOS before rearming');
assert.equal(state('run.capturePending'), false);
node('cancel').listeners.click(); await nativeFinish.message({op: 'cancelled'});

const nativeCancel = await startNative('native-cancel');
assert.equal(state('run.capturePending'), true);
node('cancel').listeners.click();
resolveCapture(); await new Promise(setImmediate);
assert.equal(state('run.capture.length'), 0, 'late cancelled capture never appends');
assert.equal(state('run.captured'), 1);
assert.equal(state('game.suspended'), true);
await nativeCancel.message({op: 'cancelled'});

holdCapture = false;
const nativeAdmission = await startNative('native-admission');
await new Promise(setImmediate);
assert.equal(state('run.captured'), 17, 'completion fills and commits a packet without RAF');
assert.equal(state('run.capturePending'), false, 'no cross-socket capture before admitted');
for (let tick = 0; tick < 3; tick++) await nativeTick();
assert.equal(state('run.captured'), 17);
await nativeAdmission.message({op: 'admitted', session_id: 'native-admission', chunk_id: 0, status: 'admitted', rpc_ms: 1});
await new Promise(setImmediate);
assert.equal(state('run.captured'), 34, 'admission immediately wakes the pump');
await nativeAdmission.message({op: 'admitted', session_id: 'native-admission', chunk_id: 1, status: 'admitted', rpc_ms: 1});
await new Promise(setImmediate);
assert.equal(state('run.captured'), 51);
await nativeAdmission.message({op: 'admitted', session_id: 'native-admission', chunk_id: 2, status: 'admitted', rpc_ms: 1});
await new Promise(setImmediate);
assert.equal(state('run.captured'), 51, 'budget bounds publication-driven capture');
assert.equal(state('run.nativeGameBlocked'), true);
state('run.displayed++');
await nativeTick();
assert.equal(state('run.captured'), 52);
assert.equal(state('run.nativeGameSegment'), 1, 'capacity release starts an explicit media segment');
node('cancel').listeners.click(); await nativeAdmission.message({op: 'cancelled'});

const nativeStartup = await startNative('native-startup', 'startup_pause');
while (state('run.captured') < 17) await nativeTick();
state('sendChunk()'); await new Promise(setImmediate);
assert.equal(state('run.startup.state'), 'waiting_batch');
assert.equal(state('run.capturePending'), false);
assert.equal(state('game.suspended'), true);
const nativeNormal = nativeStartup.sent.find(m => m.op === 'native_game_chunk');
assert.equal(nativeNormal.eos, false);
assert.equal(nativeNormal.capture_ns.length, 17);
assert.equal(nativeStartup.payloads[nativeStartup.sent.indexOf(nativeNormal)].length, 0);
await nativeStartup.message({op: 'admitted', session_id: 'native-startup', chunk_id: 0, status: 'admitted', rpc_ms: 1});
await nativeStartup.message({op: 'batch', session_id: 'native-startup', chunk_id: 0, output_id: 0,
  output_frame_start: 0, width: 1, height: 1, frame_count: 0, status: 'accepted',
  accepted_frames: 17, total_frames: 0, source_frames: []});
await new Promise(setImmediate);
assert.equal(state('run.captured'), 34, 'zero-output startup immediately resumes native sampling');
while (state('run.captured') < 34) await nativeTick();
state('sendChunk()'); await new Promise(setImmediate);
assert.equal(state('run.startup.state'), 'waiting_batch');
assert.equal(nativeStartup.sent.filter(m => m.op === 'native_game_chunk').length, 2);
assert.equal(nativeStartup.sent.some(m => m.op === 'proxy_frame'), false, 'Native game overrides negotiated incremental RGB');
node('download').listeners.click();
assert.doesNotMatch(await downloaded.text(), /secret-native-capability/);
node('cancel').listeners.click(); await nativeStartup.message({op: 'cancelled'});
console.log('PASS native game metadata-only 17/5 commits, pending capture finish/cancel, reservation pacing and zero-output startup');

await state("selectScene('boxing')");
fixture.mode = 'model';
fixture.native_game_source.fixture_only = false;
await state('initialize()');
await state("selectScene('native_game')");
assert.equal(node('fixture-reference').hidden, false);
assert.equal(node('start').disabled, true, 'model native game waits for an explicit reference choice');
const drawsBeforeChoice = referenceDraws;
await node('fixture-reference').listeners.click();
assert.equal(referenceDraws, drawsBeforeChoice + 1);
assert.equal(state('referenceSource'), 'fixture-proxy');
assert.match(node('reference-info').textContent, /画质和上传图对齐效果未验证/);
assert.equal(node('start').disabled, false);
const resetsBeforeStart = resets;
await node('start').listeners.click();
assert.ok(resets > resetsBeforeStart, 'start resets the prepared native frame again');
assert.equal(referenceDraws, drawsBeforeChoice + 2, 'start refreshes the selected preview');
const modelNativeGame = FakeSocket.all.at(-1); modelNativeGame.open();
assert.equal(modelNativeGame.sent[0].source.reference, 'prepared');
assert.equal(modelNativeGame.startPixels.length, 0);
node('cancel').listeners.click(); await modelNativeGame.message({op: 'cancelled'});

node('reference').files = [new Blob(['model native game upload'])];
await node('reference').listeners.change();
const modelUpload = state('reference');
await node('start').listeners.click();
const modelNativeGameUpload = FakeSocket.all.at(-1); modelNativeGameUpload.open();
assert.equal(modelNativeGameUpload.sent[0].source.reference, 'upload');
assert.deepEqual(Array.from(modelNativeGameUpload.startPixels), Array.from(modelUpload));
node('cancel').listeners.click(); await modelNativeGameUpload.message({op: 'cancelled'});
await state("selectScene('racing')");
assert.equal(node('fixture-reference').hidden, true);
assert.equal(node('start').disabled, true, 'model Three.js scene still needs an upload');
assert.equal(state('referenceSource'), null);
console.log('PASS model native game explicit prepared/upload reference and reset refresh; Three.js model reference gate');

const completedOutcome = state('run.outcome'), completedEvents = state('run.events.length');
state("fail(new Error('next reference failed')); input({type:'keydown', code:'KeyW'})");
node('prompt').value = 'Next session description';
node('prompt').listeners.input();
assert.equal(state('run.outcome'), completedOutcome, 'idle errors cannot change the previous receipt');
assert.equal(state('run.events.length'), completedEvents, 'idle input and settings do not belong to the previous session');

fixture.mode = 'fixture';
fixture.torcs_source = {url: './threejs-racing/index.html'};
let finishLoading;
sandbox.createTorcsSource = (canvas, options) => {
  const source = sandbox.createGame(canvas, options), snapshot = source.getSimulationState.bind(source);
  const ready = new Promise(resolve => { finishLoading = () => { options.onReady(); resolve(); }; });
  source.modes = () => [{id:'rgb', label:'RGB'}, {id:'semantic', label:'Semantic'}];
  source.getSimulationState = () => ({...snapshot(), proxyMode:'semantic', auto:false});
  source.getSourceInfo = () => ({proxy:{mode:'semantic'}});
  source.prepareRun = async () => { await ready; source.reset(); source.setControlsSuspended(true); };
  source.referenceCanvas = () => ({width:1,height:1});
  return source;
};
const loadingScene = state("selectScene('torcs')");
assert.equal(node('torcs-auto').disabled, true, 'loading source cannot accept settings');
assert.equal(node('start').disabled, true);
node('reference').files = [new Blob(['upload while game loads'])];
await node('reference').listeners.change();
assert.equal(node('start').disabled, true, 'uploaded image does not bypass source preparation');
const loadingUpload = state('reference');
finishLoading();
await loadingScene;
assert.equal(node('torcs-mode').value, 'semantic', 'selector reflects the rendered mode, not the first option');
assert.equal(state('reference'), loadingUpload, 'late game preparation preserves the uploaded reference');
assert.equal(state('referenceSource'), 'upload');
assert.equal(state('game.suspended'), false, 'superseded preparation releases controls');
assert.equal(state('sourceGate'), false);
assert.equal(node('start').disabled, false);
await assert.rejects(state("changeSource(() => Promise.reject(new Error('source failed')))"), /source failed/);
assert.equal(state('sourceGate'), false, 'source failure releases the preparation gate');
assert.equal(state('run.outcome'), completedOutcome);
assert.throws(() => state("changeSource(() => {throw new Error('sync source failed')})"), /sync source failed/);
assert.equal(state('sourceGate'), false);
delete fixture.torcs_source;
await state("selectScene('boxing')");
await state("selectScene('torcs')");
assert.equal(state('sceneId'), 'boxing', 'unavailable games cannot be selected');
for (const invalid of ['', '-1', '0.5', '2147483648']) {
  node('seed').value = invalid;
  const sockets = FakeSocket.all.length;
  await node('start').listeners.click();
  assert.equal(FakeSocket.all.length, sockets);
  assert.match(node('status').textContent, /种子/);
}
node('seed').value = '2026';
fixture.source_budget.eligible = false;
state('updateControls()');
assert.equal(node('source-budget').disabled, true);
assert.match(node('budget-help').textContent, /不支持/);
await state('initialize()');
await state('initialize()');
assert.equal(node('scene').children.length, Object.keys(sourceScenes).length, 'reinitializing never duplicates scene options');
assert.equal(node('scene').children.find(option => option.value === 'torcs').hidden, true);
console.log('PASS multi-game capabilities, actual proxy setting, upload during loading, failed preparation and completed-run isolation');
