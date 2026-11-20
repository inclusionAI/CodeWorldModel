import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {parseTrace, TimelineHistory, replayRows, ganttOption, createTimeline} from '../web/playable/gantt.mjs';
import {REPLAY} from '../web/playable/gantt-replay.mjs';

const baseNs = 6e14;
function receipt(chunk, startMs = chunk * 1000, count = 1) {
  return {session_id: 'run-a', chunk_id: chunk, output_id: chunk, frame_count: count,
    source_frames: Array.from({length: count}, (_, i) => ({source_frame: chunk * 17 + i, chunk_id: chunk})),
    orchestration: {clock: 'frontend_monotonic', stages: [
      {stage: 'proxy_encode', dispatch_ns: baseNs + startMs * 1e6, complete_ns: baseNs + (startMs + 40) * 1e6, wall_ms: 40},
      {stage: 'render', dispatch_ns: baseNs + (startMs + 50) * 1e6, complete_ns: baseNs + (startMs + 250) * 1e6, wall_ms: 200},
    ]}};
}

assert.equal(parseTrace({rpc_ms: 200}), null);
const nativeReceipt = receipt(1);
nativeReceipt.orchestration.clock = 'native_worker_monotonic';
nativeReceipt.orchestration.stages[0].stage = 'proxy_prepare';
nativeReceipt.orchestration.stages.push({stage: 'codec', dispatch_ns: baseNs + 1300e6,
  complete_ns: baseNs + 1375e6, wall_ms: 75});
assert.equal(parseTrace(nativeReceipt).clock, 'native_worker_monotonic');
const nativeHistory = new TimelineHistory();
assert.equal(nativeHistory.add(nativeReceipt, 1500), true);
assert.deepEqual(nativeHistory.rows[0].stages.map(stage => stage.stage), ['proxy_prepare', 'render', 'codec']);
const nativeOption = ganttOption(nativeHistory.rows, 400);
assert.deepEqual(nativeOption.yAxis.data, ['Proxy / 噪声准备', 'DiT render', 'VAE 并行 / D2H']);
assert.deepEqual(nativeOption.series[0].data.map(row => row.value.slice(1)), [[0, 40, 40], [50, 250, 200], [300, 375, 75]]);
for (const mutate of [
  row => { row.orchestration.clock = 'worker_local'; },
  row => { row.orchestration.stages[0].wall_ms = 100; },
  row => { row.orchestration.stages[0].dispatch_ns = NaN; },
  row => { row.orchestration.stages[0].complete_ns = -1; },
  row => { row.orchestration.stages.push(row.orchestration.stages[0]); },
  row => { row.orchestration.stages[0].stage = '__proto__'; },
  row => { row.source_frames[0].source_frame = '1'; },
  row => { row.frame_count = 0; },
]) { const row = receipt(0); mutate(row); assert.equal(parseTrace(row), null); }

const history = new TimelineHistory();
const original = receipt(0, 500);
assert.equal(history.add(original, 100), true);
assert.equal(history.add(original, 110), false);
assert.deepEqual(history.rows[0].stages.map(row => [row.start, row.end]), [[0, 40], [50, 250]]);
assert.equal(original.orchestration.stages[0].dispatch_ns, baseNs + 500e6, 'metadata is unchanged');
history.add(receipt(1, 2000, 0), 8000);
assert.equal(history.rows[1].start, 1500, 'browser receive jitter cannot move server interval');
assert.equal(history.rows[1].start - history.rows[0].end, 1250, 'real inter-request gap is preserved');
assert.equal(history.rows[1].frames, 0);
assert.deepEqual(history.rows[1].sources, []);
assert.deepEqual(history.rows[0].sources, [{source_frame: 0, chunk_id: 0}]);
assert.equal(history.add({...receipt(2), session_id: 'another'}, 10000), false);
assert.equal(history.time(8100, true), 1850);
assert.equal(history.time(9000, false), 1850, 'ended timeline freezes');
const frozenRows = history.rows.slice();
history.select(frozenRows[0].key);
history.time(40000, true);
assert.equal(history.rows.length, 0);
assert.equal(history.selected.key, frozenRows[0].key, 'one selected row survives expiry');
history.select(frozenRows[1].key, frozenRows);
assert.equal(history.selected.frames, 0, 'paused visible request remains selectable after live expiry');
const cursor = history.cursor;
history.time(41000, false);
assert.equal(history.cursor, cursor);
history.reset();
assert.equal(history.selected, null);
assert.equal(history.originNs, null);
assert.equal(history.time(100000, true), 0);
for (let i = 0; i < 200; i++) history.add(receipt(i), i);
assert.equal(history.rows.length, 128, 'memory cap applies even before time pruning');
assert.equal(history.selected.chunk, 199, 'detail follows the newest receipt until explicitly pinned');
history.select(history.rows[0].key);
history.add(receipt(200), 200);
assert.equal(history.selected.chunk, 72, 'one explicit pin survives the bounded history');
history.select('');
assert.equal(history.selected.chunk, 200, 'following latest can be restored');

assert.deepEqual(replayRows(0), [], 'completed spans only');
const replay = replayRows(5500);
assert.equal(replay.length, 3);
assert.ok(replay.every(row => row.kind === 'replay' && !Object.hasOwn(row, 'chunk')));
assert.equal(replay[0].stages[2].wallMs, REPLAY.clips[0].stages[2].wall_ms);
const muchLater = replayRows(10000000);
assert.ok(muchLater.length <= 9);
assert.ok(muchLater.every(row => row.end <= 10000000 && row.end >= 10000000 - 10000));
const looped = muchLater[0];
const fromSource = REPLAY.clips.find(row => row.clip === looped.clip);
assert.ok(Math.abs((looped.stages[2].start - looped.stages[0].start) -
  (fromSource.stages[2].start_ms - fromSource.stages[0].start_ms)) < 1e-7, 'loop preserves within-clip spacing');

const echarts = createRequire(import.meta.url)('../web/playable/vendor/echarts-6.1.0/echarts.min.js');
const chart = echarts.init(null, null, {renderer: 'svg', ssr: true, width: 1000, height: 300});
const option = ganttOption(replay, 5500, replay[0].key);
chart.setOption(option);
assert.ok(chart.renderToSVGString().includes('1036 ms'));
assert.equal(option.xAxis.max - option.xAxis.min, 10000);
const detail = ganttOption([replay[0]], 0, replay[0].key, true);
assert.equal(detail.series[0].data[0].value[1], 0, 'detail starts at request first stage');
chart.setOption(detail);
assert.ok(chart.renderToSVGString().includes('1036.2 ms'));
const clipped = ganttOption(replay, 11500);
chart.setOption(clipped);
assert.ok(chart.renderToSVGString().includes('<svg'));
chart.setOption(ganttOption([], 0));
assert.equal(chart.getOption().series[0].data.length, 0);
chart.dispose();

function element() {
  return {textContent: '', value: '', dataset: {}, children: [], listeners: {}, disabled: false,
    addEventListener(name, listener) { this.listeners[name] = listener; },
    setAttribute() {}, append(child) { this.children.push(child); },
    replaceChildren(...children) { this.children = children; }};
}
const instances = [];
globalThis.echarts = {init() {
  const item = {options: [], handlers: {}, setOption(option) { this.options.push(option); },
    on(name, handler) { this.handlers[name] = handler; }, resize() {}, dispose() { this.disposed = true; }};
  instances.push(item); return item;
}};
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.document = {createElement: element};
const elements = Object.fromEntries(['overview', 'detail', 'state', 'notice', 'selection', 'title', 'table', 'source', 'pause', 'waits'].map(key => [key, element()]));
const widget = createTimeline(elements);
assert.equal(instances.length, 0, 'collapsed diagnostics do not initialize charts');
widget.add(receipt(0), 100);
widget.add(receipt(1), 1100);
widget.paint(1200, true);
assert.match(elements.title.textContent, /chunk #1/);
elements.selection.value = 'run-a:0:0'; elements.selection.listeners.change();
const detailPaints = instances[1].options.length;
widget.paint(1300, true);
assert.equal(instances[1].options.length, detailPaints, 'pinned detail is not rebuilt every tick');
widget.addObservation({event: 'frame_draw', session_id: 'run-a', output_id: 0, output_frame: 0,
  waits: {schema: 'stream_waits/v1', browser: {decoded_ns: 10, draw_start_ns: 20, draw_end_ns: 30, ack_enqueue_ns: 40}}}, 'run-a');
widget.paint(1350, true);
assert.equal(instances[1].options.length, detailPaints, 'late waits leave existing model diagram intact');
assert.equal(elements.waits.children.length, 3, 'late waits repaint the pinned detail without changing selection');
assert.match(elements.waits.children[1].textContent, /browser performance.now/);
elements.pause.listeners.click();
const pausedPaints = instances[0].options.length;
widget.paint(50000, true);
assert.equal(instances[0].options.length, pausedPaints, 'frozen overview does not repaint identical data');
elements.selection.value = 'run-a:1:1'; elements.selection.listeners.change();
assert.ok(elements.title.textContent.includes('chunk #1'), 'paused selector works after live expiry');
elements.source.value = 'replay'; elements.source.listeners.change();
widget.paint(56000, true);
assert.ok(elements.notice.textContent.includes('非本次推理'));
assert.ok(elements.title.textContent.includes('历史 clip'));
const latestReplayKey = instances[1].options.at(-1).series[0].data[0].rowKey;
assert.equal(latestReplayKey, replayRows(6000).at(-1).key, 'replay detail also follows latest');
elements.pause.listeners.click();
const pausedReplayTime = elements.state.dataset.time;
widget.paint(96000, true);
assert.equal(elements.state.dataset.time, pausedReplayTime);
elements.pause.listeners.click();
assert.equal(elements.state.dataset.time, pausedReplayTime, 'resume does not skip paused replay time');
widget.paint(97000, true);
assert.equal(Number(elements.state.dataset.time), Number(pausedReplayTime) + 1000);
elements.source.value = 'live'; elements.source.listeners.change();
widget.paint(56001, false);
assert.ok(elements.title.textContent.includes('chunk #1'), 'replay cannot overwrite real selected request');
assert.ok(!elements.title.textContent.includes('历史'));
widget.reset(); widget.add(receipt(0), 60000); widget.paint(60000, true);
elements.pause.listeners.click();
const frozenLivePaints = instances[0].options.length;
widget.add(receipt(1), 61000); widget.paint(61000, true);
assert.equal(instances[0].options.length, frozenLivePaints, 'incoming receipts cannot mutate the paused snapshot');
assert.match(elements.title.textContent, /chunk #0/, 'paused detail stays on the frozen clip');
elements.pause.listeners.click();
assert.match(elements.title.textContent, /chunk #1/, 'resume follows the newly received clip');
widget.reset(); widget.paint(57000, false);
assert.equal(elements.selection.children.length, 0);
assert.equal(elements.table.children.length, 0);
assert.ok(elements.notice.textContent.includes('尚无有效阶段计时'));
widget.dispose();
assert.ok(instances.every(item => item.disposed));
createTimeline(elements).dispose();
assert.equal(instances.length, 2, 'disposing an unopened panel creates no charts');
console.log('PASS Gantt trace boundaries, fixed clock, zero-output provenance, cap/pin/reset, source replay parity/isolation, ECharts rendering, paused selection, detail redraw lifecycle');
