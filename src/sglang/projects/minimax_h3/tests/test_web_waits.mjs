import assert from 'node:assert/strict';
import {WAIT_SCHEMA, pendingPacketTiming, parseWaitObservation, clipWaitRows} from '../web/playable/waits.mjs';
import {TimelineHistory} from '../web/playable/gantt.mjs';

const frames = Array.from({length: 34}, (_, id) => ({id, capture_ns: id * 10, pixels_ready_ns: id * 10 + 3}));
assert.equal(pendingPacketTiming(frames, false).readyNs, 163);
assert.equal(pendingPacketTiming(frames.slice(17), false).readyNs, 333, 'second ready packet keeps its own final capture completion');
assert.equal(pendingPacketTiming(frames.slice(0, 5), true, 100).readyNs, 100, 'EOS5 only becomes legal on finish');
assert.equal(pendingPacketTiming(frames.slice(0, 5), true, 20).readyNs, 43, 'later pixels must finish before EOS is ready');
const samples = frames.map(({pixels_ready_ns, ...frame}) => ({...frame, sample_receipt_ns: pixels_ready_ns + 2}));
assert.equal(pendingPacketTiming(samples, false).readyNs, 165, 'Native game packet becomes ready at the final sample receipt');
assert.equal(pendingPacketTiming(samples.slice(0, 5), true, 20).readyNs, 45);
for (const count of [22, 34]) {
  const queue = frames.slice(0, count);
  assert.equal(pendingPacketTiming(queue, true, 500).readyNs, 163);
  queue.splice(0, 17);
  assert.equal(pendingPacketTiming(queue, true, 500).readyNs, count === 22 ? 500 : 333);
}
assert.equal(pendingPacketTiming(frames.slice(0, 4), true, 500), null);

const sid = 'waits-test';
const interval = (start_ns, end_ns, outcome = 'complete') => ({start_ns, end_ns, outcome});
const input = chunk => ({event: 'chunk_send', session_id: sid, chunk_id: chunk,
  waits: {schema: WAIT_SCHEMA, browser: {first_capture_ns: 10, packet_ready_ns: 100,
    credit_obtained_ns: 300, send_call_start_ns: 320, send_call_end_ns: 330}}});
const rows = parseWaitObservation(input(0), sid).rows;
const upload = source_frame => ({event: 'proxy_frame_enqueue', chunk_id: 0, source_frame,
  waits: {schema: WAIT_SCHEMA, browser: {proxy_frame_enqueue: interval(1, 2)}}});
assert.notEqual(parseWaitObservation(upload(0), sid).rows[0].key, parseWaitObservation(upload(1), sid).rows[0].key,
  'each uploaded frame keeps its own timeline observation');
const commit = parseWaitObservation({event: 'chunk_send', chunk_id: 0, waits: {schema: WAIT_SCHEMA,
  browser: {proxy_commit_enqueue: interval(2, 3)}, gateway: {proxy_assembly: interval(4e12, 4e12 + 7)}}}, sid);
assert.equal(commit.status, 'valid');
assert.deepEqual(commit.rows.map(row => row.spans[0].name), ['proxy_commit_enqueue', 'proxy_assembly']);
assert.deepEqual(rows[0].spans.map(span => [span.name, span.end_ns - span.start_ns]),
  [['capture_assembly', 90], ['credit_wait', 200], ['send_preparation', 20], ['send_enqueue', 10]]);
const receipt = {event: 'frame_receive', metadata: {session_id: sid, chunk_id: 4, output_id: 7, output_frame: 10,
  source_frame: 3, waits: {schema: WAIT_SCHEMA,
    gateway: {record_read: interval(10, 20), ack_window: interval(20, 30)},
    frontend: {publication: interval(8e14, 8e14 + 200), event_dequeued_ns: 8e14 + 400}}}};
const separate = parseWaitObservation(receipt, sid).rows;
assert.deepEqual(separate.map(row => row.spans.map(span => span.end_ns - span.start_ns)), [[10, 10], [200, 200]],
  'unrelated process origins never participate in a duration');
assert.equal(parseWaitObservation(receipt, 'stale-session').status, 'stale');
for (const bad of [null, [], {schema: WAIT_SCHEMA, browser: {bogus: 1}},
  {schema: WAIT_SCHEMA, browser: {packet_ready_ns: 2, credit_obtained_ns: 1}},
  {schema: WAIT_SCHEMA, frontend: {publication: interval(2, 1)}},
  {schema: WAIT_SCHEMA, gateway: {record_read: interval(1, 2, 'made_up')}},
  {schema: WAIT_SCHEMA, browser: {decoded_ns: NaN}},
  {schema: WAIT_SCHEMA, browser: {decoded_ns: '0'.repeat(2100)}}]) {
  assert.equal(parseWaitObservation({...input(0), waits: bad}, sid).status, 'malformed');
}
assert.equal(parseWaitObservation({...input(0), waits: {schema: 'future'}}, sid).status, 'unsupported');
assert.equal(parseWaitObservation({event: 'legacy'}, sid).status, 'absent');
assert.equal(clipWaitRows(rows, {session: sid, kind: 'live', chunk: 4, output: 7, frames: 1,
  sources: [{chunk_id: 0, source_frame: 0}]}).length, 1, 'input belongs to authoritative source chunk, not generated batch chunk');
assert.equal(clipWaitRows(rows, {session: sid, kind: 'live', chunk: 0, output: 0, frames: 0, sources: []}).length, 1,
  'zero-output clip retains its admission/input waits');

function clip(chunk) {
  const start = 1e12 + chunk * 17 / 24 * 1e9;
  return {session_id: sid, chunk_id: chunk, output_id: chunk, frame_count: 17,
    source_frames: Array.from({length: 17}, (_, i) => ({source_frame: chunk * 17 + i, chunk_id: chunk})),
    orchestration: {clock: 'frontend_monotonic', stages: [{stage: 'render', dispatch_ns: start,
      complete_ns: start + 10e6, wall_ms: 10}]}};
}
function playback(chunk, frame) {
  return {event: 'frame_draw', session_id: sid, chunk_id: chunk, output_id: chunk, output_frame: frame,
    waits: {schema: WAIT_SCHEMA, browser: {decoded_ns: frame * 1e9, draw_start_ns: frame * 1e9 + 1e6,
      draw_end_ns: frame * 1e9 + 2e6, ack_enqueue_ns: frame * 1e9 + 3e6}}};
}
const history = new TimelineHistory();
for (let chunk = 0; chunk < 18; chunk++) {
  history.addObservation(input(chunk), sid);
  history.add(clip(chunk), chunk * 17 / 24 * 1000);
  if (chunk === 0) history.select(history.rows[0].key);
  for (let i = 0; i < 17; i++) {
    const event = playback(chunk, chunk * 17 + i);
    for (const kind of ['frame_draw', 'frame_decode', 'frame_receive']) history.addObservation({...event, event: kind}, sid);
  }
  history.time(chunk * 17 / 24 * 1000, true);
}
assert.ok(history.rows.length >= 14, 'at 24 FPS with 17 outputs per clip, wait receipts do not evict the ten-second model overview');
assert.equal(history.selected.chunk, 0);
assert.equal(history.selected.waitRows.length, 52, 'pinned clip retains all its source and frame receipts after expiry');
const pinnedCount = history.selected.waitRows.length;
const late = {...playback(0, 16), event: 'late_draw'};
history.addObservation(late, sid);
assert.equal(history.selected.waitRows.length, pinnedCount + 1, 'late playback updates reach the same pinned clip');
history.addObservation(late, sid);
assert.equal(history.selected.waitRows.length, pinnedCount + 1, 'duplicate receipts are ignored');
assert.equal(history.addObservation(input(0), 'old'), false);
for (let i = 0; i < 200; i++) history.addObservation({...playback(0, i), event: `frame-${i}`}, sid);
assert.equal(history.selected.waitRows.length, 128);
assert.ok(history.selected.waitIntervalsOmitted > 0);
assert.ok(history.selected.waitRows.some(row => row.kind === 'chunk_send'), 'per-clip overflow keeps source-input waits');
for (let chunk = 100; chunk < 300; chunk++) history.addObservation(input(chunk), sid);
assert.equal(history.observations.length, 128);
assert.ok(history.observationsOmitted > 0);
history.reset();
assert.equal(history.observations.length, 0);
assert.equal(history.selected, null);
const zero = {...clip(0), frame_count: 0, source_frames: [], waits: {schema: WAIT_SCHEMA, frontend: {rgb_enqueued_ns: 10, rgb_taken_ns: 50}}};
delete zero.orchestration;
history.addObservation({event: 'chunk_receive', metadata: zero}, sid);
assert.equal(history.add(zero, 1), true);
assert.equal(history.selected.waitRows[0].spans[0].name, 'rgb_queue');
assert.equal(history.selected.frames, 0);
console.log('PASS packet-ready/EOS5/22/34, fixed schema, source joins, separate clocks, malformed receipts, 24 FPS overview, bounded clips/pin/pending/reset');
