import assert from 'node:assert/strict';
import {pack, unpack, packetSize, rgbToRgba, rgbaToRgb, FrameLedger, percentile, profilingLabel, CapturePacer, DisplayPacer, sourceBudgetPolicy, sourceBudgetBlocked} from '../web/playable/stream.mjs';
const payload = new Uint8Array([1, 2, 3, 255, 254, 253]);
assert.deepEqual(unpack(pack({op: '中文'}, payload).buffer), {metadata: {op: '中文'}, pixels: payload});
assert.throws(() => unpack(new ArrayBuffer(5)));
assert.throws(() => unpack(new Uint8Array([0, 1, 0, 1, 123, 125]).buffer));
assert.deepEqual(rgbaToRgb(rgbToRgba(payload)), payload);
assert.equal(percentile([10, 100, 20], .5), 20);
assert.equal(percentile([], .95), null);
assert.match(profilingLabel({status: 'baseline', source: 'health'}), /Profiler 关闭.*启动后确认/);
assert.match(profilingLabel({status: 'diagnostic', source: 'start'}), /Profiler 开启.*诊断采集.*会话启动回执/);
assert.match(profilingLabel({status: 'unknown', source: 'start'}), /状态未知.*不纳入正式对照/);
assert.match(profilingLabel({status: 'fixture', source: 'fixture'}), /不适用.*CPU 测试回显.*无模型推理/);
assert.match(profilingLabel(), /状态未知.*等待会话回执/);
const pacer = new CapturePacer(0, 'continuous', 10);
assert.equal(pacer.poll(0, false), 0);
assert.equal(pacer.poll(10, true), null);
assert.equal(pacer.poll(200, true), null);
assert.equal(pacer.waitCount, 1);
assert.equal(pacer.waitingMs(210), 200);
assert.equal(pacer.poll(210, false), 0);
assert.equal(pacer.waitMs, 200);
assert.equal(pacer.poll(220, false), 0);
assert.equal(pacer.poll(250, false), 2);
assert.equal(pacer.poll(260, true), null);
pacer.stop(280);
assert.equal(pacer.waitingMs(10000), 220);
assert.throws(() => new CapturePacer(0, 'stress').poll(0, true), /超载/);
const display = new DisplayPacer('adaptive');
for (let time = 0; time <= 5000; time += 50) display.sample(time, time % 200 === 0 ? 0 : 1);
assert.ok(display.targetFps < 24 && display.targetFps >= 19, 'repeated missed deadlines reduce the target smoothly');
const slowed = display.targetFps;
for (let time = 5050; time <= 16000; time += 50) display.sample(time, 3);
assert.equal(display.targetFps, 24, 'sustained buffered supply restores the 24 FPS target');
assert.ok(slowed < display.targetFps);
const single = new DisplayPacer('adaptive');
for (let time = 0; time <= 5000; time += 50) single.sample(time, 1);
assert.equal(single.targetFps, 24, 'one ready frame at each deadline is not an underrun');
const jitter = new DisplayPacer('adaptive');
for (let time = 0; time <= 5000; time += 50) jitter.sample(time, time % 1000 === 0 ? 0 : 2);
assert.equal(jitter.targetFps, 24, 'isolated missed deadlines do not lower the target');
const fixed = new DisplayPacer();
for (let time = 0; time <= 30000; time += 50) fixed.sample(time, 0);
assert.equal(fixed.targetFps, 24);
const budgetConfig = {source_budget: {eligible: true, reason: 'fixed_v2_bootstrap9_current5_causal'}};
assert.equal(sourceBudgetPolicy(budgetConfig).limit_frames, 51);
for (const limit of [40, 45, 51]) {
  const policy = sourceBudgetPolicy(budgetConfig, `bounded${limit}`);
  assert.equal(policy.effective, `bounded${limit}`);
  assert.equal(policy.limit_frames, limit);
  assert.equal(sourceBudgetBlocked(limit, 0, policy.limit_frames), true);
  assert.equal(sourceBudgetBlocked(limit, 1, policy.limit_frames), false);
}
assert.equal(sourceBudgetPolicy(budgetConfig, 'off').effective, 'off');
assert.equal(sourceBudgetPolicy({}, 'off').reason, 'selected_off');
assert.equal(sourceBudgetPolicy({source_budget: {eligible: false, reason: 'requires_h264'}}).reason, 'requires_h264');
assert.equal(sourceBudgetPolicy({source_budget: {eligible: false, reason: 'unsupported_recipe_or_decoder'}}).effective, 'off');
assert.equal(sourceBudgetPolicy({source_budget: {eligible: true, reason: 'fixture_h264'}}).limit_frames, 51);
assert.deepEqual(sourceBudgetPolicy({}), {selected: 'bounded51', effective: 'off', limit_frames: null,
  reason: 'unsupported_recipe_or_decoder'});
assert.throws(() => sourceBudgetPolicy(budgetConfig, 'unknown'));
assert.equal(sourceBudgetBlocked(51, 1, 51), false, 'one draw returns one capture credit');
assert.equal(sourceBudgetBlocked(51, 0, null), false);
for (let prefix = 0; prefix < 17; prefix++) {
  assert.equal(sourceBudgetBlocked(51 + prefix, prefix, 51), true, 'every partial packet obeys the same cap');
  assert.equal(sourceBudgetBlocked(50 + prefix, prefix, 51), false, 'any free credit permits the next frame');
}
for (let stopAt = 0; stopAt < 70; stopAt++) {
  let buffered = stopAt % 17, emitted = stopAt - buffered;
  let captured = stopAt, displayed = Math.max(0, stopAt - 34 - buffered);
  for (let count = 0; count < 40; count++) {
    const size = packetSize(buffered, true);
    if (size) { buffered -= size; emitted += size; if (size === 5) break; }
    if (sourceBudgetBlocked(captured, displayed, 51)) { displayed++; continue; }
    captured++;
    buffered++;
    assert.ok(captured - displayed <= 51, `stop at ${stopAt} never bypasses the cap`);
  }
  assert.equal(emitted % 17, 5, `stop at ${stopAt} ends with real five-frame EOS`);
  assert.equal(emitted, captured, 'EOS commits every captured source');
}
const ledger = new FrameLedger();
const frames = [0, 1, 2].map(id => ({id, capture_ns: id + 10, input: null}));
ledger.submit(frames, 0);
const response = {output_id: 0, output_frame_start: 0, width: 1, height: 1, frame_count: 0, source_frames: []};
assert.deepEqual(ledger.accept(response, 0, 1, 1), []);
const full = {...response, output_id: 1, frame_count: 3, status: 'complete', source_frames: frames.map(f => ({source_frame: f.id, capture_ns: f.capture_ns, chunk_id: 0}))};
assert.throws(() => ledger.accept({...full, frame_count: 2}, 6, 1, 1));
assert.throws(() => ledger.accept({...full, source_frames: [...full.source_frames].reverse()}, 9, 1, 1));
assert.equal(ledger.pending.size, 3);
assert.deepEqual(ledger.accept(full, 9, 1, 1).map(f => f.id), [0, 1, 2]);
assert.equal(ledger.pending.size, 0);
console.log('PASS binary RGB framing, geometry/provenance, delayed zero output, EOS alignment, percentile');
