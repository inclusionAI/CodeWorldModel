import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {TimingHistory, renderStageMs, chartOption} from '../web/playable/charts.mjs';

assert.equal(renderStageMs({rpc_ms: 9, render_start_ns: 0, render_submit_ns: 9000000}), null);
const stage = value => ({orchestration: {clock: 'frontend_monotonic', stages: [{stage: 'render', wall_ms: value}]}});
assert.equal(renderStageMs(stage(123.5)), 123.5);
assert.equal(renderStageMs({orchestration: {...stage(123.5).orchestration, clock: 'native_worker_monotonic'}}), 123.5);
assert.equal(renderStageMs(stage(0)), 0);
for (const invalid of [null, -1, NaN, Infinity, '2', true]) assert.equal(renderStageMs(stage(invalid)), null);
assert.equal(renderStageMs({orchestration: {...stage(1).orchestration, clock: 'worker_local'}}), null);
assert.equal(renderStageMs({orchestration: {clock: 'frontend_monotonic', stages: [stage(1).orchestration.stages[0], stage(2).orchestration.stages[0]]}}), null);

const history = new TimingHistory();
history.add('rpc', 0, 20);
history.add('rpc', 60, 0);
history.add('rpc', 61, null);
history.add('rpc', 62, 10);
history.prune(62);
assert.deepEqual(history.series.rpc.map(p => p.time), [60, 61, 62]);
const option = chartOption(history, ['rpc', 'render'], 62);
assert.equal(option.xAxis.min, 52);
assert.equal(option.xAxis.max, 62);
assert.equal(option.xAxis.max - option.xAxis.min, 10);
assert.deepEqual(option.series[0].data, [[60, 0], [61, null], [62, 10]]);
assert.deepEqual(option.series[1].data, []);
assert.equal(option.series[0].connectNulls, false);
assert.equal(option.series[0].smooth, false);
assert.equal(option.animation, false);
assert.equal(chartOption(history, ['rpc'], 61).series[0].data.at(-1)[1], null);
assert.equal(chartOption(new TimingHistory(), ['rpc'], 0).xAxis.max, 10);

const boundary = new TimingHistory();
boundary.add('rpc', 51.999, 1);
boundary.add('rpc', 52, 2);
boundary.prune(62);
assert.deepEqual(boundary.series.rpc, [{time: 52, value: 2}]);

history.add('receive', 62, 800);
history.add('receive', 62, 100);
assert.deepEqual(chartOption(history, ['receive'], 62).series[0].data, [[62, 800], [62, 100]], 'same response preserves every source frame age');
assert.ok(option.tooltip.formatter([{seriesName: 'Chunk RPC', data: [62, 0]}]).includes('0.0 ms'));
assert.ok(option.tooltip.formatter([{seriesName: 'Chunk RPC', data: [62, 10]}]).includes('62.000 s'));
assert.ok(option.tooltip.formatter([{seriesName: 'Chunk RPC', data: [62, null]}]).includes('无数据'));

const dense = new TimingHistory();
for (let i = 0; i < 10000; i++) dense.add('receive', i / 1000, i % 100);
assert.equal(dense.series.receive.length, 4096);
assert.throws(() => dense.add('receive', 0, 10), /ordered/);
dense.prune(100);
assert.equal(dense.series.receive.length, 0);

const echarts = createRequire(import.meta.url)('../web/playable/vendor/echarts-6.1.0/echarts.min.js');
assert.equal(echarts.version, '6.1.0');
const chart = echarts.init(null, null, {renderer: 'svg', ssr: true, width: 1000, height: 285});
chart.setOption(option);
chart.dispatchAction({type: 'legendToggleSelect', name: 'Chunk RPC'});
chart.setOption(chartOption(history, ['rpc', 'render'], 63));
assert.equal(chart.getOption().legend[0].selected['Chunk RPC'], false, 'refresh preserves legend selection');
chart.setOption(chartOption(new TimingHistory(), ['rpc', 'render'], 0));
assert.deepEqual(chart.getOption().series[0].data, [], 'new session clears previous samples');
assert.ok(chart.renderToSVGString().includes('<svg'));
chart.dispose();
console.log('PASS ECharts updates, legend persistence/reset, rolling expiry/cap, duplicate timestamps, gaps vs zero, Ray wall-ms contract');
