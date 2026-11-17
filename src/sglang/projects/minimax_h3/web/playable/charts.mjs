export const CHART_WINDOW_SECONDS = 10;
export const CHART_MAX_SAMPLES = 4096;

export const SERIES = {
  rpc: {label: 'Chunk RPC', color: '#78dfc4'},
  render: {label: '模型 render · 阶段墙钟', color: '#ffba75'},
  receive: {label: '采集 → 接收', color: '#83b7ff'},
  draw: {label: '采集 → 绘制提交', color: '#d5a4ff'},
};

export function renderStageMs(metadata) {
  if (!['frontend_monotonic', 'native_worker_monotonic'].includes(metadata.orchestration?.clock)) return null;
  const stages = metadata.orchestration.stages;
  if (!Array.isArray(stages)) return null;
  const rows = stages.filter(row => row?.stage === 'render');
  const value = rows.length === 1 ? rows[0].wall_ms : null;
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

export class TimingHistory {
  constructor() {
    this.series = Object.fromEntries(Object.keys(SERIES).map(key => [key, []]));
  }
  add(name, time, value) {
    const rows = this.series[name];
    if (!rows || !Number.isFinite(time) || time < 0) return;
    if (rows.length && time < rows.at(-1).time) throw new Error('Chart timestamps must be ordered');
    rows.push({time, value: typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null});
    if (rows.length > CHART_MAX_SAMPLES) rows.splice(0, rows.length - CHART_MAX_SAMPLES);
  }
  prune(time) {
    for (const rows of Object.values(this.series)) {
      const first = rows.findIndex(row => row.time >= time - CHART_WINDOW_SECONDS);
      if (first < 0) rows.length = 0;
      else if (first) rows.splice(0, first);
    }
  }
}

export function chartOption(history, names, now) {
  const end = Math.max(CHART_WINDOW_SECONDS, now), start = Math.max(0, now - CHART_WINDOW_SECONDS);
  return {
    animation: false,
    textStyle: {fontFamily: 'system-ui, sans-serif', color: '#a7b7cb'},
    grid: {left: 58, right: 22, top: 65, bottom: 36},
    legend: {top: 8, left: 8, itemWidth: 18, itemHeight: 8, itemGap: 22,
      textStyle: {color: '#c4d1e2', fontSize: 12}, inactiveColor: '#526174'},
    tooltip: {trigger: 'axis', confine: true, renderMode: 'richText',
      axisPointer: {type: 'line', lineStyle: {color: '#7c91ab', type: 'dashed'}},
      backgroundColor: '#101a29', borderColor: '#435875', padding: 12,
      textStyle: {color: '#e6edf6', fontSize: 12},
      formatter: samples => ['邻近采样 · 非批次统计', ...samples.map(sample =>
        `${sample.seriesName}\n会话 ${sample.data[0].toFixed(3)} s · ${sample.data[1] === null ? '无数据' : sample.data[1].toFixed(1) + ' ms'}`)].join('\n')},
    xAxis: {type: 'value', min: start, max: end, splitNumber: 6,
      axisLine: {lineStyle: {color: '#344358'}}, axisTick: {show: false},
      axisLabel: {color: '#91a4bd', formatter: value => `${Number(value.toFixed(1))}s`},
      splitLine: {show: false}},
    yAxis: {type: 'value', min: 0, name: 'ms', nameTextStyle: {color: '#91a4bd'},
      axisLabel: {color: '#91a4bd'}, splitNumber: 4,
      splitLine: {lineStyle: {color: '#263448', type: 'dashed'}}},
    series: names.map(name => ({id: name, name: SERIES[name].label, type: 'line',
      data: history.series[name].filter(p => p.time >= start && p.time <= now).map(p => [p.time, p.value]),
      showSymbol: false, symbol: 'circle', symbolSize: 5,
      smooth: false, connectNulls: false,
      lineStyle: {color: SERIES[name].color, width: 1.5},
      itemStyle: {color: SERIES[name].color}, emphasis: {scale: 2}})),
  };
}

const charts = new Map();

export function paintChart(container, summary, history, names, now) {
  let entry = charts.get(container);
  if (!entry) {
    const chart = globalThis.echarts.init(container);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(container);
    entry = {chart, observer};
    charts.set(container, entry);
  }
  const option = chartOption(history, names, now);
  entry.chart.setOption(option);
  summary.textContent = option.series.map(row => {
    const latest = row.data.at(-1)?.[1] ?? null;
    const count = row.data.filter(point => point[1] !== null).length;
    return `${row.name} ${latest === null ? '无数据' : latest.toFixed(1) + ' ms'} · ${count} 样本`;
  }).join('　/　');
  container.setAttribute('aria-label', summary.textContent);
}

export function disposeCharts() {
  for (const {chart, observer} of charts.values()) {
    observer.disconnect();
    chart.dispose();
  }
  charts.clear();
}
