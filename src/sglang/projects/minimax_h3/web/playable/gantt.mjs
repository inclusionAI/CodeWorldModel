import {CHART_WINDOW_SECONDS} from './charts.mjs';
import {REPLAY} from './gantt-replay.mjs';
import {WAIT_CLOCKS, parseWaitObservation, clipWaitRows} from './waits.mjs';

const STAGES = {
  text_condition: {label: '文本条件', color: '#c6a5fa'},
  proxy_encode: {label: 'Proxy 编码', color: '#78dfc4'},
  proxy_prepare: {label: 'Proxy / 噪声准备', color: '#78dfc4'},
  render: {label: 'DiT render', color: '#ffba75'},
  decode: {label: 'RGB decode', color: '#83b7ff'},
  codec: {label: 'VAE 并行 / D2H', color: '#83b7ff'},
  collect: {label: '结果收集', color: '#f1ce77'},
};
const stageNames = Object.keys(STAGES);
const WINDOW_MS = CHART_WINDOW_SECONDS * 1000;
const MAX_ROWS = 128;
const validNumber = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const validId = value => Number.isSafeInteger(value) && value >= 0;

export function parseTrace(metadata) {
  const trace = metadata?.orchestration;
  if (!['frontend_monotonic', 'native_worker_monotonic'].includes(trace?.clock) || !Array.isArray(trace.stages) ||
      !trace.stages.length || trace.stages.length > stageNames.length ||
      typeof metadata.session_id !== 'string' || !validId(metadata.chunk_id) ||
      !validId(metadata.output_id) || !validId(metadata.frame_count) ||
      !Array.isArray(metadata.source_frames) || metadata.source_frames.length !== metadata.frame_count) return null;
  const seen = new Set();
  const stages = [];
  for (const row of trace.stages) {
    if (!row || !Object.hasOwn(STAGES, row.stage) || seen.has(row.stage) ||
        !validNumber(row.dispatch_ns) || !validNumber(row.complete_ns) || !validNumber(row.wall_ms) ||
        row.complete_ns < row.dispatch_ns ||
        Math.abs((row.complete_ns - row.dispatch_ns) / 1e6 - row.wall_ms) > .05) return null;
    seen.add(row.stage);
    stages.push({stage: row.stage, startNs: row.dispatch_ns, endNs: row.complete_ns, wallMs: row.wall_ms});
  }
  const sources = [];
  for (const row of metadata.source_frames) {
    if (!row || !validId(row.source_frame) || !validId(row.chunk_id)) return null;
    sources.push({source_frame: row.source_frame, chunk_id: row.chunk_id});
  }
  return {key: `${metadata.session_id}:${metadata.chunk_id}:${metadata.output_id}`,
    kind: 'live', session: metadata.session_id, chunk: metadata.chunk_id, output: metadata.output_id,
    frames: metadata.frame_count, sources, stages, clock: trace.clock};
}

export class TimelineHistory {
  constructor() { this.reset(); }
  reset() {
    this.rows = []; this.selected = null; this.pinned = false; this.originNs = null; this.session = null;
    this.anchor = null; this.cursor = 0;
    this.observations = []; this.observationsOmitted = 0; this.clipIntervalsOmitted = 0; this.expiredClipIntervals = 0;
  }
  add(metadata, receivedMs) {
    let row = parseTrace(metadata);
    if (!row && metadata?.waits && validId(metadata.chunk_id) && validId(metadata.output_id) &&
        validId(metadata.frame_count) && metadata.frame_count <= 256 && typeof metadata.session_id === 'string' &&
        Array.isArray(metadata.source_frames) && metadata.source_frames.length === metadata.frame_count &&
        metadata.source_frames.every(source => validId(source?.source_frame) && validId(source?.chunk_id))) {
      row = {key: `${metadata.session_id}:${metadata.chunk_id}:${metadata.output_id}`, kind: 'live',
        session: metadata.session_id, chunk: metadata.chunk_id, output: metadata.output_id,
        frames: metadata.frame_count, sources: metadata.source_frames.map(({source_frame, chunk_id}) => ({source_frame, chunk_id})), stages: []};
    }
    if (!row || !validNumber(receivedMs) || (this.session !== null && row.session !== this.session)) return false;
    if (this.rows.some(item => item.key === row.key)) return false;
    row.waitRows = []; row.waitRevision = 0; row.waitIntervalsOmitted = 0;
    for (const observation of clipWaitRows(this.observations, row)) this.attachObservation(row, observation);
    this.observations = this.observations.filter(item => item.output === undefined || item.output !== row.output);
    if (!row.stages.length) {
      this.session = row.session;
      row.start = row.end = this.cursor;
      this.rows.push(row); if (!this.pinned) this.select(''); this.trim();
      return true;
    }
    const first = Math.min(...row.stages.map(item => item.startNs));
    const origin = this.originNs ?? first;
    if (first < origin) return false;
    this.originNs = origin; this.session = row.session;
    row.stages = row.stages.map(item => ({stage: item.stage,
      start: (item.startNs - origin) / 1e6, end: (item.endNs - origin) / 1e6, wallMs: item.wallMs}));
    row.start = Math.min(...row.stages.map(item => item.start));
    row.end = Math.max(...row.stages.map(item => item.end));
    this.rows.push(row);
    this.trim();
    if (!this.pinned) this.select('');
    if (!this.anchor || row.end >= this.anchor.serverMs) this.anchor = {serverMs: row.end, browserMs: receivedMs};
    this.cursor = Math.max(this.cursor, row.end);
    return true;
  }
  trim() {
    while (this.rows.length > MAX_ROWS) this.expire(this.rows.shift());
    while (this.observations.length > MAX_ROWS) this.observationsOmitted += this.observations.shift().spans.length;
  }
  expire(row) {
    if (row !== this.selected) this.expiredClipIntervals += row.waitRows.reduce((n, item) => n + item.spans.length, 0);
  }
  attachObservation(clip, row) {
    if (clip.waitRows.some(item => item.key === row.key)) return;
    clip.waitRows.push(row); clip.waitRevision++;
    if (clip.waitRows.length > MAX_ROWS) {
      const frameIndex = clip.waitRows.findIndex(item => item.frame !== undefined);
      const [removed] = clip.waitRows.splice(Math.max(0, frameIndex), 1);
      clip.waitIntervalsOmitted += removed.spans.length;
      this.clipIntervalsOmitted += removed.spans.length;
    }
  }
  addObservation(event, session, parsed = parseWaitObservation(event, session)) {
    if (!session || this.session !== null && session !== this.session || parsed.status !== 'valid') return false;
    this.session ??= session;
    const clips = this.selected && !this.rows.includes(this.selected) ? [this.selected, ...this.rows] : this.rows;
    for (const row of parsed.rows) {
      const matches = clips.filter(clip => clipWaitRows([row], clip).length);
      for (const clip of matches) this.attachObservation(clip, row);
      if ((row.output === undefined || !matches.length) && !this.observations.some(item => item.key === row.key)) this.observations.push(row);
    }
    this.trim();
    return true;
  }
  time(now, active) {
    if (active && this.anchor) this.cursor = Math.max(this.cursor,
      this.anchor.serverMs + Math.max(0, now - this.anchor.browserMs));
    this.rows = this.rows.filter(row => {
      if (row.end >= this.cursor - WINDOW_MS) return true;
      this.expire(row); return false;
    });
    return this.cursor;
  }
  select(key, rows = this.rows) {
    const previous = this.selected;
    this.pinned = Boolean(key);
    this.selected = (key ? rows.find(row => row.key === key) : rows.at(-1)) ?? this.selected;
    if (previous && previous !== this.selected && !this.rows.includes(previous)) this.expire(previous);
  }
}

export function replayRows(now) {
  const end = Math.max(...REPLAY.clips.flatMap(clip => clip.stages.map(row => row.start_ms + row.wall_ms)));
  const period = end + 1000;
  const firstCycle = Math.max(0, Math.floor((now - WINDOW_MS - end) / period));
  const rows = [];
  for (let cycle = firstCycle; cycle <= Math.floor(now / period); cycle++) {
    for (const clip of REPLAY.clips) {
      const stages = clip.stages.map(row => ({stage: row.stage, start: row.start_ms + cycle * period,
        end: row.start_ms + row.wall_ms + cycle * period, wallMs: row.wall_ms}));
      const start = Math.min(...stages.map(row => row.start)), finish = Math.max(...stages.map(row => row.end));
      if (finish > now || finish < now - WINDOW_MS) continue;
      rows.push({key: `replay:${cycle}:${clip.clip}`, kind: 'replay', clip: clip.clip,
        cycle: cycle + 1, frames: clip.frames, stages, start, end: finish});
    }
  }
  return rows;
}

function rowLabel(row) {
  return row.kind === 'replay' ? `历史 clip #${row.clip} · 第 ${row.cycle} 轮` : `chunk #${row.chunk} · output #${row.output}`;
}

function provenance(row) {
  if (row.kind === 'replay') return `${row.frames} 帧 · ${REPLAY.run} · 录制 clip 不等于当前网页 chunk`;
  if (!row.frames) return '输出 0 帧 · 本请求没有返回 RGB';
  const ids = row.sources.map(source => source.source_frame);
  const chunks = [...new Set(row.sources.map(source => source.chunk_id))];
  return `${row.frames} 帧 · 源帧 ${ids[0]}…${ids.at(-1)} · 来自输入 chunk ${chunks.join(', ')}`;
}

export function ganttOption(rows, now, selectedKey = null, detail = false) {
  const lanes = stageNames.filter(name => rows.some(row => row.stages.some(stage => stage.stage === name)));
  if (!lanes.length) lanes.push('text_condition', 'proxy_encode', 'render', 'decode', 'collect');
  const origin = detail && rows.length ? rows[0].start : 0;
  const minimum = detail ? 0 : Math.max(0, now - WINDOW_MS);
  const maximum = detail ? Math.max(1, ...rows.map(row => row.end - origin)) : Math.max(WINDOW_MS, now);
  const selectedVisible = rows.some(row => row.key === selectedKey);
  const data = rows.flatMap(row => row.stages.map(stage => ({
    value: [lanes.indexOf(stage.stage), stage.start - origin, stage.end - origin, stage.wallMs],
    rowKey: row.key, row, stage: stage.stage,
    itemStyle: {color: STAGES[stage.stage].color, opacity: selectedVisible && selectedKey !== row.key ? .55 : .95},
  })));
  return {
    animation: false,
    textStyle: {fontFamily: 'system-ui, sans-serif', color: '#a7b7cb'},
    grid: {left: 135, right: 24, top: 18, bottom: 40},
    tooltip: {trigger: 'item', confine: true, renderMode: 'richText', backgroundColor: '#101a29',
      borderColor: '#435875', textStyle: {color: '#e6edf6'},
      formatter: item => `${rowLabel(item.data.row)}\n${STAGES[item.data.stage].label} · ${item.value[3].toFixed(2)} ms\n` +
        `${item.value[1].toFixed(2)} → ${item.value[2].toFixed(2)} ms\n${provenance(item.data.row)}`},
    xAxis: {type: 'value', min: minimum, max: maximum, splitNumber: 5,
      axisLabel: {color: '#91a4bd', formatter: value => detail ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`},
      axisLine: {lineStyle: {color: '#344358'}}, splitLine: {lineStyle: {color: '#263448', type: 'dashed'}}},
    yAxis: {type: 'category', inverse: true, data: lanes.map(stage => STAGES[stage].label),
      axisLabel: {color: '#c4d1e2'}, axisLine: {show: false}, axisTick: {show: false},
      splitLine: {show: true, lineStyle: {color: '#263448'}}},
    series: [{id: 'stages', type: 'custom', data,
      renderItem(params, api) {
        const item = data[params.dataIndex];
        const left = api.coord([api.value(1), api.value(0)]), right = api.coord([api.value(2), api.value(0)]);
        const height = Math.min(32, api.size([0, 1])[1] * .64);
        const x = Math.max(left[0], params.coordSys.x);
        const width = Math.min(right[0], params.coordSys.x + params.coordSys.width) - x;
        if (width <= 0) return;
        const duration = `${api.value(3).toFixed(detail ? 1 : 0)} ms`;
        const fullLabel = `#${item.row.kind === 'replay' ? item.row.clip : item.row.chunk} · ${duration}`;
        const label = !detail && width >= fullLabel.length * 6.5 ? fullLabel : duration;
        return {type: 'group', children: [
          {type: 'rect', shape: {x, y: left[1] - height / 2, width, height, r: 4},
            style: {fill: STAGES[item.stage].color, opacity: item.itemStyle.opacity,
              stroke: item.rowKey === selectedKey ? '#edf6ff' : 'transparent', lineWidth: 1.5}},
          ...(width >= label.length * 6.5 ? [{type: 'text', silent: true,
            style: {x: x + width / 2, y: left[1], text: label, align: 'center', verticalAlign: 'middle',
              fill: '#102033', font: '600 11px system-ui'}}] : []),
        ]};
      },
    }],
  };
}

export function createTimeline(elements) {
  const {overview, detail, state, notice, selection, title, table, source, pause, waits} = elements;
  const live = new TimelineHistory();
  let overviewChart, detailChart;
  const observer = new ResizeObserver(() => { overviewChart?.resize(); detailChart?.resize(); });
  observer.observe(overview); observer.observe(detail);
  let replayStart = null, replaySelected = null, frozen = null, lastNow = 0, lastActive = false;
  let visibleRows = [], selected = null, detailKey = null, choicesKey = null, overviewKey = null, mode = 'live';
  let waitsKey = null;
  function select(key) {
    if (mode === 'live') live.select(key, key ? visibleRows : live.rows);
    else replaySelected = key ? visibleRows.find(row => row.key === key) ?? replaySelected : null;
    if (frozen) frozen.selected = key ? (mode === 'live' ? live.selected : replaySelected) : visibleRows.at(-1);
    paint(lastNow, lastActive);
  }
  selection.addEventListener('change', () => select(selection.value));
  source.addEventListener('change', () => {
    mode = source.value; replayStart = null; replaySelected = null; frozen = null;
    detailKey = null; choicesKey = null; pause.textContent = '暂停滚动';
    paint(lastNow, lastActive);
  });
  pause.addEventListener('click', () => {
    if (frozen) {
      if (mode === 'replay') replayStart = lastNow - frozen.time;
      frozen = null;
    } else frozen = {rows: visibleRows.slice(), selected, time: Number(state.dataset.time)};
    pause.textContent = frozen ? '继续滚动' : '暂停滚动';
    paint(lastNow, lastActive);
  });
  function paint(now, active) {
    if (!overviewChart) {
      overviewChart = globalThis.echarts.init(overview);
      detailChart = globalThis.echarts.init(detail);
      overviewChart.on('click', event => { if (event.data?.rowKey) select(event.data.rowKey); });
    }
    lastNow = now; lastActive = active;
    const liveTime = live.time(now, active);
    let time = liveTime;
    if (mode === 'replay') {
      replayStart ??= now;
      time = now - replayStart;
      visibleRows = replayRows(time);
      selected = replaySelected ?? visibleRows.at(-1) ?? null;
    } else {
      visibleRows = live.rows; selected = live.selected;
    }
    if (frozen) { visibleRows = frozen.rows; selected = frozen.selected; time = frozen.time; }
    state.dataset.time = String(time);
    state.textContent = `${mode === 'replay' ? '历史回放 · 非当前会话' : active ? '实时阶段回执' : '当前会话 · 已冻结'}${frozen ? ' · 暂停查看' : ''} · ${(Math.max(0, time - WINDOW_MS) / 1000).toFixed(1)}–${(time / 1000).toFixed(1)} s`;
    notice.textContent = mode === 'replay'
      ? '历史录制循环回放，非本次推理。保留原始阶段间隔，循环只平移时间原点。'
      : live.originNs === null ? '尚无有效阶段计时：等待模型完成批次；若已有 RGB 输出仍为空，说明服务没有回传阶段计时。'
        : '已完成批次的服务端墙钟区间，随输出回执更新，非执行中的 GPU 进度。游标按本地经过时间推进。' +
          (live.selected?.clock === 'native_worker_monotonic' ? ' 原生路径：Proxy 准备含当前编码（未预编码时）与噪声；VAE 并行包含 RGB 解码、下一包 Proxy 编码及 D2H，不拆分估算。' : ' Ray 路径：前端调度区间。');
    const nextOverview = `${mode}:${time}:${selected?.key}:${visibleRows.map(row => row.key).join('|')}`;
    if (nextOverview !== overviewKey) {
      overviewChart.setOption(ganttOption(visibleRows, time, selected?.key));
      overviewKey = nextOverview;
    }
    overview.setAttribute('aria-label', `${state.textContent}，${visibleRows.length} 个已完成请求`);
    const choices = selected && !visibleRows.some(row => row.key === selected.key) ? [selected, ...visibleRows] : visibleRows;
    const nextChoices = choices.map(row => row.key).join('|');
    if (nextChoices !== choicesKey) {
      selection.replaceChildren();
      if (choices.length) {
        const option = document.createElement('option'); option.value = ''; option.textContent = '跟随最新 clip';
        selection.append(option);
      }
      for (const row of choices) {
        const option = document.createElement('option'); option.value = row.key;
        option.textContent = `${rowLabel(row)} · ${row.frames} 帧${!visibleRows.includes(row) ? ' · 已固定' : ''}`;
        selection.append(option);
      }
      choicesKey = nextChoices;
    }
    selection.disabled = !choices.length;
    selection.value = (mode === 'live' ? live.pinned : replaySelected) ? selected?.key ?? '' : '';
    if (detailKey !== selected?.key) {
      detailKey = selected?.key;
      detailChart.setOption(ganttOption(selected ? [selected] : [], 0, selected?.key, true));
      title.textContent = selected ? `${rowLabel(selected)} · ${provenance(selected)}` : '点击上方条形或选择一个请求，固定查看阶段耗时。';
      detail.setAttribute('aria-label', title.textContent);
      table.replaceChildren();
      for (const span of selected?.stages ?? []) {
        const tr = document.createElement('tr');
        for (const value of [STAGES[span.stage].label, (span.start - selected.start).toFixed(2),
          (span.end - selected.start).toFixed(2), span.wallMs.toFixed(2)]) {
          const td = document.createElement('td'); td.textContent = value; tr.append(td);
        }
        table.append(tr);
      }
    }
    const nextWaits = `${selected?.key}:${selected?.waitRevision}:${mode}`;
    if (waits && waitsKey !== nextWaits) {
      waitsKey = nextWaits; waits.replaceChildren();
      const rows = selected?.waitRows ?? [];
      const note = document.createElement('p');
      note.textContent = mode === 'replay' ? '历史回放不含本次等待回执。'
        : `等待按源输入 chunk 与输出帧关联。各时钟独立取原点，不能跨时钟相减。未采集 socket 写完成、物理传输、屏幕呈现和末帧 ACK 排空。当前保留 ${rows.length} 条相关回执，本 clip 已淘汰 ${selected?.waitIntervalsOmitted ?? 0} 个区间。`;
      waits.append(note);
      for (const domain of Object.keys(WAIT_CLOCKS)) {
        const domainRows = rows.filter(row => row.domain === domain);
        if (!domainRows.length) continue;
        const origin = Math.min(...domainRows.flatMap(row => row.spans.map(span => span.start_ns)));
        const heading = document.createElement('h4'); heading.textContent = `${WAIT_CLOCKS[domain]} · 独立相对原点 · ms`;
        const grid = document.createElement('table'); grid.className = 'gantt-table';
        const header = document.createElement('tr');
        for (const value of ['回执 / 来源', '区间', '开始 ms', '结束 ms', '耗时 ms', '结果']) {
          const cell = document.createElement('th'); cell.textContent = value; header.append(cell);
        }
        grid.append(header);
        for (const row of domainRows) for (const span of row.spans) {
          const tr = document.createElement('tr');
          for (const value of [`${row.kind} · chunk ${row.chunk ?? '—'} · output ${row.output ?? '—'} · frame ${row.frame ?? '—'} · source ${row.source ?? '—'}`,
            span.name, ((span.start_ns - origin) / 1e6).toFixed(2), ((span.end_ns - origin) / 1e6).toFixed(2),
            ((span.end_ns - span.start_ns) / 1e6).toFixed(2), span.outcome]) {
            const cell = document.createElement('td'); cell.textContent = value; tr.append(cell);
          }
          grid.append(tr);
        }
        waits.append(heading); waits.append(grid);
      }
    }
  }
  return {
    add: (metadata, now) => live.add(metadata, now),
    addObservation: (event, session, parsed) => live.addObservation(event, session, parsed),
    waitCoverage: () => ({pending_receipts: live.observations.length, pending_intervals_omitted: live.observationsOmitted,
      clip_intervals_omitted: live.clipIntervalsOmitted, expired_clip_intervals: live.expiredClipIntervals,
      retention: '128 model clips plus one pin; 128 wait receipts per clip; 128 pending receipts'}),
    reset() {
      live.reset(); frozen = null; detailKey = null; choicesKey = null; overviewKey = null;
      waitsKey = null;
      if (mode === 'replay') { replayStart = null; replaySelected = null; }
      pause.textContent = '暂停滚动';
    },
    paint,
    dispose() { observer.disconnect(); overviewChart?.dispose(); detailChart?.dispose(); },
  };
}
