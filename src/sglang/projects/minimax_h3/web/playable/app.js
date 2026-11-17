import {createGame} from './game.js';
import {sourceScenes, sceneAvailable} from './sources.mjs';
import {createTorcsSource} from './torcs-source.mjs';
import {createBallpointSource} from './ballpoint-source.mjs';
import {createNativeGameSource} from './native-game-source.mjs';
import {pack, unpack, rgbaToRgb, rgbToRgba, packetSize, percentile, FrameLedger, profilingLabel, CapturePacer, DisplayPacer, sourceBudgetPolicy, sourceBudgetBlocked} from './stream.mjs';
import {CHART_WINDOW_SECONDS, CHART_MAX_SAMPLES, TimingHistory, renderStageMs, paintChart, disposeCharts} from './charts.mjs';
import {createTimeline} from './gantt.mjs';
import {prepareMedia, MediaDecoder} from './media.mjs';
import {PairedRecorder} from './recording.mjs';
import {WAIT_SCHEMA, WAIT_CLOCKS, WAIT_UNCOLLECTED, pendingPacketTiming, parseWaitObservation} from './waits.mjs';

const $ = id => document.getElementById(id);
const ns = () => Math.round(performance.now() * 1e6);
const interval = 1000 / 24;
const CHART_REFRESH_HZ = 10;
const chartRefreshInterval = 1000 / CHART_REFRESH_HZ;
const activeStates = new Set(['starting', 'running', 'finishing', 'draining', 'stopping']);
let config, game, reference, socket, run, connection;
let latestRecording = null, recordingDownloads = new Set();
let phase = 'idle', latestInput = null, inputId = 0, nextDraw = 0, previousTick = performance.now();
let lastMetrics = 0, promptRevision = 1, currentPrompt = null, referenceRevision = 0;
let sceneId = 'boxing', referenceScene = null, referenceSource = null;
const isNativeScene = (value = sceneId) => !!sourceScenes[value]?.protocol;
const isRgbScene = () => sourceScenes[sceneId].reference === 'rgb';
const canSeedReference = () => config?.mode === 'fixture' || isNativeScene() || isRgbScene();
const autoSeedReference = () => config.mode === 'fixture' || sourceScenes[sceneId].reference;
const isNativeProtocol = value => value === 'native_game_commands/v1';
let sourceGate = false, sourceTransition = 0, sourceReady = false;
const sceneName = () => ({racing: '赛车', boxing: '拳击', street: '街景'})[sceneId];
const outputContext = $('output').getContext('2d');
const timeline = createTimeline({overview: $('gantt-overview'), detail: $('gantt-detail'),
  state: $('gantt-state'), notice: $('gantt-notice'), selection: $('gantt-selection'),
  title: $('gantt-selected'), table: $('gantt-rows'), source: $('gantt-source'), pause: $('gantt-pause'),
  waits: $('gantt-waits')});

function startupFrozen() {
  return ['waiting_batch', 'draining_produced'].includes(run?.startup.state);
}

function changeSource(operation) {
  const source = game, revision = ++sourceTransition, activeRun = run;
  sourceGate = true;
  updateControls();
  const ready = () => {
    if (game !== source || revision !== sourceTransition) return;
    sourceGate = false;
    updateControls();
    if (isNativeScene()) $('source-pause').textContent = source.getSimulationState().state === 'live' ? '暂停游戏' : '继续游戏';
    pumpNativeGame();
  };
  let result;
  try { result = operation(source); } catch (error) { ready(); throw error; }
  if (!result?.then) { ready(); return; }
  const pending = result.then(ready);
  pending.catch(error => {
    if (game === source && run === activeRun && revision === sourceTransition) { ready(); fail(error); }
  });
  return pending;
}

function stopSource() { return changeSource(source => source.stop()); }
function preparedNativeReference() {
  return isNativeScene() && referenceSource === 'fixture-proxy';
}

function startupSnapshot(now = performance.now()) {
  const s = run.startup;
  return {mode: s.mode, state: s.state, simulation: run.terminalSimulation ?? game.getSimulationState(),
    committed_frames: s.committed_frames, emitted_frames: run.ledger.emitted, drawn_frames: run.displayed,
    capture_frames: run.capture.length, upload_sent: run.upload?.sent ?? null,
    chunk_id: s.chunk_id, accepted_frames: s.accepted_frames, drain_target: s.drain_target,
    paused_ms: s.paused_ms + (s.pause_started_ms === null ? 0 : now - s.pause_started_ms),
    ignored_inputs: s.ignored_inputs};
}

function startupTransition(state, reason) {
  const s = run.startup, from = s.state, now = performance.now();
  const wasFrozen = startupFrozen();
  s.state = state;
  const frozen = startupFrozen();
  if (!wasFrozen && frozen) {
    s.pause_started_ms = now;
    run.pacer.stop(now);
    changeSource(source => source.setControlsSuspended(true));
    latestInput = null;
  } else if (wasFrozen && !frozen) {
    s.paused_ms += now - s.pause_started_ms;
    s.pause_started_ms = null;
    changeSource(source => source.setControlsSuspended(false));
    previousTick = now;
    run.pacer.next = now;
  }
  const transition = {from, to: state, reason, browser_ns: ns(), ...startupSnapshot(now)};
  if (s.transitions.length === 1000) { s.transitions.shift(); s.transitions_omitted++; }
  s.transitions.push(transition);
  record('startup_transition', transition);
}

function stopStartup(reason) {
  if (!run?.startup || run.startup.state === 'resumed') return;
  const s = run.startup;
  if (s.pause_started_ms !== null) s.paused_ms += performance.now() - s.pause_started_ms;
  s.pause_started_ms = null;
  s.state = 'resumed';
  record('startup_stopped', {reason});
}

function startupCommitted(metadata, count) {
  const s = run.startup;
  s.committed_frames += count;
  if (s.state !== 'collecting' || phase !== 'running') return;
  if (count !== 17 || metadata.eos || run.capture.length || run.upload) {
    throw new Error('启动暂停要求完整的 17 帧普通包且本地采集和上传已清空');
  }
  s.chunk_id = metadata.chunk_id;
  s.accepted_frames = s.committed_frames;
  startupTransition('waiting_batch', 'normal_17_committed');
}

function startupBatch(metadata) {
  const s = run.startup;
  if (s.state !== 'waiting_batch') return;
  if (metadata.chunk_id !== s.chunk_id || metadata.status !== 'accepted' ||
      metadata.accepted_frames !== s.accepted_frames || metadata.accepted_frames !== s.committed_frames ||
      metadata.total_frames !== run.ledger.emitted) {
    throw new Error('启动追齐的批次或累计帧数不匹配');
  }
  if (metadata.frame_count === 0) {
    startupTransition('collecting', 'zero_output_needs_another_normal_17');
  } else {
    s.drain_target = metadata.total_frames;
    startupTransition('draining_produced', 'drain_emitted_snapshot_not_retained_tail');
    startupDrawn();
  }
}

function startupDrawn() {
  if (run.startup.state === 'draining_produced' && run.displayed >= run.startup.drain_target) {
    startupTransition('resumed', 'produced_watermark_drawn_and_ack_enqueued');
  }
}

function status(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

function sourceBudgetSnapshot(now = performance.now()) {
  const budget = run.sourceBudget;
  return {selected: budget.selected, effective: budget.effective, limit_frames: budget.limit_frames, reason: budget.reason,
    lead_frames: run.captured - run.displayed, peak_lead_frames: budget.peak_lead_frames,
    wait_count: budget.wait_count, wait_ms: budget.wait_ms + (budget.wait_started_ms === null ? 0 : now - budget.wait_started_ms),
    game_time_policy: 'wall_time_render; budget_wait_skips_capture_slots; startup_freeze_unchanged'};
}

function sourceBudgetWait(blocked, now) {
  const budget = run.sourceBudget;
  if (blocked && budget.wait_started_ms === null) {
    budget.wait_started_ms = now;
    budget.wait_count++;
    run.pacer.stop(now);
  } else if (!blocked && budget.wait_started_ms !== null) {
    budget.wait_ms += now - budget.wait_started_ms;
    budget.wait_started_ms = null;
    run.pacer.next = now;
  }
}

function setPhase(value) {
  if (run && !['running', 'finishing'].includes(value)) sourceBudgetWait(false, performance.now());
  if (run && value === 'idle' && activeStates.has(phase)) {
    finishRecording(run.outcome === 'complete' ? 'complete' : 'interrupted');
    run.terminalSimulation = game.getSimulationState();
    run.endedAt = performance.now();
    run.chartEnd = (run.endedAt - run.started_ns / 1e6) / 1000;
  }
  if (value !== 'running' && value !== 'starting') stopStartup(`lifecycle_${value}`);
  phase = value;
  updateControls();
}

function updateControls() {
  const busy = activeStates.has(phase), preparing = sourceGate || !sourceReady;
  $('start').disabled = busy || preparing || !reference || !config;
  $('reference-preview').hidden = !reference;
  $('reference').disabled = busy;
  $('scene').disabled = busy;
  $('reset-game').disabled = busy || preparing;
  $('seed').disabled = busy;
  $('capture-policy').disabled = busy;
  $('source-budget').disabled = busy || !config?.source_budget?.eligible;
  $('budget-help').textContent = config?.source_budget?.eligible
    ? '限制采集领先显示的帧数；等待时游戏继续运行。'
    : '当前服务不支持帧预算，本次不启用。';
  $('startup-mode').disabled = busy || !config?.media;
  $('playback-mode').disabled = busy || !config?.media;
  $('observe-waits').disabled = busy || !config?.media;
  $('record-session').disabled = busy || isNativeScene() || !globalThis.VideoEncoder || !globalThis.VideoFrame;
  if (!busy && isNativeScene()) $('record-session').checked = false;
  $('bitrate').disabled = busy || !config?.media;
  $('fixture-reference').disabled = busy || preparing;
  $('fixture-reference').hidden = !canSeedReference();
  $('torcs-mode').disabled = busy || preparing;
  $('torcs-auto').disabled = busy || preparing;
  $('source-pause').disabled = preparing || ['starting', 'finishing', 'draining', 'stopping'].includes(phase);
  $('fixture-reference').textContent = isNativeScene() ? '用本机游戏准备帧生成 proxy 参考图' : sceneId === 'torcs' ? '用赛车同帧 RGB 准备首图' : sceneId === 'ballpoint' ? '用游戏同帧 RGB 准备首图' : '用 proxy 生成测试参考图';
  $('finish').disabled = phase !== 'running';
  $('cancel').disabled = !['starting', 'running', 'finishing', 'draining'].includes(phase);
  $('download').disabled = !run;
  $('output-title').textContent = run
    ? `${busy ? '服务输出' : '上次输出'} · ${sourceScenes[run.proxy.scene].label}` : '服务输出';
  $('diagnostics-session').textContent = run
    ? `${busy ? '当前会话' : '上次会话'} · ${sourceScenes[run.proxy.scene].label}` : '尚无会话';
  $('metrics-session').textContent = $('diagnostics-session').textContent;
  const native = run ? isNativeProtocol(run.proxy_protocol) : isNativeScene();
  const sample = native ? '采样请求' : '采集';
  $('visible-age-label').textContent = native ? '可见帧的采样请求年龄' : '可见源画面年龄';
  $('visible-age-info').textContent = native
    ? '当前浏览器时间减采样请求时间；原生画面发布时间属于另一时钟，画面实际年龄未知。'
    : '当前浏览器时间减源帧采集时间；画面不变时继续增长，结束后也继续计时';
  $('latest-draw-label').textContent = `最近可见帧 · ${sample} → 绘制延迟`;
  $('receive-age-label').textContent = `10 秒事件窗口 · ${sample} → 接收 P50 / P95`;
  $('display-age-label').textContent = `10 秒事件窗口 · ${sample} → 绘制 P50 / P95`;
}

function recordingControls() {
  const ready = !!latestRecording?.result;
  $('download-proxy-video').disabled = !ready;
  $('download-rgb-video').disabled = !ready;
  $('download-video-timing').disabled = !ready;
}

function finishRecording(reason) {
  const recorder = run?.recording;
  if (!recorder || recorder.done) return;
  $('record-status').textContent = '正在完成同步录像…';
  recorder.finish(reason).then(() => { if (latestRecording === recorder) recordingControls(); });
}

function record(event, fields = {}) {
  if (!run) return;
  const row = {event, browser_ns: ns(), ...fields};
  if (run.observe_waits) {
    const parsed = parseWaitObservation(row, run.session_id);
    if (parsed.status === 'valid') run.waitCoverage.receipts++;
    if (parsed.status === 'absent' && ['started', 'chunk_send', 'chunk_admitted', 'chunk_receive', 'frame_receive',
      'frame_decode', 'frame_draw', 'complete', 'cancelled'].includes(event)) run.waitCoverage.absent_receipts++;
    if (['malformed', 'unsupported', 'stale'].includes(parsed.status)) {
      run.waitCoverage[parsed.status]++;
      row.wait_status = parsed.status;
      delete row.waits;
      if (row.metadata?.waits !== undefined) { row.metadata = {...row.metadata}; delete row.metadata.waits; }
    }
    timeline.addObservation(row, run.session_id, parsed);
  }
  if (run.events.length === 10000) {
    const omitted = run.events.shift(); run.events_omitted++;
    if (run.observe_waits) run.waitCoverage.intervals_omitted += parseWaitObservation(omitted, run.session_id).rows.reduce((n, row) => n + row.spans.length, 0);
  }
  run.events.push(row);
}

function pendingInputWaits(outcome) {
  if (!run.observe_waits || !run.capture.length) return {};
  const packet = pendingPacketTiming(run.capture, phase === 'finishing', run.finishRequestedNs);
  const started = packet?.readyNs ?? run.capture[0].capture_ns;
  const ended = ns();
  const browser = {[packet ? 'pending_input' : 'incomplete_assembly']: {start_ns: started, end_ns: ended, outcome}};
  if (packet && run.capture.length > packet.count) {
    const tail = run.capture.slice(packet.count);
    const tailPacket = pendingPacketTiming(tail, phase === 'finishing', run.finishRequestedNs);
    browser[tailPacket ? 'pending_input_tail' : 'incomplete_assembly'] = {
      start_ns: tailPacket?.readyNs ?? tail[0].capture_ns, end_ns: ended, outcome};
  }
  return {chunk_id: run.upload?.chunkId ?? run.chunkId, first_source_frame: run.capture[0].id, pending_frames: run.capture.length,
    waits: {schema: WAIT_SCHEMA, browser}};
}

function observe(kind, time) {
  const samples = run.rates[kind];
  samples.push(time);
  while (samples.length && samples[0] < time - 5000) samples.shift();
}

function input(event) {
  if (!activeStates.has(phase)) return;
  if (event.disposition === 'ignored_during_startup_pause') {
    if (run) run.startup.ignored_inputs++;
    record('input_ignored', {...event, simulation: game.getSimulationState()});
    return;
  }
  if (event.disposition === 'released_on_startup_pause') {
    record('input_released', event);
    return;
  }
  latestInput = {id: ++inputId, emitted_ns: ns(), ...event};
  record('input', latestInput);
}

function clearOutput() {
  if (run) {
    if (run.observe_waits) run.waitCoverage.browser_discarded += run.decoder?.outstanding.size ?? 0;
    run.decoder?.close(); run.playback.length = 0; run.mediaSources?.clear();
    run.latestDraw = null;
  }
  outputContext.clearRect(0, 0, $('output').width, $('output').height);
  $('empty').hidden = false;
  $('visible-age').textContent = '—';
  $('latest-draw-age').textContent = '—';
}

function cancel(reason) {
  if (!activeStates.has(phase)) return;
  finishRecording('cancelled');
  if (run.closeIntent !== 'error') run.closeIntent = 'explicit_cancel';
  record('cancel_requested', {reason, ...pendingInputWaits('cancelled')});
  run.outcome = 'cancelled';
  run.pacer.stop(performance.now());
  run.capture.length = 0;
  run.upload = null;
  clearOutput();
  stopSource();
  setPhase('stopping');
  status(`${reason}。等待后端释放会话…`, true);
  if (connection) connection.cancel();
  else if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({op: 'cancel'}));
  else if (socket) socket.close();
  else setPhase('idle');
}

function fail(error) {
  if (activeStates.has(phase)) {
    run.closeIntent = 'error';
    record('error', {error: String(error.message || error)});
    cancel(String(error.message || error));
    run.outcome = 'error';
  } else status(String(error.message || error), true);
}

function takeReference(source, sourceName = 'upload') {
  const {width, height} = config.encoder_recipe;
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  const ratio = Math.max(width / source.width, height / source.height);
  ctx.drawImage(source, (width - source.width * ratio) / 2, (height - source.height * ratio) / 2,
    source.width * ratio, source.height * ratio);
  reference = rgbaToRgb(ctx.getImageData(0, 0, width, height).data);
  referenceScene = sceneId;
  referenceSource = sourceName;
  const preview = $('reference-preview-image');
  preview.width = width; preview.height = height;
  preview.getContext('2d').putImageData(new ImageData(rgbToRgba(reference), width, height), 0, 0);
  $('reference-preview-caption').textContent = sourceName === 'fixture-proxy' ? 'Proxy 测试参考图' : '上传参考图';
  $('reference-info').textContent = `已准备 ${width} × ${height} RGB 参考图，居中裁剪。`;
  setPhase(phase);
  if (!activeStates.has(phase)) status('首帧已就绪，可以开始串流。');
}

$('reference').addEventListener('change', async () => {
  if (activeStates.has(phase)) return;
  const revision = ++referenceRevision;
  const file = $('reference').files[0];
  reference = null;
  referenceScene = referenceSource = null;
  $('reference-info').textContent = file ? '正在处理上传参考图…' : '请选择参考图片。';
  setPhase(phase);
  if (!file) return;
  try {
    if (file.size > 20 * 1024 * 1024) throw new Error('参考图片上限 20 MiB');
    const bitmap = await createImageBitmap(file);
    try { if (revision === referenceRevision && referenceScene === null) takeReference(bitmap); } finally { bitmap.close(); }
  } catch (error) {
    if (revision === referenceRevision) {
      $('reference-info').textContent = '参考图处理失败，请重新选择图片。';
      fail(error);
    }
  }
});

async function seedProxyReference() {
  if (activeStates.has(phase) || !canSeedReference()) return;
  const revision = ++referenceRevision, source = game;
  try {
    const prepared = changeSource(source => source.prepareRun());
    if (prepared) await prepared;
    if (game !== source || revision !== referenceRevision || activeStates.has(phase)) return;
    referenceFromProxy();
    return true;
  } catch (error) { if (game === source && revision === referenceRevision) fail(error); }
  finally {
    // An upload can supersede the prepared reference while the source is loading.
    if (game === source && !activeStates.has(phase)) changeSource(source => source.setControlsSuspended(false));
  }
}

function referenceFromProxy() {
  if (isRgbScene()) {
    const spec=config.encoder_recipe;
    takeReference(game.referenceCanvas(spec.width,spec.height),`${sceneId}-rgb`);
    $('reference-preview-caption').textContent=sceneId==='torcs'?'TORCS 同帧 RGB 首图':'Ballpoint 同帧 RGB 首图';
    $('reference-info').textContent='原始 RGB 与 Proxy 来自同一重置状态；全画幅缩放，无裁切。模型画质需另行验收。';
    return;
  }
  if (isNativeScene()) {
    reference = new Uint8Array();
    referenceScene = sceneId; referenceSource = 'fixture-proxy';
    const preview = $('reference-preview-image');
    preview.width = config.encoder_recipe.width; preview.height = config.encoder_recipe.height;
    game.drawReference(preview);
    $('reference-preview-caption').textContent = '本机游戏准备帧 JPEG 预览';
    $('reference-info').textContent = '网关将准备帧 JPEG 裁成 proxy 画幅，再放大为首图；画质和上传图对齐效果未验证。浏览器不提取 Proxy RGB。';
    setPhase(phase);
    return;
  }
  const pixels = game.capture();
  if (!pixels) throw new Error('Proxy 首帧尚未准备好');
  const canvas = document.createElement('canvas');
  canvas.width = game.width; canvas.height = game.height;
  canvas.getContext('2d').putImageData(new ImageData(rgbToRgba(pixels), game.width, game.height), 0, 0);
  takeReference(canvas, 'fixture-proxy');
  $('reference-info').textContent += ' 测试参考图，不代表真实外观条件。';
}
$('fixture-reference').addEventListener('click', seedProxyReference);

function boxingStatus(state) {
  $('player-hp').textContent = state.playerHp;
  $('opponent-hp').textContent = state.opponentHp;
  $('boxing-state').textContent = state.phase === 'ready' ? '按 WASD 或 J/K 开始'
    : state.phase === 'result' ? (state.winner === 'player' ? '胜利' : '落败') : '对战中';
}

function racingStatus(state) {
  $('race-state').textContent = state.phase === 'ready' ? '按 W 或 ↑ 开始'
    : state.phase === 'countdown' ? `${state.countdown} 秒后出发` : state.offroad ? '驶离赛道' : '比赛中';
  $('race-speed').textContent = `${state.speed} km/h`;
  $('race-laps').textContent = `${state.laps} 圈`;
}

function mountGame() {
  game?.dispose();
  sourceTransition++; sourceGate = false;
  const previous = $('proxy'), canvas = previous.cloneNode(false);
  previous.replaceWith(canvas);
  const {proxy_width: width, proxy_height: height} = config.encoder_recipe;
  const scene = sourceScenes[sceneId];
  sourceReady = !isRgbScene();
  const create = {synthetic: createGame, torcs: createTorcsSource, ballpoint: createBallpointSource, native: createNativeGameSource}[scene.kind];
  game = create(canvas, {width, height, scene: sceneId, onInput: input,
    url: config[scene.configKey]?.url,
    onReady: () => {
      if(game!==mounted)return;
      sourceReady = true;
      if(sceneId==='torcs'){
        $('torcs-mode').replaceChildren(...game.modes().map(mode=>new Option(mode.label,mode.id)));
        $('torcs-mode').value = game.getSimulationState().proxyMode;
        $('torcs-auto').checked=game.getSimulationState().auto;
      }
      setPhase(phase);
    },
    onError: error => { if (game === mounted) fail(error); },
    onPointerStatus: message => { if (game === mounted && sceneId === 'ballpoint') $('scene-help').textContent = `${sourceScenes.ballpoint.help} ${message}`; },
    onStatus: sceneId === 'boxing' ? boxingStatus : racingStatus,
    canReset: () => !activeStates.has(phase)});
  const mounted = game;
  game.render(0);
  $('boxing-hud').hidden = sceneId !== 'boxing';
  $('racing-hud').hidden = sceneId !== 'racing';
  $('torcs-controls').hidden = sceneId !== 'torcs';
  $('reset-game').hidden = !scene.resetLabel;
  $('source-pause').hidden = !isNativeScene();
  $('reset-game').textContent = scene.resetLabel ?? '';
  $('scene-help').textContent = sourceScenes[sceneId].help;
}

async function selectScene(value) {
  if (activeStates.has(phase) || !sceneAvailable(value, config) || value === sceneId) {
    $('scene').value = sceneId;
    return;
  }
  sceneId = value;
  $('scene').value = value;
  referenceRevision++;
  reference = null;
  referenceScene = referenceSource = null;
  $('reference').value = '';
  mountGame();
  setPhase(phase);
  if (autoSeedReference()) {
    if (await seedProxyReference()) status('场景与同帧首图已就绪，可以开始串流。');
  } else if (isNativeScene()) {
    $('reference-info').textContent = '选择本机游戏准备帧 proxy 参考图，或上传自己的图片。上传图与原生第 0 帧的对齐效果未验证。';
    status('本机游戏场景已就绪。选择首图后开始串流。');
  } else {
    const name = sceneName();
    $('reference-info').textContent = `${name}请上传匹配场景的参考图。`;
    status(`${name}场景已就绪。上传参考图后开始串流。`);
  }
}
$('scene').addEventListener('change', () => selectScene($('scene').value));
$('torcs-mode').addEventListener('change', async () => {
  if(sceneId!=='torcs'||activeStates.has(phase))return;
  try{game.setProxyMode($('torcs-mode').value);if(referenceSource==='torcs-rgb')await seedProxyReference();}
  catch(error){fail(error);}
});
$('torcs-auto').addEventListener('change', () => {
  if(sceneId!=='torcs'||activeStates.has(phase))return;
  try{game.setAutopilot($('torcs-auto').checked);}catch(error){fail(error);}
});
$('reset-game').addEventListener('click', () => {
  if (activeStates.has(phase)) return;
  changeSource(source => source.reset());
});
$('source-pause').addEventListener('click', () => {
  if (!isNativeScene() || sourceGate) return;
  const pause = game.getSimulationState().state === 'live';
  changeSource(source => source.setControlsSuspended(pause));
});

$('prompt').addEventListener('input', () => {
  if ($('prompt').value.trim()) {
    currentPrompt = {revision: ++promptRevision, text: $('prompt').value, emitted_ns: ns()};
    $('prompt-info').textContent = `描述修订 ${promptRevision}，将在下次 chunk 提交时生效。`;
    if (activeStates.has(phase)) record('prompt', currentPrompt);
  } else {
    $('prompt-info').textContent = '描述为空，尚未提交；继续使用上一次非空描述。';
    if (activeStates.has(phase)) record('prompt_rejected', {reason: 'blank', retained_revision: currentPrompt?.revision});
  }
});

function sendChunk() {
  if (run.capturePending || (!run.credit && !run.upload) || !['running', 'finishing'].includes(phase)) return;
  const count = packetSize(run.capture.length, phase === 'finishing');
  if (run.proxy_protocol === 'proxy_frames/v1') {
    const limit = config.max_message_bytes || 128 * 1024 * 1024;
    const target = count || Math.min(run.capture.length, 17);
    while ((run.upload?.sent ?? 0) < target) {
      const frame = run.capture[run.upload?.sent ?? 0];
      if (socket.bufferedAmount + frame.pixels.byteLength + 4 > limit) return;
      const chunkId = run.upload?.chunkId ?? run.chunkId;
      const metadata = {op: 'proxy_frame', chunk_id: chunkId, source_frame: frame.id, capture_ns: frame.capture_ns};
      const body = pack(metadata, frame.pixels);
      if (socket.bufferedAmount + body.byteLength > limit) return;
      const started = run.observe_waits ? ns() : null;
      socket.send(body);
      if (!run.upload) { run.upload = {chunkId: run.chunkId++, sent: 0}; run.credit = false; }
      run.upload.sent++;
      frame.pixels = null;
      if (run.observe_waits) record('proxy_frame_enqueue', {...metadata, payload_bytes: body.byteLength,
        waits: {schema: WAIT_SCHEMA, browser: {proxy_frame_enqueue: {start_ns: started, end_ns: ns()}}}});
    }
    if (!count) return;
    const frames = run.capture.slice(0, count);
    const metadata = {op: 'proxy_commit', chunk_id: run.upload.chunkId, first_source_frame: frames[0].id,
      capture_ns: frames.map(f => f.capture_ns), assembled_ns: ns(), eos: count === 5};
    if (currentPrompt) metadata.prompt = currentPrompt;
    const body = pack(metadata);
    if (socket.bufferedAmount + body.byteLength > limit) return;
    const started = run.observe_waits ? ns() : null;
    socket.send(body);
    run.ledger.submit(frames, run.upload.chunkId);
    run.admissionChunk = run.upload.chunkId;
    run.upload = null;
    run.capture.splice(0, count);
    const fields = {...metadata, input_ids: frames.map(f => f.input?.id ?? null),
      payload_bytes: count * game.width * game.height * 3, proxy_protocol: run.proxy_protocol, commit_bytes: body.byteLength};
    if (run.observe_waits) {
      const packet = pendingPacketTiming(frames, phase === 'finishing', run.finishRequestedNs);
      fields.waits = {schema: WAIT_SCHEMA, browser: {first_capture_ns: packet.firstCaptureNs,
        packet_ready_ns: packet.readyNs, proxy_commit_enqueue: {start_ns: started, end_ns: ns()}}};
    }
    record('chunk_send', fields);
    startupCommitted(metadata, count);
    if (metadata.eos) { run.pacer.stop(performance.now()); stopSource(); setPhase('draining'); status('EOS 已发送，等待剩余画面…'); }
    return;
  }
  if (!count) return;
  const packet = run.observe_waits ? pendingPacketTiming(run.capture, phase === 'finishing', run.finishRequestedNs) : null;
  const frames = run.capture.splice(0, count);
  const native = isNativeProtocol(run.proxy_protocol);
  const pixels = new Uint8Array(native ? 0 : frames[0].pixels.length * count);
  if (!native) frames.forEach((frame, i) => pixels.set(frame.pixels, i * frame.pixels.length));
  const chunkId = run.chunkId++;
  const metadata = {op: native ? 'native_game_chunk' : 'chunk', chunk_id: chunkId, first_source_frame: frames[0].id,
    capture_ns: frames.map(f => f.capture_ns), assembled_ns: ns(), eos: count === 5};
  if (currentPrompt) metadata.prompt = currentPrompt;
  run.ledger.submit(frames, chunkId);
  run.credit = false;
  run.admissionChunk = chunkId;
  const fields = {...metadata, input_ids: frames.map(f => f.input?.id ?? null), payload_bytes: pixels.length};
  if (packet) {
    const body = pack(metadata, pixels);
    const sendStart = ns();
    socket.send(body);
    fields.waits = {schema: WAIT_SCHEMA, browser: {first_capture_ns: packet.firstCaptureNs,
      packet_ready_ns: packet.readyNs, credit_obtained_ns: Math.max(packet.readyNs, run.creditGrantedNs),
      send_call_start_ns: sendStart, send_call_end_ns: ns()}};
    record('chunk_send', fields);
  } else {
    record('chunk_send', fields);
    socket.send(pack(metadata, pixels));
  }
  startupCommitted(metadata, count);
  if (metadata.eos) { run.pacer.stop(performance.now()); stopSource(); setPhase('draining'); status('EOS 已发送，等待剩余画面…'); }
}

function receiveFrame(frame, now, fields = {}) {
  run.received++;
  observe('receive', now);
  run.charts.add('receive', (now - run.started_ns / 1e6) / 1000, now - frame.capture_ns / 1e6);
  record('frame_receive', {source_frame: frame.id, capture_ns: frame.capture_ns, input_id: frame.input?.id ?? null, ...fields});
}

async function receive(m, pixels) {
  if (m.op === 'error') {
    record('remote_error', {metadata: m});
    if (run.observe_waits) run.waitCoverage.remote = m.wait_coverage ?? null;
    throw new Error(m.error);
  }
  if (m.op === 'cancelled') {
    record('cancelled', m);
    if (run.observe_waits) run.waitCoverage.remote = m.wait_coverage ?? null;
    socket.close();
    return;
  }
  if (phase === 'stopping') return;
  if (m.op === 'started') {
    if (phase !== 'starting') throw new Error('会话启动响应顺序错误');
    run.session_id = m.session_id;
    if (run.recording) run.recording.sessionId = m.session_id;
    run.start_response = m;
    Object.assign(run.sourceBudget, sourceBudgetPolicy({source_budget: m.source_budget}, run.sourceBudget.selected));
    if (isNativeProtocol(run.proxy_protocol) && (m.source?.protocol !== run.proxy_protocol ||
        m.source.budget_frames !== run.sourceBudget.limit_frames ||
        m.source.instance_id !== run.proxy.prepared_source?.instance_id ||
        m.source.epoch !== run.proxy.prepared_source?.epoch ||
        m.source.frame_id !== run.proxy.prepared_source?.frame_id)) throw new Error('本机游戏来源绑定回执不匹配');
    if (run.mediaRequest) {
      if (m.media?.bitrate_bps !== run.mediaRequest.video.bitrate_bps) throw new Error('服务生效码率与本次目标不一致');
      const activeRun = run, activeSocket = socket, activeConnection = connection, {width, height} = config.encoder_recipe;
      run.media = m.media;
      run.observe_waits = m.wait_observability?.schema === WAIT_SCHEMA && m.wait_observability.enabled === true;
      run.waitCoverage.confirmed = run.observe_waits;
      run.decoder = new MediaDecoder(m.media, width, height, {
        observe_waits: run.observe_waits,
        frame: frame => { if (run === activeRun && socket === activeSocket) activeRun.playback.push(frame); else frame.video.close(); },
        decoded: (metadata, decode_ms, browser) => {
          if (run === activeRun && socket === activeSocket) record('frame_decode', {metadata, decode_ms,
            ...(browser ? {waits: {schema: WAIT_SCHEMA, browser}} : {})});
        },
        ack: output_frame => {
          if (run !== activeRun || socket !== activeSocket) throw new Error('旧会话不能确认新会话的帧');
          if (socket?.readyState !== WebSocket.OPEN) throw new Error('释放 H.264 帧时连接已关闭');
          activeConnection.send({op: 'frame_ack', output_frame});
          const enqueued = activeRun.observe_waits ? ns() : undefined;
          record('frame_ack', {output_frame});
          return enqueued;
        },
        error: error => { if (run === activeRun && socket === activeSocket) fail(error); }
      });
    } else if (m.media) throw new Error('服务意外返回 H.264 会话');
    run.profiling = m.profiling ?? {status: 'unknown', source: 'start'};
    $('profiling-status').textContent = profilingLabel(run.profiling);
    run.started_ns = ns();
    run.credit = true;
    if (run.observe_waits) run.creditGrantedNs = ns();
    latestInput = null;
    run.pacer = new CapturePacer(performance.now(), run.capture_policy);
    previousTick = performance.now();
    record('started', m);
    run.pacer.poll(previousTick, false);
    const activeRun = run, source = game;
    await captureFrame(previousTick);
    if (run !== activeRun || game !== source || phase !== 'starting') return;
    if (run.captured !== 1) throw new Error('准备帧采样未完成');
    const resumed = changeSource(source => source.setControlsSuspended(false));
    if (resumed) await resumed;
    if (run !== activeRun || game !== source || phase !== 'starting') return;
    setPhase('running');
    status('持续串流中。点击左侧画面操控；停止后保留模型，可再次开始。');
    $('proxy').focus();
    record('startup_start', startupSnapshot());
    pumpNativeGame();
    return;
  }
  if (run.mediaRequest) {
    if (m.session_id !== run.session_id || !['running', 'finishing', 'draining'].includes(phase) ||
        (run.completing && m.op !== 'admitted')) {
      throw new Error('H.264 响应会话或状态不匹配');
    }
    const now = performance.now(), {width, height} = config.encoder_recipe;
    if (m.op === 'admitted') {
      if (m.chunk_id !== run.admissionChunk || run.credit || m.status !== 'admitted') throw new Error('chunk 接收入队回执不匹配');
      run.admissionChunk = null;
      run.credit = !run.completing && phase !== 'draining';
      if (run.observe_waits) run.creditGrantedNs = ns();
      run.lastRpc = m.rpc_ms;
      run.charts.add('rpc', (now - run.started_ns / 1e6) / 1000, m.rpc_ms);
      record('chunk_admitted', {metadata: m, rpc_kind: 'admission'});
      sendChunk();
      pumpNativeGame();
    } else if (m.op === 'batch') {
      if (pixels.length || !['accepted', 'complete'].includes(m.status) ||
          !Number.isSafeInteger(m.chunk_id) || m.chunk_id < 0 || m.chunk_id >= run.chunkId) {
        throw new Error('H.264 批次状态、chunk 或 RGB 字节无效');
      }
      const frames = run.ledger.acceptMetadata(m, width, height);
      startupBatch(m);
      frames.forEach((frame, i) => run.mediaSources.set(m.output_frame_start + i,
        {...frame, output_id: m.output_id, batch_chunk_id: m.chunk_id}));
      record('chunk_receive', {metadata: m, payload_bytes: 0, encoded: true});
      timeline.add(m, now);
      run.charts.add('render', (now - run.started_ns / 1e6) / 1000, renderStageMs(m));
    } else if (m.op === 'video_frame') {
      const frame = run.mediaSources.get(m.output_frame);
      if (!frame || m.chunk_id !== frame.batch_chunk_id) throw new Error('H.264 帧缺少对应批次');
      run.decoder.decode(m, pixels, frame);
      run.mediaSources.delete(m.output_frame);
      run.encodedBytes += pixels.length;
      receiveFrame(frame, now, {metadata: m, encoded: true, payload_bytes: pixels.length});
    } else if (m.op === 'complete') {
      if (phase !== 'draining' || m.status !== 'complete' || m.total_frames !== run.received || m.total_frames !== run.ledger.emitted ||
          m.total_frames !== run.captured || run.ledger.pending.size || run.mediaSources.size) throw new Error('H.264 完成回执仍有未返回帧');
      const activeRun = run;
      run.completing = true;
      record('complete', m);
      if (run.observe_waits) run.waitCoverage.remote = m.wait_coverage ?? null;
      await run.decoder.flush();
      if (run !== activeRun || phase === 'stopping' || run.decoder.closed) return;
      run.outcome = 'complete';
      run.backend_complete = true;
      status('后端已完成，正在排空播放队列…');
    } else throw new Error('未知 H.264 服务响应');
    return;
  }
  if (m.op !== 'output' || !['running', 'finishing', 'draining'].includes(phase)) throw new Error('未知或过期的服务响应');
  if (m.session_id !== run.session_id || m.chunk_id !== run.chunkId - 1 || run.credit) throw new Error('响应会话或 chunk 不匹配');
  const now = performance.now(), {width, height} = config.encoder_recipe;
  const frames = run.ledger.accept(m, pixels.length, width, height);
  record('chunk_receive', {metadata: m, payload_bytes: pixels.length});
  timeline.add(m, now);
  run.lastRpc = m.rpc_ms;
  const elapsed = (now - run.started_ns / 1e6) / 1000;
  run.charts.add('rpc', elapsed, m.rpc_ms);
  run.charts.add('render', elapsed, renderStageMs(m));
  const bytesPerFrame = width * height * 3;
  const playbackBound = Math.max(1, Math.min(48, Math.floor(64 * 1024 * 1024 / bytesPerFrame)));
  frames.forEach((frame, index) => {
    receiveFrame(frame, now);
    if (run.playback.length >= playbackBound) {
      const dropped = run.playback.shift();
      run.recording?.forget(dropped.id);
      run.dropped++;
      record('display_drop', {source_frame: dropped.id});
    }
    run.playback.push({...frame, pixels: pixels.slice(index * bytesPerFrame, (index + 1) * bytesPerFrame)});
  });
  if (m.status === 'complete') {
    run.outcome = 'complete';
    run.backend_complete = true;
    status('后端已完成，正在排空播放队列…');
  } else {
    run.credit = true;
    sendChunk();
  }
}

$('start').addEventListener('click', async () => {
  let startingRun;
  try {
    if (activeStates.has(phase)) return;
    if (!sourceReady || sourceGate || !game || !reference || referenceScene !== sceneId) {
      status('请先准备当前场景的首帧参考图', true);
      return;
    }
    if (config.mode !== 'fixture' && referenceSource !== 'upload'
        && !preparedNativeReference() && !(isRgbScene() && referenceSource === `${sceneId}-rgb`)) {
      status('当前场景的模型模式需要上传匹配的首帧图片', true);
      return;
    }
    const seed = Number($('seed').value);
    if (!$('seed').value.trim() || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) throw new Error('种子必须在 0 到 2147483647 之间');
    if (config.media?.proxy_protocol !== undefined && config.media.proxy_protocol !== 'proxy_frames/v1') {
      throw new Error('不支持服务 proxy 上传协议');
    }
    if ($('record-session').checked && isNativeScene()) throw new Error('本机游戏来源暂不支持精确同步录像');
    if ($('record-session').checked && latestRecording?.done && (!latestRecording.result && !latestRecording.error || latestRecording.result && recordingDownloads.size < 2)) {
      status('请先等待录像完成并分别下载两个视频，再开始新的录像', true); return;
    }
    stopStartup('new_run');
    timeline.reset();
    inputId = 0; latestInput = null;
    run?.decoder?.close();
    socket?.close();
    connection = null;
    run = {schema: 'playable_web_run/v1', config, created_at: new Date().toISOString(), outcome: 'running',
      proxy: Object.freeze({scene: sceneId, style: sourceScenes[sceneId].style, reference_source: referenceSource}),
      transport_mode: config.media ? 'h264_annexb_v1' : 'legacy_rgb24',
      proxy_protocol: isNativeScene() ? sourceScenes[sceneId].protocol : config.media?.proxy_protocol ?? 'whole_chunk', upload: null,
      capturePending: false, captureCommandBytes: 0,
      bitrate_target_bps: config.media ? Number($('bitrate').value) * 1e6 : null,
      encodedBytes: 0, mediaSources: new Map(),
      observe_waits: false,
      waitCoverage: {requested: !!config.media && $('observe-waits').checked === true, confirmed: false,
        malformed: 0, unsupported: 0, stale: 0, receipts: 0, absent_receipts: 0,
        intervals_omitted: 0, browser_discarded: 0, remote: null},
      capture_policy: $('capture-policy').value, pacer: new CapturePacer(performance.now(), $('capture-policy').value),
      sourceBudget: {...sourceBudgetPolicy(config, $('source-budget').value), peak_lead_frames: 0,
        wait_count: 0, wait_ms: 0, wait_started_ms: null},
      playback_mode: config.media ? $('playback-mode').value : 'fixed',
      displayPacer: new DisplayPacer(config.media ? $('playback-mode').value : 'fixed'),
      profiling: {status: config.mode === 'fixture' ? 'fixture' : 'unknown', source: config.mode === 'fixture' ? 'fixture' : null},
      started_ns: ns(), events: [], events_omitted: 0, capture: [], playback: [], ledger: new FrameLedger(),
      socketClosure: {first: null, media: null, control: null}, closeIntent: null,
      rates: {capture: [], receive: [], display: []}, latestDraw: null, credit: false,
      charts: new TimingHistory(), chartPainted: null,
      startup: {mode: config.media ? $('startup-mode').value : 'baseline',
        state: config.media && $('startup-mode').value === 'startup_pause' ? 'collecting' : 'resumed',
        committed_frames: 0, chunk_id: null, accepted_frames: null, drain_target: null,
        paused_ms: 0, pause_started_ms: null, ignored_inputs: 0, transitions: [], transitions_omitted: 0},
      captured: 0, received: 0, displayed: 0, dropped: 0, missed: 0, chunkId: 0, lastInputDrawn: 0};
    startingRun = run;
    $('profiling-status').textContent = profilingLabel(run.profiling);
    clearOutput();
    setPhase('starting');
    updateMetrics(performance.now());
    const activeRun = run;
    if ($('record-session').checked) {
      const spec = config.encoder_recipe;
      const recorder = await PairedRecorder.create({proxyWidth:spec.proxy_width, proxyHeight:spec.proxy_height,
        width:spec.width, height:spec.height, onStatus: message => {
          if (run === activeRun || latestRecording === activeRun.recording) $('record-status').textContent = message;
          if (latestRecording === activeRun.recording) recordingControls();
        }});
      if (run !== activeRun || phase !== 'starting') { recorder.finish('cancelled'); return; }
      activeRun.recording = latestRecording = recorder; recordingDownloads = new Set(); recordingControls();
      $('record-status').textContent = '正在录制；两段视频使用相同的实际显示时间。停止并排空后可下载。';
    }
    status('正在重置并冻结 Proxy 首帧…');
    const prepared = changeSource(source => source.prepareRun());
    if (prepared) await prepared;
    if (run !== activeRun || phase !== 'starting') return;
    if (isNativeScene()) run.proxy = Object.freeze({...run.proxy, prepared_source: game.getSimulationState().source});
    if (preparedNativeReference() || isRgbScene() && referenceSource === `${sceneId}-rgb`) referenceFromProxy();
    if (isRgbScene()) run.proxy = Object.freeze({...run.proxy, source:game.getSourceInfo()});
    let sourceBinding = isNativeScene() ? {...game.takeBinding(),
      reference: preparedNativeReference() ? 'prepared' : 'upload', budget: run.sourceBudget.effective} : null;
    status(config.media ? '正在检查浏览器 H.264 解码能力…' : '正在建立模型会话…');
    run.mediaRequest = await prepareMedia(config.media, $('bitrate').value, config.encoder_recipe.width, config.encoder_recipe.height);
    if (run !== activeRun || phase !== 'starting') return;
    status('正在建立模型会话…');
    const streamUrl = new URL('stream', location.href);
    streamUrl.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const split = config.media?.control_protocol === 'media_control/v1';
    if (config.media?.control_protocol && !split) throw new Error('不支持服务控制协议');
    const ws = new WebSocket(streamUrl.href);
    socket = ws; ws.binaryType = 'arraybuffer';
    run.control_protocol = split ? 'media_control/v1' : 'in_band';
    let control = null, bound = false, bindTimer = null;
    const current = () => run === activeRun && socket === ws;
    const observeSocket = (channel, kind, event) => {
      const observation = {channel, kind, browser_ns: ns(), phase, outcome: run.outcome,
        local_intent: run.closeIntent, code: kind === 'close' ? event?.code ?? null : null,
        reason: kind === 'close' ? String(event?.reason ?? '').slice(0, 120) : null,
        wasClean: kind === 'close' ? event?.wasClean ?? null : null};
      run.socketClosure.first ??= observation;
      if (kind === 'close') run.socketClosure[channel] ??= observation;
    };
    const close = () => {
      clearTimeout(bindTimer);
      ws.close();
      control?.close();
    };
    const send = message => {
      const target = split ? control : ws;
      if (!current() || split && !bound || target?.readyState !== WebSocket.OPEN) throw new Error('控制连接已关闭');
      target.send(JSON.stringify(message));
    };
    connection = {send, close, cancel: () => {
      if (split && !bound || ws.readyState !== WebSocket.OPEN) close();
      else send({op: 'cancel'});
    }};
    const start = () => {
      if (!current() || phase !== 'starting') { close(); return; }
      ws.send(pack({op: 'start', seed, condition_seed: seed,
        ...(sourceBinding ? {source: sourceBinding} : {}),
        ...(run.mediaRequest ? {video: run.mediaRequest.video, observe_waits: run.waitCoverage.requested} : {})}, reference));
      sourceBinding = null;
    };
    ws.onopen = () => {
      if (!current() || phase !== 'starting') { close(); return; }
      if (!split) start();
      else bindTimer = setTimeout(() => {
        if (current() && !bound) { fail(new Error('控制连接绑定超时')); close(); }
      }, 5000);
    };
    ws.onmessage = async event => {
      if (!current()) return;
      try {
        const {metadata, pixels} = typeof event.data === 'string'
          ? {metadata: JSON.parse(event.data), pixels: new Uint8Array()} : unpack(event.data);
        if (metadata.op === 'control_offer') {
          if (phase !== 'starting') return;
          if (!split || control || metadata.protocol !== 'media_control/v1' ||
              typeof metadata.capability !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(metadata.capability)) {
            throw new Error('控制连接协商失败');
          }
          let capability = metadata.capability;
          const controlUrl = new URL('control', location.href);
          controlUrl.protocol = streamUrl.protocol;
          const channel = new WebSocket(controlUrl.href);
          control = channel;
          const currentControl = () => current() && control === channel;
          channel.onopen = () => {
            if (!currentControl() || phase !== 'starting') { capability = null; close(); return; }
            channel.send(JSON.stringify({op: 'bind', capability}));
            capability = null;
          };
          channel.onmessage = async message => {
            if (!currentControl()) return;
            try {
              const reply = JSON.parse(message.data);
              if (reply.op === 'bound' && !bound && reply.protocol === 'media_control/v1') {
                bound = true;
                clearTimeout(bindTimer);
                if (phase === 'starting') start();
              } else if (bound && ['cancelled', 'error'].includes(reply.op)) {
                try { await receive(reply, new Uint8Array()); } finally { close(); }
              } else throw new Error('控制连接协议失败');
            } catch {
              if (currentControl()) { fail(new Error('控制连接协议失败')); close(); }
            }
          };
          channel.onerror = () => {
            if (currentControl()) { observeSocket('control', 'error'); fail(new Error('控制连接失败')); close(); }
          };
          channel.onclose = event => {
            if (run === activeRun && control === channel) observeSocket('control', 'close', event);
            capability = null;
            if (currentControl()) close();
          };
          return;
        }
        if (split && !bound) throw new Error('控制连接尚未绑定');
        await receive(metadata, pixels);
      } catch (error) { if (current() && phase !== 'stopping') fail(error); }
    };
    ws.onerror = () => { if (current()) { observeSocket('media', 'error'); fail(new Error('WebSocket 连接失败，请检查服务日志')); } };
    ws.onclose = event => {
      if (!current()) return;
      observeSocket('media', 'close', event);
      socket = null;
      connection = null;
      clearTimeout(bindTimer);
      control?.close();
      const finished = split ? run.backend_complete && !run.playback.length : run.backend_complete || run.completing;
      const pending = pendingInputWaits('cancelled');
      run.capture.length = 0;
      run.upload = null;
      if (!finished) {
        clearOutput();
        if (run.outcome === 'running' || split && run.outcome === 'complete') run.outcome = 'disconnected';
      }
      else if (run.backend_complete && !run.playback.length) run.decoder?.close();
      record('socket_closed', {outcome: run.outcome, ...pending});
      stopSource();
      setPhase(finished && (run.playback.length || !run.backend_complete) ? 'draining' : 'idle');
      status(finished ? '后端已完成，播放队列排空后可重新开始。' : '会话已关闭。可下载记录；后端清理完毕后可以重试。', !finished);
    };
  } catch (error) { if (!startingRun || run === startingRun) fail(error); }
});

$('finish').addEventListener('click', () => {
  if (phase !== 'running') return;
  run.closeIntent = 'normal_eos';
  if (run.observe_waits) run.finishRequestedNs = ns();
  if (startupFrozen() || isNativeScene() && game.getSimulationState().state === 'held') changeSource(source => source.setControlsSuspended(false));
  setPhase('finishing');
  record('finish_requested');
  status('继续采集到协议边界，以 5 个真实源帧结束…');
  sendChunk();
  pumpNativeGame();
});
$('cancel').addEventListener('click', () => cancel('用户取消'));

for (const [button, index, name] of [['download-proxy-video', 0, 'proxy'], ['download-rgb-video', 1, 'rgb']]) {
  $(button).addEventListener('click', () => {
    if (!latestRecording?.result) return;
    const url = URL.createObjectURL(latestRecording.result[index]);
    const anchor = document.createElement('a'); anchor.href = url;
    anchor.download = `h3-${name}-${latestRecording.sessionId || 'recording'}.webm`; anchor.click();
    recordingDownloads.add(index); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
$('download-video-timing').addEventListener('click', () => {
  if (!latestRecording?.result) return;
  const metadata = {...latestRecording.metadata(), session_id:latestRecording.sessionId ?? null};
  const url = URL.createObjectURL(new Blob([JSON.stringify(metadata, null, 2)], {type:'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url;
  anchor.download = `h3-video-timing-${latestRecording.sessionId || 'recording'}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('download').addEventListener('click', () => {
  const latency = latencySnapshot(performance.now());
  const receipt = {schema: run.schema, config: run.config, session_id: run.session_id,
    socket_closure: run.socketClosure,
    startup: {schema: 's06_startup_catchup/v1', snapshot: startupSnapshot(),
      transitions: run.startup.transitions, transitions_omitted: run.startup.transitions_omitted},
    playback: {mode: run.playback_mode, effective_target_fps: run.displayPacer.targetFps,
      target_range_fps: run.playback_mode === 'adaptive' ? [8, 24] : [24, 24],
      actual_unique_draw_fps_5s: run.rates.display.filter(time => time >= (run.endedAt ?? performance.now()) - 5000).length /
        Math.max(.001, Math.min(5, ((run.endedAt ?? performance.now()) - run.started_ns / 1e6) / 1000)),
      actual_fps_population: 'unique canvas draw submissions in final five session seconds; not target FPS or model capacity'},
    proxy: run.proxy,
    recording: run.recording?.metadata() ?? null,
    transport_mode: run.transport_mode,
    control_protocol: run.control_protocol,
    proxy_protocol: run.proxy_protocol,
    source_transport: isNativeProtocol(run.proxy_protocol) ? {protocol: run.proxy_protocol,
      scheduling: 'native-publication-media-slots', proxy_upload_bytes: 0, acknowledged_capture_command_bytes: run.captureCommandBytes,
      command_bytes_population: 'capture commands with active-run replies; unanswered or cancelled replies excluded',
      latency_semantics: 'sample-request-to-receive/draw; not native publication age'} : null,
    media_target: run.bitrate_target_bps === null ? null : {bitrate_bps: run.bitrate_target_bps},
    media_effective: run.media ?? null,
    media_received: {encoded_bytes: run.encodedBytes,
      actual_bitrate_bps: run.media && run.received ? run.encodedBytes * 8 * run.media.fps / run.received : null,
      bitrate_population: 'received access-unit bytes / received frame media duration'},
    profiling: run.profiling,
    wait_observability: {schema: WAIT_SCHEMA, enabled: run.observe_waits, clocks: WAIT_CLOCKS,
      gateway_request_boundary: {started: 'complete message after unpack',
        chunk: run.proxy_protocol === 'proxy_frames/v1' ? 'proxy_commit dispatch before unpack; prior frame bytes already staged' : 'complete message dispatch before unpack'},
      ...run.waitCoverage, timeline: timeline.waitCoverage(),
      browser_complete: run.outcome === 'complete' && !!run.backend_complete && !run.waitCoverage.browser_discarded &&
        !run.playback.length && !run.decoder?.outstanding.size,
      remote_complete: run.waitCoverage.remote?.frontend_complete === true && run.waitCoverage.remote?.gateway_complete === true,
      remote_omissions: run.waitCoverage.remote ?? 'unknown', uncollected: WAIT_UNCOLLECTED},
    capture_policy: run.capture_policy,
    source_budget: sourceBudgetSnapshot(run.endedAt ?? performance.now()),
    created_at: run.created_at, start_response: run.start_response, outcome: run.outcome,
    counts: {captured: run.captured, received: run.received, displayed: run.displayed,
      display_dropped: run.dropped, missed_capture_slots: run.missed, pending_source_frames: run.ledger.pending.size,
      capture_backpressure_count: run.pacer.waitCount, capture_backpressure_ms: run.pacer.waitingMs(run.endedAt ?? performance.now())},
    timing: {clock: 'browser performance.now nanoseconds', display: 'canvas draw submission, not physical presentation',
      percentile_population: 'receive/draw events within the inclusive last 10 session seconds by event time, not capture time; at most 4096 samples per series',
      percentile_window: 'current browser wall time while active; frozen chartEnd after completion or closure',
      visible_source_age: isNativeProtocol(run.proxy_protocol) ? 'sample request age only; native publication age unknown'
        : 'snapshot time minus capture time of the source image still on canvas; continues increasing after completion; null after canvas clear',
      latest_draw_latency: isNativeProtocol(run.proxy_protocol) ? 'browser sample-request to canvas draw submission'
        : 'last visible source capture to canvas draw submission; unchanged while that image remains; null after canvas clear',
      latency_snapshot: latency, fps_window_seconds: 5, fps_population: 'independent source capture, received output and submitted draw event counts; not model capacity',
      chart_window_seconds: CHART_WINDOW_SECONDS, chart_refresh_hz: CHART_REFRESH_HZ,
      chart_max_samples_per_series: CHART_MAX_SAMPLES,
      rpc_series: run.mediaRequest ? 'input admission HTTP only, excludes model completion' : 'full RGB chunk HTTP response',
      render_series: 'orchestration render.wall_ms; native worker completion or Ray frontend dispatch interval'},
    events_omitted: run.events_omitted, events: run.events};
  const url = URL.createObjectURL(new Blob([JSON.stringify(receipt, null, 2)], {type: 'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `h3-web-${run.session_id || 'failed'}.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function latencySnapshot(now) {
  const end = run.chartEnd ?? Math.max(0, (now - run.started_ns / 1e6) / 1000);
  const start = Math.max(0, end - CHART_WINDOW_SECONDS);
  run.charts.prune(end);
  const population = name => {
    const values = run.charts.series[name].filter(row => row.time >= start && row.time <= end && row.value !== null).map(row => row.value);
    return {samples: values.length, p50_ms: percentile(values, .5), p95_ms: percentile(values, .95)};
  };
  return {clock: 'browser performance.now', snapshot_ns: Math.round(now * 1e6),
    latest_draw: run.latestDraw ? {...run.latestDraw, capture_to_draw_ms: (run.latestDraw.draw_ns - run.latestDraw.capture_ns) / 1e6} : null,
    visible_source: run.latestDraw ? {source_frame: run.latestDraw.source_frame, capture_ns: run.latestDraw.capture_ns,
      age_ms: now - run.latestDraw.capture_ns / 1e6} : null,
    rolling: {clock: 'session seconds since started_ns', started_ns: run.started_ns,
      start_seconds: start, end_seconds: end, frozen: run.chartEnd !== undefined,
      receive: population('receive'), draw: population('draw')}};
}

function updateMetrics(now) {
  if (!run) return;
  const startup = startupSnapshot(now);
  const startupState = {collecting: '采集完整包', waiting_batch: '等待本批完成',
    draining_produced: '绘制已产生画面', resumed: '已恢复'}[startup.state];
  $('startup-status').textContent = `${activeStates.has(phase) ? '' : '上次会话 · '}${startup.mode === 'startup_pause' ? '启动追齐' : '连续启动'} · ${startupState} · 暂停 ${startup.paused_ms.toFixed(0)} ms · 忽略输入 ${startup.ignored_inputs}`;
  $('playback-target').textContent = `${run.displayPacer.targetFps.toFixed(1)} FPS · ${run.playback_mode === 'adaptive' ? '自适应目标' : '固定目标'}`;
  const rateEnd = run.endedAt ?? now;
  const elapsed = Math.max(.001, Math.min(5, (rateEnd - run.started_ns / 1e6) / 1000));
  const fps = ['capture', 'receive', 'display'].map(kind => {
    const times = run.rates[kind];
    while (times.length && times[0] < rateEnd - 5000) times.shift();
    return (times.length / elapsed).toFixed(1);
  });
  const latency = latencySnapshot(now);
  const age = population => population.samples ? `${population.p50_ms.toFixed(0)} / ${population.p95_ms.toFixed(0)} ms` : '—';
  $('fps').textContent = fps.join(' / ');
  $('receive-age').textContent = age(latency.rolling.receive);
  $('display-age').textContent = age(latency.rolling.draw);
  $('latency-window').textContent = `${latency.rolling.frozen ? '已结束 · 保留最后 10 秒' : '最近 10 秒 · 按接收 / 绘制事件时间'} · ${latency.rolling.start_seconds.toFixed(1)}–${latency.rolling.end_seconds.toFixed(1)} s · 接收 ${latency.rolling.receive.samples} / 绘制 ${latency.rolling.draw.samples} 样本`;
  $('visible-age').textContent = latency.visible_source ? `${latency.visible_source.age_ms.toFixed(0)} ms · 源帧 ${latency.visible_source.source_frame}` : '—';
  $('latest-draw-age').textContent = latency.latest_draw ? `${latency.latest_draw.capture_to_draw_ms.toFixed(0)} ms · 源帧 ${latency.latest_draw.source_frame}` : '—';
  $('rpc').textContent = Number.isFinite(run.lastRpc) ? `${run.lastRpc.toFixed(1)} ms` : '—';
  $('queues').textContent = `${run.capture.length} / ${run.decoder?.outstanding.size ?? run.playback.length}`;
  const budget = sourceBudgetSnapshot(run.endedAt ?? now);
  $('source-lead').textContent = `${budget.lead_frames} / ${budget.limit_frames ?? '不限'} · 峰值 ${budget.peak_lead_frames}`;
  $('source-budget-wait').textContent = `${budget.wait_ms.toFixed(0)} ms / ${budget.wait_count} 次`;
  $('source-budget-status').textContent = budget.limit_frames !== null ? `${budget.limit_frames} 帧预算生效`
    : `预算关闭 · ${{selected_off: '已选择原采集基线', requires_h264: '当前传输不支持',
      unsupported_recipe_or_decoder: '当前配方或解码器不支持'}[budget.reason]}`;
  $('media-bytes').textContent = run.transport_mode === 'h264_annexb_v1' ? `${(run.encodedBytes / 1024).toFixed(1)} KiB` : 'RGB24';
  $('media-rate').textContent = run.media && run.received
    ? `${(run.encodedBytes * 8 * run.media.fps / run.received / 1e6).toFixed(2)} Mbps 实际 / ${(run.media.bitrate_bps / 1e6).toFixed(2)} Mbps 目标`
    : (run.transport_mode === 'legacy_rgb24' ? '旧版未编码传输' : '等待编码帧');
  $('drops').textContent = `${run.dropped} / ${run.missed}`;
  $('capture-wait').textContent = `${run.pacer.waitingMs(run.endedAt ?? now).toFixed(0)} ms / ${run.pacer.waitCount} 次`;
  const chartTime = latency.rolling.end_seconds;
  if ($('diagnostics').open && run.chartPainted !== chartTime) {
    paintChart($('processing-chart'), $('processing-summary'), run.charts, ['rpc', 'render'], chartTime);
    paintChart($('latency-chart'), $('latency-summary'), run.charts, ['receive', 'draw'], chartTime);
    $('chart-state').textContent = `${run.chartEnd === undefined ? '实时滚动' : '已结束 · 保留最后一屏'} · ${Math.max(0, chartTime - CHART_WINDOW_SECONDS).toFixed(1)}–${chartTime.toFixed(1)} s`;
    run.chartPainted = chartTime;
  }
}

function captureFrame(now) {
  if (isNativeProtocol(run.proxy_protocol)) {
    if (run.capturePending) return;
    const activeRun = run, source = game, capture_ns = ns(), id = run.captured, input = latestInput;
    run.capturePending = true;
    return source.captureNative(id, capture_ns, run.nativeGameSegment ?? 0).then(sample => {
      if (run !== activeRun || game !== source || !['starting', 'running', 'finishing'].includes(phase)) return;
      run.captureCommandBytes += sample?.command_bytes ?? 0;
      if (!sample || sample.blocked) {
        activeRun.nativeGameBlocked = true; record('source_sample_blocked'); return;
      }
      if (sample.waiting) { record('source_sample_waiting'); return; }
      const frame = {id, capture_ns, input, sample_receipt_ns: sample.browser_receipt_ns ?? ns()};
      run.captured++;
      run.capture.push(frame);
      run.sourceBudget.peak_lead_frames = Math.max(run.sourceBudget.peak_lead_frames, run.captured - run.displayed);
      observe('capture', performance.now());
      record('frame_capture', {source_frame: id, capture_ns, input_id: input?.id ?? null,
        source: sample.source, sample: {...sample, source: undefined},
        sample_receipt_ns: frame.sample_receipt_ns, proxy_upload_bytes: 0,
        capture_semantics: 'browser-sample-request'});
    }).finally(() => {
      activeRun.capturePending = false;
    }).then(() => {
      if (run === activeRun && game === source && ['running', 'finishing'].includes(phase)) {
        sendChunk();
        if (!activeRun.nativeGameBlocked) pumpNativeGame();
      }
    }).catch(error => {
      if (run === activeRun && game === source && ['starting', 'running', 'finishing'].includes(phase)) throw error;
    });
  }
  const capture_ns = ns(), pixels = game.capture();
  if (!pixels) { record('source_sample_missing'); return; }
  const frame = {id: run.captured++, capture_ns, input: latestInput, pixels};
  run.recording?.capture(frame.id, pixels, capture_ns);
  run.sourceBudget.peak_lead_frames = Math.max(run.sourceBudget.peak_lead_frames, run.captured - run.displayed);
  if (run.observe_waits) frame.pixels_ready_ns = ns();
  run.capture.push(frame);
  observe('capture', now);
  record('frame_capture', {source_frame: frame.id, capture_ns: frame.capture_ns, input_id: frame.input?.id ?? null,
    lead_frames: run.captured - run.displayed,
    ...(game.getCaptureEvidence ? {source: game.getCaptureEvidence()} : {}),
    ...(run.observe_waits ? {pixels_ready_ns: frame.pixels_ready_ns} : {})});
}

function pumpNativeGame() {
  if (!isNativeProtocol(run?.proxy_protocol) || run.capturePending ||
      !['running', 'finishing'].includes(phase) || startupFrozen() || sourceGate) return;
  sendChunk();
  if (run.admissionChunk != null || !['running', 'finishing'].includes(phase) || startupFrozen() || sourceGate) return;
  if (phase === 'finishing' && [5, 17, 22, 34].includes(run.capture.length)) return;
  const now = performance.now();
  const budgetBlocked = sourceBudgetBlocked(run.captured, run.displayed, run.sourceBudget.limit_frames);
  sourceBudgetWait(budgetBlocked, now);
  const limit = run.startup.state === 'collecting' ? 17 : (config.max_capture_frames || 34);
  if (game.getSimulationState().state === 'held' && run.captured > 0 || budgetBlocked || run.capture.length >= limit ||
      socket.bufferedAmount > (config.max_message_bytes || 128 * 1024 * 1024)) {
    run.nativeGameBlocked = true;
    return;
  }
  if (run.nativeGameBlocked) { run.nativeGameSegment = (run.nativeGameSegment ?? 0) + 1; run.nativeGameBlocked = false; }
  captureFrame(now)?.catch(fail);
}

function tick(now) {
  requestAnimationFrame(tick);
  const delta = Math.min((now - previousTick) / 1000, .1);
  previousTick = now;
  if (!game) return;
  try {
    if (!startupFrozen()) game.render(delta);
    if (isNativeProtocol(run?.proxy_protocol)) pumpNativeGame();
    else if (['running', 'finishing'].includes(phase) && !startupFrozen() && !sourceGate) {
      const needTail = phase !== 'finishing' || ![5, 17, 22, 34].includes(run.capture.length);
      if (needTail) {
        const captureLimit = run.startup.state === 'collecting' ? 17 : (config.max_capture_frames || 34);
        const blocked = run.capturePending || run.capture.length >= captureLimit
          || socket.bufferedAmount > (config.max_message_bytes || 128 * 1024 * 1024);
        const budgetBlocked = sourceBudgetBlocked(run.captured + Number(run.capturePending), run.displayed, run.sourceBudget.limit_frames);
        sourceBudgetWait(!blocked && budgetBlocked && now >= run.pacer.next, now);
        const missed = budgetBlocked && !blocked ? null : run.pacer.poll(now, blocked);
        if (missed !== null) {
          run.missed += missed;
          const pending = captureFrame(now);
          if (pending?.catch) pending.catch(fail);
        }
      } else sourceBudgetWait(false, now);
      sendChunk();
    }
    if (run && now >= nextDraw && run.playback_mode === 'adaptive' && run.displayed > 0 &&
        run.startup.state === 'resumed' && ['running', 'finishing'].includes(phase)) {
      const previousTarget = run.displayPacer.targetFps;
      run.displayPacer.sample(now, run.playback.length);
      if (run.displayPacer.targetFps !== previousTarget) record('playback_target',
        {target_fps: run.displayPacer.targetFps, decoded_buffer_frames: run.playback.length});
    }
    if (run?.playback.length && now >= nextDraw && phase !== 'stopping') {
      const frame = run.playback.shift(), {width, height} = config.encoder_recipe;
      const drawStarted = run.observe_waits ? ns() : null;
      let drawEnded, ackEnqueued;
      if (frame.video) {
        try { outputContext.drawImage(frame.video, 0, 0, width, height); }
        finally {
          if (run.observe_waits) drawEnded = ns();
          ackEnqueued = run.decoder.release(frame.output_frame);
        }
      } else outputContext.putImageData(new ImageData(rgbToRgba(frame.pixels), width, height), 0, 0);
      const draw = performance.now();
      run.recording?.draw(frame.id, $('output'), draw);
      run.displayed++;
      observe('display', draw);
      run.latestDraw = {source_frame: frame.id, capture_ns: frame.capture_ns, draw_ns: Math.round(draw * 1e6)};
      run.charts.add('draw', (draw - run.started_ns / 1e6) / 1000, draw - frame.capture_ns / 1e6);
      record('frame_draw', {source_frame: frame.id, capture_ns: frame.capture_ns, input_id: frame.input?.id ?? null,
        lead_frames: run.captured - run.displayed,
        ...(frame.video ? {output_frame: frame.output_frame, output_id: frame.output_id, chunk_id: frame.batch_chunk_id, encoded: true} : {}),
        ...(run.observe_waits ? {waits: {schema: WAIT_SCHEMA, browser: {decoded_ns: frame.decoded_ns,
          draw_start_ns: drawStarted, draw_end_ns: drawEnded, ack_enqueue_ns: ackEnqueued}}} : {})});
      if (frame.input && frame.input.id > run.lastInputDrawn) {
        run.lastInputDrawn = frame.input.id;
        record('input_first_draw', {input_id: frame.input.id, emitted_ns: frame.input.emitted_ns,
          source_frame: frame.id, input_to_draw_ms: draw - frame.input.emitted_ns / 1e6});
      }
      startupDrawn();
      pumpNativeGame();
      $('empty').hidden = true;
      $('output-info').textContent = `已接收 ${run.received} 帧 · 已绘制 ${run.displayed} 帧 · 当前源帧 ${frame.id} · 显示丢帧 ${run.dropped}`;
      const drawInterval = run.playback_mode === 'adaptive' ? run.displayPacer.interval() : interval;
      nextDraw += (Math.max(0, Math.floor((now - nextDraw) / drawInterval)) + 1) * drawInterval;
    }
    if (run?.backend_complete && !run.playback.length && phase === 'draining') {
      run.decoder?.close();
      if (socket) { if (run.control_protocol !== 'media_control/v1') socket.close(); }
      else { setPhase('idle'); status('本次串流已完成，可下载记录或重新开始。'); }
    }
    if (now - lastMetrics >= chartRefreshInterval) {
      updateMetrics(performance.now());
      if ($('diagnostics').open) timeline.paint(now, activeStates.has(phase));
      lastMetrics = now - (now - lastMetrics) % chartRefreshInterval;
    }
  } catch (error) { fail(error); }
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden && activeStates.has(phase)) cancel('页面进入后台，采集计时已失效');
});
window.addEventListener('pagehide', () => { if (run) run.closeIntent = 'pagehide'; stopStartup('pagehide'); run?.decoder?.close(); connection?.close(); socket?.close(); game?.dispose(); disposeCharts(); timeline.dispose(); });
window.addEventListener('pageshow', event => { if (event.persisted) location.reload(); });

$('diagnostics').addEventListener('toggle', () => {
  if (!$('diagnostics').open) return;
  const now = performance.now();
  if (run) updateMetrics(now);
  else {
    const emptyHistory = new TimingHistory();
    paintChart($('processing-chart'), $('processing-summary'), emptyHistory, ['rpc', 'render'], 0);
    paintChart($('latency-chart'), $('latency-summary'), emptyHistory, ['receive', 'draw'], 0);
  }
  timeline.paint(now, activeStates.has(phase));
});

async function initialize() {
  const response = await fetch('./config');
  if (!response.ok) throw new Error(`服务配置读取失败 HTTP ${response.status}`);
  config = await response.json();
  const spec = config.encoder_recipe;
  for (const key of ['width', 'height', 'proxy_width', 'proxy_height']) {
    if (!Number.isInteger(spec[key]) || spec[key] < 1 || spec[key] > 4096) throw new Error('artifact 图像尺寸无效');
  }
  referenceRevision++;
  const fixture = config.mode === 'fixture';
  if (!config.media) { $('startup-mode').value = 'baseline'; $('playback-mode').value = 'fixed'; }
  if (!config.source_budget?.eligible) $('source-budget').value = 'off';
  if (reference && (referenceScene !== sceneId || (!fixture && referenceSource === 'fixture-proxy' && !isNativeScene()))) {
    referenceRevision++;
    reference = null;
    referenceScene = referenceSource = null;
  }
  const codec = config.codec;
  const decoderLabel = codec?.decoder ? ` · ${codec.decoder} decoder · ${codec.backend}`
    + (codec.compile ? ' + compile' : '') + (codec.cuda_graph ? ' + CUDA Graph' : '') : '';
  $('mode').textContent = fixture ? '测试回显 · 无模型推理' : `真实模型 serving${decoderLabel}`;
  $('mode').className = `badge ${fixture ? 'fixture' : 'model'}`;
  $('profiling-status').textContent = profilingLabel(config.profiling);
  $('configuration').textContent = JSON.stringify(config, null, 2);
  $('source-geometry').textContent = `${spec.proxy_width} × ${spec.proxy_height} · 24 FPS 目标`;
  $('output-geometry').textContent = `${spec.width} × ${spec.height} · ${config.media ? 'H.264 / WebCodecs' : '旧版 RGB24'}`;
  $('transport-info').textContent = config.media ? 'H.264 低延迟编码；浏览器解码后按本次会话的播放目标绘制，媒体 PTS 保持 24 FPS。'
    : '后端未声明媒体能力，使用旧版 RGB24 传输。';
  $('rpc-label').textContent = config.media ? '最近一次 chunk 入队 RPC' : '最近一次 chunk RPC';
  $('rpc-info').textContent = config.media ? '只计输入接收入队，不含模型完成或完整 RGB 回传' : '网关 HTTP 往返，不能当作 DiT 耗时';
  $('processing-info').textContent = config.media
    ? 'RPC 为输入接收入队 HTTP 往返，在 admitted 回执到达时标点。模型 render 在批次到达时标点；原生服务计 DiT 完成区间，Ray 包含调度与 worker 等待，均非纯 kernel 耗时。'
    : '按响应到达时间标点。RPC 包含 HTTP 往返；模型 render 为服务端阶段墙钟，非纯 kernel 耗时。未提供该字段时留空。';
  $('drops-info').textContent = config.media ? '编码帧窗口最多 4 帧，解码与播放共享背压，不淘汰参考帧' : '显示淘汰旧帧以限制积压';
  if (config.media) {
    $('bitrate').value = config.media.bitrate_default_bps / 1e6;
    $('bitrate').min = config.media.bitrate_min_bps / 1e6;
    $('bitrate').max = config.media.bitrate_max_bps / 1e6;
  }
  $('output').width = spec.width; $('output').height = spec.height;
  $('output').style.aspectRatio = `${spec.width}/${spec.height}`;
  const streamingText = config.recipe.protocol.includes('streaming_text');
  $('prompt').disabled = !streamingText;
  $('prompt-info').textContent = streamingText ? '完整替换描述，在下次 chunk 提交时生效。' : '当前 artifact 固定文本；键盘输入通过 proxy 画面传入。';
  $('prompt').value = spec.system_prompt ?? '';
  if (streamingText) {
    currentPrompt = {revision: 1, text: spec.system_prompt, emitted_ns: ns()};
  }
  const requestedScene = new URL(location.href).searchParams.get('scene') || $('scene').value;
  $('scene').replaceChildren(...Object.entries(sourceScenes).map(([id, scene]) => {
    const option = new Option(scene.label, id);
    option.id = `${id}-option`;
    option.disabled = option.hidden = !sceneAvailable(id, config);
    return option;
  }));
  sceneId = sceneAvailable(requestedScene, config) ? requestedScene : 'boxing';
  $('scene').value = sceneId;
  mountGame();
  setPhase('idle');
  status(fixture ? '测试场景和首帧已就绪，可直接开始串流或先游玩。' : '场景已就绪。上传参考图后开始串流。');
  if (autoSeedReference() && !reference) {
    if (await seedProxyReference()) status('场景与同帧首图已就绪，可以开始串流。');
  }
  else if (!fixture && isNativeScene() && !reference) {
    $('reference-info').textContent = '选择本机游戏准备帧 proxy 参考图，或上传自己的图片。上传图与原生第 0 帧的对齐效果未验证。';
  }
  else if (!fixture && !reference) {
    $('reference-info').textContent = `${sceneName()}请上传匹配场景的参考图。`;
  }
}
requestAnimationFrame(tick);
initialize().catch(fail);
