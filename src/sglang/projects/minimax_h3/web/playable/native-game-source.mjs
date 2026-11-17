import {unpack} from './stream.mjs';

const KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ShiftLeft', 'ShiftRight', 'Space', 'KeyF', 'KeyJ', 'KeyK', 'KeyQ', 'KeyV', 'KeyT']);

export function createNativeGameSource(canvas, {width, height, scene = 'native_game', onInput, onError, canReset}) {
  canvas.width = width; canvas.height = height;
  canvas.style.aspectRatio = `${width}/${height}`;
  const ctx = canvas.getContext('2d');
  const url = new URL(scene, location.href);
  url.protocol = 'ws:';
  const socket = new WebSocket(url.href);
  socket.binaryType = 'arraybuffer';
  let state = 'connecting', serial = 0, generation = 0, latest = null, drag = null, binding = null;
  let chain = Promise.resolve(), aliveTimer = null, evidence = null, previewLoop = null;
  const pending = new Map();
  const keys = new Set();
  let openResolve, openReject;
  const opened = new Promise((resolve, reject) => { openResolve = resolve; openReject = reject; });
  opened.catch(() => {});
  const openTimer = setTimeout(() => broken(new Error('Native game connection timeout')), 5000);

  function send(message) {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  }
  function releaseInput() {
    keys.clear(); drag = null;
    send({op: 'release'});
  }
  function broken(error) {
    if (state === 'disposed' || state === 'failed') return;
    state = 'failed'; generation++; latest = null; evidence = null; binding = null;
    clearTimeout(openTimer); clearTimeout(aliveTimer);
    releaseInput(); openReject(error);
    for (const request of pending.values()) { clearTimeout(request.timer); request.reject(error); }
    pending.clear();
    socket.close(); onError(error);
  }
  function keepalive() {
    send({op: 'keepalive'});
    aliveTimer = setTimeout(keepalive, 500);
  }
  socket.onopen = () => {
    clearTimeout(openTimer);
    if (state === 'disposed' || state === 'failed') return;
    if (state === 'connecting') state = 'held';
    keepalive(); openResolve();
  };
  socket.onerror = () => broken(new Error('Native game source connection failed'));
  socket.onclose = () => broken(new Error('Native game source disconnected'));
  socket.onmessage = event => {
    try {
      if (typeof event.data === 'string') throw new Error(JSON.parse(event.data).error || 'Invalid native game response');
      const {metadata, pixels} = unpack(event.data);
      const request = pending.get(metadata.id);
      if (!request || metadata.op !== 'result') throw new Error('Unexpected native game response');
      pending.delete(metadata.id); clearTimeout(request.timer);
      request.resolve({...metadata, jpeg: pixels, command_bytes: request.commandBytes});
    } catch (error) { broken(error); }
  };
  async function request(op, fields = {}) {
    await opened;
    if (state === 'disposed' || state === 'failed') throw new Error('Native game source is closed');
    if (pending.size >= 2 || [...pending.values()].some(value => (value.op === 'frame') === (op === 'frame'))) {
      throw new Error('Native game request lane is occupied');
    }
    return new Promise((resolve, reject) => {
      const id = ++serial;
      pending.set(id, {op, resolve, reject, commandBytes: JSON.stringify({op, id, ...fields}).length,
        timer: setTimeout(() => broken(new Error('Native game operation timeout')), 15000)});
      send({op, id, ...fields});
    });
  }
  function enqueue(operation) {
    const result = chain.then(operation);
    chain = result.catch(() => {});
    return result;
  }
  async function accept(frame, expected) {
    if (expected !== generation || state === 'disposed' || state === 'failed') return false;
    if (!frame.jpeg.length || frame.jpeg.length > 4 * 1024 * 1024 || frame.evidence?.atomic !== true ||
        !Number.isSafeInteger(frame.evidence.frame_id) || frame.evidence.frame_id <= 0 ||
        !Number.isSafeInteger(frame.evidence.epoch) || frame.evidence.epoch < 0 ||
        typeof frame.evidence.instance_id !== 'string' || !frame.evidence.instance_id ||
        frame.evidence.paused !== ['preparing', 'suspending'].includes(state) ||
        (latest && (frame.evidence.instance_id !== latest.evidence.instance_id ||
          frame.evidence.epoch !== latest.evidence.epoch ||
          frame.evidence.frame_id < latest.evidence.frame_id))) {
      throw new Error('Invalid native game frame');
    }
    const bitmap = await createImageBitmap(new Blob([frame.jpeg], {type: 'image/jpeg'}));
    try {
      if (expected !== generation || state === 'disposed' || state === 'failed') return false;
      if (bitmap.width > 4096 || bitmap.height > 4096) throw new Error('Native game image exceeds geometry limit');
      const ratio = Math.max(width / bitmap.width, height / bitmap.height);
      ctx.drawImage(bitmap, (width - bitmap.width * ratio) / 2, (height - bitmap.height * ratio) / 2,
        bitmap.width * ratio, bitmap.height * ratio);
      latest = {receipt_ms: performance.now(), evidence: frame.evidence};
      return true;
    } finally { bitmap.close(); }
  }
  function poll() {
    if (state !== 'live' || previewLoop) return;
    const expected = generation;
    previewLoop = (async () => {
      while (state === 'live' && expected === generation) {
        const frame = await request('frame', {after_frame_id: latest?.evidence.frame_id ?? 0});
        if (expected !== generation || state !== 'live') return;
        if (frame.cancelled) return;
        if (!frame.unchanged) await accept(frame, expected);
      }
    })().catch(broken).finally(() => { previewLoop = null; });
  }
  function setControlsSuspended(value) {
    if (state === 'disposed' || state === 'failed') return Promise.resolve();
    const expected = ++generation;
    state = value ? 'suspending' : 'resuming';
    releaseInput();
    return enqueue(async () => {
      if (state === 'disposed' || state === 'failed') return;
      if (!value && expected !== generation) return;
      const result = await request(value ? 'pause' : 'resume');
      if (expected !== generation) return;
      if (value) {
        if (await accept(result, expected)) state = 'held';
      }
      else {
        if (previewLoop) await previewLoop;
        if (expected !== generation) return;
        let frame;
        do {
          frame = await request('frame', {after_frame_id: latest?.evidence.frame_id ?? 0});
          if (expected !== generation || frame.cancelled) return;
        } while (frame.unchanged);
        await accept(frame, expected);
        if (expected === generation) { state = 'live'; poll(); }
      }
    });
  }
  function prepareRun() {
    if (state === 'disposed' || state === 'failed') return Promise.reject(new Error('Reconnect the native game source'));
    const expected = ++generation;
    state = 'preparing'; latest = null; evidence = null; binding = null;
    releaseInput();
    return enqueue(async () => {
      if (state === 'disposed' || state === 'failed' || expected !== generation) return;
      const result = await request('prepare');
      if (await accept(result, expected)) { binding = result.binding; state = 'held'; }
    });
  }
  async function reset() { await prepareRun(); if (state === 'held') await setControlsSuspended(false); }
  function stop() { return setControlsSuspended(true); }
  function captureNative(source_frame, capture_ns, segment = 0) {
    const expected = generation;
    return enqueue(async () => {
      if (expected !== generation || !['held', 'live'].includes(state)) return null;
      const result = await request('capture', {source_frame, capture_ns, segment});
      if (expected !== generation || !['held', 'live'].includes(state)) return null;
      if (result.cancelled) return null;
      if (result.waiting === true) return {waiting: true, command_bytes: result.command_bytes};
      if (result.blocked === 'source_budget') return {blocked: true, command_bytes: result.command_bytes};
      const sample = result.sample;
      if (result.jpeg.length || sample?.source_frame !== source_frame || sample.capture_ns !== capture_ns ||
          sample.source?.instance_id !== latest?.evidence.instance_id || sample.source?.epoch !== latest?.evidence.epoch) {
        throw new Error('Invalid native game sample receipt');
      }
      evidence = {...sample, browser_receipt_ns: Math.round(performance.now() * 1e6), command_bytes: result.command_bytes};
      return evidence;
    });
  }
  function inputState(dx = 0, dy = 0) {
    let buttons = 0;
    if (keys.has('ShiftLeft') || keys.has('ShiftRight')) buttons |= 1;
    if (keys.has('Space')) buttons |= 2 | 1024;
    for (const [key, bit] of [['KeyF', 16], ['KeyJ', 4], ['KeyK', 8], ['KeyQ', 128], ['KeyV', 32], ['KeyT', 64]]) {
      if (keys.has(key)) buttons |= bit;
    }
    if (latest?.evidence.seated_vehicle) {
      if (keys.has('KeyW')) buttons |= 256;
      if (keys.has('KeyS')) buttons |= 512;
    }
    send({op: 'input', value: {move_lr: Number(keys.has('KeyD')) - Number(keys.has('KeyA')),
      move_ud: Number(keys.has('KeyS')) - Number(keys.has('KeyW')),
      look_lr: Math.max(-1, Math.min(1, dx / 50)), look_ud: Math.max(-1, Math.min(1, dy / 50)), buttons}});
  }
  function key(event) {
    if (event.code === 'KeyR' && event.type === 'keydown' && !event.repeat && canReset()) { reset().catch(broken); return; }
    if (!KEYS.has(event.code)) return;
    event.preventDefault();
    if (state !== 'live') return;
    if (event.type === 'keydown') keys.add(event.code); else keys.delete(event.code);
    inputState(); onInput({kind: event.type, key: event.code});
  }
  const handlers = {keydown: key, keyup: key, blur: releaseInput,
    pointerdown(event) {
      if (event.button !== 0) return;
      canvas.focus();
      if (state === 'held' && canReset()) setControlsSuspended(false).catch(broken);
      if (state !== 'live') return;
      drag = event.pointerId; canvas.setPointerCapture(drag);
    },
    pointermove(event) {
      if (state === 'live' && drag === event.pointerId) {
        inputState(event.movementX, event.movementY);
        onInput({kind: 'pointer', dx: event.movementX, dy: event.movementY});
      }
    }, pointerup: releaseInput, pointercancel: releaseInput, lostpointercapture: releaseInput};
  for (const [name, handler] of Object.entries(handlers)) canvas.addEventListener(name, handler);
  const visibility = () => { if (document.hidden) releaseInput(); };
  window.addEventListener('blur', releaseInput);
  document.addEventListener('visibilitychange', visibility);
  function dispose() {
    releaseInput(); state = 'disposed'; generation++; latest = null; evidence = null; binding = null;
    clearTimeout(openTimer); clearTimeout(aliveTimer);
    const error = new Error('Native game source disposed');
    openReject(error);
    for (const request of pending.values()) { clearTimeout(request.timer); request.reject(error); }
    pending.clear();
    socket.close();
    for (const [name, handler] of Object.entries(handlers)) canvas.removeEventListener(name, handler);
    window.removeEventListener('blur', releaseInput);
    document.removeEventListener('visibilitychange', visibility);
  }
  return {width, height, render() {}, reset, prepareRun, captureNative, setControlsSuspended, releaseInput, dispose,
    takeBinding() { const value = binding; binding = null; return value; },
    drawReference(target) { target.getContext('2d').drawImage(canvas, 0, 0, target.width, target.height); },
    stop,
    get captureReady() { return state === 'live' || state === 'held'; },
    getCaptureEvidence: () => evidence,
    getSimulationState: () => ({scene, state, held_keys: [...keys], source: latest?.evidence ?? null})};
}
