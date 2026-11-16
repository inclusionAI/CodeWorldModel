const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', {fatal: true});

export class CapturePacer {
  constructor(start, policy = 'continuous', interval = 1000 / 24) {
    this.next = start;
    this.policy = policy;
    this.interval = interval;
    this.waitStarted = null;
    this.waitMs = 0;
    this.waitCount = 0;
  }
  poll(now, blocked) {
    if (now < this.next) return null;
    if (blocked) {
      if (this.policy === 'stress') throw new Error('输入或网络队列超载，停止本次 24 FPS 压力测试');
      if (this.waitStarted === null) { this.waitStarted = now; this.waitCount++; }
      return null;
    }
    if (this.waitStarted !== null) {
      this.waitMs += now - this.waitStarted;
      this.waitStarted = null;
      this.next = now;
    }
    const missed = Math.max(0, Math.floor((now - this.next) / this.interval));
    this.next += (missed + 1) * this.interval;
    return missed;
  }
  waitingMs(now) { return this.waitMs + (this.waitStarted === null ? 0 : now - this.waitStarted); }
  stop(now) { this.waitMs = this.waitingMs(now); this.waitStarted = null; }
}

export class DisplayPacer {
  constructor(mode = 'fixed') {
    this.mode = mode;
    this.targetFps = 24;
    this.windowStart = null;
    this.nextMiss = 0;
    this.misses = 0;
    this.buffered = 0;
  }
  sample(now, depth) {
    if (this.mode !== 'adaptive') return;
    this.windowStart ??= now;
    if (!depth && now >= this.nextMiss) {
      this.misses++;
      this.nextMiss = now + this.interval();
    } else if (depth >= 3) this.buffered++;
    if (depth) this.nextMiss = 0;
    if (now - this.windowStart < 1000) return;
    if (this.misses >= 3) this.targetFps = Math.max(8, this.targetFps - 1);
    else if (this.buffered >= 3) this.targetFps = Math.min(24, this.targetFps + .5);
    this.windowStart = now;
    this.misses = this.buffered = 0;
  }
  interval() { return 1000 / this.targetFps; }
}

export function profilingLabel(profiling) {
  const labels = {
    baseline: 'Profiler 关闭 · 普通测量',
    diagnostic: 'Profiler 开启 · 诊断采集，单独统计',
    unknown: 'Profiler 状态未知 · 不纳入正式对照',
    fixture: 'Profiler 不适用 · CPU 测试回显'
  };
  const sources = {health: '服务健康快照，启动后确认', start: '本次会话启动回执', fixture: '无模型推理'};
  return `${labels[profiling?.status] ?? labels.unknown} · ${sources[profiling?.source] ?? '等待会话回执'}`;
}

export function pack(metadata, pixels = new Uint8Array()) {
  const header = encoder.encode(JSON.stringify(metadata));
  if (header.length > 65536) throw new Error('消息头超过 64 KiB');
  const body = new Uint8Array(4 + header.length + pixels.length);
  new DataView(body.buffer).setUint32(0, header.length);
  body.set(header, 4);
  body.set(pixels, 4 + header.length);
  return body;
}

export function unpack(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 6 || buffer.byteLength > 128 * 1024 * 1024) {
    throw new Error('服务响应长度无效');
  }
  const size = new DataView(buffer).getUint32(0);
  if (size < 2 || size > 65536 || size + 4 > buffer.byteLength) throw new Error('服务消息头无效');
  const metadata = JSON.parse(decoder.decode(new Uint8Array(buffer, 4, size)));
  if (!metadata || typeof metadata !== 'object' || Array.isArray(metadata)) throw new Error('服务元数据无效');
  return {metadata, pixels: new Uint8Array(buffer, size + 4)};
}

export function rgbToRgba(rgb) {
  if (rgb.length % 3) throw new Error('RGB 字节未对齐');
  const rgba = new Uint8ClampedArray(rgb.length / 3 * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    rgba[j] = rgb[i]; rgba[j + 1] = rgb[i + 1]; rgba[j + 2] = rgb[i + 2]; rgba[j + 3] = 255;
  }
  return rgba;
}

export function rgbaToRgb(rgba) {
  const rgb = new Uint8Array(rgba.length / 4 * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i]; rgb[j + 1] = rgba[i + 1]; rgb[j + 2] = rgba[i + 2];
  }
  return rgb;
}

export function packetSize(buffered, finishing) {
  if (!finishing) return buffered >= 17 ? 17 : 0;
  if (buffered === 5) return 5;
  return buffered >= 17 ? 17 : 0;
}

export function sourceBudgetPolicy(config, selected = 'bounded51') {
  const limits = {off: null, bounded51: 51, bounded45: 45, bounded40: 40};
  if (!Object.hasOwn(limits, selected)) throw new Error('不支持的在途帧预算');
  const reason = selected === 'off' ? 'selected_off'
    : config.source_budget?.reason ?? 'unsupported_recipe_or_decoder';
  const enabled = selected !== 'off' && config.source_budget?.eligible === true;
  return {selected, effective: enabled ? selected : 'off', limit_frames: enabled ? limits[selected] : null, reason};
}

export function sourceBudgetBlocked(captured, displayed, limit) {
  return limit !== null && captured - displayed >= limit;
}

export function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(fraction * sorted.length) - 1];
}

export class FrameLedger {
  constructor() { this.pending = new Map(); this.emitted = 0; this.outputId = 0; }
  submit(frames, chunkId) {
    for (const frame of frames) {
      if (this.pending.has(frame.id)) throw new Error('源帧重复提交');
      this.pending.set(frame.id, {id: frame.id, capture_ns: frame.capture_ns, input: frame.input, chunk_id: chunkId});
    }
  }
  accept(m, byteLength, width, height) {
    if (byteLength !== m.frame_count * width * height * 3) throw new Error('返回 RGB 字节数不匹配');
    return this.acceptMetadata(m, width, height);
  }
  acceptMetadata(m, width, height) {
    if (m.output_id !== this.outputId || m.output_frame_start !== this.emitted ||
        m.width !== width || m.height !== height || !Number.isSafeInteger(m.frame_count) || m.frame_count < 0 ||
        !Array.isArray(m.source_frames) ||
        m.source_frames.length !== m.frame_count || m.frame_count > this.pending.size) {
      throw new Error('返回 RGB 的序号、尺寸或帧数不匹配');
    }
    const expected = [...this.pending.values()].slice(0, m.frame_count);
    for (let i = 0; i < expected.length; i++) {
      const source = m.source_frames[i], frame = expected[i];
      if (source.source_frame !== frame.id || source.capture_ns !== frame.capture_ns || source.chunk_id !== frame.chunk_id) {
        throw new Error('返回 RGB 的源帧映射不匹配');
      }
    }
    if (m.status === 'complete' && m.frame_count !== this.pending.size) throw new Error('EOS 仍有未返回源帧');
    for (const frame of expected) this.pending.delete(frame.id);
    this.emitted += expected.length;
    this.outputId++;
    return expected;
  }
}
