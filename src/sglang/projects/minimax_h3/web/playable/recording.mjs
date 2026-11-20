import {rgbToRgba} from './stream.mjs';

const text = new TextEncoder();
const byteLimit = 512 * 1024 * 1024;
const pendingLimit = 128 * 1024 * 1024;

function bytes(hex) { return Uint8Array.from(hex.match(/../g), part => parseInt(part, 16)); }
function uint(value) {
  let n = BigInt(value), out = [];
  do { out.unshift(Number(n & 255n)); n >>= 8n; } while (n);
  return Uint8Array.from(out);
}
function size(value) {
  const n = BigInt(value);
  for (let width = 1; width <= 8; width++) {
    if (n < (1n << BigInt(width * 7)) - 1n) {
      const out = new Uint8Array(width); let v = n | (1n << BigInt(width * 7));
      for (let i = width - 1; i >= 0; i--) { out[i] = Number(v & 255n); v >>= 8n; }
      return out;
    }
  }
  throw Error('录像文件过大');
}
function element(id, payload) {
  const body = payload instanceof Blob ? payload : new Blob([payload]);
  return new Blob([bytes(id), size(body.size), body]);
}
function float64(value) { const b = new ArrayBuffer(8); new DataView(b).setFloat64(0, value); return new Uint8Array(b); }
function webm(frames, width, height, durationMs) {
  const e = element, u = (id, n) => e(id, uint(n)), s = (id, v) => e(id, text.encode(v));
  const header = e('1a45dfa3', new Blob([u('4286', 1), u('42f7', 1), u('42f2', 4), u('42f3', 8),
    s('4282', 'webm'), u('4287', 4), u('4285', 2)]));
  const info = e('1549a966', new Blob([u('2ad7b1', 1000000), s('4d80', 'invoker-render Recorder'), s('5741', 'invoker-render Recorder'),
    e('4489', float64(durationMs))]));
  const video = e('e0', new Blob([u('b0', width), u('ba', height)]));
  const track = e('ae', new Blob([u('d7', 1), u('73c5', 1), u('83', 1),
    u('23e383', Math.max(1, Math.round(durationMs * 1000000 / frames.length))), s('86', 'V_VP8'), video]));
  const clusters = []; let start = -1, blocks = [];
  const flush = () => { if (start >= 0) clusters.push(e('1f43b675', new Blob([u('e7', start), ...blocks]))); blocks = []; };
  for (const frame of frames) {
    const ms = Math.floor(frame.timestamp / 1000);
    if (start < 0 || ms - start > 30000) { flush(); start = ms; }
    const offset = ms - start;
    blocks.push(e('a3', new Blob([Uint8Array.of(0x81, (offset >> 8) & 255, offset & 255, frame.key ? 0x80 : 0), frame.data])));
  }
  flush();
  const segment = e('18538067', new Blob([info, e('1654ae6b', track), ...clusters]));
  return new Blob([header, segment], {type: 'video/webm'});
}

export class PairedRecorder {
  static async create({proxyWidth, proxyHeight, width, height, onStatus = () => {}}) {
    if (!globalThis.VideoEncoder || !globalThis.VideoFrame) throw Error('浏览器不支持 WebCodecs 录像');
    const configs = [
      {codec:'vp8', width:proxyWidth, height:proxyHeight, bitrate:4000000, framerate:24},
      {codec:'vp8', width, height, bitrate:10000000, framerate:24},
    ];
    const supported = await Promise.all(configs.map(config => VideoEncoder.isConfigSupported(config)));
    if (supported.some(item => !item.supported)) throw Error('浏览器不支持当前尺寸的 VP8 录像');
    return new PairedRecorder(configs, onStatus);
  }
  constructor(configs, onStatus) {
    this.configs = configs; this.onStatus = onStatus; this.pending = new Map(); this.pendingBytes = 0;
    this.outputs = [[], []]; this.outputBytes = 0; this.pairs = []; this.skipped = 0;
    this.timestampSkipped = 0; this.unmatched = 0;
    this.startedAt = null; this.reason = null; this.error = null; this.done = null; this.result = null;
    this.proxyCanvas = document.createElement('canvas');
    this.proxyCanvas.width = configs[0].width; this.proxyCanvas.height = configs[0].height;
    this.proxyContext = this.proxyCanvas.getContext('2d');
    this.encoders = [];
    try {
      for (const [index, config] of configs.entries()) {
        const encoder = new VideoEncoder({output: chunk => this.output(index, chunk), error: error => this.fail(error)});
        this.encoders.push(encoder); encoder.configure(config);
      }
    } catch (error) {
      for (const encoder of this.encoders) if (encoder.state !== 'closed') encoder.close();
      throw error;
    }
  }
  fail(error) {
    if (this.error) return;
    this.error = String(error?.message || error); this.reason = 'encoder_error';
    this.onStatus(`录像编码失败：${this.error}`); this.finish();
  }
  output(index, chunk) {
    const data = new Uint8Array(chunk.byteLength); chunk.copyTo(data);
    this.outputs[index].push({timestamp:chunk.timestamp, key:chunk.type === 'key', data});
    this.outputBytes += data.byteLength;
    if (this.outputBytes >= byteLimit && !this.done) {
      this.onStatus('录像达到编码数据停止阈值，正在排空已提交画面。'); this.finish('size_limit');
    }
  }
  capture(id, pixels, captureNs) {
    if (this.done) return;
    if (this.pendingBytes + pixels.byteLength > pendingLimit) {
      this.onStatus('录像缓存已满，正在处理已录制部分。'); this.finish('proxy_buffer_limit'); return;
    }
    this.pending.set(id, {pixels, captureNs}); this.pendingBytes += pixels.byteLength;
  }
  forget(id) {
    const item = this.pending.get(id);
    if (item) { this.pendingBytes -= item.pixels.byteLength; this.pending.delete(id); }
    return item;
  }
  draw(id, outputCanvas, nowMs) {
    const source = this.forget(id);
    if (!source || this.done) return;
    if (this.encoders.some(encoder => encoder.encodeQueueSize >= 8)) { this.skipped++; return; }
    this.startedAt ??= nowMs;
    const timestamp = Math.round((nowMs - this.startedAt) * 1000);
    const previous = this.pairs.at(-1);
    if (previous && timestamp <= previous.timestamp_us) { this.timestampSkipped++; return; }
    try {
      this.proxyContext.putImageData(new ImageData(rgbToRgba(source.pixels), this.configs[0].width, this.configs[0].height), 0, 0);
      const forceKey = !previous || Math.floor(timestamp / 2000000) > Math.floor(previous.timestamp_us / 2000000);
      for (const [index, canvas] of [this.proxyCanvas, outputCanvas].entries()) {
        const frame = new VideoFrame(canvas, {timestamp});
        try { this.encoders[index].encode(frame, {keyFrame:forceKey}); } finally { frame.close(); }
      }
      this.pairs.push({source_frame:id, capture_ns:source.captureNs, timestamp_us:timestamp, draw_ns:Math.round(nowMs*1e6)});
    } catch (error) { this.fail(error); }
  }
  finish(reason = 'complete') {
    if (this.done) return this.done;
    this.reason ??= reason; this.unmatched = this.pending.size; this.pending.clear(); this.pendingBytes = 0;
    this.done = (async () => {
      try {
        await Promise.all(this.encoders.map(encoder => encoder.state === 'configured' ? encoder.flush() : Promise.reject(Error('录像编码器已关闭'))));
        if (this.error) throw Error(this.error);
        if (!this.pairs.length) throw Error('没有可录制的同步画面');
        for (const output of this.outputs) {
          if (output.length !== this.pairs.length || output.some((frame, i) => frame.timestamp !== this.pairs[i].timestamp_us)) {
            throw Error('录像帧数或时间戳不匹配');
          }
        }
        const last = this.pairs.at(-1).timestamp_us;
        const step = this.pairs.length > 1 ? last - this.pairs.at(-2).timestamp_us : 1000000 / 24;
        const durationMs = (last + step) / 1000;
        this.result = this.outputs.map((frames, i) => webm(frames, this.configs[i].width, this.configs[i].height, durationMs));
        const partial = {cancelled:'（包含取消前的画面）', interrupted:'（会话中断，仅包含此前画面）',
          size_limit:'（达到 512 MiB 编码数据停止阈值，包含排空画面）', proxy_buffer_limit:'（缓存达到 128 MiB 上限，仅包含此前画面）'}[this.reason] ?? '';
        this.onStatus(`同步录像已就绪：${this.pairs.length} 对帧${this.skipped ? `，编码繁忙跳过 ${this.skipped} 对` : ''}${this.timestampSkipped ? `，时间戳重复跳过 ${this.timestampSkipped} 对` : ''}${partial}。`);
        return this.result;
      } catch (error) {
        this.error = String(error?.message || error);
        this.onStatus(`录像未能保存：${this.error}`); return null;
      } finally {
        for (const encoder of this.encoders) if (encoder.state !== 'closed') encoder.close();
        this.outputs = [[], []];
      }
    })();
    return this.done;
  }
  metadata() { return {schema:'paired_recording/v1', clock:'display_wall_time', reason:this.reason,
    timestamp_origin:'first recorded canvas draw; browser performance.now, not physical presentation',
    population:'displayed source/RGB pairs accepted by both encoders; excludes display drops, busy-encoder skips and unmatched proxy frames',
    limits:{pending_proxy_bytes:pendingLimit, encoded_stop_threshold_bytes:byteLimit, encoder_queue_frames:8},
    limit_semantics:'encoded threshold stops new pairs; queued encoder output and WebM packaging may exceed it; not a total browser memory bound',
    encoded_bytes:this.outputBytes, unmatched_proxy_frames:this.unmatched,
    pairs:this.pairs, skipped_encoder:this.skipped, skipped_timestamp:this.timestampSkipped, error:this.error}; }
}
