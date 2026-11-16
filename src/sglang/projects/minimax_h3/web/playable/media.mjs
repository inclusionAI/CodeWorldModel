const protocol = 'h264_annexb_v1';
const codec = 'avc1.42C028';

function decoderConfig(media, width, height) {
  if (media?.protocol !== protocol || media.codec !== codec || media.fps !== 24 || media.frame_window !== 4 ||
      media.width !== width || media.height !== height) throw new Error('H.264 会话协议或尺寸不匹配');
  return {codec, codedWidth: width, codedHeight: height, optimizeForLatency: true};
}

export async function prepareMedia(media, mbps, width, height, runtime = globalThis) {
  if (!media) return null;
  const config = decoderConfig(media, width, height);
  const bitrate = Number(mbps) * 1e6;
  if (!Number.isSafeInteger(media.bitrate_min_bps) || !Number.isSafeInteger(media.bitrate_max_bps) ||
      !Number.isSafeInteger(media.bitrate_default_bps) || !Number.isSafeInteger(bitrate) ||
      bitrate < media.bitrate_min_bps || bitrate > media.bitrate_max_bps ||
      bitrate < 500000 || bitrate > 20000000) throw new Error('目标码率必须在 0.5 到 20 Mbps 之间');
  if (!runtime.VideoDecoder || !runtime.EncodedVideoChunk) throw new Error('当前浏览器不支持 WebCodecs H.264 解码');
  const support = await runtime.VideoDecoder.isConfigSupported(config);
  if (!support.supported) throw new Error('当前浏览器不支持 avc1.42C028 H.264 解码');
  return {video: {bitrate_bps: bitrate}, decoderConfig: config};
}

export class MediaDecoder {
  constructor(media, width, height, callbacks, runtime = globalThis) {
    const config = decoderConfig(media, width, height);
    if (!Number.isSafeInteger(media.bitrate_bps) || media.bitrate_bps < 500000 || media.bitrate_bps > 20000000) {
      throw new Error('服务返回的生效码率无效');
    }
    this.callbacks = callbacks;
    this.runtime = runtime;
    this.pending = new Map();
    this.outstanding = new Map();
    this.nextFrame = 0;
    this.decoded = 0;
    this.closed = false;
    this.decoder = new runtime.VideoDecoder({
      output: frame => this.output(frame),
      error: error => { if (!this.closed) { this.close(); callbacks.error(error); } }
    });
    try { this.decoder.configure(config); } catch (error) { this.close(); throw error; }
  }
  decode(metadata, payload, source) {
    if (this.closed) throw new Error('H.264 解码器已关闭');
    const m = metadata;
    if (!source || m.output_frame !== this.nextFrame || m.output_id !== source.output_id ||
        m.source_frame !== source.id || m.capture_ns !== source.capture_ns ||
        m.timestamp_us !== Math.floor(m.output_frame * 1000000 / 24) ||
        !Number.isSafeInteger(m.duration_us) || m.duration_us < 1 || typeof m.key_frame !== 'boolean' ||
        !Number.isFinite(m.encode_start_ns) || !Number.isFinite(m.encode_end_ns) ||
        m.encode_start_ns < 0 || m.encode_end_ns < m.encode_start_ns ||
        (m.output_frame === 0 && !m.key_frame) || m.encoded_bytes !== payload.length || !payload.length ||
        payload.length > 2 * 1024 * 1024) throw new Error('H.264 帧序号、源帧映射或字节数不匹配');
    if (this.outstanding.size >= 4) throw new Error('H.264 解码与播放窗口超过 4 帧');
    const entry = {metadata: m, source, receivedAt: performance.now(), video: null};
    this.pending.set(m.timestamp_us, entry);
    this.outstanding.set(m.output_frame, entry);
    this.nextFrame++;
    this.decoder.decode(new this.runtime.EncodedVideoChunk({type: m.key_frame ? 'key' : 'delta',
      timestamp: m.timestamp_us, duration: m.duration_us, data: payload}));
  }
  output(video) {
    if (this.closed) { video.close(); return; }
    const entry = this.pending.get(video.timestamp);
    if (!entry || entry.metadata.output_frame !== this.decoded) {
      video.close();
      this.close();
      this.callbacks.error(new Error('H.264 解码输出时间戳或顺序不匹配'));
      return;
    }
    this.pending.delete(video.timestamp);
    entry.video = video;
    this.decoded++;
    try {
      const decodedAt = performance.now();
      const browser = this.callbacks.observe_waits ? {received_ns: Math.round(entry.receivedAt * 1e6),
        decoded_ns: Math.round(decodedAt * 1e6)} : undefined;
      this.callbacks.decoded(entry.metadata, decodedAt - entry.receivedAt, browser);
      this.callbacks.frame({...entry.source, output_frame: entry.metadata.output_frame, video,
        ...(browser ? {decoded_ns: browser.decoded_ns} : {})});
    } catch (error) { this.close(); this.callbacks.error(error); }
  }
  release(outputFrame) {
    const first = this.outstanding.keys().next().value;
    const entry = this.outstanding.get(outputFrame);
    if (first !== outputFrame || !entry?.video) throw new Error('H.264 帧释放顺序错误');
    entry.video.close();
    this.outstanding.delete(outputFrame);
    return this.callbacks.ack(outputFrame);
  }
  async flush() {
    await this.decoder.flush();
    if (!this.closed && this.pending.size) throw new Error('H.264 完成后仍有未解码帧');
  }
  close() {
    if (this.closed) return;
    this.closed = true;
    for (const entry of this.outstanding.values()) entry.video?.close();
    this.outstanding.clear();
    this.pending.clear();
    if (this.decoder.state !== 'closed') this.decoder.close();
  }
}
