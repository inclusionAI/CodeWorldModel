import assert from 'node:assert/strict';
import {prepareMedia, MediaDecoder} from '../web/playable/media.mjs';
import {FrameLedger} from '../web/playable/stream.mjs';

const capability = {protocol: 'h264_annexb_v1', codec: 'avc1.42C028', fps: 24, frame_window: 4,
  width: 640, height: 360, bitrate_min_bps: 500000, bitrate_max_bps: 20000000, bitrate_default_bps: 8000000};
const effective = {...capability, bitrate_bps: 8000000};
class FakeDecoder {
  static supported = true;
  static async isConfigSupported(config) { return {supported: this.supported, config}; }
  constructor(callbacks) { this.callbacks = callbacks; this.state = 'unconfigured'; this.chunks = []; this.frames = []; }
  configure(config) { this.config = config; this.state = 'configured'; }
  decode(chunk) { this.chunks.push(chunk); }
  emit(timestamp) {
    const frame = {timestamp, closeCount: 0, close() { this.closeCount++; }};
    this.frames.push(frame);
    this.callbacks.output(frame);
    return frame;
  }
  async flush() { for (const chunk of this.chunks.splice(0)) this.emit(chunk.timestamp); }
  close() { this.state = 'closed'; }
}
class FakeChunk { constructor(data) { Object.assign(this, data); } }
const runtime = {VideoDecoder: FakeDecoder, EncodedVideoChunk: FakeChunk};
assert.equal(await prepareMedia(null, 'bad', 640, 360, {}), null, 'raw legacy needs no WebCodecs');
assert.deepEqual((await prepareMedia(capability, 8, 640, 360, runtime)).video, {bitrate_bps: 8000000});
for (const bitrate of [0, .49, 20.01, NaN, Infinity, '']) {
  await assert.rejects(prepareMedia(capability, bitrate, 640, 360, runtime), /码率/);
}
await assert.rejects(prepareMedia(capability, 8, 640, 360, {}), /WebCodecs/);
await assert.rejects(prepareMedia({...capability, codec: 'avc1.42E01E'}, 8, 640, 360, runtime), /协议/);
FakeDecoder.supported = false;
await assert.rejects(prepareMedia(capability, 8, 640, 360, runtime), /不支持/);
FakeDecoder.supported = true;

const payload = new Uint8Array([0, 0, 0, 1, 65]);
const source = ordinal => ({id: ordinal + 50, capture_ns: 1000000 + ordinal, chunk_id: 0, output_id: 2});
const metadata = ordinal => ({op: 'video_frame', session_id: 'media-a', chunk_id: 1, output_id: 2,
  output_frame: ordinal, source_frame: source(ordinal).id, capture_ns: source(ordinal).capture_ns,
  timestamp_us: Math.floor(ordinal * 1000000 / 24), duration_us: 41667, key_frame: ordinal === 0,
  encode_start_ns: 2000000, encode_end_ns: 2100000, encoded_bytes: payload.length});
const playback = [], decoded = [], acknowledgements = [], errors = [];
let player;
player = new MediaDecoder(effective, 640, 360, {
  frame: frame => playback.push(frame), decoded: m => decoded.push(m),
  ack: ordinal => {
    assert.equal(playback[ordinal].video.closeCount, 1, 'VideoFrame closes before ACK');
    acknowledgements.push(ordinal);
  }, error: error => errors.push(error)
}, runtime);
for (let i = 0; i < 4; i++) player.decode(metadata(i), payload, source(i));
assert.equal(player.pending.size, 4);
assert.equal(player.outstanding.size, 4);
assert.throws(() => player.decode(metadata(4), payload, source(4)), /4 帧/);
assert.equal(player.decoder.chunks.length, 4, 'window includes pending decoder frames');
await player.flush();
assert.equal(playback.length, 4, 'flush retains delta frames for paced playback');
assert.equal(player.pending.size, 0);
assert.equal(player.outstanding.size, 4, 'decode alone does not release a permit');
assert.deepEqual(acknowledgements, []);
assert.deepEqual(decoded.map(m => m.source_frame), [50, 51, 52, 53]);
assert.throws(() => player.release(1), /顺序/);
for (let i = 0; i < 4; i++) player.release(i);
assert.deepEqual(acknowledgements, [0, 1, 2, 3]);
assert.equal(player.outstanding.size, 0);
assert.throws(() => player.release(3), /顺序/);
player.decode(metadata(4), payload, source(4));
player.decoder.chunks.length = 0;
const last = player.decoder.emit(metadata(4).timestamp_us);
player.close();
player.close();
assert.equal(last.closeCount, 1, 'cancel closes decoded frames exactly once');
assert.equal(player.decoder.state, 'closed');
assert.equal(player.outstanding.size, 0);
assert.equal(player.decoder.emit(999999).closeCount, 1, 'late decoder callback closes its frame');
assert.deepEqual(acknowledgements, [0, 1, 2, 3], 'cancel does not acknowledge undrawn frames');
assert.deepEqual(errors, []);

for (const change of [{source_frame: 1}, {timestamp_us: 1}, {output_id: 0}, {encoded_bytes: 99},
  {key_frame: false}, {duration_us: 0}, {encode_start_ns: undefined}, {encode_end_ns: 1}]) {
  const invalid = new MediaDecoder(effective, 640, 360, {error() {}}, runtime);
  assert.throws(() => invalid.decode({...metadata(0), ...change}, payload, source(0)), /不匹配/);
  assert.equal(invalid.pending.size, 0, 'invalid mapping never reaches decoder');
  invalid.close();
}
const badOutput = new MediaDecoder(effective, 640, 360, {error: error => errors.push(error)}, runtime);
badOutput.decode(metadata(0), payload, source(0));
assert.equal(badOutput.decoder.emit(1).closeCount, 1);
assert.equal(badOutput.closed, true, 'unexpected decoded timestamp closes the decoder');
assert.equal(errors.length, 1);

const ledger = new FrameLedger();
ledger.submit([source(0), source(1)], 0);
const batch = {output_id: 0, output_frame_start: 0, width: 640, height: 360,
  frame_count: 2, status: 'complete', source_frames: [source(0), source(1)].map(f => ({
    source_frame: f.id, capture_ns: f.capture_ns, chunk_id: 0}))};
assert.throws(() => ledger.acceptMetadata({...batch, source_frames: batch.source_frames.toReversed()}, 640, 360), /映射/);
assert.equal(ledger.pending.size, 2);
assert.deepEqual(ledger.acceptMetadata(batch, 640, 360).map(f => f.id), [50, 51]);
assert.equal(ledger.pending.size, 0, 'encoded metadata validates provenance without fabricated RGB bytes');
console.log('PASS media capability, bitrate, source timestamps, four-frame window, flush, FIFO close/ACK, cancellation');
