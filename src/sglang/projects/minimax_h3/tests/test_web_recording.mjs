import assert from 'node:assert/strict';
import {PairedRecorder} from '../web/playable/recording.mjs';

class FakeVideoFrame {
  constructor(canvas, {timestamp}) { this.canvas = canvas; this.timestamp = timestamp; }
  close() { this.closed = true; }
}
class FakeEncoder {
  static async isConfigSupported() { return {supported:true}; }
  constructor(callbacks) { this.callbacks=callbacks;this.state='unconfigured';this.encodeQueueSize=0;this.frames=[]; }
  configure(config) { this.config=config;this.state='configured'; }
  encode(frame, {keyFrame}) {
    if (this.throwOnEncode) throw Error('encoder failed');
    this.frames.push(frame.timestamp);
    this.callbacks.output({timestamp:frame.timestamp, type:keyFrame?'key':'delta', byteLength:2, copyTo(data){data.set([1,2]);}});
  }
  async flush() {}
  close() { this.state='closed'; }
}
globalThis.VideoEncoder=FakeEncoder;
globalThis.VideoFrame=FakeVideoFrame;
globalThis.ImageData=class { constructor(data,width,height) {Object.assign(this,{data,width,height});} };
globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>({putImageData(){}})})};

const make = () => PairedRecorder.create({proxyWidth:2,proxyHeight:1,width:4,height:2});
const pixels = () => new Uint8Array([1,2,3,4,5,6]);
const recorder = await make();
recorder.capture(0,pixels(),100);
recorder.draw(0,{},1000);
recorder.capture(1,pixels(),200);
recorder.encoders[0].encodeQueueSize=8;
recorder.draw(1,{},1020);
recorder.encoders[0].encodeQueueSize=0;
recorder.capture(2,pixels(),300);
recorder.draw(2,{},1150);
assert.deepEqual(recorder.pairs.map(pair=>[pair.source_frame,pair.timestamp_us]),[[0,0],[2,150000]]);
assert.equal(recorder.skipped,1);
assert.equal(recorder.pendingBytes,0);
const files=await recorder.finish();
assert.equal(files.length,2);
assert.ok(files.every(file=>file.type==='video/webm'&&file.size>50));
assert.deepEqual(recorder.encoders[0].frames,recorder.encoders[1].frames);
assert.ok(recorder.encoders.every(encoder=>encoder.state==='closed'));

const failure=await make();
failure.encoders[1].throwOnEncode=true;
failure.capture(0,pixels(),100);
failure.draw(0,{},1000);
assert.equal(await failure.done,null,'a partial pair must not be exported');
assert.match(failure.error,/encoder failed/);
assert.ok(failure.encoders.every(encoder=>encoder.state==='closed'));

const limited=await make();
limited.capture(0,{byteLength:128*1024*1024+1},100);
assert.equal(await limited.done,null);
assert.equal(limited.reason,'proxy_buffer_limit');

const timestamped = await make();
timestamped.capture(0,pixels(),100);
timestamped.draw(0,{},1000);
timestamped.capture(1,pixels(),200);
timestamped.draw(1,{},1000);
timestamped.capture(2,pixels(),300);
timestamped.encoders[1].encodeQueueSize=8;
timestamped.draw(2,{},1020);
timestamped.encoders[1].encodeQueueSize=0;
timestamped.capture(3,pixels(),400);
timestamped.forget(3); // The application discarded this frame before display.
timestamped.capture(4,pixels(),500);
timestamped.draw(4,{},1200);
timestamped.capture(5,pixels(),600); // Never matched to a displayed RGB frame.
const cancelled = await timestamped.finish('cancelled');
assert.equal(cancelled.length,2);
assert.equal(await timestamped.finish('complete'),cancelled,'finish is idempotent');
const timing = timestamped.metadata();
assert.equal(timing.reason,'cancelled');
assert.equal(timing.skipped_encoder,1,'either busy encoder skips the whole pair');
assert.equal(timing.skipped_timestamp,1,'duplicate timestamps are not encoder overload');
assert.equal(timing.unmatched_proxy_frames,1);
assert.deepEqual(timing.pairs.map(pair=>pair.source_frame),[0,4]);
assert.equal(timing.encoded_bytes,8);
assert.equal(timing.limits.pending_proxy_bytes,128*1024*1024);
assert.equal(timing.limits.encoded_stop_threshold_bytes,512*1024*1024);
assert.match(timing.limit_semantics,/not a total browser memory bound/);
timestamped.capture(6,pixels(),700);
timestamped.draw(6,{},1250);
assert.equal(timestamped.pendingBytes,0,'finished recorders retain no new captures');
assert.equal(timestamped.pairs.length,2);

const partial = await make();
partial.capture(0,pixels(),100);
partial.draw(0,{},1000);
partial.capture(1,{byteLength:128*1024*1024+1},200);
assert.equal((await partial.done).length,2,'pending limit keeps already encoded pairs');
assert.equal(partial.reason,'proxy_buffer_limit');

const encodedLimit = await make();
encodedLimit.outputBytes=512*1024*1024-4;
encodedLimit.capture(0,pixels(),100);
encodedLimit.draw(0,{},1000);
assert.equal((await encodedLimit.done).length,2,'encoded threshold flushes the accepted pair');
assert.equal(encodedLimit.reason,'size_limit');
assert.ok(encodedLimit.encoders.every(encoder=>encoder.state==='closed'));

const mismatched = await make();
mismatched.capture(0,pixels(),100);
mismatched.draw(0,{},1000);
mismatched.outputs[1][0].timestamp++;
assert.equal(await mismatched.finish(),null,'timestamp mismatch cannot yield a pair');
assert.match(mismatched.error,/时间戳不匹配/);

const empty = await make();
assert.equal(await empty.finish(),null);
assert.match(empty.error,/没有可录制/);
assert.ok(empty.encoders.every(encoder=>encoder.state==='closed'));
console.log('PASS matching timestamps, paired overload skips, encoder failure and bounded proxy cache');
