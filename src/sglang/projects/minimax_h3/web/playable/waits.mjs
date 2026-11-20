export const WAIT_SCHEMA = 'stream_waits/v1';
export const WAIT_CLOCKS = Object.freeze({browser: 'browser performance.now',
  gateway: 'gateway process perf_counter_ns', frontend: 'serving frontend process perf_counter_ns'});
export const WAIT_UNCOLLECTED = ['socket_write_completion', 'physical_network_transmission',
  'physical_screen_presentation', 'terminal_ack_drain'];
const stamp = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const id = value => Number.isSafeInteger(value) && value >= 0;
const object = value => value && typeof value === 'object' && !Array.isArray(value);
const outcomes = new Set(['complete', 'cancelled', 'timeout', 'error']);
const fields = {
  browser: ['first_capture_ns', 'packet_ready_ns', 'credit_obtained_ns', 'send_call_start_ns', 'send_call_end_ns',
    'received_ns', 'decoded_ns', 'draw_start_ns', 'draw_end_ns', 'ack_enqueue_ns', 'pending_input', 'pending_input_tail', 'incomplete_assembly',
    'proxy_frame_enqueue', 'proxy_commit_enqueue'],
  gateway: ['request_complete_ns', 'rpc_start_ns', 'rpc_end_ns', 'record_read', 'ack_window', 'send_lock', 'admission_rpc', 'proxy_assembly'],
  frontend: ['request_complete_ns', 'enqueued_ns', 'taken_ns', 'rgb_reserve', 'prepared_ns',
    'rgb_enqueued_ns', 'rgb_taken_ns', 'publication', 'event_dequeued_ns'],
};
const intervals = {
  browser: ['pending_input', 'pending_input_tail', 'incomplete_assembly', 'proxy_frame_enqueue', 'proxy_commit_enqueue'],
  gateway: ['record_read', 'ack_window', 'send_lock', 'admission_rpc', 'proxy_assembly'],
  frontend: ['rgb_reserve', 'publication'],
};
const pairs = {
  browser: [['capture_assembly', 'first_capture_ns', 'packet_ready_ns'], ['credit_wait', 'packet_ready_ns', 'credit_obtained_ns'],
    ['send_preparation', 'credit_obtained_ns', 'send_call_start_ns'], ['send_enqueue', 'send_call_start_ns', 'send_call_end_ns'],
    ['decode', 'received_ns', 'decoded_ns'], ['playback_queue', 'decoded_ns', 'draw_start_ns'],
    ['draw_submission', 'draw_start_ns', 'draw_end_ns'], ['release_ack_enqueue', 'draw_end_ns', 'ack_enqueue_ns']],
  gateway: [['request_scheduling', 'request_complete_ns', 'rpc_start_ns'], ['admission_rpc', 'rpc_start_ns', 'rpc_end_ns']],
  frontend: [['request_enqueue', 'request_complete_ns', 'enqueued_ns'], ['mailbox_queue', 'enqueued_ns', 'taken_ns'],
    ['admission_total', 'request_complete_ns', 'prepared_ns'], ['rgb_queue', 'rgb_enqueued_ns', 'rgb_taken_ns']],
};

export function pendingPacketTiming(frames, finishing, finishRequestedNs) {
  const count = finishing && frames.length === 5 ? 5 : frames.length >= 17 ? 17 : 0;
  if (!count) return null;
  return {count, readyNs: Math.max(frames[count - 1].sample_receipt_ns ?? frames[count - 1].pixels_ready_ns,
    count === 5 ? finishRequestedNs : 0), firstCaptureNs: frames[0].capture_ns};
}

export function parseWaitObservation(event, session) {
  const m = event.metadata ?? event;
  const waits = event.waits ?? m.waits;
  if (waits === undefined) return {status: 'absent', rows: []};
  const invalid = status => ({status, rows: []});
  if (!object(waits)) return invalid('malformed');
  if (waits.schema !== WAIT_SCHEMA) return invalid('unsupported');
  if (JSON.stringify(waits).length > 2048 || Object.keys(waits).some(key => !['schema', ...Object.keys(fields)].includes(key))) return invalid('malformed');
  if (!session || m.session_id !== undefined && m.session_id !== session) return invalid('stale');
  for (const key of ['chunk_id', 'output_id', 'output_frame', 'source_frame', 'first_source_frame']) {
    if (m[key] !== undefined && !id(m[key])) return invalid('malformed');
  }
  const rows = [];
  for (const domain of Object.keys(fields)) {
    const data = waits[domain];
    if (data === undefined) continue;
    if (!object(data) || Object.keys(data).some(key => !fields[domain].includes(key))) return invalid('malformed');
    const spans = [];
    const add = (name, start_ns, end_ns, outcome = 'complete') => {
      if (!stamp(start_ns) || !stamp(end_ns) || end_ns < start_ns || !outcomes.has(outcome)) return false;
      spans.push({name, start_ns, end_ns, outcome}); return true;
    };
    for (const [key, value] of Object.entries(data)) {
      if (intervals[domain].includes(key)) {
        if (!object(value) || Object.keys(value).some(k => !['start_ns', 'end_ns', 'outcome'].includes(k)) ||
            !add(key, value.start_ns, value.end_ns, value.outcome)) return invalid('malformed');
      } else if (!stamp(value)) return invalid('malformed');
    }
    for (const [name, start, end] of pairs[domain]) {
      if (data[start] !== undefined && data[end] !== undefined && !add(name, data[start], data[end])) return invalid('malformed');
    }
    if (domain === 'frontend') {
      if (data.rgb_reserve?.outcome === 'complete' && data.prepared_ns !== undefined &&
          !add('preparation', data.rgb_reserve.end_ns, data.prepared_ns)) return invalid('malformed');
      if (data.publication?.outcome === 'complete' && data.event_dequeued_ns !== undefined &&
          !add('event_residence', data.publication.end_ns, data.event_dequeued_ns)) return invalid('malformed');
      if (m.op === 'video_frame' && stamp(m.encode_start_ns) && stamp(m.encode_end_ns) &&
          !add('encode', m.encode_start_ns, m.encode_end_ns)) return invalid('malformed');
    }
    if (spans.length) rows.push({key: [session, domain, event.event, m.chunk_id, m.output_id, m.output_frame, m.source_frame].join(':'),
      domain, kind: event.event, session, chunk: m.chunk_id, output: m.output_id, frame: m.output_frame,
      source: m.source_frame ?? m.first_source_frame, spans});
  }
  return {status: 'valid', rows};
}

export function clipWaitRows(rows, clip) {
  if (!clip || clip.kind === 'replay') return [];
  const chunks = new Set(clip.sources.map(source => source.chunk_id));
  if (!clip.frames) chunks.add(clip.chunk);
  return rows.filter(row => row.session === clip.session &&
    (row.output !== undefined ? row.output === clip.output : chunks.has(row.chunk)));
}
