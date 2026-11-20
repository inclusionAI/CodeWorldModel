"""Fixed, process-local receipts for the existing media queue owners."""
import json


WAIT_SCHEMA = 'stream_waits/v1'
MAX_WAITS = 2048


def with_waits(metadata, owner, fields, counts):
    previous = metadata.get('waits', {})
    if not isinstance(previous, dict) or previous.get('schema', WAIT_SCHEMA) != WAIT_SCHEMA:
        counts['unsupported'] = counts.get('unsupported', 0) + 1
        previous = {}
    owned = previous.get(owner, {})
    waits = dict(previous, schema=WAIT_SCHEMA)
    waits[owner] = dict(owned if isinstance(owned, dict) else {}, **fields)
    result = dict(metadata, waits=waits)
    if (len(json.dumps(waits, separators=(',', ':')).encode()) > MAX_WAITS
            or len(json.dumps(result, separators=(',', ':'), ensure_ascii=False).encode()) > 65536):
        counts['omitted'] = counts.get('omitted', 0) + 1
        result.pop('waits')
    return result


def interval(start, end, outcome='complete'):
    return dict(start_ns=start, end_ns=end, outcome=outcome)
