"""Complete replacement instructions and bounded, boundary-polled JSONL sources."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

MAX_PROMPT_BYTES = 16 << 10
MAX_TEXT_ROWS = 4096
MAX_JSON_LINE_BYTES = 128 << 10


@dataclass(frozen=True)
class AppliedPrompt:
    text: str
    sha256: str
    first_chunk: int
    rows: int


@dataclass(frozen=True)
class PromptedLatent:
    video: object
    prompt: AppliedPrompt


def validate_prompt(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('prompt must be a nonblank complete string')
    try:
        encoded = text.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError('prompt must contain valid Unicode') from exc
    if len(encoded) > MAX_PROMPT_BYTES:
        raise ValueError(f'prompt exceeds {MAX_PROMPT_BYTES} UTF-8 bytes')
    return text


def own_document(document, *, static=None):
    import torch
    if not isinstance(document, dict) or set(document) != {'prompt_embeds', 'text_token_tags'}:
        raise ValueError('Qwen document needs exactly prompt_embeds and text_token_tags')
    embeds, tags = document['prompt_embeds'], document['text_token_tags']
    if (not isinstance(embeds, torch.Tensor) or embeds.ndim != 2 or min(embeds.shape) < 1
            or not embeds.is_floating_point() or not torch.isfinite(embeds).all()):
        raise ValueError('Qwen embeddings must be a nonempty finite floating matrix')
    if (not isinstance(tags, torch.Tensor) or tags.shape != (len(embeds),)
            or tags.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or not ((tags == 0) | (tags == 1)).all()):
        raise ValueError('Qwen tags must be aligned integral 0/1 rows')
    if static is not None:
        if len(embeds) > MAX_TEXT_ROWS:
            raise ValueError(f'text document exceeds {MAX_TEXT_ROWS} rows')
        if embeds.shape[1] != static['prompt_embeds'].shape[1] or embeds.dtype != static['prompt_embeds'].dtype:
            raise ValueError('text document width and dtype must match static conditioning')
        if tags.dtype != static['text_token_tags'].dtype:
            raise ValueError('text tag dtype must match static conditioning')
        if not (tags == 1).all():
            raise ValueError('dynamic text document must have all-1 text tags')
    return {key: value.detach().clone() for key, value in document.items()}


def encode_instruction(encoders, text):
    """Check real tokenizer and model limits before the frozen Qwen forward."""
    validate_prompt(text)
    rows = len(encoders.processor.tokenizer(text, add_special_tokens=False)['input_ids'])
    context = encoders.qwen.model.config.text_config.max_position_embeddings
    if not 1 <= rows <= min(MAX_TEXT_ROWS, context):
        raise ValueError(f'text tokenizer rows {rows} exceed supported range 1..{min(MAX_TEXT_ROWS, context)}')
    return encoders.condition(text)


def _record(line, fields):
    if len(line) > MAX_JSON_LINE_BYTES:
        raise ValueError('prompt JSONL line exceeds 128 KiB')
    try:
        value = json.loads(line.decode('utf-8'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('malformed completed prompt JSONL record') from exc
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f'prompt JSONL record requires exactly {sorted(fields)}')
    validate_prompt(value['prompt'])
    return value


class LivePromptSource:
    """Read only the file prefix present at poll entry; keep an incomplete line."""
    def __init__(self, path):
        self.path = Path(path)
        info = self.path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('prompt updates must be a regular append-only file')
        self.identity = (info.st_dev, info.st_ino)
        self.cursor = 0
        self.buffer = b''
        digest = hashlib.sha256()
        with self.path.open('rb') as handle:
            remaining = info.st_size
            while remaining:
                block = handle.read(min(64 << 10, remaining))
                if not block:
                    raise ValueError('prompt updates truncated during initialization')
                digest.update(block)
                remaining -= len(block)
        self.prefix_bytes, self.prefix_sha256 = info.st_size, digest.hexdigest()
        self.provenance = dict(kind='live_append_jsonl', path=str(self.path.resolve()),
                               initial_prefix_bytes=info.st_size, initial_prefix_sha256=digest.hexdigest())

    def __call__(self, chunk):
        info = self.path.stat()
        if (info.st_dev, info.st_ino) != self.identity:
            raise ValueError('prompt updates file was replaced')
        if info.st_size < self.prefix_bytes:
            raise ValueError('prompt updates file was truncated')
        endpoint = info.st_size
        latest = None
        with self.path.open('rb') as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != self.identity:
                raise ValueError('prompt updates file was replaced')
            digest = hashlib.sha256()
            remaining = self.prefix_bytes
            while remaining:
                block = handle.read(min(64 << 10, remaining))
                if not block:
                    raise ValueError('prompt updates file was truncated')
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != self.prefix_sha256:
                raise ValueError('prompt updates consumed prefix was modified or rewritten')
            handle.seek(self.cursor)
            while self.cursor < endpoint:
                block = handle.read(min(64 << 10, endpoint-self.cursor))
                if not block:
                    raise ValueError('prompt updates file was truncated')
                digest.update(block[max(0, self.prefix_bytes-self.cursor):])
                self.cursor += len(block)
                pieces = (self.buffer+block).split(b'\n')
                self.buffer = pieces.pop()
                for line in pieces:
                    latest = _record(line, {'prompt'})['prompt']
                if len(self.buffer) > MAX_JSON_LINE_BYTES:
                    raise ValueError('prompt JSONL line exceeds 128 KiB')
        self.prefix_bytes, self.prefix_sha256 = endpoint, digest.hexdigest()
        if chunk == 0 and latest is None:
            raise ValueError('chunk 0 requires a completed initial prompt record')
        return latest


class ReplayPromptSource:
    def __init__(self, path, chunks):
        self.records = []
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            while line := handle.readline(MAX_JSON_LINE_BYTES+2):
                digest.update(line)
                value = _record(line, {'chunk', 'prompt'})
                index = value['chunk']
                if (type(index) is not int or not 0 <= index < chunks
                        or self.records and index <= self.records[-1]['chunk']):
                    raise ValueError('replay chunks must be strictly increasing unique integers in range')
                self.records.append(value)
        if not self.records or self.records[0]['chunk'] != 0:
            raise ValueError('replay requires an initial chunk 0 prompt')
        self.cursor = 0
        self.provenance = dict(kind='replay_jsonl', path=str(Path(path).resolve()), sha256=digest.hexdigest())

    def __call__(self, chunk):
        if self.cursor < len(self.records) and self.records[self.cursor]['chunk'] == chunk:
            value = self.records[self.cursor]['prompt']
            self.cursor += 1
            return value
        return None


def prompt_source(config, chunks):
    if config['kind'] == 'live_append_jsonl':
        return LivePromptSource(config['path'])
    if config['kind'] == 'replay_jsonl':
        return ReplayPromptSource(config['path'], chunks)
    raise ValueError('unsupported prompt source')
