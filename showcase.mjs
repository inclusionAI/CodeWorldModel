export const showcaseManifestSource = './media/showcase-v1/manifest.json';
const groups = ['hero', 'granularity', 'routes', 'correspondence', 'games', 'wall'];
const routes = [null, 'a', 'b'];

export function parseShowcaseManifest(value, manifestURL) {
  const base = new URL('.', manifestURL);
  const localSource = (source, extension) => {
    if (typeof source !== 'string' || !/^[A-Za-z0-9_./-]+$/.test(source)
      || source.split('/').some((part) => !part || part === '.' || part === '..') || !source.endsWith(extension)) {
      throw new Error('Showcase media must use relative paths inside its manifest directory');
    }
    const resolved = new URL(source, base);
    if (resolved.origin !== base.origin || !resolved.pathname.startsWith(base.pathname)) throw new Error('Showcase media escaped its manifest directory');
    return resolved.href;
  };
  const integer = (number, min, max) => Number.isInteger(number) && number >= min && number <= max;
  const text = (item) => typeof item === 'string' && item.trim() !== '';
  const streams = value?.streams;
  if (value?.schema !== 'showcase/1' || !streams || typeof streams !== 'object' || Array.isArray(streams)
    || Object.keys(streams).length === 0 || !Object.values(streams).every(text)) throw new Error('Invalid showcase manifest');
  const clips = {};
  for (const [id, clip] of Object.entries(value.clips ?? {})) {
    const { size, tiles } = clip ?? {};
    if (!/^[a-z][a-z0-9_-]*$/.test(id) || !groups.includes(clip?.group)
      || !Array.isArray(size) || size.length !== 2 || !integer(size[0], 1, 8192) || !integer(size[1], 1, 8192)
      || !integer(clip.fps, 1, 60) || !integer(clip.frames, 2, 3600) || !/^[0-9a-f]{64}$/.test(clip.sha256 ?? '')
      || !Array.isArray(tiles) || tiles.length < 1 || tiles.length > 8) throw new Error(`Invalid showcase clip ${id}`);
    // Rebuilt clips keep their file names; the content hash in the URL keeps browsers from showing a cached old cut.
    const version = `?v=${clip.sha256.slice(0, 12)}`;
    clips[id] = {
      id, group: clip.group, size, fps: clip.fps, frames: clip.frames,
      src: localSource(clip.src, '.mp4') + version, poster: localSource(clip.poster, '.jpg') + version,
      tiles: tiles.map((tile) => {
        const rect = tile?.rect;
        if (!Object.hasOwn(streams, tile?.stream) || !text(tile.scene) || !text(tile.style) || !routes.includes(tile.route)
          || !Array.isArray(rect) || rect.length !== 4 || !rect.every(Number.isInteger)
          || rect[0] < 0 || rect[1] < 0 || rect[2] < 1 || rect[3] < 1
          || rect[0] + rect[2] > size[0] || rect[1] + rect[3] > size[1]) throw new Error(`Invalid showcase tile in ${id}`);
        return { stream: tile.stream, rect, route: tile.route, scene: tile.scene, style: tile.style, name: streams[tile.stream] };
      }),
    };
    const parsed = clips[id].tiles;
    // A route names a tile only when the clip's tiles differ by route.
    const routed = new Set(parsed.map((tile) => tile.route)).size > 1;
    for (const tile of parsed) tile.label = routed ? `${tile.name} · ${tile.route.toUpperCase()}` : tile.name;
  }
  return { streams, clips };
}

/** Manifest rects are in video pixels; a poster may be downscaled. */
export function scaleRect(rect, naturalWidth, clipWidth) {
  const scale = naturalWidth / clipWidth;
  return rect.map((part) => part * scale);
}

export function wrapIndex(index, delta, length) {
  return (((index + delta) % length) + length) % length;
}

/** Source sub-rect that fills a width x height target without distortion, centred. */
export function coverRect([x, y, w, h], width, height) {
  const target = width / height;
  if (w / h > target) {
    const cropped = h * target;
    return [x + (w - cropped) / 2, y, cropped, h];
  }
  const cropped = w / target;
  return [x, y + (h - cropped) / 2, w, cropped];
}

/** CSS background size and position that show one tile of a clip poster. */
export function tileBackground(clip, tile) {
  const [x, y, w, h] = tile.rect;
  const [width, height] = clip.size;
  const offset = (start, part, whole) => (whole === part ? 0 : start / (whole - part) * 100);
  return {
    size: `${width / w * 100}% ${height / h * 100}%`,
    position: `${offset(x, w, width)}% ${offset(y, h, height)}%`,
  };
}

const SLIDER_STEP = 0.05;
const SLIDER_PAGE = 0.2;

/** Returns the next slider position in 0..1 for a key, or null when the key is not a slider key. */
export function sliderKey(value, key) {
  const moves = {
    ArrowLeft: value - SLIDER_STEP, ArrowDown: value - SLIDER_STEP, PageDown: value - SLIDER_PAGE,
    ArrowRight: value + SLIDER_STEP, ArrowUp: value + SLIDER_STEP, PageUp: value + SLIDER_PAGE,
    Home: 0, End: 1,
  };
  if (!Object.hasOwn(moves, key)) return null;
  return Math.round(Math.max(0, Math.min(1, moves[key])) * 1000) / 1000;
}

let showcase = null;
export function loadShowcase() {
  const manifestURL = new URL(showcaseManifestSource, document.baseURI);
  showcase ??= fetch(manifestURL, { cache: 'no-cache' })
    .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`Manifest ${response.status}`))))
    .then((value) => parseShowcaseManifest(value, manifestURL));
  return showcase;
}

export function fitCanvas(canvas) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return canvas.getContext('2d');
}

const liveViews = new Set();
if (typeof window !== 'undefined') window.addEventListener('resize', () => liveViews.forEach((view) => view.draw()), { passive: true });

/**
 * One clip's video (or its poster until frames decode) drawn through `paint(media, rectOf, clip)`.
 * `rectOf(tile)` maps a manifest rect into the current media's pixel space.
 */
export function createTileView(clip, paint, { preload = 'none', loop = true, autoDraw = true, onError } = {}) {
  const video = document.createElement('video');
  video.muted = true;
  video.loop = loop;
  video.playsInline = true;
  video.preload = preload;
  const poster = new Image();
  poster.decoding = 'async';
  poster.src = clip.poster;
  let frame = 0;
  let frameKind = null;
  let started = false;
  let pendingTime = null;
  let released = false;

  function draw() {
    if (released) return false;
    if (video.readyState >= 2) paint(video, (tile) => tile.rect, clip);
    else if (poster.complete && poster.naturalWidth) paint(poster, (tile) => scaleRect(tile.rect, poster.naturalWidth, clip.size[0]), clip);
    else return false;
    return true;
  }
  function cancel() {
    if (frame) {
      if (frameKind === 'video') video.cancelVideoFrameCallback(frame);
      else window.cancelAnimationFrame(frame);
    }
    frame = 0;
  }
  function tick() {
    frame = 0;
    draw();
    if (released || video.paused || video.ended) return;
    frameKind = typeof video.requestVideoFrameCallback === 'function' ? 'video' : 'raf';
    frame = frameKind === 'video' ? video.requestVideoFrameCallback(tick) : window.requestAnimationFrame(tick);
  }
  const fail = () => { if (!released) onError?.(); };

  video.addEventListener('loadedmetadata', () => {
    if (pendingTime !== null && Number.isFinite(video.duration)) video.currentTime = pendingTime % video.duration;
    pendingTime = null;
  });
  video.addEventListener('loadeddata', draw);
  video.addEventListener('seeked', draw);
  video.addEventListener('error', fail);
  if (autoDraw) video.addEventListener('playing', () => { cancel(); tick(); });
  poster.addEventListener('load', () => { if (video.readyState < 2) draw(); });

  const view = {
    clip,
    video,
    draw,
    load() {
      if (started || released) return;
      started = true;
      video.src = clip.src;
      video.load();
    },
    play() {
      view.load();
      return Promise.resolve(video.play()).catch((failure) => { if (failure?.name !== 'AbortError') fail(); });
    },
    pause() {
      video.pause();
      cancel();
    },
    get time() { return pendingTime ?? video.currentTime; },
    seek(seconds) {
      if (video.readyState >= 1 && Number.isFinite(video.duration)) video.currentTime = seconds % video.duration;
      else pendingTime = seconds;
    },
    release() {
      released = true;
      liveViews.delete(view);
      cancel();
      video.pause();
      video.removeAttribute('src');
      video.load();
      poster.removeAttribute('src');
    },
  };
  liveViews.add(view);
  return view;
}

/** Draws tile i of the clip into canvases[i]. */
export function tilePainter(canvases) {
  return (media, rectOf, clip) => clip.tiles.forEach((tile, index) => {
    const canvas = canvases[index];
    const context = fitCanvas(canvas);
    const [sx, sy, sw, sh] = coverRect(rectOf(tile), canvas.width, canvas.height);
    context.drawImage(media, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  });
}

/** Builds (or reuses) one labelled canvas per tile in `container` and returns a tile view for the clip. */
export function mountClip(container, clip, options = {}) {
  let cells = [...container.children];
  if (cells.length !== clip.tiles.length) {
    cells = clip.tiles.map(() => {
      const cell = document.createElement('div');
      cell.className = 'tile';
      const canvas = document.createElement('canvas');
      canvas.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span');
      label.className = 'tile-label mono';
      cell.append(canvas, label);
      return cell;
    });
    container.replaceChildren(...cells);
  }
  clip.tiles.forEach((tile, index) => {
    cells[index].dataset.stream = tile.stream;
    cells[index].dataset.route = tile.route ?? '';
    cells[index].style.setProperty('--tile-aspect', `${tile.rect[2]} / ${tile.rect[3]}`);
    cells[index].querySelector('.tile-label').textContent = tile.label;
  });
  container.dataset.group = clip.group;
  container.setAttribute('role', 'img');
  container.setAttribute('aria-label', `${clip.tiles[0].scene}, ${clip.tiles[0].style} proxy. Synced views: ${clip.tiles.map((tile) => tile.label).join(', ')}.`);
  return createTileView(clip, tilePainter(cells.map((cell) => cell.querySelector('canvas'))), options);
}
