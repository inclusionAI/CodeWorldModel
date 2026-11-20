import { mediaKinds, pendingClips } from './content.mjs?rev=design-18';
import { coverRect, createTileView, fitCanvas, tileBackground, tilePainter } from './showcase.mjs?rev=showcase-7';
import { createSnowView } from './snow.mjs?rev=design-8';

/** Per clip group: one framed output per slot, each with its proxy inset; `split` layouts open from one shared first frame. */
export const chapterLayouts = {
  granularity: { split: false, frames: [{ output: { stream: 'sft' }, inset: { stream: 'proxy' } }] },
  games: { split: false, frames: [{ output: { stream: 'sft' }, inset: { stream: 'proxy' } }] },
  routes: { split: true, frames: [
    { output: { stream: 'sft', route: 'a' }, inset: { stream: 'proxy', route: 'a' } },
    { output: { stream: 'sft', route: 'b' }, inset: { stream: 'proxy', route: 'b' } },
  ] },
  // `blend` draws the proxy over the output in one frame instead of as an inset.
  correspondence: { split: false, blend: true, frames: [{ output: { stream: 'sft' }, inset: { stream: 'proxy' } }] },
};

// Pending clips are 672 x 384 per tile, with A/B routes.
const pendingTile = [0, 0, 672, 384];
const pendingTiles = { routes: [['proxy', 'a'], ['proxy', 'b'], ['sft', 'a'], ['sft', 'b']] };

/** The manifest clip, or for a pending clip a stand-in of the same shape whose frames draw snow. */
export function chapterClip(showcase, id) {
  if (showcase.clips[id]) return showcase.clips[id];
  const spec = pendingClips[id];
  if (!spec) throw new Error(`${id} is neither in the manifest nor pending`);
  const tiles = pendingTiles[spec.group].map(([stream, route]) => ({
    stream, route, rect: pendingTile, scene: spec.scene, style: 'Training proxy',
    name: showcase.streams[stream], label: `${showcase.streams[stream]} · ${route.toUpperCase()}`,
  }));
  return { id, group: spec.group, pending: true, tiles };
}

export function resolveFrames(clip, layout) {
  const tileFor = (slot) => {
    const matches = clip.tiles.filter((tile) => tile.stream === slot.stream && (!slot.route || tile.route === slot.route));
    if (matches.length !== 1) throw new Error(`${clip.id}: ${matches.length} tiles match ${slot.stream}${slot.route ? ` route ${slot.route}` : ''}`);
    return matches[0];
  };
  return layout.frames.map(({ output, inset }) => ({ output: tileFor(output), inset: tileFor(inset) }));
}

/** Tab items for one chapter; style tabs are unnamed proxy thumbnails, labelled by position for screen readers. */
export function chapterTabs(chapter, showcase) {
  return chapter.clips.map((id, index) => {
    const clip = chapterClip(showcase, id);
    const proxy = clip.tiles.find((tile) => tile.stream === 'proxy');
    return {
      id,
      label: chapter.tabBy === 'style' ? `Proxy ${index + 1}` : clip.tiles[0][chapter.tabBy],
      thumb: chapter.tabBy === 'style' && proxy && !clip.pending ? { src: clip.poster, ...tileBackground(clip, proxy) } : null,
    };
  });
}

const SPLIT_HOLD_MS = 400;
const SPLIT_FALLBACK_MS = 1100;

function frameElement() {
  const root = document.createElement('div');
  root.className = 'frame';
  const output = document.createElement('canvas');
  output.setAttribute('aria-hidden', 'true');
  const outputLabel = document.createElement('span');
  outputLabel.className = 'tile-label mono';
  const inset = document.createElement('div');
  inset.className = 'frame-inset';
  const insetCanvas = document.createElement('canvas');
  insetCanvas.setAttribute('aria-hidden', 'true');
  const insetLabel = document.createElement('span');
  insetLabel.className = 'frame-inset-label mono';
  inset.append(insetCanvas, insetLabel);
  root.append(output, outputLabel, inset);
  return { root, canvases: [output, insetCanvas], labels: [outputLabel, insetLabel] };
}

/**
 * `phase` is 'merged' (one centred shared first frame, paused at 0), 'splitting' (frames moving apart)
 * or 'split' (side by side, playable). The stage exposes only merged/split to CSS.
 */
export function initChapter(section, chapter, showcase, reducedMotion) {
  const stage = section.querySelector('[data-chapter-stage]');
  const poster = section.querySelector('[data-chapter-poster]');
  const label = section.querySelector('[data-chapter-label]');
  const error = section.querySelector('[data-chapter-error]');
  const toggle = section.querySelector('[data-chapter-toggle]');
  const narrow = window.matchMedia('(max-width: 700px)');
  const layout = chapterLayouts[chapterClip(showcase, chapter.clips[0]).group];
  const frames = layout.frames.map(frameElement);
  const blend = section.querySelector('[data-chapter-blend]');
  const badge = section.querySelector('[data-chapter-badge]');
  const paint = layout.blend ? (media, rectOf, { tiles: [output, proxy] }) => {
    const canvas = frames[0].canvases[0];
    const context = fitCanvas(canvas);
    const draw = (tile) => context.drawImage(media, ...coverRect(rectOf(tile), canvas.width, canvas.height), 0, 0, canvas.width, canvas.height);
    draw(output);
    context.globalAlpha = blend.value / 100;
    draw(proxy);
    context.globalAlpha = 1;
  } : tilePainter(frames.flatMap((frame) => frame.canvases));
  const observed = typeof IntersectionObserver === 'function';
  const state = { near: !observed, visible: false, inView: false, userPaused: reducedMotion.matches };
  const animates = () => layout.split && observed && !reducedMotion.matches && !narrow.matches;
  let phase = animates() ? 'merged' : 'split';
  let timer = 0;
  let view = null;
  stage.prepend(...frames.map((frame) => frame.root));
  stage.dataset.frames = String(frames.length);
  blend?.addEventListener('input', () => view.draw());

  function sync() {
    stage.dataset.phase = phase === 'merged' ? 'merged' : 'split';
    if (state.near) {
      view.load();
      for (const thumb of section.querySelectorAll('[data-thumb]')) thumb.style.backgroundImage = `url("${thumb.dataset.thumb}")`;
    }
    if (phase === 'split' && state.visible && !state.userPaused && !document.hidden) view.play();
    else view.pause();
    toggle.hidden = false;
    toggle.textContent = state.userPaused ? 'Play' : 'Pause';
  }

  function settle() {
    if (phase !== 'splitting') return;
    window.clearTimeout(timer);
    view.seek(0);
    phase = 'split';
    sync();
  }

  function scheduleSplit() {
    if (phase !== 'merged' || !state.inView) return;
    window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      phase = 'splitting';
      sync();
      timer = window.setTimeout(settle, SPLIT_FALLBACK_MS);
    }, SPLIT_HOLD_MS);
  }

  function merge() {
    window.clearTimeout(timer);
    if (!animates()) return;
    stage.dataset.snap = '';
    phase = 'merged';
    sync();
    void stage.offsetWidth;
    // Two frames so the snapped position is painted before transitions return.
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      delete stage.dataset.snap;
      scheduleSplit();
    }));
  }

  function show(id) {
    const clip = chapterClip(showcase, id);
    const time = layout.split ? 0 : view?.time ?? 0;
    const resolved = resolveFrames(clip, layout);
    const tiles = resolved.flatMap(({ output, inset }) => [output, inset]);
    view?.release();
    error.hidden = true;
    frames.forEach((frame, index) => {
      const { output, inset } = resolved[index];
      // Until the new poster decodes the canvases would still show the previous scene.
      for (const canvas of frame.canvases) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
      frame.root.style.setProperty('--tile-aspect', `${output.rect[2]} / ${output.rect[3]}`);
      frame.root.style.setProperty('--inset-aspect', `${inset.rect[2]} / ${inset.rect[3]}`);
      frame.labels[0].textContent = layout.blend ? `${output.label} + ${inset.label}` : output.label;
      frame.labels[1].textContent = inset.label;
    });
    stage.setAttribute('role', 'img');
    stage.setAttribute('aria-label', `${clip.tiles[0].scene}. ${resolved.map(({ output, inset }) => `${output.label} with ${inset.label} inset`).join('; ')}. Frame-synced.`);
    view = clip.pending
      ? createSnowView(frames.flatMap((frame) => frame.canvases), chapter.clips.indexOf(id) + 1)
      : createTileView(clip, (media, rectOf) => paint(media, rectOf, { tiles }), { preload: 'metadata', onError: () => { error.hidden = false; } });
    view.seek(time);
    label.textContent = clip.tiles[0].scene;
    stage.hidden = false;
    poster.hidden = true;
    // Real output is disclosed once at the top of the page; only placeholders carry a badge.
    badge.textContent = mediaKinds.placeholder;
    badge.hidden = !clip.pending;
    if (layout.split) merge();
    sync();
  }

  frames[0].root.addEventListener('transitionend', (event) => {
    if (event.target === frames[0].root && event.propertyName === 'translate') settle();
  });
  narrow.addEventListener('change', () => {
    if (!narrow.matches || phase === 'split') return;
    window.clearTimeout(timer);
    phase = 'split';
    sync();
  });
  toggle.addEventListener('click', () => {
    state.userPaused = !state.userPaused;
    sync();
  });
  document.addEventListener('visibilitychange', sync);
  if (observed) {
    new IntersectionObserver(([entry]) => { state.near ||= entry.isIntersecting; sync(); }, { rootMargin: '600px 0px' }).observe(section);
    new IntersectionObserver(([entry]) => {
      state.visible = entry.intersectionRatio >= 0.25;
      state.inView = entry.intersectionRatio >= 0.5;
      if (state.inView) scheduleSplit();
      else if (phase === 'merged') window.clearTimeout(timer);
      sync();
    }, { threshold: [0, 0.25, 0.5] }).observe(stage);
  }
  show(chapter.clips[0]);
  return { select: show };
}
