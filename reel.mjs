import { coverRect, createTileView, fitCanvas, sliderKey, wrapIndex } from './showcase.mjs?rev=showcase-7';

const SWEEP_PERIOD = 10;
const SWEEP_CENTER = 0.5;
const SWEEP_REACH = 0.34;
const IDLE_MS = 3500;
const FADE_MS = 900;

export function reelPlayback({ ready, failed, userPaused, reducedMotion, hidden, visible }) {
  if (!ready || failed || reducedMotion) return 'static';
  if (hidden || !visible) return 'suspended';
  return userPaused ? 'paused' : 'playing';
}

/** Auto-sweep position of the proxy reveal line (0 = all output, 1 = all proxy). */
export function sweepPosition(seconds) {
  return SWEEP_CENTER - SWEEP_REACH * Math.cos(2 * Math.PI * seconds / SWEEP_PERIOD);
}

/** Sweep clock time at which the line sits at `value`, on the rising half, so a resumed sweep starts where the user left it. */
export function sweepSeconds(value) {
  const cos = Math.max(-1, Math.min(1, (SWEEP_CENTER - value) / SWEEP_REACH));
  return Math.acos(cos) * SWEEP_PERIOD / (2 * Math.PI);
}

export function initReel(root, showcase, ids, reducedMotion) {
  const clips = ids.map((id) => showcase.clips[id]).filter(Boolean);
  if (!clips.length) return;
  const media = root.querySelector('[data-reel-media]');
  const layers = [...root.querySelectorAll('[data-reel-canvas]')].map((canvas) => ({ canvas, view: null, index: -1 }));
  const split = root.querySelector('[data-reel-split]');
  const handle = root.querySelector('[role="slider"]');
  const ticks = root.querySelector('[data-reel-ticks]');
  const toggle = root.querySelector('[data-reel-toggle]');
  const error = root.querySelector('[data-reel-error]');
  const state = { index: 0, front: 0, position: 0.5, auto: !reducedMotion.matches, userPaused: false, visible: true, failed: false };
  let sweepClock = sweepSeconds(state.position);
  let frame = 0;
  let lastNow = null;
  let idle = 0;
  let fading = 0;

  for (const label of root.querySelectorAll('[data-reel-stream]')) label.textContent = showcase.streams[label.dataset.reelStream];

  const playback = () => reelPlayback({ ready: true, failed: state.failed, userPaused: state.userPaused,
    reducedMotion: reducedMotion.matches, hidden: document.hidden, visible: state.visible });

  const paint = (layer) => (source, rectOf, clip) => {
    const context = fitCanvas(layer.canvas);
    const { width, height } = layer.canvas;
    const output = coverRect(rectOf(clip.tiles.find((tile) => tile.stream === 'sft')), width, height);
    context.drawImage(source, ...output, 0, 0, width, height);
    const edge = Math.round(state.position * width);
    if (edge <= 0) return;
    const [sx, sy, sw, sh] = coverRect(rectOf(clip.tiles.find((tile) => tile.stream === 'proxy')), width, height);
    context.drawImage(source, sx, sy, sw * edge / width, sh, 0, 0, edge, height);
  };

  function mount(layer, index, preload) {
    layer.view?.release();
    layer.index = index;
    layer.view = createTileView(clips[index], paint(layer), { preload, loop: false, autoDraw: false, onError: fail });
    if (preload !== 'none') layer.view.load();
    layer.view.video.addEventListener('ended', () => { if (layer === layers[state.front]) show(wrapIndex(state.index, 1, clips.length)); });
  }

  function render() {
    split.style.setProperty('--split', `${state.position * 100}%`);
    handle.setAttribute('aria-valuenow', String(Math.round(state.position * 100)));
    handle.setAttribute('aria-valuetext', `${Math.round(state.position * 100)}% proxy input revealed`);
    const front = layers[state.front];
    if (fading) layers[1 - state.front].view?.draw();
    front.view?.draw();
    const video = front.view?.video;
    if (video?.duration) ticks.style.setProperty('--progress', String(video.currentTime / video.duration));
  }

  function loop(now) {
    frame = 0;
    if (playback() !== 'playing') return;
    if (state.auto && lastNow !== null) {
      sweepClock += (now - lastNow) / 1000;
      state.position = sweepPosition(sweepClock);
    }
    lastNow = now;
    render();
    frame = window.requestAnimationFrame(loop);
  }

  function sync() {
    const mode = playback();
    root.dataset.playback = mode;
    toggle.hidden = mode === 'static';
    toggle.textContent = state.userPaused ? 'Play reel' : 'Pause reel';
    const front = layers[state.front].view;
    if (mode === 'playing') {
      front.play();
      if (!frame) frame = window.requestAnimationFrame(loop);
    } else {
      layers.forEach((layer) => layer.view?.pause());
      if (frame) window.cancelAnimationFrame(frame);
      frame = 0;
      lastNow = null;
      render();
    }
  }

  function updateMeta() {
    for (const [index, button] of [...ticks.children].entries()) {
      if (index === state.index) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    }
    ticks.style.setProperty('--progress', '0');
  }

  function show(index) {
    if (index === state.index && layers[state.front].view) return;
    state.index = index;
    updateMeta();
    if (playback() === 'static') {
      mount(layers[state.front], index, 'none');
      render();
      return;
    }
    const back = layers[1 - state.front];
    if (back.index !== index || !back.view) mount(back, index, 'auto');
    back.view.seek(0);
    state.front = 1 - state.front;
    media.dataset.front = String(state.front);
    window.clearTimeout(fading);
    fading = window.setTimeout(() => {
      fading = 0;
      mount(layers[1 - state.front], wrapIndex(state.index, 1, clips.length), 'auto');
    }, FADE_MS);
    sync();
  }

  function fail() {
    if (state.failed) return;
    state.failed = true;
    error.hidden = false;
    sync();
  }

  function setPosition(value) {
    state.position = value;
    state.auto = false;
    window.clearTimeout(idle);
    if (!reducedMotion.matches) {
      idle = window.setTimeout(() => {
        sweepClock = sweepSeconds(state.position);
        state.auto = true;
      }, IDLE_MS);
    }
    render();
  }

  clips.forEach((clip, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'reel-tick';
    button.setAttribute('aria-label', `Show ${clip.tiles[0].scene}`);
    button.addEventListener('click', () => show(index));
    ticks.append(button);
  });
  ticks.hidden = false;

  const pointerPosition = (event) => {
    const bounds = media.getBoundingClientRect();
    return Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
  };
  media.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;
    media.setPointerCapture(event.pointerId);
    setPosition(pointerPosition(event));
  });
  media.addEventListener('pointermove', (event) => { if (media.hasPointerCapture(event.pointerId)) setPosition(pointerPosition(event)); });
  handle.addEventListener('keydown', (event) => {
    const next = sliderKey(state.position, event.key);
    if (next === null) return;
    event.preventDefault();
    setPosition(next);
  });
  toggle.addEventListener('click', () => {
    state.userPaused = !state.userPaused;
    sync();
  });
  document.addEventListener('visibilitychange', sync);
  reducedMotion.addEventListener('change', () => {
    state.auto = !reducedMotion.matches;
    if (reducedMotion.matches) state.position = 0.5;
    sync();
  });
  if (typeof IntersectionObserver === 'function') {
    new IntersectionObserver(([entry]) => { state.visible = entry.isIntersecting; sync(); }).observe(media);
  }

  mount(layers[0], 0, reducedMotion.matches ? 'none' : 'auto');
  layers[0].view.video.addEventListener('playing', () => {
    if (layers[1].index < 0 && clips.length > 1) mount(layers[1], 1, 'auto');
  }, { once: true });
  media.dataset.front = '0';
  split.hidden = false;
  root.classList.add('has-reel');
  updateMeta();
  sync();
}
