/**
 * "Play it live": one recorded session, output full frame with the proxy and controller baked into the corners.
 * The clip is stored as fragmented-MP4 parts under 8 MB (some Git hosts reject files over 10 MiB), listed in an index.
 * Nothing is fetched until the section scrolls into view (muted autoplay) or the user presses play; the parts are then
 * appended in order to a MediaSource, so playback starts with the first part. Without MediaSource (iPhone Safari) the
 * parts are joined into one Blob first. It pauses again when scrolled away.
 */
export async function liveSource(index, fetchBytes, MediaSourceType = globalThis.MediaSource) {
  const base = new URL('.', index.url);
  const urls = index.parts.map((part) => new URL(part.src, base).href);
  if (!MediaSourceType?.isTypeSupported?.(index.type)) {
    const parts = await Promise.all(urls.map(fetchBytes));
    return { url: URL.createObjectURL(new Blob(parts, { type: 'video/mp4' })), done: Promise.resolve() };
  }
  const source = new MediaSourceType();
  const url = URL.createObjectURL(source);
  const done = new Promise((resolve, reject) => {
    source.addEventListener('sourceopen', async () => {
      try {
        source.duration = index.duration;
        const buffer = source.addSourceBuffer(index.type);
        let next = fetchBytes(urls[0]);
        for (let i = 0; i < urls.length; i += 1) {
          const bytes = await next;
          if (i + 1 < urls.length) next = fetchBytes(urls[i + 1]);
          await new Promise((appended, failed) => {
            buffer.addEventListener('updateend', appended, { once: true });
            buffer.addEventListener('error', failed, { once: true });
            buffer.appendBuffer(bytes);
          });
        }
        source.endOfStream();
        resolve();
      } catch (error) {
        reject(error);
      }
    }, { once: true });
  });
  return { url, done };
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.arrayBuffer();
}

export function initLive(section, session, reducedMotion) {
  const frame = section.querySelector('[data-live-frame]');
  const poster = frame.querySelector('[data-live-poster]');
  const error = section.querySelector('[data-live-error]');
  const video = document.createElement('video');
  Object.assign(video, { muted: true, defaultMuted: true, playsInline: true, controls: true, preload: 'none', poster: session.poster });
  video.className = 'live-video';
  video.setAttribute('aria-label', session.label);
  video.width = session.width;
  video.height = session.height;
  const state = { visible: false, userPaused: reducedMotion.matches, loading: null };

  function fail() {
    video.remove();
    poster.hidden = false;
    error.hidden = false;
  }

  function load() {
    state.loading ??= fetch(session.segments)
      .then((response) => {
        if (!response.ok) throw new Error(`${session.segments}: ${response.status}`);
        return response.json();
      })
      .then((index) => liveSource({ ...index, url: new URL(session.segments, document.baseURI).href }, fetchBytes))
      .then(({ url, done }) => {
        video.src = url;
        sync();
        state.requested = false;
        return done;
      })
      .catch(fail);
    return state.loading;
  }

  video.addEventListener('error', fail);
  // Native controls: a pause while on screen is the user's choice and sticks until they play again.
  video.addEventListener('pause', () => { if (state.visible && !document.hidden && !video.ended) state.userPaused = true; });
  video.addEventListener('play', () => { state.userPaused = false; load(); });
  // Before the parts load the video has no src, so its play control does nothing; a press on it asks to play.
  function request(event) {
    if (video.src || (event.type === 'keydown' && event.key !== ' ' && event.key !== 'Enter')) return;
    state.userPaused = false;
    state.requested = true;
    load();
  }
  video.addEventListener('pointerdown', request);
  video.addEventListener('keydown', request);

  function sync() {
    if ((state.visible || state.requested) && !state.userPaused && !document.hidden) {
      load();
      if (video.src) video.play().catch(() => {});
    } else video.pause();
  }

  poster.hidden = true;
  frame.append(video);
  document.addEventListener('visibilitychange', sync);
  if (typeof IntersectionObserver === 'function') {
    new IntersectionObserver(([entry]) => {
      state.visible = entry.intersectionRatio >= 0.5;
      sync();
    }, { threshold: [0, 0.5] }).observe(frame);
  }
}
