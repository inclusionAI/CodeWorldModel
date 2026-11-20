// TV snow for frames whose model media is not ready. Same interface as a tile view, so a player
// swaps in the real clip without changing its own logic.
const WIDTH = 168;
const HEIGHT = 96;
const FRAME_MS = 83;

export function createSnowView(canvases, seed = 1) {
  let random = seed >>> 0 || 1;
  let timer = 0;
  let released = false;
  const targets = canvases.map((canvas) => {
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.dataset.snow = '';
    const context = canvas.getContext('2d', { alpha: false });
    return { context, pixels: context?.createImageData(WIDTH, HEIGHT) };
  });

  function draw() {
    if (released) return false;
    for (const { context, pixels } of targets) {
      if (!context) continue;
      const data = pixels.data;
      for (let index = 0; index < data.length; index += 4) {
        random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
        const value = 40 + (random >>> 25);
        data[index] = value;
        data[index + 1] = value;
        data[index + 2] = value + 6;
        data[index + 3] = 255;
      }
      context.putImageData(pixels, 0, 0);
    }
    return true;
  }
  function stop() {
    window.clearTimeout(timer);
    timer = 0;
  }
  function loop() {
    draw();
    timer = window.setTimeout(loop, FRAME_MS);
  }

  draw();
  return {
    draw,
    load() {},
    play() {
      if (!timer && !released) loop();
      return Promise.resolve();
    },
    pause: stop,
    get time() { return 0; },
    seek() {},
    release() {
      released = true;
      stop();
      for (const canvas of canvases) delete canvas.dataset.snow;
    },
  };
}
