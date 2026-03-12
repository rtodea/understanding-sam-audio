/**
 * Canvas-based frequency visualizer driven by a Web Audio AnalyserNode.
 * Lifecycle: attach() to start drawing, detach() to stop and clear.
 */
export class AudioVisualizer {
  #root;
  #canvas;
  #ctx2d;
  #analyser = null;
  #dataArray = null;
  #animationId = null;

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `<canvas class="visualizer" width="600" height="80"></canvas>`;
    this.#canvas = this.#root.querySelector('canvas');
    this.#ctx2d = this.#canvas.getContext('2d');
    this.#drawIdle();
  }

  /** @param {AnalyserNode} analyserNode */
  attach(analyserNode) {
    this.#analyser = analyserNode;
    this.#dataArray = new Uint8Array(this.#analyser.frequencyBinCount);
    this.#animationId = requestAnimationFrame(() => this.#drawLoop());
  }

  #drawLoop() {
    this.#animationId = requestAnimationFrame(() => this.#drawLoop());
    this.#analyser.getByteFrequencyData(this.#dataArray);

    const { width, height } = this.#canvas;
    const ctx = this.#ctx2d;
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    const barWidth = width / this.#dataArray.length;
    this.#dataArray.forEach((value, i) => {
      const barHeight = (value / 255) * height;
      const hue = 195 + (i / this.#dataArray.length) * 55;
      ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
      ctx.fillRect(i * barWidth, height - barHeight, Math.max(barWidth - 1, 1), barHeight);
    });
  }

  #drawIdle() {
    const { width, height } = this.#canvas;
    this.#ctx2d.fillStyle = '#0f172a';
    this.#ctx2d.fillRect(0, 0, width, height);
  }

  detach() {
    if (this.#animationId !== null) {
      cancelAnimationFrame(this.#animationId);
      this.#animationId = null;
    }
    this.#analyser = null;
    this.#drawIdle();
  }
}
