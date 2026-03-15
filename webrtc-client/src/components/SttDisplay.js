/**
 * Displays two side-by-side speech-to-text streams:
 *   "raw"       — mic audio fed directly to Azure (near real-time)
 *   "separated" — SAM-separated audio fed to Azure (post-inference latency)
 *
 * Call update(stream, type, text) on each incoming "stt" WebSocket event.
 * Call reset() at session start.
 */
export class SttDisplay {
  #root;
  #panel;
  #panels = {};   // { raw: { textEl, partial, recognized }, separated: {...} }

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `
      <div class="stt-display" hidden>
        <div class="stt-panel" id="stt-panel-raw">
          <div class="stt-panel-header">
            <span class="stt-stream-label">Raw mic</span>
            <span class="stt-hint">direct to Azure</span>
          </div>
          <div class="stt-text" id="stt-text-raw"></div>
        </div>
        <div class="stt-panel" id="stt-panel-separated">
          <div class="stt-panel-header">
            <span class="stt-stream-label">Separated</span>
            <span class="stt-hint">post-SAM inference</span>
          </div>
          <div class="stt-text" id="stt-text-separated"></div>
        </div>
      </div>`;

    this.#panel = this.#root.querySelector('.stt-display');

    for (const stream of ['raw', 'separated']) {
      this.#panels[stream] = {
        textEl:     this.#root.querySelector(`#stt-text-${stream}`),
        recognized: '',
        partial:    '',
      };
    }
  }

  /**
   * @param {'raw'|'separated'} stream
   * @param {'recognizing'|'recognized'} type
   * @param {string} text
   */
  update(stream, type, text) {
    const p = this.#panels[stream];
    if (!p) return;

    if (type === 'recognized') {
      p.recognized = text;
      p.partial    = '';
    } else {
      p.partial = text;
    }

    this.#panel.hidden = false;
    this.#renderPanel(stream);
  }

  reset() {
    for (const p of Object.values(this.#panels)) {
      p.recognized = '';
      p.partial    = '';
      p.textEl.innerHTML = '';
    }
    this.#panel.hidden = true;
  }

  #renderPanel(stream) {
    const { textEl, recognized, partial } = this.#panels[stream];

    let html = '';
    if (recognized) {
      html += `<span class="stt-recognized">${_esc(recognized)}</span>`;
    }
    if (partial) {
      if (html) html += ' ';
      html += `<span class="stt-partial">${_esc(partial)}</span>`;
    }
    textEl.innerHTML = html || '<span class="stt-empty">—</span>';
  }
}

function _esc(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
