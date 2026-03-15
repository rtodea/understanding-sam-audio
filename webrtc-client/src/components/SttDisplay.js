/**
 * SttDisplay — timestamps, delta badges, and chronological timeline.
 *
 * Layout:
 *   ┌─ Live ──────────────────────────────────────────────┐
 *   │  RAW  "Hello how are—"          (current partial)   │
 *   │  SEP  "Hello, how are you—"     (current partial)   │
 *   ├─ Timeline ──────────────────────────────────────────┤
 *   │  +1.2s  RAW  "Hello."                               │
 *   │  +3.8s  RAW  "How are you today?"                   │
 *   │  +7.4s  SEP  "Hello, how are you today."  △ +3.6s  │
 *   └─────────────────────────────────────────────────────┘
 *
 * Delta (△) on SEP events = time elapsed since the most recent RAW
 * recognized event — a direct measure of SAM inference latency overhead.
 *
 * Public API (unchanged from previous version):
 *   update(stream, type, text)
 *   reset()
 */
export class SttDisplay {
  #root;
  #wrap;
  #liveEls   = {};   // { raw: el, separated: el }
  #panelEls  = {};   // { raw: el, separated: el }
  #timelineEl = null;

  #sessionStart = null;
  #partials     = { raw: null, separated: null };
  #recognized   = { raw: '',   separated: '' };
  #lastRecognizedAt = { raw: null, separated: null };
  #entries      = [];   // { stream, text, relSec, delta } — for download

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#build();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * @param {'raw'|'separated'} stream
   * @param {'recognizing'|'recognized'} type
   * @param {string} text
   */
  update(stream, type, text) {
    const now = performance.now();
    if (this.#sessionStart === null) this.#sessionStart = now;
    const relSec = (now - this.#sessionStart) / 1000;

    this.#wrap.hidden = false;

    if (type === 'recognizing') {
      this.#partials[stream] = text;
      this.#renderLive();
      this.#renderPanel(stream);
    } else {
      // Compute delta for separated events.
      let delta = null;
      if (stream === 'separated' && this.#lastRecognizedAt.raw !== null) {
        delta = relSec - this.#lastRecognizedAt.raw;
      }
      this.#lastRecognizedAt[stream] = relSec;
      this.#recognized[stream] = text;
      this.#partials[stream]   = null;
      this.#entries.push({ stream, text, relSec, delta });
      this.#root.querySelector('#stt-dl-btns').hidden = false;
      this.#renderLive();
      this.#renderPanel(stream);
      this.#appendTimelineEntry(stream, text, relSec, delta);
    }
  }

  reset() {
    this.#sessionStart = null;
    this.#partials     = { raw: null, separated: null };
    this.#recognized   = { raw: '',   separated: '' };
    this.#lastRecognizedAt = { raw: null, separated: null };
    this.#entries      = [];
    for (const el of Object.values(this.#liveEls))  el.innerHTML = '';
    for (const el of Object.values(this.#panelEls)) el.innerHTML = '';
    this.#timelineEl.innerHTML = '';
    this.#root.querySelector('#stt-dl-btns').hidden = true;
    this.#wrap.hidden = true;
  }

  // ── DOM construction ───────────────────────────────────────────────────────

  #build() {
    this.#root.innerHTML = `
      <div class="stt-wrap" hidden>

        <div class="stt-header">
          <span class="stt-title">Speech recognition</span>
          <div class="stt-dl-btns" id="stt-dl-btns" hidden>
            <button class="stt-dl-btn" id="stt-dl-txt">↓ TXT</button>
            <button class="stt-dl-btn" id="stt-dl-csv">↓ CSV</button>
          </div>
        </div>

        <div class="stt-panels">
          <div class="stt-panel">
            <div class="stt-panel-header">
              <span class="stt-badge stt-badge-raw">RAW</span>
              <span class="stt-hint">direct to Azure</span>
            </div>
            <div class="stt-panel-body" id="stt-panel-raw"></div>
          </div>
          <div class="stt-panel">
            <div class="stt-panel-header">
              <span class="stt-badge stt-badge-separated">SEP</span>
              <span class="stt-hint">post-SAM inference</span>
            </div>
            <div class="stt-panel-body" id="stt-panel-separated"></div>
          </div>
        </div>

        <div class="stt-live">
          <div class="stt-live-row" id="stt-live-raw"></div>
          <div class="stt-live-row" id="stt-live-separated"></div>
        </div>

        <div class="stt-timeline" id="stt-timeline"></div>

      </div>`;

    this.#wrap       = this.#root.querySelector('.stt-wrap');
    this.#timelineEl = this.#root.querySelector('#stt-timeline');
    this.#liveEls.raw       = this.#root.querySelector('#stt-live-raw');
    this.#liveEls.separated = this.#root.querySelector('#stt-live-separated');
    this.#panelEls = {
      raw:       this.#root.querySelector('#stt-panel-raw'),
      separated: this.#root.querySelector('#stt-panel-separated'),
    };

    this.#root.querySelector('#stt-dl-txt').addEventListener('click', () => this.#downloadTxt());
    this.#root.querySelector('#stt-dl-csv').addEventListener('click', () => this.#downloadCsv());
  }

  // ── Rendering ──────────────────────────────────────────────────────────────

  #renderLive() {
    for (const stream of ['raw', 'separated']) {
      const text = this.#partials[stream];
      const el   = this.#liveEls[stream];
      if (text) {
        el.innerHTML =
          `${_badge(stream)}` +
          `<span class="stt-partial-text">${_esc(text)}</span>`;
      } else {
        el.innerHTML = '';
      }
    }
  }

  #renderPanel(stream) {
    const el         = this.#panelEls[stream];
    const recognized = this.#recognized[stream];
    const partial    = this.#partials[stream];
    let html = '';
    if (recognized) html += `<span class="stt-recognized">${_esc(recognized)}</span>`;
    if (partial)    html += `${recognized ? ' ' : ''}<span class="stt-partial">${_esc(partial)}</span>`;
    el.innerHTML = html || '<span class="stt-empty">—</span>';
  }

  #downloadTxt() {
    const lines = [
      `SAM-Audio STT Log`,
      `Generated: ${new Date().toISOString()}`,
      ``,
      `${'TIME'.padEnd(8)}  ${'STREAM'.padEnd(10)}  ${'DELTA'.padEnd(10)}  TEXT`,
      `${'─'.repeat(70)}`,
      ...this.#entries.map(({ stream, text, relSec, delta }) => {
        const t = `+${relSec.toFixed(1)}s`.padEnd(8);
        const s = (stream === 'raw' ? '[RAW]' : '[SEP]').padEnd(10);
        const d = (delta !== null ? `△ +${delta.toFixed(1)}s` : '').padEnd(10);
        return `${t}  ${s}  ${d}  ${text}`;
      }),
    ];
    _triggerDownload(lines.join('\n'), 'text/plain', 'stt-log', 'txt');
  }

  #downloadCsv() {
    const rows = [
      ['time_sec', 'stream', 'delta_sec', 'text'],
      ...this.#entries.map(({ stream, text, relSec, delta }) => [
        relSec.toFixed(2),
        stream,
        delta !== null ? delta.toFixed(2) : '',
        `"${text.replace(/"/g, '""')}"`,
      ]),
    ];
    _triggerDownload(rows.map(r => r.join(',')).join('\n'), 'text/csv', 'stt-log', 'csv');
  }

  #appendTimelineEntry(stream, text, relSec, delta) {
    const row = document.createElement('div');
    row.className = 'stt-row';

    const timeStr  = `+${relSec.toFixed(1)}s`;
    const deltaHtml = delta !== null ? _deltaHtml(delta) : '';

    row.innerHTML =
      `<span class="stt-ts">${_esc(timeStr)}</span>` +
      `${_badge(stream)}` +
      `<span class="stt-text">${_esc(text)}</span>` +
      deltaHtml;

    this.#timelineEl.appendChild(row);
    this.#timelineEl.scrollTop = this.#timelineEl.scrollHeight;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _badge(stream) {
  const label = stream === 'raw' ? 'RAW' : 'SEP';
  return `<span class="stt-badge stt-badge-${stream}">${label}</span>`;
}

function _deltaHtml(delta) {
  const cls =
    delta < 3 ? 'delta-fast' :
    delta < 6 ? 'delta-mid'  : 'delta-slow';
  return `<span class="stt-delta ${cls}">△ +${delta.toFixed(1)}s</span>`;
}

function _esc(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function _triggerDownload(content, mimeType, stem, ext) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `${stem}-${new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-')}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}
