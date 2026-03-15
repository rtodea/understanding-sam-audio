/**
 * Renders the prompt input, Start / Stop buttons, replay section, and status badge.
 * Translates user actions into SessionState events — knows nothing about media or WebSockets.
 */
export class ControlPanel {
  #root;
  #state;
  #startBtn;
  #stopBtn;
  #promptInput;
  #statusBadge;
  #errorBox;
  #replayFileInput;
  #replayFilename;
  #replayBtn;
  #replayProgress;
  #replayFill;
  #replayTime;

  /**
   * @param {HTMLElement}  root
   * @param {SessionState} state
   */
  constructor(root, state) {
    this.#root  = root;
    this.#state = state;
    this.#render();
    this.#bind();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Update the replay progress bar. */
  setProgress(currentTime, duration) {
    const pct = duration > 0 ? (currentTime / duration) * 100 : 0;
    this.#replayFill.style.width = `${pct}%`;
    this.#replayTime.textContent = `${_fmt(currentTime)} / ${_fmt(duration)}`;
    this.#replayProgress.hidden  = false;
  }

  hideProgress() {
    this.#replayProgress.hidden = true;
    this.#replayFill.style.width = '0%';
  }

  // ── Rendering ──────────────────────────────────────────────────────────────

  #render() {
    this.#root.innerHTML = `
      <div class="control-panel">

        <div class="prompt-row">
          <label for="description">What to keep:</label>
          <input
            id="description"
            type="text"
            class="prompt-input"
            value="${this.#state.description}"
            placeholder="e.g. person speaking, guitar, thunder"
            autocomplete="off"
          >
        </div>

        <div class="button-row">
          <button id="start-btn" class="btn btn-start">Start Recording</button>
          <button id="stop-btn"  class="btn btn-stop" disabled>Stop</button>
        </div>

        <div class="panel-divider"></div>

        <div class="replay-row">
          <span class="replay-label">Replay golden file:</span>
          <label for="replay-file-input" class="btn btn-replay-pick">Choose file</label>
          <input
            type="file"
            id="replay-file-input"
            class="sr-only"
            accept="audio/*,video/*,.webm,.mp4,.ogv,.ogg"
          >
          <span class="replay-filename" id="replay-filename">No file selected</span>
          <button id="replay-btn" class="btn btn-replay" disabled>▶ Replay</button>
        </div>

        <div class="replay-progress" id="replay-progress" hidden>
          <div class="replay-bar">
            <div class="replay-fill" id="replay-fill"></div>
          </div>
          <span class="replay-time" id="replay-time">—</span>
        </div>

        <div id="status-badge" class="status-badge status-idle">Idle</div>
        <div id="error-box"   class="error-box" hidden></div>

      </div>
    `;

    this.#startBtn        = this.#root.querySelector('#start-btn');
    this.#stopBtn         = this.#root.querySelector('#stop-btn');
    this.#promptInput     = this.#root.querySelector('#description');
    this.#statusBadge     = this.#root.querySelector('#status-badge');
    this.#errorBox        = this.#root.querySelector('#error-box');
    this.#replayFileInput = this.#root.querySelector('#replay-file-input');
    this.#replayFilename  = this.#root.querySelector('#replay-filename');
    this.#replayBtn       = this.#root.querySelector('#replay-btn');
    this.#replayProgress  = this.#root.querySelector('#replay-progress');
    this.#replayFill      = this.#root.querySelector('#replay-fill');
    this.#replayTime      = this.#root.querySelector('#replay-time');
  }

  #bind() {
    this.#promptInput.addEventListener('input', () => {
      this.#syncButtons();
    });

    this.#startBtn.addEventListener('click', () => {
      this.#errorBox.hidden = true;
      this.#state.dispatchEvent(new CustomEvent('session:start'));
    });

    this.#stopBtn.addEventListener('click', () => {
      this.#state.dispatchEvent(new CustomEvent('session:stop'));
    });

    this.#replayFileInput.addEventListener('change', () => {
      const file = this.#replayFileInput.files[0];
      this.#replayFilename.textContent = file ? file.name : 'No file selected';
      this.#syncButtons();
    });

    this.#replayBtn.addEventListener('click', () => {
      const file = this.#replayFileInput.files[0];
      if (!file) return;
      this.#errorBox.hidden = true;
      this.#state.dispatchEvent(new CustomEvent('session:replay', { detail: { file } }));
    });

    this.#state.addEventListener('statuschange', ({ detail: { status } }) => {
      this.#applyStatus(status);
    });

    this.#state.addEventListener('apperror', ({ detail: { error } }) => {
      this.#errorBox.textContent = error;
      this.#errorBox.hidden = false;
    });
  }

  #syncButtons() {
    const isIdle      = this.#state.status === 'idle';
    const hasPrompt   = this.#promptInput.value.trim() !== '';
    const hasFile     = !!this.#replayFileInput.files[0];
    this.#startBtn.disabled  = !isIdle || !hasPrompt;
    this.#replayBtn.disabled = !isIdle || !hasFile || !hasPrompt;
  }

  #applyStatus(status) {
    const isReplay = this.#state.replayMode;
    const labels = {
      idle:       'Idle',
      connecting: 'Connecting…',
      active:     isReplay ? 'Replaying…' : 'Live',
      stopping:   'Stopping…',
    };
    this.#statusBadge.textContent = labels[status] ?? status;
    this.#statusBadge.className   = `status-badge status-${status}`;

    const isIdle   = status === 'idle';
    const isActive = status === 'active';
    const hasPrompt = this.#promptInput.value.trim() !== '';
    const hasFile   = !!this.#replayFileInput.files[0];

    this.#startBtn.disabled        = !isIdle || !hasPrompt;
    this.#stopBtn.disabled         = !isActive;
    this.#replayBtn.disabled       = !isIdle || !hasFile || !hasPrompt;
    this.#replayFileInput.disabled = !isIdle;
    this.#promptInput.disabled     = !isIdle;

    if (isIdle) this.hideProgress();
  }
}

function _fmt(s) {
  const m  = Math.floor(s / 60);
  const ss = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${ss}`;
}
