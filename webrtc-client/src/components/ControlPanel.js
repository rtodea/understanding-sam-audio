/**
 * Renders the prompt input, Start / Stop buttons, and status badge.
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

  /**
   * @param {HTMLElement}    root
   * @param {SessionState}   state
   */
  constructor(root, state) {
    this.#root  = root;
    this.#state = state;
    this.#render();
    this.#bind();
  }

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
        <div id="status-badge" class="status-badge status-idle">Idle</div>
        <div id="error-box"   class="error-box" hidden></div>
      </div>
    `;

    this.#startBtn    = this.#root.querySelector('#start-btn');
    this.#stopBtn     = this.#root.querySelector('#stop-btn');
    this.#promptInput = this.#root.querySelector('#description');
    this.#statusBadge = this.#root.querySelector('#status-badge');
    this.#errorBox    = this.#root.querySelector('#error-box');
  }

  #bind() {
    this.#promptInput.addEventListener('input', () => {
      const val = this.#promptInput.value.trim();
      this.#state.setDescription(val);
      this.#startBtn.disabled = val === '' || this.#state.status !== 'idle';
    });

    this.#startBtn.addEventListener('click', () => {
      this.#errorBox.hidden = true;
      this.#state.dispatchEvent(new CustomEvent('session:start'));
    });

    this.#stopBtn.addEventListener('click', () => {
      this.#state.dispatchEvent(new CustomEvent('session:stop'));
    });

    this.#state.addEventListener('statuschange', ({ detail: { status } }) => {
      this.#applyStatus(status);
    });

    this.#state.addEventListener('apperror', ({ detail: { error } }) => {
      this.#errorBox.textContent = error;
      this.#errorBox.hidden = false;
    });
  }

  #applyStatus(status) {
    const labels = {
      idle:       'Idle',
      connecting: 'Connecting…',
      active:     'Live',
      stopping:   'Stopping…',
    };
    this.#statusBadge.textContent = labels[status] ?? status;
    this.#statusBadge.className = `status-badge status-${status}`;

    const isIdle   = status === 'idle';
    const isActive = status === 'active';
    this.#startBtn.disabled = !isIdle || this.#promptInput.value.trim() === '';
    this.#stopBtn.disabled  = !isActive;
    this.#promptInput.disabled = !isIdle;
  }
}
