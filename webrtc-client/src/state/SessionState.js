/**
 * Single source of truth for the recording session.
 * Components listen via addEventListener / dispatchEvent — no third-party bus needed.
 *
 * @fires SessionState#statuschange
 * @fires SessionState#error
 */
export class SessionState extends EventTarget {
  /** @type {'idle'|'connecting'|'active'|'stopping'} */
  #status = 'idle';
  #description = 'person speaking';
  #error = null;

  get status()      { return this.#status; }
  get description() { return this.#description; }
  get error()       { return this.#error; }

  setStatus(status) {
    this.#status = status;
    this.dispatchEvent(new CustomEvent('statuschange', { detail: { status } }));
  }

  setDescription(description) {
    this.#description = description;
  }

  setError(error) {
    this.#error = error;
    this.dispatchEvent(new CustomEvent('apperror', { detail: { error } }));
  }
}
