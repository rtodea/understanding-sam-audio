/**
 * Thin wrapper around the native WebSocket API.
 * Emits DOM events so consumers can use addEventListener without coupling to this class.
 *
 * @fires WebSocketService#open
 * @fires WebSocketService#close
 * @fires WebSocketService#error
 * @fires WebSocketService#message  — detail is raw ArrayBuffer (binary) or string
 */
export class WebSocketService extends EventTarget {
  #ws = null;

  /**
   * @param {string} url  ws:// or wss:// URL
   */
  connect(url) {
    this.#ws = new WebSocket(url);
    this.#ws.binaryType = 'arraybuffer';

    this.#ws.onopen    = ()  => this.dispatchEvent(new Event('open'));
    this.#ws.onclose   = ()  => this.dispatchEvent(new Event('close'));
    this.#ws.onerror   = (e) => this.dispatchEvent(new CustomEvent('error', { detail: e }));
    this.#ws.onmessage = (e) => this.dispatchEvent(
      new MessageEvent('message', { data: e.data })
    );
  }

  /** @param {object} obj */
  sendJson(obj) {
    if (this.#ws?.readyState === WebSocket.OPEN) {
      this.#ws.send(JSON.stringify(obj));
    }
  }

  /** @param {ArrayBuffer} buffer */
  sendBinary(buffer) {
    if (this.#ws?.readyState === WebSocket.OPEN) {
      this.#ws.send(buffer);
    }
  }

  disconnect() {
    this.#ws?.close();
    this.#ws = null;
  }
}
