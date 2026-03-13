/**
 * Wraps MediaRecorder to emit audio chunks as ArrayBuffers.
 *
 * @fires AudioRecorderService#chunk  — data is ArrayBuffer
 */
export class AudioRecorderService extends EventTarget {
  static #MIME = 'audio/webm;codecs=opus';
  #recorder = null;

  /**
   * Returns true if the current browser supports the required codec.
   * Safari does not — show a user-facing error before starting.
   * @returns {boolean}
   */
  static isSupported() {
    return typeof MediaRecorder !== 'undefined' &&
      MediaRecorder.isTypeSupported(AudioRecorderService.#MIME);
  }

  /**
   * @param {MediaStream} stream
   * @param {number} [chunkMs=3000]  How often to fire a chunk event.
   */
  start(stream, chunkMs = 3000) {
    this.#recorder = new MediaRecorder(stream, { mimeType: AudioRecorderService.#MIME });

    this.#recorder.ondataavailable = async (e) => {
      if (e.data.size > 0) {
        const buffer = await e.data.arrayBuffer();
        this.dispatchEvent(new CustomEvent('chunk', { detail: buffer }));
      }
    };

    this.#recorder.start(chunkMs);
  }

  stop() {
    this.#recorder?.stop();
    this.#recorder = null;
  }
}
