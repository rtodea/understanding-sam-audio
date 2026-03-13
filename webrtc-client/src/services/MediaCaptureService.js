/**
 * Owns getUserMedia lifecycle.
 * Single responsibility: acquire and release the A/V stream.
 */
export class MediaCaptureService {
  #stream = null;

  /**
   * @returns {Promise<MediaStream>}
   */
  async start() {
    this.#stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: true,
    });
    return this.#stream;
  }

  stop() {
    this.#stream?.getTracks().forEach(t => t.stop());
    this.#stream = null;
  }

  get stream() { return this.#stream; }
}
