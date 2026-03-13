/**
 * Captures the original camera+microphone stream for later download.
 */
export class OriginalRecordingService {
  static #MIME_CANDIDATES = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ];

  #recorder = null;
  #chunks = [];
  #mimeType = '';
  #stopPromise = null;

  static isSupported() {
    return typeof MediaRecorder !== 'undefined';
  }

  start(stream) {
    void this.stop(true);

    const mimeType = OriginalRecordingService.#MIME_CANDIDATES.find(
      (candidate) => MediaRecorder.isTypeSupported(candidate),
    ) ?? '';

    this.#mimeType = mimeType;
    this.#chunks = [];
    this.#recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    this.#recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.#chunks.push(event.data);
      }
    };

    this.#stopPromise = new Promise((resolve) => {
      this.#recorder.onstop = () => {
        const type = this.#mimeType || this.#chunks[0]?.type || 'video/webm';
        const blob = this.#chunks.length > 0
          ? new Blob(this.#chunks, { type })
          : null;
        resolve(blob);
      };
    });

    this.#recorder.start(1000);
  }

  async stop(force = false) {
    if (!this.#recorder) {
      return null;
    }

    const recorder = this.#recorder;
    const stopPromise = this.#stopPromise;
    this.#recorder = null;
    this.#stopPromise = null;

    if (recorder.state !== 'inactive') {
      recorder.stop();
      return stopPromise;
    }

    if (force) {
      const type = this.#mimeType || this.#chunks[0]?.type || 'video/webm';
      return this.#chunks.length > 0 ? new Blob(this.#chunks, { type }) : null;
    }

    return stopPromise;
  }
}
