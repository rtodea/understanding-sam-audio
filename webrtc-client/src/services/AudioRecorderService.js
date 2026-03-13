/**
 * Streams raw mono PCM chunks from Web Audio instead of MediaRecorder blobs.
 *
 * @fires AudioRecorderService#chunk  — detail is ArrayBuffer containing float32 PCM
 */
export class AudioRecorderService extends EventTarget {
  static #TARGET_SAMPLE_RATE = 48000;
  #context = null;
  #source = null;
  #processor = null;
  #sink = null;
  #flushTimer = null;
  #pendingBuffers = [];
  #pendingSamples = 0;
  #sampleRate = AudioRecorderService.#TARGET_SAMPLE_RATE;

  static isSupported() {
    return typeof window !== 'undefined' &&
      typeof (window.AudioContext || window.webkitAudioContext) !== 'undefined';
  }

  get sampleRate() {
    return this.#sampleRate;
  }

  /**
   * @param {MediaStream} stream
   * @param {number} [chunkMs=250]  How often to flush accumulated PCM.
   */
  start(stream, chunkMs = 250) {
    this.stop();

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    this.#context = new AudioContextCtor({
      sampleRate: AudioRecorderService.#TARGET_SAMPLE_RATE,
    });
    this.#sampleRate = this.#context.sampleRate;

    this.#source = this.#context.createMediaStreamSource(stream);
    this.#processor = this.#context.createScriptProcessor(4096, 1, 1);
    this.#sink = this.#context.createGain();
    this.#sink.gain.value = 0;

    this.#processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      if (input.length === 0) return;

      this.#pendingBuffers.push(input.slice());
      this.#pendingSamples += input.length;
    };

    this.#source.connect(this.#processor);
    // ScriptProcessor must stay connected to keep firing in Chrome.
    this.#processor.connect(this.#sink);
    this.#sink.connect(this.#context.destination);

    void this.#context.resume();
    this.#flushTimer = window.setInterval(
      () => this.#flush(),
      Math.max(50, chunkMs),
    );
  }

  stop() {
    this.#flush();

    if (this.#flushTimer !== null) {
      window.clearInterval(this.#flushTimer);
      this.#flushTimer = null;
    }

    this.#processor?.disconnect();
    this.#source?.disconnect();
    this.#sink?.disconnect();
    this.#processor = null;
    this.#source = null;
    this.#sink = null;

    if (this.#context) {
      void this.#context.close();
      this.#context = null;
    }

    this.#pendingBuffers = [];
    this.#pendingSamples = 0;
  }

  #flush() {
    if (this.#pendingSamples === 0) return;

    const chunk = new Float32Array(this.#pendingSamples);
    let offset = 0;
    for (const buffer of this.#pendingBuffers) {
      chunk.set(buffer, offset);
      offset += buffer.length;
    }

    this.#pendingBuffers = [];
    this.#pendingSamples = 0;
    this.dispatchEvent(new CustomEvent('chunk', { detail: chunk.buffer }));
  }
}
