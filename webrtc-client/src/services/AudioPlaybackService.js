/**
 * Schedules decoded WAV chunks for gapless playback via Web Audio API.
 * Also accumulates decoded PCM data for final WAV download.
 */
export class AudioPlaybackService {
  #ctx = null;
  #analyser = null;
  #nextPlayTime = 0;
  #pendingDecode = Promise.resolve();

  /** @type {Float32Array[]} */
  #decodedChunks = [];

  /**
   * Must be called once (or after teardown) before enqueue().
   * Deferred from the constructor so AudioContext creation is in response to a user gesture.
   */
  init() {
    this.#ctx = new AudioContext({ sampleRate: 48000 });
    this.#analyser = this.#ctx.createAnalyser();
    this.#analyser.fftSize = 256;
    this.#analyser.connect(this.#ctx.destination);
    this.#nextPlayTime = 0;
    this.#decodedChunks = [];
  }

  /**
   * Decode an incoming WAV ArrayBuffer, schedule it for playback,
   * and store the PCM data for download.
   *
   * decodeAudioData() transfers (detaches) the ArrayBuffer it receives,
   * so we do not store the raw bytes — we store the decoded Float32Array instead.
   *
   * @param {ArrayBuffer} arrayBuffer
   */
  async enqueue(arrayBuffer) {
    this.#pendingDecode = this.#pendingDecode.then(async () => {
      const audioBuffer = await this.#ctx.decodeAudioData(arrayBuffer);

      // Keep a copy of the PCM for the download encoder.
      this.#decodedChunks.push(audioBuffer.getChannelData(0).slice());

      // Schedule playback with a small look-ahead to avoid gaps.
      const source = this.#ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.#analyser);

      const now = this.#ctx.currentTime;
      if (this.#nextPlayTime < now) {
        this.#nextPlayTime = now + 0.05;
      }
      source.start(this.#nextPlayTime);
      this.#nextPlayTime += audioBuffer.duration;
    });

    return this.#pendingDecode;
  }

  async drain() {
    await this.#pendingDecode;
  }

  /** @returns {AnalyserNode} */
  getAnalyserNode() { return this.#analyser; }

  /** @returns {Float32Array[]}  Decoded PCM chunks for WAV encoding. */
  getDecodedChunks() { return this.#decodedChunks; }

  /** Reset accumulated chunks and playback cursor without closing the AudioContext. */
  reset() {
    this.#decodedChunks = [];
    this.#nextPlayTime = 0;
    this.#pendingDecode = Promise.resolve();
  }

  teardown() {
    this.#ctx?.close();
    this.#ctx = null;
    this.#analyser = null;
  }
}
