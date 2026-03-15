/**
 * Decodes an audio/video file and replays it as a stream of PCM chunks,
 * matching the interface of AudioRecorderService so the same WebSocket
 * pipeline can consume it unchanged.
 *
 * @fires AudioReplayService#chunk  — detail is ArrayBuffer (float32 PCM, 48 kHz mono)
 * @fires AudioReplayService#end    — fired when all samples have been emitted
 */
export class AudioReplayService extends EventTarget {
  static #TARGET_SR = 48_000;

  #pcm      = null;   // Float32Array — full decoded mono at TARGET_SR
  #duration = 0;
  #offset   = 0;
  #timer    = null;

  get sampleRate()  { return AudioReplayService.#TARGET_SR; }
  get duration()    { return this.#duration; }
  get currentTime() { return this.#pcm ? this.#offset / AudioReplayService.#TARGET_SR : 0; }
  get loaded()      { return this.#pcm !== null; }

  /**
   * Decode an audio/video File into memory.
   * Resamples to 48 kHz and mixes down to mono.
   * @param {File} file
   * @returns {Promise<{ duration: number }>}
   */
  async load(file) {
    const arrayBuffer = await file.arrayBuffer();
    const ctx = new AudioContext({ sampleRate: AudioReplayService.#TARGET_SR });

    let audioBuffer;
    try {
      audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    } finally {
      await ctx.close();
    }

    // Mix down to mono.
    if (audioBuffer.numberOfChannels === 1) {
      this.#pcm = audioBuffer.getChannelData(0).slice();
    } else {
      const ch0 = audioBuffer.getChannelData(0);
      const ch1 = audioBuffer.getChannelData(1);
      this.#pcm = new Float32Array(ch0.length);
      for (let i = 0; i < ch0.length; i++) {
        this.#pcm[i] = (ch0[i] + ch1[i]) * 0.5;
      }
    }

    this.#duration = this.#pcm.length / AudioReplayService.#TARGET_SR;
    this.#offset   = 0;
    return { duration: this.#duration };
  }

  /**
   * Start emitting chunks at real-time pace.
   * @param {number} [chunkMs=250]
   */
  start(chunkMs = 250) {
    if (!this.#pcm) throw new Error('Call load() before start()');
    this.stop();
    this.#offset = 0;

    const sr             = AudioReplayService.#TARGET_SR;
    const samplesPerChunk = Math.floor(sr * chunkMs / 1000);

    this.#timer = setInterval(() => {
      if (this.#offset >= this.#pcm.length) {
        this.stop();
        this.dispatchEvent(new Event('end'));
        return;
      }
      const end   = Math.min(this.#offset + samplesPerChunk, this.#pcm.length);
      const chunk = this.#pcm.slice(this.#offset, end).buffer;
      this.dispatchEvent(new CustomEvent('chunk', { detail: chunk }));
      this.#offset = end;
    }, chunkMs);
  }

  stop() {
    if (this.#timer !== null) {
      clearInterval(this.#timer);
      this.#timer = null;
    }
  }

  unload() {
    this.stop();
    this.#pcm      = null;
    this.#duration = 0;
    this.#offset   = 0;
  }
}
