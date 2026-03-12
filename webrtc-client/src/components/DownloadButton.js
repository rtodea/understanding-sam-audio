/**
 * Assembles received PCM chunks into a single WAV file and offers a download link.
 * Shown only after the session stops.
 */
export class DownloadButton {
  #root;
  #panel;
  #link;
  #currentUrl = null;

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `
      <div class="download-panel" hidden>
        <p class="download-label">Separated audio ready:</p>
        <a id="dl-link" class="btn btn-download" download="separated.wav">
          Download separated.wav
        </a>
      </div>
    `;
    this.#panel = this.#root.querySelector('.download-panel');
    this.#link  = this.#root.querySelector('#dl-link');
  }

  /**
   * @param {Float32Array[]} decodedChunks  Mono PCM at 48 kHz
   */
  present(decodedChunks) {
    if (decodedChunks.length === 0) return;

    if (this.#currentUrl) {
      URL.revokeObjectURL(this.#currentUrl);
    }

    const wavBuffer = encodeWav(decodedChunks, 48000);
    const blob = new Blob([wavBuffer], { type: 'audio/wav' });
    this.#currentUrl = URL.createObjectURL(blob);
    this.#link.href = this.#currentUrl;
    this.#panel.hidden = false;
  }

  hide() {
    this.#panel.hidden = true;
    if (this.#currentUrl) {
      URL.revokeObjectURL(this.#currentUrl);
      this.#currentUrl = null;
    }
  }
}

// ---------------------------------------------------------------------------
// PCM float32 WAV encoder
// Spec: http://soundfile.sapp.org/doc/WaveFormat/
// ---------------------------------------------------------------------------

/**
 * @param {Float32Array[]} chunks  Mono 32-bit float PCM chunks
 * @param {number}         sampleRate
 * @returns {ArrayBuffer}  Valid .wav file
 */
function encodeWav(chunks, sampleRate) {
  const numChannels  = 1;
  const bitsPerSample = 32;          // float32
  const blockAlign   = numChannels * (bitsPerSample / 8);
  const byteRate     = sampleRate * blockAlign;
  const totalSamples = chunks.reduce((s, c) => s + c.length, 0);
  const dataSize     = totalSamples * blockAlign;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view   = new DataView(buffer);

  writeAscii(view,  0, 'RIFF');
  view.setUint32(   4, 36 + dataSize, true);
  writeAscii(view,  8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(  16, 16, true);          // subchunk1 size
  view.setUint16(  20,  3, true);          // audio format: IEEE float = 3
  view.setUint16(  22, numChannels, true);
  view.setUint32(  24, sampleRate,  true);
  view.setUint32(  28, byteRate,    true);
  view.setUint16(  32, blockAlign,  true);
  view.setUint16(  34, bitsPerSample, true);
  writeAscii(view, 36, 'data');
  view.setUint32(  40, dataSize, true);

  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      view.setFloat32(offset, chunk[i], true);
      offset += 4;
    }
  }

  return buffer;
}

function writeAscii(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}
