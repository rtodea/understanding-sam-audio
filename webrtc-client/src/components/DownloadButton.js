/**
 * Offers downloads for the separated WAV output and original captured media.
 * Shown only after the session stops.
 */
export class DownloadButton {
  #root;
  #panel;
  #separatedRow;
  #originalRow;
  #separatedLink;
  #originalLink;
  #separatedUrl = null;
  #originalUrl = null;

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `
      <div class="download-panel" hidden>
        <div id="separated-row">
          <p class="download-label">Separated audio ready:</p>
          <a id="dl-separated" class="btn btn-download" download="separated.wav">
            Download separated.wav
          </a>
        </div>
        <div id="original-row">
          <p class="download-label">Original recording ready:</p>
          <a id="dl-original" class="btn btn-download" download="original.webm">
            Download original
          </a>
        </div>
      </div>
    `;
    this.#panel = this.#root.querySelector('.download-panel');
    this.#separatedRow = this.#root.querySelector('#separated-row');
    this.#originalRow = this.#root.querySelector('#original-row');
    this.#separatedLink = this.#root.querySelector('#dl-separated');
    this.#originalLink = this.#root.querySelector('#dl-original');
  }

  /**
   * @param {{ separatedChunks: Float32Array[], originalBlob: Blob | null }} media
   */
  present({ separatedChunks, originalBlob }) {
    this.#revokeUrls();

    const hasSeparated = separatedChunks.length > 0;
    const hasOriginal = originalBlob instanceof Blob && originalBlob.size > 0;
    if (!hasSeparated && !hasOriginal) return;

    this.#separatedRow.hidden = !hasSeparated;
    this.#originalRow.hidden = !hasOriginal;

    if (hasSeparated) {
      const wavBuffer = encodeWav(separatedChunks, 48000);
      const blob = new Blob([wavBuffer], { type: 'audio/wav' });
      this.#separatedUrl = URL.createObjectURL(blob);
      this.#separatedLink.href = this.#separatedUrl;
    }

    if (hasOriginal) {
      this.#originalUrl = URL.createObjectURL(originalBlob);
      this.#originalLink.href = this.#originalUrl;
      this.#originalLink.download = fileNameForBlob(originalBlob);
    }

    this.#panel.hidden = false;
  }

  hide() {
    this.#panel.hidden = true;
    this.#revokeUrls();
  }

  #revokeUrls() {
    if (this.#separatedUrl) {
      URL.revokeObjectURL(this.#separatedUrl);
      this.#separatedUrl = null;
    }
    if (this.#originalUrl) {
      URL.revokeObjectURL(this.#originalUrl);
      this.#originalUrl = null;
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

function fileNameForBlob(blob) {
  const type = blob.type || 'video/webm';
  if (type.includes('mp4')) return 'original.mp4';
  if (type.includes('ogg')) return 'original.ogv';
  return 'original.webm';
}
