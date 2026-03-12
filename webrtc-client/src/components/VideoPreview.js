/**
 * Renders a <video> element and manages its MediaStream attachment.
 * Pure display concern — knows nothing about recording or WebSockets.
 */
export class VideoPreview {
  #root;
  #video;

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `
      <video class="video-preview" autoplay muted playsinline></video>
    `;
    this.#video = this.#root.querySelector('video');
  }

  /** @param {MediaStream} stream */
  attachStream(stream) {
    this.#video.srcObject = stream;
  }

  detach() {
    this.#video.srcObject = null;
  }
}
