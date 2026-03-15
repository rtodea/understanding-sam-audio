/**
 * Renders a <video> element and manages its source.
 *
 * Two modes:
 *   attachStream(stream)  — live camera feed, muted to avoid feedback
 *   attachFile(file)      — replay from a local file, audio audible
 *
 * Call play() to start playback in file mode (gives you control over timing).
 * Call detach() to stop and clean up in both modes.
 */
export class VideoPreview {
  #root;
  #video;
  #objectUrl = null;

  /** @param {HTMLElement} root */
  constructor(root) {
    this.#root = root;
    this.#render();
  }

  #render() {
    this.#root.innerHTML = `<video class="video-preview" playsinline></video>`;
    this.#video = this.#root.querySelector('video');
  }

  /** Live camera stream — muted (avoid speaker feedback). */
  attachStream(stream) {
    this.#revoke();
    this.#video.muted    = true;
    this.#video.autoplay = true;
    this.#video.srcObject = stream;
    this.#video.src       = '';
  }

  /**
   * Local file for replay — audio on.
   * Call play() when you want playback to start.
   * @param {File} file
   */
  attachFile(file) {
    this.#revoke();
    this.#objectUrl       = URL.createObjectURL(file);
    this.#video.muted     = false;
    this.#video.autoplay  = false;
    this.#video.srcObject = null;
    this.#video.src       = this.#objectUrl;
    this.#video.load();
  }

  /** Start playback (used in file mode to sync with replay chunk start). */
  play() {
    return this.#video.play();
  }

  detach() {
    this.#video.pause();
    this.#video.srcObject = null;
    this.#video.src       = '';
    this.#video.load();
    this.#revoke();
  }

  #revoke() {
    if (this.#objectUrl) {
      URL.revokeObjectURL(this.#objectUrl);
      this.#objectUrl = null;
    }
  }
}
