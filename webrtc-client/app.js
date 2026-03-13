/**
 * app.js — Root bootstrap.
 * Imports components and services, wires the event graph, starts nothing automatically.
 */

import { SessionState }         from './src/state/SessionState.js';
import { MediaCaptureService }  from './src/services/MediaCaptureService.js';
import { WebSocketService }     from './src/services/WebSocketService.js';
import { AudioRecorderService } from './src/services/AudioRecorderService.js';
import { AudioPlaybackService } from './src/services/AudioPlaybackService.js';
import { ControlPanel }         from './src/components/ControlPanel.js';
import { VideoPreview }         from './src/components/VideoPreview.js';
import { AudioVisualizer }      from './src/components/AudioVisualizer.js';
import { DownloadButton }       from './src/components/DownloadButton.js';

// ── Browser capability check ───────────────────────────────────────────────

if (!AudioRecorderService.isSupported()) {
  document.getElementById('app').innerHTML = `
    <p style="color:#f87171;padding:2rem;font-size:1.1rem;">
      This app requires <strong>Chrome</strong> or <strong>Firefox</strong>
      (audio/webm;codecs=opus is not supported in this browser).
    </p>`;
  throw new Error('Unsupported browser');
}

// ── Services ───────────────────────────────────────────────────────────────

const state        = new SessionState();
const mediaCapture = new MediaCaptureService();
const wsService    = new WebSocketService();
const recorder     = new AudioRecorderService();
const playback     = new AudioPlaybackService();

// ── Components ─────────────────────────────────────────────────────────────

const controlPanel = new ControlPanel(
  document.getElementById('control-panel-container'), state
);
const videoPreview = new VideoPreview(
  document.getElementById('video-preview')
);
const visualizer   = new AudioVisualizer(
  document.getElementById('visualizer-container')
);
const downloadBtn  = new DownloadButton(
  document.getElementById('download-container')
);

// ── WebSocket URL (relative — NGINX proxies /api → server) ─────────────────

const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/ws/separate`;

// ── Session start ──────────────────────────────────────────────────────────

state.addEventListener('session:start', async () => {
  state.setStatus('connecting');
  downloadBtn.hide();

  let stream;
  try {
    stream = await mediaCapture.start();
  } catch (err) {
    state.setError(`Camera/microphone access denied: ${err.message}`);
    state.setStatus('idle');
    return;
  }

  videoPreview.attachStream(stream);
  playback.init();
  visualizer.attach(playback.getAnalyserNode());

  recorder.addEventListener('chunk', (e) => {
    wsService.sendBinary(e.detail);
  });

  wsService.addEventListener('open', () => {
    wsService.sendJson({ description: state.description });
    // Some browsers are stricter about MediaRecorder mime types when the stream
    // contains both audio and video tracks. Record audio only to avoid
    // NotSupportedError while still keeping video for the preview element.
    const audioOnlyStream = new MediaStream(stream.getAudioTracks());
    recorder.start(audioOnlyStream, 3000);
    state.setStatus('active');
  }, { once: true });

  wsService.addEventListener('message', async (e) => {
    if (e.data instanceof ArrayBuffer) {
      await playback.enqueue(e.data);
    }
  });

  wsService.addEventListener('error', () => {
    state.setError('WebSocket connection error — is the server running?');
    teardownSession();
  }, { once: true });

  wsService.addEventListener('close', () => {
    if (state.status !== 'idle') teardownSession();
  }, { once: true });

  // Connect AFTER attaching listeners to avoid missing 'open' on fast connections
  wsService.connect(wsUrl);
});

// ── Session stop ───────────────────────────────────────────────────────────

state.addEventListener('session:stop', teardownSession);

function teardownSession() {
  if (state.status === 'idle') return;
  state.setStatus('stopping');

  recorder.stop();
  wsService.disconnect();
  mediaCapture.stop();
  videoPreview.detach();
  visualizer.detach();

  downloadBtn.present(playback.getDecodedChunks());
  playback.reset();

  state.setStatus('idle');
}
