/**
 * app.js — Root bootstrap.
 * Imports components and services, wires the event graph, starts nothing automatically.
 */

import { SessionState }            from './src/state/SessionState.js';
import { MediaCaptureService }     from './src/services/MediaCaptureService.js';
import { WebSocketService }        from './src/services/WebSocketService.js';
import { AudioRecorderService }    from './src/services/AudioRecorderService.js';
import { AudioPlaybackService }    from './src/services/AudioPlaybackService.js';
import { AudioReplayService }      from './src/services/AudioReplayService.js';
import { OriginalRecordingService} from './src/services/OriginalRecordingService.js';
import { ControlPanel }            from './src/components/ControlPanel.js';
import { VideoPreview }            from './src/components/VideoPreview.js';
import { AudioVisualizer }         from './src/components/AudioVisualizer.js';
import { DownloadButton }          from './src/components/DownloadButton.js';
import { SttDisplay }              from './src/components/SttDisplay.js';

// ── Browser capability check ───────────────────────────────────────────────

if (!AudioRecorderService.isSupported()) {
  document.getElementById('app').innerHTML = `
    <p style="color:#f87171;padding:2rem;font-size:1.1rem;">
      This app requires a browser with <strong>Web Audio</strong> and
      <strong>getUserMedia</strong> support.
    </p>`;
  throw new Error('Unsupported browser');
}

// ── Services ───────────────────────────────────────────────────────────────

const state             = new SessionState();
const mediaCapture      = new MediaCaptureService();
const wsService         = new WebSocketService();
const recorder          = new AudioRecorderService();
const replay            = new AudioReplayService();
const playback          = new AudioPlaybackService();
const originalRecording = new OriginalRecordingService();
let   originalRecordingPromise = null;
let   progressTimer            = null;

// ── Components ─────────────────────────────────────────────────────────────

const chunkStatsContainer = document.getElementById('chunk-stats-container');

// ── Chunk latency tracking ──────────────────────────────────────────────────

const ADVANCE_SECONDS  = 1.5;
const INTERVAL_HISTORY = 10;

let chunkCount = 0;
let lastChunkAt = null;
const recentIntervals = [];

function resetChunkStats() {
  chunkCount = 0;
  lastChunkAt = null;
  recentIntervals.length = 0;
  chunkStatsContainer.innerHTML = '';
}

function recordChunk() {
  const now = performance.now();
  chunkCount++;

  if (lastChunkAt !== null) {
    const interval = (now - lastChunkAt) / 1000;
    recentIntervals.push(interval);
    if (recentIntervals.length > INTERVAL_HISTORY) recentIntervals.shift();
  }
  lastChunkAt = now;

  const last = recentIntervals.at(-1);
  const avg  = recentIntervals.length
    ? recentIntervals.reduce((a, b) => a + b, 0) / recentIntervals.length
    : null;

  const fmt = (s) => s.toFixed(2) + 's';
  const cls = (s) => s <= ADVANCE_SECONDS ? 'stat-fast' : 'stat-slow';

  chunkStatsContainer.innerHTML = `
    <div class="chunk-stats">
      <span>chunks: <span class="stat-value">${chunkCount}</span></span>
      ${last != null ? `<span>last interval: <span class="stat-value ${cls(last)}">${fmt(last)}</span></span>` : ''}
      ${avg  != null ? `<span>avg interval: <span class="stat-value ${cls(avg)}">${fmt(avg)}</span></span>`  : ''}
      <span>target: <span class="stat-value">${ADVANCE_SECONDS}s</span></span>
    </div>`;
}

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
const sttDisplay   = new SttDisplay(
  document.getElementById('stt-container')
);

// ── WebSocket URL ───────────────────────────────────────────────────────────
// Default: relative to current host (NGINX proxies /api → server).
// Override: pass ?api_url=http://other-host:8000 to connect to an external server.

const _apiUrl = new URLSearchParams(location.search).get('api_url');
const wsUrl = _apiUrl
  ? `${_apiUrl.replace(/^http/, 'ws').replace(/\/$/, '')}/api/ws/separate`
  : `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/ws/separate`;

if (_apiUrl) {
  console.info('[app] Using external api_url:', _apiUrl, '→', wsUrl);
}

// ── Audio chunk routing ─────────────────────────────────────────────────────
// Both live recorder and replay service emit 'chunk' — both pipe to wsService.

recorder.addEventListener('chunk', (e) => wsService.sendBinary(e.detail));
replay.addEventListener('chunk',   (e) => wsService.sendBinary(e.detail));

// When replay finishes, trigger a clean session stop automatically.
replay.addEventListener('end', () => {
  if (state.status === 'active') {
    state.dispatchEvent(new CustomEvent('session:stop'));
  }
});

wsService.addEventListener('message', async (e) => {
  if (e.data instanceof ArrayBuffer) {
    recordChunk();
    await playback.enqueue(e.data);
  } else if (typeof e.data === 'string') {
    try {
      const msg = JSON.parse(e.data);
      if (msg.event === 'stt') sttDisplay.update(msg.stream, msg.type, msg.text);
    } catch { /* ignore malformed frames */ }
  }
});

// ── Session start (live mic) ───────────────────────────────────────────────

state.addEventListener('session:start', async () => {
  state.setStatus('connecting');
  state.setReplayMode(false);
  downloadBtn.hide();
  resetChunkStats();
  sttDisplay.reset();

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
  originalRecordingPromise = null;
  if (OriginalRecordingService.isSupported()) {
    originalRecording.start(stream);
  }

  wsService.addEventListener('open', () => {
    const audioOnlyStream = new MediaStream(stream.getAudioTracks());
    recorder.start(audioOnlyStream, 250);
    visualizer.attach(recorder.getAnalyserNode());
    wsService.sendJson({
      description: state.description,
      encoding:    'pcm_f32le',
      sampleRate:  recorder.sampleRate,
    });
    state.setStatus('active');
  }, { once: true });

  wsService.addEventListener('error', () => {
    state.setError('WebSocket connection error — is the server running?');
    void finalizeSession();
  }, { once: true });

  wsService.addEventListener('close', () => {
    if (state.status !== 'idle') void finalizeSession();
  }, { once: true });

  wsService.connect(wsUrl);
});

// ── Session start (replay) ─────────────────────────────────────────────────

state.addEventListener('session:replay', async ({ detail: { file } }) => {
  state.setStatus('connecting');
  state.setReplayMode(true);
  downloadBtn.hide();
  resetChunkStats();
  sttDisplay.reset();

  try {
    await replay.load(file);
  } catch (err) {
    state.setError(`Failed to decode file: ${err.message}`);
    state.setStatus('idle');
    state.setReplayMode(false);
    return;
  }

  videoPreview.attachFile(file);
  playback.init();

  wsService.addEventListener('open', () => {
    replay.start(250);
    videoPreview.play();
    wsService.sendJson({
      description: state.description,
      encoding:    'pcm_f32le',
      sampleRate:  replay.sampleRate,
    });
    state.setStatus('active');
    // Tick progress bar while replaying.
    progressTimer = setInterval(
      () => controlPanel.setProgress(replay.currentTime, replay.duration),
      100,
    );
  }, { once: true });

  wsService.addEventListener('error', () => {
    state.setError('WebSocket connection error — is the server running?');
    void finalizeSession();
  }, { once: true });

  wsService.addEventListener('close', () => {
    if (state.status !== 'idle') void finalizeSession();
  }, { once: true });

  wsService.connect(wsUrl);
});

// ── Session stop ───────────────────────────────────────────────────────────

state.addEventListener('session:stop', teardownSession);

function teardownSession() {
  if (state.status !== 'active') return;
  state.setStatus('stopping');

  if (state.replayMode) {
    replay.stop();
    clearInterval(progressTimer);
    progressTimer = null;
    videoPreview.detach();
  } else {
    recorder.stop();
    originalRecordingPromise = originalRecording.stop();
    mediaCapture.stop();
    videoPreview.detach();
    visualizer.detach();
  }

  wsService.sendJson({ event: 'stop' });
}

async function finalizeSession() {
  if (state.status === 'idle') return;

  let originalBlob = null;

  if (state.replayMode) {
    replay.stop();
    clearInterval(progressTimer);
    progressTimer = null;
    videoPreview.detach();
  } else {
    recorder.stop();
    if (!originalRecordingPromise) {
      originalRecordingPromise = originalRecording.stop();
    }
    mediaCapture.stop();
    videoPreview.detach();
    visualizer.detach();
    originalBlob = await originalRecordingPromise;
    originalRecordingPromise = null;
  }

  wsService.disconnect();

  await playback.drain();
  downloadBtn.present({
    separatedChunks: playback.getDecodedChunks(),
    originalBlob,
  });
  playback.reset();

  state.setReplayMode(false);
  state.setStatus('idle');
}
