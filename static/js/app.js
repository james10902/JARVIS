/* ── JARVIS UI — app.js ───────────────────────────────────────────────────
 * Desktop: uses SocketIO + server-side Whisper (microphone captured by Python)
 * Mobile:  uses MediaRecorder → POST /transcribe → server-side Whisper
 *          (phone mic → webm audio → server transcribes → response JSON)
 * ─────────────────────────────────────────────────────────────────────── */

const socket = io();

// ── DOM refs ──────────────────────────────────────────────
const statusDot       = document.getElementById('statusDot');
const statusText      = document.getElementById('statusText');
const timeDisplay     = document.getElementById('timeDisplay');
const transcriptLabel = document.getElementById('transcriptLabel');
const transcriptResp  = document.getElementById('transcriptResponse');
const micBtn          = document.getElementById('micBtn');
const stopBtn         = document.getElementById('stopBtn');
const mobileHint      = document.getElementById('mobileHint');
const historyList     = document.getElementById('historyList');
const waveCanvas      = document.getElementById('waveCanvas');
const particleCanvas  = document.getElementById('particles');

// ── Detect mobile ─────────────────────────────────────────
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
              || window.innerWidth <= 600;

// ── Clock ─────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const h = String(now.getHours()).padStart(2,'0');
  const m = String(now.getMinutes()).padStart(2,'0');
  const s = String(now.getSeconds()).padStart(2,'0');
  timeDisplay.textContent = `${h}:${m}:${s}`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Status helper ─────────────────────────────────────────
function setStatus(state, label) {
  statusDot.className  = `status-dot ${state}`;
  statusText.className = `status-text ${state}`;
  statusText.textContent = label;
}

// ── Waveform canvas ───────────────────────────────────────
const wCtx = waveCanvas.getContext('2d');
let waveState = 'idle';
let wavePhase = 0;

function resizeWave() {
  waveCanvas.width  = waveCanvas.offsetWidth  * devicePixelRatio;
  waveCanvas.height = waveCanvas.offsetHeight * devicePixelRatio;
}
resizeWave();
window.addEventListener('resize', resizeWave);

function drawWave() {
  const W = waveCanvas.width;
  const H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  const cy = H / 2;

  if (waveState === 'idle') {
    wCtx.beginPath();
    wCtx.moveTo(0, cy);
    wCtx.lineTo(W, cy);
    wCtx.strokeStyle = 'rgba(79,195,247,0.15)';
    wCtx.lineWidth = 1.5;
    wCtx.stroke();
    requestAnimationFrame(drawWave);
    return;
  }

  const isRecording = waveState === 'recording';
  const layers = isRecording ? 3 : (waveState === 'listening' ? 3 : 4);
  const colors  = isRecording
    ? ['rgba(239,83,80,0.8)', 'rgba(239,83,80,0.4)', 'rgba(239,83,80,0.2)']
    : waveState === 'listening'
    ? ['rgba(0,229,255,0.7)', 'rgba(0,229,255,0.4)', 'rgba(0,229,255,0.2)']
    : ['rgba(79,195,247,0.8)', 'rgba(41,121,255,0.6)', 'rgba(79,195,247,0.4)', 'rgba(41,121,255,0.2)'];

  for (let l = 0; l < layers; l++) {
    const amp   = (waveState === 'speaking' ? 28 : 18) * (1 - l * 0.18);
    const freq  = 0.012 + l * 0.004;
    const speed = (waveState === 'speaking' ? 0.06 : 0.04) * (1 + l * 0.3);
    const phase = wavePhase * speed + l * 1.2;

    wCtx.beginPath();
    for (let x = 0; x <= W; x += 2) {
      const y = cy
        + Math.sin(x * freq + phase) * amp
        + Math.sin(x * freq * 1.7 + phase * 1.3) * amp * 0.4;
      x === 0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
    }
    wCtx.strokeStyle = colors[l];
    wCtx.lineWidth   = 2 - l * 0.3;
    wCtx.shadowColor = colors[0];
    wCtx.shadowBlur  = waveState === 'speaking' ? 12 : 6;
    wCtx.stroke();
    wCtx.shadowBlur  = 0;
  }
  wavePhase++;
  requestAnimationFrame(drawWave);
}
drawWave();

// ── Particle background ───────────────────────────────────
const pCtx = particleCanvas.getContext('2d');
const particles = [];

function resizeParticles() {
  particleCanvas.width  = window.innerWidth;
  particleCanvas.height = window.innerHeight;
}
resizeParticles();
window.addEventListener('resize', resizeParticles);

for (let i = 0; i < 60; i++) {
  particles.push({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    r: Math.random() * 1.5 + 0.3,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    alpha: Math.random() * 0.5 + 0.1,
  });
}

function drawParticles() {
  pCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy;
    if (p.x < 0) p.x = particleCanvas.width;
    if (p.x > particleCanvas.width) p.x = 0;
    if (p.y < 0) p.y = particleCanvas.height;
    if (p.y > particleCanvas.height) p.y = 0;
    pCtx.beginPath();
    pCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    pCtx.fillStyle = `rgba(79,195,247,${p.alpha})`;
    pCtx.fill();
  });
  requestAnimationFrame(drawParticles);
}
drawParticles();

// ── Transcript helpers ────────────────────────────────────
function setTranscript(text) {
  const styled = text.replace(/(\b\w+\b)/, '<span class="highlight">$1</span>');
  transcriptLabel.innerHTML = styled;
}

function setResponse(text) {
  transcriptResp.textContent = text;
  transcriptResp.classList.add('visible');
}

function clearResponse() {
  transcriptResp.classList.remove('visible');
  setTimeout(() => { transcriptResp.textContent = ''; }, 500);
}

// ── History ───────────────────────────────────────────────
function addHistory(userText, jarvisText, action) {
  const entry = document.createElement('div');
  entry.className = 'history-entry';
  entry.innerHTML = `
    <div class="history-user"><span>YOU</span> — ${escHtml(userText)}</div>
    <div class="history-jarvis">${escHtml(jarvisText)}</div>
    ${action ? `<div class="history-action">→ ${escHtml(action)}</div>` : ''}
  `;
  historyList.appendChild(entry);
  historyList.scrollTop = historyList.scrollHeight;
}

function escHtml(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ══════════════════════════════════════════════════════════
//  MOBILE PATH — MediaRecorder → POST /transcribe
// ══════════════════════════════════════════════════════════
let mediaRecorder = null;
let audioChunks   = [];
let mStream       = null;

function initMobile() {
  mobileHint.style.display = 'block';

  micBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      // Already recording — stop (handled by onstop)
      return;
    }

    try {
      mStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setTranscript('Microphone access denied. Please allow mic access.');
      return;
    }

    // Pick a supported MIME type
    const mimeType = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','audio/mp4']
      .find(m => MediaRecorder.isTypeSupported(m)) || '';

    mediaRecorder = new MediaRecorder(mStream, mimeType ? { mimeType } : {});
    audioChunks   = [];

    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };

    mediaRecorder.onstop = async () => {
      // Stop mic tracks
      mStream.getTracks().forEach(t => t.stop());
      mStream = null;

      // Hide stop button, show mic button
      stopBtn.style.display  = 'none';
      micBtn.style.display   = 'flex';
      micBtn.classList.remove('recording');
      waveState = 'thinking';
      setStatus('thinking', 'THINKING');
      setTranscript('Processing...');

      // Build FormData and POST to server
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      const fd   = new FormData();
      fd.append('audio', blob, 'recording.webm');

      try {
        const res  = await fetch('/transcribe', { method: 'POST', body: fd });
        const data = await res.json();

        if (data.error) {
          setTranscript(data.error === 'No speech detected'
            ? 'I did not catch that, Sir. Please try again.'
            : `Error: ${data.error}`);
          waveState = 'idle';
          setStatus('', 'STANDBY');
          return;
        }

        // Show transcript and response
        setTranscript(data.user_text || '');
        setResponse(data.response || '');
        addHistory(data.user_text || '', data.response || '', data.action || null);

        // Speak the response using browser TTS (no server audio needed on phone)
        mobileSpeech(data.response || '');

        waveState = 'speaking';
        setStatus('speaking', 'SPEAKING');
      } catch (err) {
        setTranscript(`Network error: ${err.message}`);
        waveState = 'idle';
        setStatus('error', 'ERROR');
      }
    };

    // Start recording
    mediaRecorder.start();
    micBtn.style.display  = 'none';
    stopBtn.style.display = 'flex';
    micBtn.classList.add('recording');
    waveState = 'recording';
    setStatus('recording', 'RECORDING');
    setTranscript('Listening, Sir...');
    clearResponse();
  });

  stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
  });
}

// Mobile TTS using the Web Speech API
function mobileSpeech(text) {
  if (!window.speechSynthesis) return;
  speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);

  // Prefer a deep British male voice if available
  const voices = speechSynthesis.getVoices();
  const preferred = voices.find(v =>
    v.lang === 'en-GB' && /male|daniel|george|oliver/i.test(v.name)
  ) || voices.find(v => v.lang.startsWith('en'));

  if (preferred) utt.voice = preferred;
  utt.rate   = 0.95;
  utt.pitch  = 0.85;
  utt.volume = 1.0;

  utt.onstart = () => { waveState = 'speaking'; setStatus('speaking', 'SPEAKING'); };
  utt.onend   = () => { waveState = 'idle'; setStatus('', 'STANDBY'); };

  speechSynthesis.speak(utt);
}

// ══════════════════════════════════════════════════════════
//  DESKTOP PATH — SocketIO (server handles mic via Whisper)
// ══════════════════════════════════════════════════════════
function initDesktop() {
  micBtn.addEventListener('click', () => {
    const state = statusText.textContent.trim();
    if (state === 'SPEAKING') {
      micBtn.classList.add('active');
      socket.emit('interrupt');
    } else if (state === 'LISTENING') {
      // already listening
    } else {
      micBtn.classList.add('active');
      socket.emit('start_listen');
    }
  });

  socket.on('connect', () => {
    setStatus('', 'ONLINE');
    setTranscript('Awaiting your command, Sir.');
  });

  socket.on('interrupted', () => {
    setTranscript('Yes, Sir?');
    clearResponse();
  });

  socket.on('partial_transcript', ({ text }) => {
    transcriptLabel.innerHTML = text.replace(/(\b\w+\b)/, '<span class="highlight">$1</span>');
    clearResponse();
  });

  socket.on('transcript', ({ text }) => {
    transcriptLabel.innerHTML = text.replace(/(\b\w+\b)/, '<span class="highlight">$1</span>');
    clearResponse();
  });

  socket.on('response', ({ text, action, user_text }) => {
    setResponse(text);
    if (user_text) addHistory(user_text, text, action || null);
  });

  socket.on('error_msg', ({ text }) => {
    setStatus('error', 'ERROR');
    setTranscript(text);
    waveState = 'idle';
  });

  socket.on('status', ({ state, label }) => {
    setStatus(state, label);
    waveState = state === 'listening' ? 'listening'
              : state === 'speaking'  ? 'speaking'
              : 'idle';

    if (state === 'listening') {
      micBtn.classList.add('active');
      micBtn.disabled = false;
    } else if (state === 'speaking') {
      micBtn.classList.remove('active');
      micBtn.disabled = false;
    } else {
      micBtn.classList.remove('active');
      micBtn.disabled = false;
    }
  });
}

// ── Bootstrap ─────────────────────────────────────────────
if (isMobile) {
  initMobile();
} else {
  initDesktop();
}

// Load voices for mobile TTS (browsers require this async)
if (window.speechSynthesis) {
  speechSynthesis.getVoices();
  speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
}
