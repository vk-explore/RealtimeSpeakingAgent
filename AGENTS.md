# AGENTS.md — AI Agent Guide for RealtimeSpeakingAgent

## Project Overview

**Rythu Mitrudu** — a real-time Telugu AI agriculture assistant with a 3D avatar. Users speak to the avatar via microphone; Google Gemini Live API handles voice conversation, and NVIDIA Audio2Face (A2F) drives lip-sync and facial animation in real time.

## Architecture

```
Browser (index.html)
  ├── Gemini Live API ──── WebSocket (direct to Google)
  ├── Three.js ─────────── 3D avatar with ARKit blendshapes
  ├── Web Audio API ────── mic capture (16 kHz) + playback (24 kHz)
  └── A2F Proxy (WS) ──── backend/a2f_proxy.mjs ──► NVIDIA A2F NIM (gRPC)
```

- Browser connects **directly** to Gemini via WebSocket for voice conversation.
- Audio responses from Gemini are sent to the **A2F proxy** over a second WebSocket; the proxy forwards audio via **gRPC bidirectional streaming** to NVIDIA's cloud A2F NIM, then relays blendshape animation + emotion data back to the browser.
- Three.js applies ARKit blendshapes to the avatar mesh each frame.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Single-file HTML/CSS/JS (`index.html`), ES modules, no build step |
| 3D rendering | Three.js r170 — GLTFLoader, DRACOLoader, OrbitControls, RGBELoader |
| Voice AI | Google Gemini Live API (`gemini-3.1-flash-live-preview`) via WebSocket |
| Lip-sync | NVIDIA Audio2Face NIM (cloud gRPC) |
| Audio | Web Audio API — 16 kHz mic capture, 24 kHz playback |
| Backend proxy | Node.js 20, ES modules (`.mjs`) |
| Backend libs | `ws`, `@grpc/grpc-js`, `@grpc/proto-loader`, `protobufjs` |
| Protocols | gRPC (NVIDIA ACE protobuf), WebSocket (Gemini + A2F proxy) |
| Deployment | Fly.io (backend), GitHub Pages (frontend) |

## File Structure

| Path | Role |
|---|---|
| `index.html` | **Entire frontend** — UI, Three.js scene, Gemini WS, A2F WS, audio pipeline, blendshape animation |
| `avatar.glb` | 3D avatar model with ARKit blendshapes and idle animation (gitignored) |
| `encrypt_key.mjs` | CLI tool to AES-GCM encrypt the Google API key |
| `readme.md` | User-facing documentation |
| `backend/a2f_proxy.mjs` | WebSocket-to-gRPC proxy bridging browser to NVIDIA A2F NIM |
| `backend/package.json` | Node.js dependencies |
| `backend/fly.toml` | Fly.io deployment config (Mumbai `bom` region) |
| `backend/Dockerfile` | Container image (node:20-slim) |
| `backend/protos/` | NVIDIA ACE protobuf definitions (.proto files) |

## Running Locally

1. Set environment variables (or create `.env` in `backend/`):
   - `NVIDIA_API_KEY` — NGC API key for A2F NIM
   - `A2F_FUNCTION_ID` — Audio2Face function ID from build.nvidia.com
2. Start the backend proxy:
   ```bash
   cd backend && npm install && npm start
   ```
   Runs on `ws://localhost:8766`
3. Serve the frontend:
   ```bash
   npx serve . 
   # or: python3 -m http.server 8000
   ```
4. Open in browser, enter Google API key / password, click **Start Session**

## Deployment

- **Frontend**: Push to GitHub → GitHub Pages serves `index.html` + static assets
- **Backend**: `cd backend && flyctl deploy` → Fly.io (auto-stop/start, 256 MB RAM)

## Environment Variables

| Variable | Where | Required | Default |
|---|---|---|---|
| `NVIDIA_API_KEY` | Backend | Yes | — |
| `A2F_FUNCTION_ID` | Backend | Yes | — |
| `A2F_GRPC_URI` | Backend | No | `grpc.nvcf.nvidia.com:443` |
| `A2F_PROXY_PORT` | Backend | No | `8766` |
| `GOOGLE_API_KEY` | Browser | Yes | Entered via UI or encrypted in source |

## Coding Conventions

- **Single-file frontend**: all HTML, CSS, JS in one `index.html` — no framework, no build step
- **ES modules** everywhere (`.mjs` backend, `import` in frontend via import map)
- **UPPER_SNAKE_CASE** for constants (`SEND_SAMPLE_RATE`, `A2F_PROXY_URL`)
- **camelCase** for functions and variables (`loadAvatar`, `avatarLoaded`)
- Section headers use box-style comments: `// ════════`, `// ───`
- Three.js loaded via CDN import map (no npm/bundler)
- Protobuf `.proto` files loaded at runtime via `@grpc/proto-loader`

## Key Technical Details

- Avatar uses ARKit 52-blendshape standard + custom emotion shapes (happy_M, angry_M, etc.)
- Head/neck look-at is additive: euler decompose → add yaw/pitch offset → recompose, applied per frame on top of animation mixer output
- A2F blendshapes are queued and time-synced to audio playback via `audioStartTime`
- Emotion data (14 categories, 0–10 scale) displayed in a real-time panel
- Audio playback uses GainNode for volume control (`PLAYBACK_VOLUME`)
- HDRI environment lighting from Poly Haven

## Lipsync Architecture (CORRECT — do not revert)

**How audio→lipsync sync works:**

1. Each Gemini audio chunk arrives → **immediately sent to A2F** via `{ type: 'audio', data: base64PCM }` over WebSocket (no batching, no timer)
2. First audio chunk of a turn → sends `{ type: 'start' }` first to open a gRPC stream on NVIDIA's side
3. NVIDIA A2F NIM streams blendshape frames back in **real-time** as audio arrives — each frame has a `timeCode` (seconds offset from clip start) and 52 ARKit blendshape values (0–1)
4. Blendshape frames are queued in `a2fBsQueue[]`
5. **Playback trigger**: when `a2fBsQueue[last].timeCode >= 1.0` (1 second of blendshapes buffered), `startAudioPlayback()` fires — sets `audioStartTime = performance.now()` and starts draining `playbackQueue`
6. Every render frame: `elapsed = (now - audioStartTime) / 1000` — dequeues blendshape frames whose `timeCode <= elapsed` and applies them to the avatar
7. On Gemini `turnComplete` → sends `{ type: 'end' }` to A2F to flush remaining blendshape frames
8. **Fallback**: if A2F hasn't delivered 1s of blendshapes within 1s of first audio (`firstAudioChunkTime`), audio starts anyway without lipsync

**Key variables:**
- `audioStartTime` — `performance.now()` when playback started (the sync clock)
- `firstAudioChunkTime` — `performance.now()` when first audio chunk arrived (for fallback timeout)
- `a2fBsQueue` — `{bsDict, timeCode, headRotation, emotions}` frames
- `a2fStreamReady` — true while a gRPC stream is open to NVIDIA

**WRONG pattern (do not implement):** Batching audio into `a2fAudioBuffer` and flushing all at once on `turnComplete` — this adds full-utterance latency before any lipsync begins.

## Guidelines

- **ALWAYS ask clarifying questions before implementing any critical logic** — especially audio/video sync, WebSocket protocol changes, or anything touching the lipsync pipeline. A wrong assumption here caused a full re-implementation.
- Do not execute code changes when not sure, try to understand the problem, ask question if needed so user can help corner the problem.
- Only apply the fix when you are sure that was the problem.