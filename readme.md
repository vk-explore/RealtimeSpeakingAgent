# Signova AI Avatar

Real-time AI avatar that speaks with farmers using Gemini Live API for voice conversation and NVIDIA Audio2Face for lip-sync and facial animation.

## Architecture

```
Browser (index.html)
  ├── Gemini Live API (WebSocket, direct)
  ├── Three.js avatar with ARKit blendshapes
  ├── Web Audio API (mic capture + playback)
  └── A2F Proxy (WebSocket) ──► backend/a2f_proxy.mjs ──► NVIDIA A2F NIM (gRPC)
```

- **Frontend** (`index.html`) — Single-file web app. Connects directly to Gemini via WebSocket. Sends audio to the A2F proxy for blendshape animation.
- **Backend** (`backend/`) — Lightweight Node.js WebSocket-to-gRPC proxy. Required because browsers cannot make gRPC bidirectional streaming calls.

## Prerequisites

- [Node.js](https://nodejs.org/) 18+
- A Google API key with Gemini API access
- An NVIDIA NGC API key with Audio2Face NIM access
- An A2F Function ID (from [build.nvidia.com](https://build.nvidia.com/nvidia/audio2face-3d))

## Setup

### 1. Environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key
NVIDIA_API_KEY=your_nvidia_api_key
A2F_FUNCTION_ID=your_a2f_function_id
```

### 2. Install backend dependencies

```bash
cd backend
npm install
```

### 3. Place your avatar

Put your avatar GLB file (with ARKit blendshapes) as `avatar.glb` in the project root.

## Running Locally

### Start the A2F proxy

```bash
cd backend
npm start
```

This starts the WebSocket proxy on `ws://localhost:8766`.

### Serve the frontend

Open `index.html` in a browser. For local development, use any static file server:

```bash
# From the project root
npx serve .
# or
python3 -m http.server 8000
```

Then open `http://localhost:8000` (or `http://localhost:3000` with `npx serve`).

Enter your Google API key in the config panel and click **Start Session**.

## Deployment

### Frontend

Host `index.html` and `avatar.glb` on any static hosting (GitHub Pages, Netlify, Vercel, S3, etc.).

### Backend (A2F Proxy) — Docker (Local / Self-hosted)

#### Build the image

```bash
cd backend
docker build -t a2f-proxy .
```

#### Run as a persistent container

Use `--restart unless-stopped` so the container automatically restarts after a machine reboot or Docker daemon restart — and stays stopped if you explicitly stop it.

```bash
docker run -d \
  --name a2f-proxy \
  --restart unless-stopped \
  -p 8766:8766 \
  -e NVIDIA_API_KEY=nvapi-... \
  -e A2F_FUNCTION_ID=your_function_id \
  a2f-proxy
```

Or with a `.env` file (create `backend/.env` with your keys):

```bash
docker run -d \
  --name a2f-proxy \
  --restart unless-stopped \
  -p 8766:8766 \
  --env-file backend/.env \
  a2f-proxy
```

#### Using a self-hosted NVIDIA NIM (same machine)

If you're running the NVIDIA Audio2Face NIM locally (e.g. on port `52000`), add these to your `.env` or `-e` flags. Because the proxy runs in Docker, use `host.docker.internal` to reach the host machine:

```bash
docker run -d \
  --name a2f-proxy \
  --restart unless-stopped \
  -p 8766:8766 \
  -e A2F_LOCAL=true \
  -e A2F_GRPC_URI=host.docker.internal:52000 \
  a2f-proxy
```

`NVIDIA_API_KEY` and `A2F_FUNCTION_ID` are not required when `A2F_LOCAL=true`.

#### Stop / Start the container

```bash
docker stop a2f-proxy   # stop (won't auto-restart until you start it again)
docker start a2f-proxy  # start again — no flags needed
```

#### View logs

```bash
docker logs -f a2f-proxy
```

#### Remove the container

```bash
docker rm -f a2f-proxy
```

The proxy listens on `ws://localhost:8766`. Update `A2F_PROXY_URL` in `index.html` accordingly when serving the frontend locally.

---

### Backend (A2F Proxy) — Fly.io (Mumbai)

The proxy is deployed on [Fly.io](https://fly.io) in the `bom` (Mumbai) region on a free `shared-cpu-1x` machine that auto-starts on the first WebSocket connection and shuts down after idle.

#### First-time setup

1. Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
2. Sign in: `flyctl auth login`

#### Deploy

```bash
cd backend

# Set secrets (only needed once)
flyctl secrets set NVIDIA_API_KEY=nvapi-... A2F_FUNCTION_ID=your_function_id

# Deploy
flyctl deploy
```

#### Scale to a single machine

By default Fly.io creates 2 machines for HA. Scale down to 1:

```bash
flyctl scale count 1
```

#### fly.toml reference

```toml
app = 'farmer-helper-agent-proxy-for-a2f'
primary_region = 'bom'

[build]

[[services]]
  internal_port = 8766
  protocol = "tcp"
  auto_stop_machines = "stop"   # stop when idle (no active connections)
  auto_start_machines = true    # wake on first connection
  min_machines_running = 0

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [[services.ports]]
    port = 80
    handlers = ["http"]

[[vm]]
  size = "shared-cpu-1x"
  memory = "256mb"
```

The machine stops automatically when there are no active WebSocket connections (managed by Fly.io's platform). `min_machines_running = 0` ensures it fully shuts down rather than staying on standby.

#### Update frontend URL

After deploying, `A2F_PROXY_URL` in `index.html` should point to your Fly app:

```js
const A2F_PROXY_URL = 'wss://farmer-helper-agent-proxy-for-a2f.fly.dev/';
```

## Project Structure

```
├── index.html              # Frontend — single-file web app
├── avatar.glb              # 3D avatar with ARKit blendshapes
├── .env                    # API keys (not committed)
├── backend/
│   ├── a2f_proxy.mjs       # WebSocket-to-gRPC proxy server
│   ├── package.json
│   └── protos/             # NVIDIA ACE protobuf definitions
│       ├── nvidia_ace.services.a2f_controller.v1.proto
│       ├── nvidia_ace.controller.v1.proto
│       ├── nvidia_ace.a2f.v1.proto
│       ├── nvidia_ace.animation_data.v1.proto
│       └── ...
└── readme.md
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | Yes | — | NVIDIA NGC API key |
| `A2F_FUNCTION_ID` | Yes | — | Audio2Face NIM function ID |
| `A2F_GRPC_URI` | No | `grpc.nvcf.nvidia.com:443` | A2F gRPC endpoint |
| `A2F_PROXY_PORT` | No | `8766` | WebSocket proxy port |
| `GOOGLE_API_KEY` | No | — | Entered in browser UI; only needed in `.env` for reference |
