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

### Backend (A2F Proxy)

Deploy the `backend/` folder as a Node.js service. Recommended: **Google Cloud Run** (scales to zero).

#### Cloud Run deployment

```bash
cd backend

# Create a Dockerfile
cat > Dockerfile <<'EOF'
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 8766
CMD ["node", "a2f_proxy.mjs"]
EOF

# Deploy
gcloud run deploy a2f-proxy \
  --source . \
  --port 8766 \
  --allow-unauthenticated \
  --set-env-vars "NVIDIA_API_KEY=your_key,A2F_FUNCTION_ID=your_id"
```

> **Note:** Cloud Run uses HTTP, but the proxy uses raw WebSocket. You may need to use Cloud Run's WebSocket support or deploy behind a load balancer that supports WebSocket upgrades.

After deploying, update the `A2F_PROXY_URL` in `index.html` to point to your Cloud Run URL (use `wss://` for HTTPS).

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
