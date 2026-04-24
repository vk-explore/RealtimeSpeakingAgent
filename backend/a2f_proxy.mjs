/**
 * A2F WebSocket-to-gRPC Proxy
 *
 * Bridges browser WebSocket clients to NVIDIA Audio2Face NIM gRPC service.
 * Browser sends JSON messages over WebSocket, proxy forwards to A2F via gRPC
 * bidirectional streaming and relays blendshape animation data back.
 *
 * Usage:
 *   node a2f_proxy.mjs
 *
 * Environment variables (or .env file):
 *   NVIDIA_API_KEY   — NGC API key
 *   A2F_FUNCTION_ID  — A2F NIM function ID
 *   A2F_GRPC_URI     — gRPC endpoint (default: grpc.nvcf.nvidia.com:443)
 *   A2F_PROXY_PORT   — WebSocket port (default: 8766)
 */

import { WebSocketServer, WebSocket } from 'ws';
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import protobuf from 'protobufjs';
import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ─── Load .env ───
function loadEnv() {
    // Check backend/.env first, then project root .env
    for (const p of [resolve(__dirname, '.env'), resolve(__dirname, '..', '.env')]) {
        if (!existsSync(p)) continue;
        for (const line of readFileSync(p, 'utf8').split('\n')) {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) continue;
            const eq = trimmed.indexOf('=');
            if (eq < 0) continue;
            const key = trimmed.slice(0, eq).trim();
            const val = trimmed.slice(eq + 1).trim();
            if (!process.env[key]) process.env[key] = val;
        }
    }
}
loadEnv();

const NVIDIA_API_KEY = process.env.NVIDIA_API_KEY || '';
const A2F_FUNCTION_ID = process.env.A2F_FUNCTION_ID || '';
const A2F_GRPC_URI = process.env.A2F_GRPC_URI || 'grpc.nvcf.nvidia.com:443';
const PROXY_PORT = parseInt(process.env.A2F_PROXY_PORT || '8766', 10);
// Set A2F_LOCAL=true when using a self-hosted NIM (disables TLS and auth headers)
const A2F_LOCAL = process.env.A2F_LOCAL === 'true';

// ─── A2F audio constants ───
const A2F_SAMPLE_RATE = 24000;
const BITS_PER_SAMPLE = 16;
const CHANNEL_COUNT = 1;

// ─── Face config defaults (matching original Python) ───
const DEFAULT_FACE_CONFIG = {
    face_params: {
        upperFaceStrength: 5,
        upperFaceSmoothing: 0.02,
        lowerFaceStrength: 3,
        lowerFaceSmoothing: 0.02,
        faceMaskLevel: 0.6,
        faceMaskSoftness: 0.01,
        skinStrength: 1.0,
        eyelidOpenOffset: 0.0,
        lipOpenOffset: -0.3,
    },
    post_processing: {
        emotion_contrast: 2,
        live_blend_coef: 0.7,
        enable_preferred_emotion: false,  // let Audio2Emotion model infer from audio
        preferred_emotion_strength: 0.5,
        emotion_strength: 1.0,
        max_emotions: 3,
    },
};

// ─── Load protobuf definitions ───
function loadProtos() {
    const protoRoot = resolve(__dirname, 'protos');
    const a2fControllerProto = resolve(protoRoot, 'nvidia_ace.services.a2f_controller.v1.proto');

    if (!existsSync(a2fControllerProto)) {
        console.error(`Proto files not found at ${protoRoot}/`);
        console.error('Download them from: https://github.com/NVIDIA/ACE/tree/main/microservices/audio_2_face_microservice/1.2/proto/protobuf_files');
        process.exit(1);
    }

    const packageDef = protoLoader.loadSync(a2fControllerProto, {
        keepCase: true,
        longs: Number,
        enums: String,
        defaults: true,
        oneofs: true,
        includeDirs: [protoRoot],
    });

    return grpc.loadPackageDefinition(packageDef);
}

// ─── Load protobufjs root for Any decoding ───
let EmotionAggregateType = null;
function loadPbRoot() {
    const protoRoot = resolve(__dirname, 'protos');
    // protobufjs needs google/protobuf/any.proto — load the emotion protos via it
    const pbRoot = new protobuf.Root();
    pbRoot.resolvePath = (origin, target) => {
        // Let protobufjs find files relative to protoRoot
        if (target.startsWith('google/')) {
            // Use protobufjs's bundled well-known types
            return protobuf.common[target] ? null : resolve(protoRoot, target);
        }
        return resolve(protoRoot, target);
    };
    pbRoot.loadSync(resolve(protoRoot, 'nvidia_ace.emotion_aggregate.v1.proto'), { keepCase: true });
    pbRoot.resolveAll();
    EmotionAggregateType = pbRoot.lookupType('nvidia_ace.emotion_aggregate.v1.EmotionAggregate');
    console.log('[A2F Proxy] EmotionAggregate type loaded');
}

function decodeEmotionAny(anyMsg) {
    if (!anyMsg || !anyMsg.value || !EmotionAggregateType) return null;
    try {
        const ea = EmotionAggregateType.decode(anyMsg.value);
        // Debug: log which arrays have data
        const hasSmoothed = ea.a2f_smoothed_output?.length > 0;
        const hasA2E = ea.a2e_output?.length > 0;
        const hasInput = ea.input_emotions?.length > 0;
        if (hasSmoothed || hasA2E || hasInput) {
            console.log(`[A2F Proxy] Emotion arrays: smoothed=${ea.a2f_smoothed_output?.length}, a2e=${ea.a2e_output?.length}, input=${ea.input_emotions?.length}`);
        }
        const emotions = {};
        // Prefer a2e_output (raw A2E model output) over smoothed when smoothed is all zeros
        const source = hasSmoothed ? ea.a2f_smoothed_output
                     : hasA2E     ? ea.a2e_output
                     : hasInput   ? ea.input_emotions
                     : [];
        for (const etc of source) {
            if (etc.emotion) Object.assign(emotions, etc.emotion);
        }
        const hasNonZero = Object.values(emotions).some(v => v > 0.01);
        if (!hasNonZero && hasA2E) {
            // smoothed is all-zero, fall back to raw a2e
            for (const etc of ea.a2e_output) {
                if (etc.emotion) Object.assign(emotions, etc.emotion);
            }
        }
        return Object.keys(emotions).length > 0 ? emotions : null;
    } catch (e) {
        console.warn('[A2F Proxy] Failed to decode EmotionAggregate Any:', e.message);
        return null;
    }
}

// ─── Main ───
async function main() {
    if (!NVIDIA_API_KEY && !A2F_LOCAL) {
        console.error('NVIDIA_API_KEY not set (required for cloud NIM; set A2F_LOCAL=true for self-hosted)');
        process.exit(1);
    }

    loadPbRoot();
    const proto = loadProtos();

    // Navigate to the service
    const A2FService = proto.nvidia_ace.services.a2f_controller.v1.A2FControllerService;

    // Kill any existing process on the proxy port
    try {
        const pids = execSync(`lsof -ti:${PROXY_PORT}`, { encoding: 'utf8' }).trim();
        if (pids) {
            console.log(`[A2F Proxy] Port ${PROXY_PORT} in use (PID ${pids.replace(/\n/g, ', ')}), killing...`);
            execSync(`kill -9 ${pids.replace(/\n/g, ' ')}`);
            // Brief wait for OS to release the port
            await new Promise(r => setTimeout(r, 500));
        }
    } catch {
        // No process on port — good
    }

    const wss = new WebSocketServer({ port: PROXY_PORT });
    console.log(`[A2F Proxy] WebSocket server listening on ws://localhost:${PROXY_PORT}`);
    console.log(`[A2F Proxy] gRPC target: ${A2F_GRPC_URI}`);

    wss.on('connection', (ws, req) => {
        console.log(`[A2F Proxy] Client connected from ${req.socket.remoteAddress}`);
        handleClient(ws, A2FService);
    });
}

function handleClient(ws, A2FService) {
    // Create a new gRPC client and bidi stream per WebSocket connection
    const client = new A2FService(A2F_GRPC_URI,
        A2F_LOCAL
            ? grpc.credentials.createInsecure()
            : grpc.credentials.combineChannelCredentials(
                grpc.credentials.createSsl(),
                grpc.credentials.createFromMetadataGenerator((_, cb) => {
                    const meta = new grpc.Metadata();
                    meta.add('function-id', A2F_FUNCTION_ID);
                    meta.add('authorization', `Bearer ${NVIDIA_API_KEY}`);
                    cb(null, meta);
                })
            )
    );

    let stream = null;
    let streamActive = false;
    let streamGeneration = 0; // prevents stale callbacks from clobbering new streams

    function startStream() {
        // End old stream if any (fire-and-forget, don't await its end callback)
        if (stream) {
            try { stream.end(); } catch {}
        }
        stream = null;
        streamActive = false;

        const gen = ++streamGeneration;
        const newStream = client.ProcessAudioStream();
        stream = newStream;
        streamActive = true;
        console.log(`[A2F Proxy] gRPC bidi stream #${gen} started`);

        // Read responses and forward to WebSocket
        newStream.on('data', (message) => {
            // Ignore data from stale streams
            if (gen !== streamGeneration || ws.readyState !== WebSocket.OPEN) return;

            // Convert protobuf to JSON-friendly format
            const json = {};

            if (message.animation_data_stream_header) {
                const header = message.animation_data_stream_header;
                json.type = 'header';
                json.blendShapes = header.skel_animation_header?.blend_shapes || [];
                json.joints = header.skel_animation_header?.joints || [];
                console.log(`[A2F Proxy] Stream #${gen} header: ${json.blendShapes.length} blendshapes, ${json.joints.length} joints`);
            } else if (message.animation_data) {
                const ad = message.animation_data;
                json.type = 'animation';
                json.frames = [];

                if (ad.skel_animation?.blend_shape_weights) {
                    for (const frame of ad.skel_animation.blend_shape_weights) {
                        json.frames.push({
                            timeCode: frame.time_code,
                            values: Array.from(frame.values),
                        });
                    }
                }

                // Head rotations
                if (ad.skel_animation?.rotations?.length > 0) {
                    const r = ad.skel_animation.rotations[0];
                    if (r.values?.length > 0) {
                        const q = r.values[0];
                        json.headRotation = { real: q.real, i: q.i, j: q.j, k: q.k };
                    }
                }

                // Emotions from metadata (map<string, google.protobuf.Any>)
                if (ad.metadata) {
                    const metaKeys = Object.keys(ad.metadata);
                    if (metaKeys.length > 0) {
                        console.log(`[A2F Proxy] metadata keys: ${metaKeys.join(', ')}`);
                        const anyMsg = ad.metadata['emotion_aggregate'];
                        if (anyMsg) {
                            console.log(`[A2F Proxy] emotion_aggregate Any: type_url=${anyMsg.type_url}, bytes=${anyMsg.value?.length}`);
                            const emotions = decodeEmotionAny(anyMsg);
                            if (emotions) json.emotions = emotions;
                        }
                    }
                }
            } else if (message.event) {
                // A2F event (e.g. END_OF_A2F_AUDIO_PROCESSING) — skip
                return;
            } else if (message.status) {
                json.type = 'status';
                json.code = message.status.code;
                json.message = message.status.message;
                console.log(`[A2F Proxy] Stream #${gen} status: ${json.message} (${json.code})`);
            }

            ws.send(JSON.stringify(json));
        });

        newStream.on('error', (err) => {
            console.error(`[A2F Proxy] Stream #${gen} gRPC error:`, err.message);
            if (gen === streamGeneration) {
                streamActive = false;
                stream = null;
            }
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'error', message: err.message }));
            }
        });

        newStream.on('end', () => {
            console.log(`[A2F Proxy] Stream #${gen} gRPC ended`);
            // Only clear if this is still the current stream
            if (gen === streamGeneration) {
                streamActive = false;
                stream = null;
            }
        });

        // Send the header
        const header = {
            audio_stream_header: {
                audio_header: {
                    samples_per_second: A2F_SAMPLE_RATE,
                    bits_per_sample: BITS_PER_SAMPLE,
                    channel_count: CHANNEL_COUNT,
                    audio_format: 0, // PCM
                },
                emotion_post_processing_params: DEFAULT_FACE_CONFIG.post_processing,
                face_params: {
                    float_params: DEFAULT_FACE_CONFIG.face_params,
                },
                blendshape_params: {
                    bs_weight_multipliers: {},
                    bs_weight_offsets: {},
                },
            },
        };
        newStream.write(header);
        console.log(`[A2F Proxy] Stream #${gen} header sent`);
    }

    ws.on('message', (data) => {
        let msg;
        try {
            msg = JSON.parse(data.toString());
        } catch {
            console.warn('[A2F Proxy] Invalid JSON from client');
            return;
        }

        if (msg.type === 'start') {
            // Start a new gRPC stream
            startStream();
            return;
        }

        if (msg.type === 'audio' && stream && streamActive) {
            // Decode base64 PCM audio and forward to A2F.
            // Do NOT send hardcoded input emotions — let A2F's Audio2Emotion
            // model infer emotions from the audio signal automatically.
            const audioBytes = Buffer.from(msg.data, 'base64');
            stream.write({
                audio_with_emotion: {
                    audio_buffer: audioBytes,
                },
            });
            return;
        }

        if (msg.type === 'end' && stream && streamActive) {
            // Send EndOfAudio message, then half-close the gRPC write side.
            // The half-close is required for NVIDIA NIM to flush the remaining
            // tail frames immediately — without it the server waits ~5-9s before
            // flushing. Use the write callback to ensure END_STREAM is sent only
            // after end_of_audio is actually flushed to the transport.
            const s = stream;
            streamActive = false;
            s.write({ end_of_audio: {} }, (err) => {
                if (err) console.warn('[A2F Proxy] end_of_audio write error:', err.message);
                s.end();
                console.log('[A2F Proxy] End of audio sent + gRPC stream half-closed');
            });
            return;
        }

        if (msg.type === 'stop') {
            // Close the stream
            if (stream) {
                stream.end();
                stream = null;
                streamActive = false;
            }
            return;
        }
    });

    ws.on('close', () => {
        console.log('[A2F Proxy] Client disconnected');
        if (stream) {
            stream.end();
            stream = null;
            streamActive = false;
        }
        client.close();
    });
}

main().catch((err) => {
    console.error('Fatal:', err);
    process.exit(1);
});
