"""
NVIDIA Audio2Face integration module.

Streams audio from Gemini Live API to NVIDIA Audio2Face NIM (cloud or local)
via gRPC and receives ARKit blendshape weights + emotions in real-time.
Blendshapes are forwarded to connected WebSocket clients (e.g., a Three.js avatar viewer).

Requirements:
    pip install nvidia_ace-1.0.0-py3-none-any.whl   # from NVIDIA ACE repo
    # OR generate from protos — see NVIDIA ACE repo for instructions

Environment variables:
    NVIDIA_API_KEY       — NGC API key (for cloud NIM)
    A2F_FUNCTION_ID      — Function ID for A2F NIM (cloud)
    A2F_GRPC_URI         — gRPC endpoint (default: grpc.nvcf.nvidia.com:443 for cloud,
                           localhost:50000 for local)
    A2F_USE_SSL          — "true" for cloud, "false" for local (default: true)
"""

import os
import asyncio
import json
import struct
import traceback
import threading
import http.server
import functools
from collections import deque

import grpc
import websockets

# NVIDIA ACE protobuf imports
from nvidia_ace.services.a2f_controller.v1_pb2_grpc import A2FControllerServiceStub
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.a2f.v1_pb2 import (
    AudioWithEmotion,
    EmotionPostProcessingParameters,
    FaceParameters,
    BlendShapeParameters,
)
from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader
from nvidia_ace.emotion_aggregate.v1_pb2 import EmotionAggregate
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode

# Audio format constants (A2F expects 16-bit PCM mono)
A2F_SAMPLE_RATE = 24000  # Match Gemini output rate
BITS_PER_SAMPLE = 16
CHANNEL_COUNT = 1


class Audio2FaceClient:
    """
    Streams audio to NVIDIA Audio2Face and receives blendshape animations.
    Forwards blendshapes to WebSocket clients for 3D avatar rendering.
    """

    def __init__(
        self,
        api_key: str | None = None,
        function_id: str | None = None,
        grpc_uri: str | None = None,
        use_ssl: bool | None = None,
        ws_port: int = 8765,
        face_config: dict | None = None,
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.function_id = function_id or os.environ.get("A2F_FUNCTION_ID", "")
        self.grpc_uri = grpc_uri or os.environ.get(
            "A2F_GRPC_URI", "grpc.nvcf.nvidia.com:443"
        )
        self.use_ssl = use_ssl if use_ssl is not None else (
            os.environ.get("A2F_USE_SSL", "true").lower() == "true"
        )
        self.ws_port = ws_port

        # Face/emotion config defaults
        self.face_config = face_config or self._default_face_config()

        # Audio buffer for streaming to A2F
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        # WebSocket clients
        self._ws_clients: set = set()

        # Latest blendshapes for polling
        self.latest_blendshapes: dict | None = None
        self.latest_emotions: dict | None = None

        # Playback timing: track when audio feeding started to pace blendshape output
        self._playback_start_time: float | None = None
        self._audio_bytes_fed: int = 0  # total PCM bytes fed since start

        # State
        self._stream = None
        self._running = False

    @staticmethod
    def _default_face_config() -> dict:
        return {
            "face_parameters": {
                "upperFaceStrength": 5,
                "upperFaceSmoothing": 0.02,
                "lowerFaceStrength": 2,
                "lowerFaceSmoothing": 0.02,
                "faceMaskLevel": 0.6,
                "faceMaskSoftness": 0.01,
                "skinStrength": 1.0,
                "eyelidOpenOffset": 0.0,
                "lipOpenOffset": 0.0,
            },
            "post_processing_parameters": {
                "emotion_contrast": 1,
                "live_blend_coef": 0.7,
                "enable_preferred_emotion": True,
                "preferred_emotion_strength": 0.75,
                "emotion_strength": 1.2,
                "max_emotions": 3,
            },
            "blendshape_multipliers": {},
            "blendshape_offsets": {},
        }

    def _create_channel(self) -> grpc.aio.Channel:
        """Create gRPC channel to A2F service."""
        if self.use_ssl:
            creds = grpc.ssl_channel_credentials()
            if self.api_key and self.function_id:
                metadata = [
                    ("function-id", self.function_id),
                    ("authorization", f"Bearer {self.api_key}"),
                ]

                def metadata_callback(context, callback):
                    callback(metadata, None)

                auth_creds = grpc.metadata_call_credentials(metadata_callback)
                creds = grpc.composite_channel_credentials(creds, auth_creds)
            return grpc.aio.secure_channel(self.grpc_uri, creds)
        else:
            return grpc.aio.insecure_channel(self.grpc_uri)

    def _build_stream_header(self) -> AudioStream:
        """Build the initial AudioStreamHeader message."""
        fc = self.face_config
        return AudioStream(
            audio_stream_header=AudioStreamHeader(
                audio_header=AudioHeader(
                    samples_per_second=A2F_SAMPLE_RATE,
                    bits_per_sample=BITS_PER_SAMPLE,
                    channel_count=CHANNEL_COUNT,
                    audio_format=AudioHeader.AUDIO_FORMAT_PCM,
                ),
                emotion_post_processing_params=EmotionPostProcessingParameters(
                    **fc["post_processing_parameters"]
                ),
                face_params=FaceParameters(
                    float_params=fc["face_parameters"]
                ),
                blendshape_params=BlendShapeParameters(
                    bs_weight_multipliers=fc.get("blendshape_multipliers", {}),
                    bs_weight_offsets=fc.get("blendshape_offsets", {}),
                ),
            )
        )

    def feed_audio(self, audio_data: bytes):
        """
        Feed audio data (PCM 16-bit mono) to the A2F pipeline.
        Call this with the same audio chunks sent to the speaker.
        """
        if self._running:
            import time
            if self._playback_start_time is None:
                self._playback_start_time = time.monotonic()
                self._audio_bytes_fed = 0
            self._audio_bytes_fed += len(audio_data)
            self._audio_queue.put_nowait(audio_data)

    def stop_audio(self):
        """Signal end of current audio stream."""
        if self._running:
            self._audio_queue.put_nowait(None)
            self._playback_start_time = None
            self._audio_bytes_fed = 0

    async def _write_audio_stream(self, stream):
        """Write audio chunks to the A2F gRPC stream."""
        # Send header first
        await stream.write(self._build_stream_header())
        print("[A2F] Stream header sent")

        first_chunk = True
        while True:
            chunk = await self._audio_queue.get()
            if chunk is None:
                # End of audio
                await stream.write(
                    AudioStream(end_of_audio=AudioStream.EndOfAudio())
                )
                print("[A2F] End of audio sent")
                break

            msg = AudioStream(
                audio_with_emotion=AudioWithEmotion(
                    audio_buffer=chunk,
                    emotions=[EmotionWithTimeCode(
                        time_code=0.0,
                        emotion={"Joy": 1.0},
                    )],
                )
            )
            await stream.write(msg)

            if first_chunk:
                print("[A2F] First audio chunk sent")
                first_chunk = False

    async def _read_animation_stream(self, stream):
        """Read blendshape animation data from A2F and broadcast via WebSocket, paced to audio playback."""
        import time
        bs_names = []

        while True:
            message = await stream.read()
            if message == grpc.aio.EOF:
                print("[A2F] Stream ended")
                break

            if message.HasField("animation_data_stream_header"):
                header: AnimationDataStreamHeader = message.animation_data_stream_header
                bs_names = list(
                    header.skel_animation_header.blend_shapes
                )
                print(f"[A2F] Received header with {len(bs_names)} blendshapes")

            elif message.HasField("animation_data"):
                animation_data: AnimationData = message.animation_data

                # Extract emotions
                emotions = {}
                emotion_aggregate = EmotionAggregate()
                if (
                    "emotion_aggregate" in animation_data.metadata
                    and animation_data.metadata["emotion_aggregate"].Unpack(
                        emotion_aggregate
                    )
                ):
                    for etc in emotion_aggregate.a2f_smoothed_output:
                        emotions = dict(etc.emotion)

                self.latest_emotions = emotions

                # Extract blendshapes — pace to match audio playback
                for bs_frame in animation_data.skel_animation.blend_shape_weights:
                    bs_dict = dict(zip(bs_names, bs_frame.values))
                    self.latest_blendshapes = bs_dict

                    # Wait until the right time to send this frame
                    if self._playback_start_time is not None:
                        frame_time = bs_frame.time_code
                        elapsed = time.monotonic() - self._playback_start_time
                        delay = frame_time - elapsed
                        if delay > 0.001:
                            await asyncio.sleep(delay)

                    # Broadcast to WebSocket clients
                    payload = json.dumps({
                        "type": "blendshapes",
                        "timeCode": bs_frame.time_code,
                        "blendShapes": bs_dict,
                        "emotions": emotions,
                    })
                    await self._broadcast_ws(payload)

            elif message.HasField("status"):
                status = message.status
                print(f"[A2F] Status: {status.message} (code: {status.code})")

    async def _broadcast_ws(self, message: str):
        """Send message to all connected WebSocket clients."""
        if not self._ws_clients:
            return
        disconnected = set()
        for ws in self._ws_clients:
            try:
                await ws.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(ws)
        self._ws_clients -= disconnected

    async def _ws_handler(self, websocket):
        """Handle a new WebSocket connection."""
        self._ws_clients.add(websocket)
        print(f"[A2F-WS] Client connected ({len(self._ws_clients)} total)")
        try:
            async for msg in websocket:
                pass  # We only send, not receive
        finally:
            self._ws_clients.discard(websocket)
            print(f"[A2F-WS] Client disconnected ({len(self._ws_clients)} total)")

    async def run_stream(self):
        """
        Main loop: connects to A2F, streams audio, receives blendshapes.
        Reconnects automatically for each new audio segment.
        """
        self._running = True
        channel = self._create_channel()
        stub = A2FControllerServiceStub(channel)

        try:
            while self._running:
                print("[A2F] Starting new stream...")
                stream = stub.ProcessAudioStream()
                self._stream = stream

                writer = asyncio.create_task(self._write_audio_stream(stream))
                reader = asyncio.create_task(self._read_animation_stream(stream))

                # Run both concurrently; if one fails, cancel the other
                done, pending = await asyncio.wait(
                    [writer, reader], return_when=asyncio.FIRST_EXCEPTION
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    if task.exception():
                        # Try to get server error details
                        try:
                            code = await stream.code()
                            details = await stream.details()
                            print(f"[A2F] gRPC error: code={code}, details={details}")
                        except Exception:
                            pass
                        raise task.exception()

                print("[A2F] Stream cycle complete, waiting for new audio...")

                # Clear queue for next round
                while not self._audio_queue.empty():
                    self._audio_queue.get_nowait()

        except asyncio.CancelledError:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            await channel.close()
            self._running = False

    async def start(self, loop=None):
        """Start the A2F client and WebSocket server."""
        self._running = True

        # Start HTTP server for avatar_viewer.html on port 8000 (background thread)
        serve_dir = os.path.dirname(os.path.abspath(__file__))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)
        httpd = http.server.HTTPServer(("localhost", 8000), handler)
        http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        http_thread.start()
        print(f"[A2F] Avatar viewer at http://localhost:8000/avatar_viewer.html")

        # Start WebSocket server for avatar viewer
        self._ws_server = await websockets.serve(
            self._ws_handler, "localhost", self.ws_port
        )
        print(f"[A2F-WS] WebSocket server on ws://localhost:{self.ws_port}")

        # Run gRPC stream (reconnects on failure)
        while self._running:
            try:
                await self.run_stream()
            except Exception:
                traceback.print_exc()
                if self._running:
                    print("[A2F] Retrying in 3 seconds...")
                    await asyncio.sleep(3)

    def stop(self):
        """Stop the A2F client."""
        self._running = False
        self._audio_queue.put_nowait(None)
