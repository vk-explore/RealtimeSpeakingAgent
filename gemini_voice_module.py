import os
import asyncio
import traceback
import time
import pyaudio
import array
import math

import argparse

from google import genai
from google.genai import types
from google.genai.types import Type
from dotenv import load_dotenv

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-3.1-flash-live-preview"
# NOTE: The bidi streaming currently officially works on gemini-2.0-flash, but you can try gemini-3.1-flash-preview if you have access
# MODEL = "models/gemini-3.1-flash-live-preview"

DEFAULT_MODE = "none"  # audio only

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GOOGLE_API_KEY"), # updated to use the right .env key name you set
)


CONFIG = types.LiveConnectConfig.model_validate(
    {
        "responseModalities": ["AUDIO"],
        "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
        "realtimeInputConfig": {
            "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            "turnCoverage": "TURN_INCLUDES_ONLY_ACTIVITY",
            "automaticActivityDetection": {
                "disabled": False,
                "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                "silenceDurationMs": 400,
            },
        },
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": "Zephyr"}
            }
        },
        "contextWindowCompression": {
            "triggerTokens": 104857,
            "slidingWindow": {"targetTokens": 52428},
        },
        "systemInstruction": {
            "parts": [
                {
                    "text": "You are a helpful AI assistant in a real-time voice conversation. Keep your answers concise and natural."
                }
            ]
        },
    }
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.audio_out_queue = None  # mic audio only

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.audio_stream = None
        self.is_model_speaking = False
        self.model_speaking_until = 0.0
        self.barge_in_until = 0.0
        self.suppress_model_audio_until = 0.0
        self.interrupt_threshold = 2500
        self.interrupt_chunk_count = 0
        self.interrupt_required_chunks = 3
        
        self._loop = None  # Will be set when run() starts

    def _pcm_rms(self, data):
        samples = array.array("h")
        samples.frombytes(data)
        if not samples:
            return 0.0
        square_sum = sum(sample * sample for sample in samples)
        return math.sqrt(square_sum / len(samples))

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            if self.session is not None:
                await self.session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part.from_text(text=text or ".")]),
                    turn_complete=True
                )

    async def send_audio(self):
        """Sends mic audio to Gemini in realtime."""
        assert self.audio_out_queue is not None
        while True:
            data = await self.audio_out_queue.get()
            if self.session is not None:
                await self.session.send_realtime_input(
                    audio={"data": data, "mime_type": "audio/pcm"}
                )

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            now = time.monotonic()
            mic_level = self._pcm_rms(data)

            if self.is_model_speaking or now < self.model_speaking_until:
                if mic_level >= self.interrupt_threshold:
                    self.interrupt_chunk_count += 1
                else:
                    self.interrupt_chunk_count = 0

                if self.interrupt_chunk_count >= self.interrupt_required_chunks:
                    self.barge_in_until = now + 0.8
                    self.is_model_speaking = False
                    self.model_speaking_until = 0.0
                    self.suppress_model_audio_until = now + 1.0
                    self.interrupt_chunk_count = 0
                    if self.audio_in_queue is not None:
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()
                    if self.session is not None:
                        await self.session.send_realtime_input(activity_start={})
                elif now >= self.barge_in_until:
                    continue
            else:
                self.interrupt_chunk_count = 0

            if self.audio_out_queue is not None:
                await self.audio_out_queue.put(data)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        assert self.audio_in_queue is not None
        while True:
            if self.session is not None:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        if time.monotonic() < self.suppress_model_audio_until:
                            continue
                        self.is_model_speaking = True
                        self.model_speaking_until = time.monotonic() + 0.35
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="")

                self.is_model_speaking = False
                self.model_speaking_until = time.monotonic() + 0.15

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            if self.audio_in_queue is not None:
                bytestream = await self.audio_in_queue.get()
                self.is_model_speaking = True
                self.model_speaking_until = time.monotonic() + 0.35
                await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        self._loop = asyncio.get_running_loop()  # Capture loop before spawning threads
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.audio_out_queue = asyncio.Queue()          # unbounded mic audio queue

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_audio())
                tg.create_task(self.listen_audio())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream is not None:
                self.audio_stream.close()
                traceback.print_exception(EG)


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY environment variable is not set. Please create a .env file.")
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())
