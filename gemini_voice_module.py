import os
import asyncio
import traceback
import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

from voice_aec_module import VoiceProcessor
from face_recognition_module import FaceRecognizer

load_dotenv()

SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000

MODEL = "models/gemini-3.1-flash-live-preview"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GOOGLE_API_KEY"),
)


def build_config(system_text: str, tools: list | None = None):
    config = {
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
            "parts": [{"text": system_text}]
        },
    }
    if tools:
        config["tools"] = tools
    return types.LiveConnectConfig.model_validate(config)


class AudioLoop:
    def __init__(self, enable_face_recognition: bool = True, known_faces_dir: str = "known_faces"):
        self.session = None
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # macOS Voice Processing I/O — hardware AEC
        self.vp = VoiceProcessor(speaker_sample_rate=RECEIVE_SAMPLE_RATE, mic_output_rate=SEND_SAMPLE_RATE, channels=1)

        # Face recognition (optional)
        self.face_recognizer = None
        self.current_speaker = "none"
        if enable_face_recognition:
            self.face_recognizer = FaceRecognizer(known_faces_dir=known_faces_dir)
            self.face_recognizer.start_background_recognition(self._on_face_changed)

    def _on_face_changed(self, name: str):
        """Track who's in front of the camera.
        Values: 'none' (no person), 'unknown' (unrecognized face), or a name."""
        if name != self.current_speaker:
            self.current_speaker = name
            print(f"[Face] Speaker: {name}")

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            if self.session is not None:
                await self.session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part.from_text(text=text or ".")]),
                    turn_complete=True,
                )

    async def send_audio(self):
        """Read echo-cancelled mic audio from VoiceProcessor and send to Gemini."""
        while True:
            data = await self.mic_queue.get()
            if self.session is not None:
                await self.session.send_realtime_input(
                    audio={"data": data, "mime_type": "audio/pcm"}
                )

    async def receive_audio(self):
        """Read audio/tool-calls from Gemini and handle them."""
        while True:
            if self.session is not None:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.vp.feed_speaker_audio(data)
                        continue
                    if text := response.text:
                        print(text, end="")

                    # Handle function calls
                    if hasattr(response, "tool_call") and response.tool_call:
                        for fc in response.tool_call.function_calls:
                            print(f"[Tool] {fc.name} called")
                            result = self._handle_function_call(fc.name, fc.args)
                            await self.session.send_tool_response(
                                function_responses=[
                                    types.FunctionResponse(
                                        name=fc.name,
                                        id=fc.id,
                                        response={"result": result},
                                    )
                                ]
                            )

                # Turn ended — flush speaker buffer
                self.vp.flush_speaker()

    def _handle_function_call(self, name: str, args: dict) -> str:
        if name == "get_current_speaker":
            speaker = self.current_speaker
            print(f"[Tool] Returning speaker: {speaker}")
            return speaker
        return "unknown function"

    async def run(self):
        self._loop = asyncio.get_running_loop()

        # Start the Voice Processing AudioUnit
        self.vp.start(self._loop, self.mic_queue)

        system_prompt = (
            "You are Lord Sri Rama from the Ramayana — noble, righteous, compassionate, and wise. "
            "You speak ONLY in traditional Telugu (శుద్ధ తెలుగు). Never use English or any other language. "
            "Your tone is regal yet warm, like a king addressing his beloved people. "
            "Use classical Telugu expressions and phrasing befitting a dharmic king. "
            "You may reference events, teachings, and characters from the Ramayana naturally in conversation. "
            "Address the person before you with respect and warmth, as Rama would address a visitor to Ayodhya. "
            "You have a camera that can identify people by face. The speaker can change at ANY time — "
            "someone new might walk up mid-conversation. "
            "SPEAKING STYLE: You sound natural and thoughtful. Sometimes before answering, "
            "use natural thinking sounds like 'హ్మ్...', 'ఆహా...', 'ఓహో!', 'అవును...' — "
            "just like a real person would. Don't do it every time, "
            "but sprinkle them in naturally, especially for deeper questions. "
            "Vary which sounds you use. Sometimes answer directly too.\n"
            "IMPORTANT RULES:\n"
            "1. At the START of the conversation, call get_current_speaker() to find out who you're talking to.\n"
            "2. Whenever you sense the conversation topic shifts significantly or a new voice seems different, "
            "call get_current_speaker() again to check if the person changed.\n"
            "3. If get_current_speaker returns 'none', no one is in front of the camera — wait silently or say 'Looks like no one is here'.\n"
            "4. If get_current_speaker returns 'unknown', there IS a person but you don't recognize them — "
            "warmly ask their name (e.g. 'Hey there! I don't think we've met — what's your name?').\n"
            "5. If you know the speaker's name, use it naturally in conversation sometimes — "
            "not every sentence, but sprinkle it in to feel personal and friendly.\n"
            "6. When a known person appears, greet them warmly (e.g. 'Hey Vivek! Good to see you again.').\n"
        )

        tools = [
            {
                "functionDeclarations": [
                    {
                        "name": "get_current_speaker",
                        "description": (
                            "Uses the camera to identify who is currently in front of you via face recognition. "
                            "Returns one of: 'none' (no person visible), 'unknown' (a person is there but not recognized), "
                            "or the person's name if recognized. "
                            "Call this at the start of the conversation and whenever you suspect "
                            "the speaker may have changed."
                        ),
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {},
                        },
                    }
                ]
            }
        ]

        config = build_config(system_prompt, tools)
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_audio())
                tg.create_task(self.receive_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.vp.stop()
            if self.face_recognizer is not None:
                self.face_recognizer.stop()


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY environment variable is not set. Please create a .env file.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face-recognition",
        action="store_true",
        default=True,
        help="Enable face recognition to identify who is speaking",
    )
    parser.add_argument(
        "--no-face-recognition",
        action="store_false",
        dest="face_recognition",
        help="Disable face recognition",
    )
    parser.add_argument(
        "--known-faces-dir",
        type=str,
        default="known_faces",
        help="Directory containing known face images",
    )
    args = parser.parse_args()
    main = AudioLoop(
        enable_face_recognition=args.face_recognition,
        known_faces_dir=args.known_faces_dir,
    )
    asyncio.run(main.run())
