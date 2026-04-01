import os
import asyncio
import traceback
import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

from voice_aec_module import VoiceProcessor
from face_recognition_module import FaceRecognizer
from audio2face_module import Audio2FaceClient

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
    def __init__(self, enable_face_recognition: bool = True, known_faces_dir: str = "known_faces",
                 enable_a2f: bool = False):
        self.session = None
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # macOS Voice Processing I/O — hardware AEC
        self.vp = VoiceProcessor(speaker_sample_rate=RECEIVE_SAMPLE_RATE, mic_output_rate=SEND_SAMPLE_RATE, channels=1)

        # NVIDIA Audio2Face (optional)
        self.a2f: Audio2FaceClient | None = None
        if enable_a2f:
            self.a2f = Audio2FaceClient()
            print("[A2F] Audio2Face enabled")

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
                        # Also feed to Audio2Face for avatar animation
                        if self.a2f:
                            self.a2f.feed_audio(data)
                        continue
                    if text := response.text:
                        print(text, end="")

                    # Log Google Search grounding
                    if hasattr(response, "server_content") and response.server_content:
                        sc = response.server_content
                        if hasattr(sc, "grounding_metadata") and sc.grounding_metadata:
                            print("[Google Search] Grounding used")

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
                # Signal end of audio segment to A2F
                if self.a2f:
                    self.a2f.stop_audio()

    def _handle_function_call(self, name: str, args: dict) -> str:
        if name == "get_current_speaker":
            speaker = self.current_speaker
            print(f"[Tool] Returning speaker: {speaker}")
            return speaker
        if name == "lookup_signova_products":
            query = args.get("query", "")
            print(f"[Tool] Product lookup: {query}")
            return self._product_data
        return "unknown function"

    def _load_product_data(self) -> str:
        path = os.path.join(os.path.dirname(__file__), "signova.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Product data file not found."

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._product_data = self._load_product_data()

        # Start the Voice Processing AudioUnit
        self.vp.start(self._loop, self.mic_queue)

        system_prompt = (
            "You are a friendly, experienced farmer helper (రైతు మిత్రుడు) from Signova Fertilizers. "
            "You speak ONLY in Telugu (తెలుగు). Never use English or any other language. "
            "Your tone is warm, down-to-earth, and caring — like a trusted neighbor who knows farming inside out. "
            "You genuinely care about farmers and their crops. "
            "You are an expert on all Signova fertilizer products and crop nutrition. "
            "When a farmer describes a problem (yellowing leaves, stunted growth, poor yield, pest issues, etc.), "
            "you should: diagnose the likely cause, recommend the right Signova product, "
            "and explain the dosage and application method clearly. "
            "Use the lookup_signova_products tool to get detailed product info before recommending. "
            "Always suggest specific Signova products by name with correct dosage. "
            "If you're unsure about the crop issue, ask clarifying questions about the crop type, "
            "symptoms, soil type, and growth stage. "
            "You have a camera that can identify people by face. The speaker can change at ANY time — "
            "someone new might walk up mid-conversation. "
            "SPEAKING STYLE: You sound natural and warm. Sometimes before answering, "
            "use natural thinking sounds like 'హ్మ్...', 'ఆహా...', 'ఓహో!', 'అవును...', 'అలాగా...' — "
            "just like a real person would. Don't do it every time, "
            "but sprinkle them in naturally. Vary which sounds you use. Sometimes answer directly too.\n"
            "IMPORTANT RULES:\n"
            "1. At the START of the conversation, call get_current_speaker() to find out who you're talking to.\n"
            "2. Whenever you sense the conversation topic shifts significantly or a new voice seems different, "
            "call get_current_speaker() again to check if the person changed.\n"
            "3. If get_current_speaker returns 'none', no one is in front of the camera — "
            "say something like 'ఎవరూ కనిపించడం లేదు, ఎవరైనా ఉంటే రండి!' (no one visible, come if anyone is there).\n"
            "4. If get_current_speaker returns 'unknown', there IS a person but you don't recognize them — "
            "warmly ask their name in Telugu (e.g. 'నమస్కారం! మీ పేరు చెప్పగలరా?').\n"
            "5. If you know the speaker's name, use it naturally in conversation sometimes — "
            "not every sentence, but sprinkle it in to feel personal and friendly.\n"
            "6. When a known person appears, greet them warmly in Telugu.\n"
            "7. ALWAYS call lookup_signova_products before recommending any product, "
            "so you give accurate names, dosages, and usage info.\n"
            "8. For weather queries, crop season info, market prices, regional agricultural advice, "
            "or any real-time information, use Google Search. When a farmer asks about weather in their area, "
            "current crop prices, best crops for the season, or pest/disease outbreaks, search for it.\n"
        )

        tools = [
            {"googleSearch": {}},
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
                    },
                    {
                        "name": "lookup_signova_products",
                        "description": (
                            "Looks up Signova fertilizer product catalog including all products, their uses, "
                            "dosages, available sizes, and which crops/deficiencies they address. "
                            "Call this before recommending any Signova product to a farmer so you give "
                            "accurate product names, dosages, and application methods."
                        ),
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                "query": {
                                    "type": "STRING",
                                    "description": "The farmer's problem or crop issue to look up products for.",
                                }
                            },
                        },
                    },
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

                # Start Audio2Face streaming if enabled
                if self.a2f:
                    tg.create_task(self.a2f.start())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            self.vp.stop()
            if self.a2f:
                self.a2f.stop()
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
    parser.add_argument(
        "--a2f",
        action="store_true",
        default=False,
        help="Enable NVIDIA Audio2Face for 3D avatar animation",
    )
    args = parser.parse_args()
    main = AudioLoop(
        enable_face_recognition=args.face_recognition,
        known_faces_dir=args.known_faces_dir,
        enable_a2f=args.a2f,
    )
    asyncio.run(main.run())
