import os
import asyncio
import io
import pyaudio
from google import genai
from google.genai import types

# Requires GOOGLE_API_KEY to be set in the environment

class GeminiVoiceChat:
    def __init__(self):
        self.model = "models/gemini-2.0-flash-exp" # Adjust to gemini-3.0-flash if officially available
        self.client = genai.Client()
        
        # Audio configuration
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000 # Gemini expects 16kHz
        self.pyaudio_instance = pyaudio.PyAudio()

    async def _audio_streamer(self, session):
        """Captures microphone audio and sends it to the Gemini session."""
        stream = self.pyaudio_instance.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        try:
            print("[Microphone active - Speak now]")
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                # Send the audio snippet to Gemini
                await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()

    async def _audio_receiver(self, session):
        """Receives audio chunks from Gemini and plays them out loud."""
        stream = self.pyaudio_instance.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK
        )
        try:
            async for response in session.receive():
                server_content = response.server_content
                if server_content is not None:
                    interrupted = server_content.interrupted
                    if interrupted:
                        # Handle interruption logic if needed
                        continue
                    
                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                # Play received audio
                                stream.write(part.inline_data.data)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        finally:
            stream.stop_stream()
            stream.close()

    async def start_conversation(self, person_name="Unknown"):
        """Starts a live voice connection contextualized with the person's name."""
        print(f"Starting Gemini Voice Conversation with context: talking to {person_name}...")
        
        system_instruction = types.Content(
            parts=[types.Part.from_text(f"You are a helpful AI assistant. You are currently talking to someone named {person_name}. You must respond directly via voice. Keep your answers concise for quick real-time conversation.")]
        )
        
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=system_instruction
        )
        
        try:
            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                # Run mic capture and speaker playback concurrently
                print("Connection established. Start speaking!")
                streamer_task = asyncio.create_task(self._audio_streamer(session))
                receiver_task = asyncio.create_task(self._audio_receiver(session))
                
                await asyncio.gather(streamer_task, receiver_task)
                
        except Exception as e:
            print(f"Error during conversation: {e}")

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    # Load environment variables from a .env file
    load_dotenv()
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("Please create a .env file and add: GOOGLE_API_KEY='your_api_key_here'")
        sys.exit(1)
        
    chat = GeminiVoiceChat()
    
    # Normally this name defaults to what face recognition provides
    person_id = "Vivek"
    
    try:
        asyncio.run(chat.start_conversation(person_name=person_id))
    except KeyboardInterrupt:
        print("\nConversation ended.")
