"""
Standalone Audio2Face test — speak into your mic and see blendshapes in real-time.
Opens the WebSocket server so you can also view the avatar in avatar_viewer.html.

Usage:
    python a2f_test.py
    # Then open avatar_viewer.html in a browser

Requires: NVIDIA_API_KEY and A2F_FUNCTION_ID in .env
"""

import os
import asyncio
import struct
from dotenv import load_dotenv

from voice_aec_module import VoiceProcessor
from audio2face_module import Audio2FaceClient

load_dotenv()

SAMPLE_RATE = 24000  # A2F expects 24kHz, VPIO runs at 24kHz


async def main():
    print("=== Audio2Face Standalone Test ===")
    print("Speak into your mic. Blendshapes will print to console.")
    print("Open avatar_viewer.html in a browser to see the 3D avatar.")
    print("Press Ctrl+C to stop.\n")

    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Voice processor for mic input (with AEC so we don't feed back speaker audio)
    vp = VoiceProcessor(speaker_sample_rate=SAMPLE_RATE, mic_output_rate=SAMPLE_RATE, channels=1)

    # Audio2Face client
    a2f = Audio2FaceClient()

    # Override the read loop to also print blendshapes
    original_latest = [None]

    vp.start(loop, mic_queue)
    print("[Mic] Voice processor started (24kHz)")

    # Start A2F + WebSocket server
    a2f_task = asyncio.create_task(a2f.start())
    print("[A2F] Connecting to A2F service...")

    # Wait a moment for A2F to connect
    await asyncio.sleep(1)

    # Read mic audio and feed to A2F
    frame_count = 0
    try:
        while True:
            data = await mic_queue.get()
            a2f.feed_audio(data)
            frame_count += 1

            # Print blendshapes periodically
            if a2f.latest_blendshapes and a2f.latest_blendshapes != original_latest[0]:
                original_latest[0] = a2f.latest_blendshapes
                # Show top 5 active blendshapes
                top = sorted(a2f.latest_blendshapes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                bs_str = "  ".join(f"{k}:{v:.2f}" for k, v in top)
                print(f"\r[BS] {bs_str}     ", end="", flush=True)

            if a2f.latest_emotions and frame_count % 50 == 0:
                top_emo = sorted(a2f.latest_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                emo_str = "  ".join(f"{k}:{v:.2f}" for k, v in top_emo if v > 0.05)
                if emo_str:
                    print(f"\n[Emotion] {emo_str}")

    except asyncio.CancelledError:
        pass
    finally:
        a2f.stop()
        vp.stop()
        print("\nStopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDone.")
