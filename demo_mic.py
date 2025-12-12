#!/usr/bin/env python3
"""
Simple demo script for WhisperS2T microphone transcription
"""

import os

import sys
import whisper_s2t
import numpy as np
import pyaudio
import wave
import tempfile
import time

def record_audio(duration=3, device_index=None):
    """Record audio from microphone for specified duration."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    audio = pyaudio.PyAudio()

    # Show selected device info
    if device_index is not None:
        try:
            device_info = audio.get_device_info_by_index(device_index)
            print(f"🎤 Using device: {device_info['name']} (index {device_index})")
        except:
            print(f"⚠️  Warning: Could not get info for device {device_index}")
    else:
        try:
            default_device = audio.get_default_input_device_info()
            print(f"🎤 Using default device: {default_device['name']} (index {default_device['index']})")
        except:
            print("🎤 Using system default microphone")

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )

    print(f"Recording for {duration} seconds... Press Ctrl+C to stop early")

    frames = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            elapsed = time.time() - start_time
            print(f"\rRecording: {elapsed:.1f}s / {duration}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        print("\nRecording finished.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()

    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return temp_file.name

def main():
    """Demo of microphone transcription."""
    import argparse

    parser = argparse.ArgumentParser(description="WhisperS2T Microphone Demo")
    parser.add_argument("--device", type=int, default=None,
                       help="Microphone device index (use list_microphones.py to see options)")
    parser.add_argument("--duration", type=float, default=3.0,
                       help="Recording duration in seconds (default: 3.0)")

    args = parser.parse_args()

    print("WhisperS2T Microphone Demo")
    print("=" * 40)

    try:
        # Load model
        print("Loading Whisper model (tiny, CUDA enabled)...")
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = whisper_s2t.load_model(
            model_identifier="base",  # Use base for better accuracy (you can change to "small", "medium", or "large-v2")
            backend='CTranslate2',
            device=device
        )
        print("Model loaded successfully!")

        # Record audio
        audio_file = record_audio(duration=args.duration, device_index=args.device)

        # Transcribe
        print("Transcribing...")
        start_time = time.time()

        out = model.transcribe_with_vad(
            audio_files=[audio_file],
            lang_codes=['en'],
            tasks=['transcribe'],
            initial_prompts=[None],
            batch_size=1
        )

        end_time = time.time()

        # Display result
        if out and len(out) > 0 and len(out[0]) > 0:
            transcription = out[0][0]['text'].strip()
            duration = end_time - start_time

            print("\nTranscription:")
            print(f"'{transcription}'")
            print(".2f")
        else:
            print("No transcription result")

        # Clean up
        os.unlink(audio_file)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nDemo completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
