#!/usr/bin/env python3
"""
WhisperS2T Microphone Transcription Script

This script records audio from your microphone and transcribes it using WhisperS2T.
Press Ctrl+C to stop recording and transcribe.

Requirements:
- pyaudio (for microphone recording)
- whisper_s2t (the main transcription library)
- numpy
"""

import os

import sys
import tempfile
import wave
import time
from typing import Optional

import numpy as np
import pyaudio
import whisper_s2t

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz
RECORD_SECONDS = 5  # Default recording duration

class MicrophoneTranscriber:
    def __init__(self, model_size: str = "base", backend: str = "CTranslate2"):
        """Initialize the transcriber with specified model.

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            backend: Backend to use ('CTranslate2', 'HuggingFace', 'OpenAI')
        """
        print(f"Loading WhisperS2T model: {model_size} ({backend})...")
        # Use CUDA if available, otherwise CPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.model = whisper_s2t.load_model(
            model_identifier=model_size,
            backend=backend,
            device=device
        )
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        print("Model loaded successfully!")

    def record_audio(self, duration: int = RECORD_SECONDS, device_index: Optional[int] = None) -> str:
        """Record audio from microphone for specified duration.

        Args:
            duration: Recording duration in seconds
            device_index: Specific device index to use (None for default)

        Returns:
            Path to the recorded WAV file
        """
        if device_index is not None:
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
                print(f"🎤 Using device: {device_info['name']} (index {device_index})")
            except:
                print(f"⚠️  Warning: Could not get info for device {device_index}")
        else:
            try:
                default_device = self.audio.get_default_input_device_info()
                print(f"🎤 Using default device: {default_device['name']} (index {default_device['index']})")
            except:
                print("🎤 Using system default microphone")

        print(f"Recording for {duration} seconds... (Press Ctrl+C to stop early)")

        # Open audio stream
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )

        frames = []

        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                # Show progress
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                print(f"\rRecording: {elapsed:.1f}s / {duration}s", end="", flush=True)

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        finally:
            print("\nRecording finished.")

        # Stop and close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()

        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        return temp_file.name

    def transcribe_file(self, audio_file: str, language: str = "en") -> str:
        """Transcribe an audio file.

        Args:
            audio_file: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            Transcribed text
        """
        print("Transcribing...")

        # Use the transcribe_with_vad method for better results
        out = self.model.transcribe_with_vad(
            audio_files=[audio_file],
            lang_codes=[language],
            tasks=["transcribe"],
            initial_prompts=[None],
            batch_size=1  # Small batch size for single file
        )

        # Extract the text from the result
        if out and len(out) > 0 and len(out[0]) > 0:
            transcription = out[0][0]['text']
            return transcription.strip()
        else:
            return ""

    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main():
    """Main function to run the microphone transcriber."""
    import argparse

    parser = argparse.ArgumentParser(description="WhisperS2T Microphone Transcription")
    parser.add_argument("--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--backend", default="CTranslate2",
                       choices=["CTranslate2", "HuggingFace", "OpenAI"],
                       help="Backend to use (default: CTranslate2)")
    parser.add_argument("--duration", type=int, default=RECORD_SECONDS,
                       help=f"Recording duration in seconds (default: {RECORD_SECONDS})")
    parser.add_argument("--language", default="en",
                       help="Language code for transcription (default: en)")
    parser.add_argument("--device", type=int, default=None,
                       help="Microphone device index (use list_microphones.py to see options)")
    parser.add_argument("--continuous", action="store_true",
                       help="Run in continuous mode (record and transcribe repeatedly)")

    args = parser.parse_args()

    try:
        # Initialize transcriber
        transcriber = MicrophoneTranscriber(model_size=args.model, backend=args.backend)

        if args.continuous:
            print("Starting continuous transcription mode...")
            print("Press Ctrl+C to exit.\n")

            while True:
                try:
                    # Record audio
                    audio_file = transcriber.record_audio(duration=args.duration, device_index=args.device)

                    # Transcribe
                    transcription = transcriber.transcribe_file(audio_file, args.language)

                    # Display result
                    print("Transcription:")
                    print(f"'{transcription}'")
                    print("-" * 50)

                    # Clean up temporary file
                    os.unlink(audio_file)

                except KeyboardInterrupt:
                    print("\nStopping continuous mode...")
                    break
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    continue
        else:
            # Single recording mode
            try:
                # Record audio
                audio_file = transcriber.record_audio(duration=args.duration, device_index=args.device)

                # Transcribe
                transcription = transcriber.transcribe_file(audio_file, args.language)

                # Display result
                print("\nTranscription:")
                print(f"'{transcription}'")

                # Clean up temporary file
                os.unlink(audio_file)

            except KeyboardInterrupt:
                print("\nRecording cancelled.")
            except Exception as e:
                print(f"Error: {e}")
                return 1

    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        return 1

    finally:
        if 'transcriber' in locals():
            transcriber.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
