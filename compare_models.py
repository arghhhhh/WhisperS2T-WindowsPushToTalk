#!/usr/bin/env python3
"""
Script to compare different Whisper models for accuracy and speed
"""

import os

import time
import whisper_s2t
import numpy as np
import tempfile
import wave

def create_test_audio():
    """Create a simple test audio file with a sine wave."""
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_signal = np.sin(frequency * 2 * np.pi * t)

    # Convert to 16-bit PCM
    audio_signal = (audio_signal * 32767).astype(np.int16)

    # Save as WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()

    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_signal.tobytes())

    return temp_file.name

def test_model(model_name, audio_file):
    """Test a specific model and return timing and result."""
    print(f"\n🧪 Testing {model_name} model...")

    try:
        # Load model
        start_load = time.time()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper_s2t.load_model(
            model_identifier=model_name,
            backend='CTranslate2',
            device=device
        )
        load_time = time.time() - start_load

        # Transcribe
        start_transcribe = time.time()
        out = model.transcribe_with_vad(
            audio_files=[audio_file],
            lang_codes=['en'],
            tasks=['transcribe'],
            initial_prompts=[None],
            batch_size=1
        )
        transcribe_time = time.time() - start_transcribe

        # Get result
        if out and len(out) > 0 and len(out[0]) > 0:
            transcription = out[0][0]['text'].strip()
        else:
            transcription = "No result"

        return {
            'model': model_name,
            'load_time': load_time,
            'transcribe_time': transcribe_time,
            'total_time': load_time + transcribe_time,
            'transcription': transcription,
            'success': True
        }

    except Exception as e:
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }

def main():
    """Compare different models."""
    print("🚀 WhisperS2T Model Comparison")
    print("=" * 50)
    print("Testing different model sizes on your RTX 4080...")
    print("Note: Larger models = better accuracy but slower speed")
    print()

    # Create test audio
    audio_file = create_test_audio()
    print(f"Created test audio: {audio_file}")

    # Test models (start with smaller ones first)
    models_to_test = ['tiny', 'base', 'small']  # Add 'medium', 'large-v2' if you want to test larger models

    results = []

    for model_name in models_to_test:
        result = test_model(model_name, audio_file)
        results.append(result)

    # Clean up
    os.unlink(audio_file)

    # Display results
    print("\n📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print("<12")
    print("-" * 60)

    for result in results:
        if result['success']:
            print("<12")
        else:
            print("<12")

    print("\n💡 RECOMMENDATIONS:")
    print("- For real-time conversation: Use 'base' or 'small'")
    print("- For maximum accuracy: Use 'large-v2' or 'large-v3'")
    print("- For fastest speed: Use 'tiny' (acceptable for clear speech)")
    print("\n🔄 To change the model in demo_mic.py, edit the model_identifier parameter")

if __name__ == "__main__":
    main()
