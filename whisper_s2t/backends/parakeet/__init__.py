"""
Parakeet ASR Backend

NVIDIA Parakeet TDT (Token-and-Duration Transducer) models for state-of-the-art
English speech recognition. Uses the NVIDIA NeMo toolkit.

Supported models:
    - nvidia/parakeet-tdt-0.6b-v2 (recommended)
    - nvidia/parakeet-tdt-1.1b
    - Local .nemo files

Usage:
    import whisper_s2t
    
    # Load from NGC (downloads automatically)
    model = whisper_s2t.load_model("nvidia/parakeet-tdt-0.6b-v2", backend="Parakeet")
    
    # Load from local .nemo file
    model = whisper_s2t.load_model("models/parakeet-tdt-0.6b-v2.nemo", backend="Parakeet")
    
    # Transcribe
    result = model.transcribe_with_vad(["audio.wav"])
"""

from .model import ParakeetModel

__all__ = ["ParakeetModel"]
