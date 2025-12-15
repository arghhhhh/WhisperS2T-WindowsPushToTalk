"""
Parakeet ASR Model Backend

Wraps NVIDIA NeMo's Parakeet TDT models to provide a compatible interface
with the WhisperS2T transcription system.
"""

import os
from typing import List, Optional, Dict, Any

import torch

from ..base import ASRModelBase


class ParakeetModel(ASRModelBase):
    """
    Parakeet TDT (Token-and-Duration Transducer) ASR model.
    
    NVIDIA's state-of-the-art English speech recognition model.
    Achieves better accuracy than Whisper large-v3 on English benchmarks
    while being faster for real-time transcription.
    
    Note: Parakeet models are English-only. For multilingual support,
    use a Whisper backend instead.
    
    Args:
        model_name_or_path: Either:
            - NGC model name (e.g., "nvidia/parakeet-tdt-0.6b-v2")
            - Path to local .nemo file
        device: Device to run inference on ("cuda" or "cpu")
        device_index: GPU index if using CUDA
        compute_type: Precision for inference ("float16", "float32", "bfloat16")
        **kwargs: Additional arguments (ignored for compatibility)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "float16",
        **kwargs
    ):
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.model_path = model_name_or_path
        
        # Import NeMo (deferred to avoid import overhead if not using Parakeet)
        try:
            import nemo.collections.asr as nemo_asr
            self.nemo_asr = nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for Parakeet backend. "
                "Install with: pip install nemo_toolkit[asr]"
            )
        
        # Load model
        self.model = self._load_model(model_name_or_path)
        
        # Move to device
        if device == "cuda":
            if torch.cuda.is_available():
                torch.cuda.set_device(device_index)
                self.model = self.model.cuda()
            else:
                print("Warning: CUDA not available, falling back to CPU")
                self.device = "cpu"
        
        # Note: Don't convert model precision here - NeMo handles this internally
        # and manual .half() conversion can break inference for some models
        
        # Set to eval mode
        self.model.eval()
        
    def _load_model(self, model_name_or_path: str):
        """Load model from local file or NGC."""
        
        # Check if it's a local .nemo file
        if os.path.isfile(model_name_or_path) and model_name_or_path.endswith('.nemo'):
            print(f"Loading Parakeet model from local file: {model_name_or_path}")
            return self.nemo_asr.models.ASRModel.restore_from(model_name_or_path)
        
        # Check if it's a local directory with a .nemo file
        if os.path.isdir(model_name_or_path):
            nemo_files = [f for f in os.listdir(model_name_or_path) if f.endswith('.nemo')]
            if nemo_files:
                nemo_path = os.path.join(model_name_or_path, nemo_files[0])
                print(f"Loading Parakeet model from: {nemo_path}")
                return self.nemo_asr.models.ASRModel.restore_from(nemo_path)
        
        # Try to load from NGC/HuggingFace
        print(f"Downloading Parakeet model: {model_name_or_path}")
        return self.nemo_asr.models.ASRModel.from_pretrained(model_name_or_path)
    
    def _convert_output_format(
        self, 
        nemo_outputs: List, 
        include_timestamps: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Convert NeMo transcription output to WhisperS2T format.
        
        NeMo returns Hypothesis objects with:
            - .text: transcribed text
            - .timestep: dict with 'word', 'segment', 'char' keys (note: timestep, not timestamp)
        
        Args:
            nemo_outputs: List of NeMo Hypothesis objects
            include_timestamps: Whether timestamps were requested
            
        Returns:
            List of segment lists in WhisperS2T format
        """
        responses = []
        
        for output in nemo_outputs:
            segments = []
            
            # Get text - handle both Hypothesis objects and plain strings
            text = ''
            if hasattr(output, 'text') and output.text:
                text = output.text
            elif hasattr(output, 'words') and output.words:
                # Fall back to words attribute if text is empty
                text = ' '.join(output.words) if isinstance(output.words, list) else str(output.words)
            # Note: Don't fall back to str(output) - that produces ugly Hypothesis repr
            
            # Check if we have timestamp information (NeMo uses 'timestep' not 'timestamp')
            timestep_data = None
            if hasattr(output, 'timestep'):
                timestep_data = output.timestep
            elif hasattr(output, 'timestamp'):
                timestep_data = output.timestamp
            
            if include_timestamps and timestep_data:
                # Use segment-level timestamps if available
                segment_stamps = timestep_data.get('segment', [])
                
                if segment_stamps:
                    for stamp in segment_stamps:
                        seg_text = stamp.get('segment', '') if isinstance(stamp, dict) else str(stamp)
                        start = stamp.get('start', 0) if isinstance(stamp, dict) else 0
                        end = stamp.get('end', 0) if isinstance(stamp, dict) else 0
                        segments.append({
                            'text': seg_text.strip() if isinstance(seg_text, str) else str(seg_text),
                            'start_time': round(float(start), 3),
                            'end_time': round(float(end), 3),
                        })
                else:
                    # Fall back to word-level timestamps aggregated
                    word_stamps = timestep_data.get('word', [])
                    if word_stamps:
                        # Group words into segments (simple approach: one segment)
                        words = []
                        for w in word_stamps:
                            if isinstance(w, dict):
                                words.append(w.get('word', ''))
                            else:
                                words.append(str(w))
                        full_text = ' '.join(words)
                        
                        first_stamp = word_stamps[0] if word_stamps else {}
                        last_stamp = word_stamps[-1] if word_stamps else {}
                        start = first_stamp.get('start', 0) if isinstance(first_stamp, dict) else 0
                        end = last_stamp.get('end', 0) if isinstance(last_stamp, dict) else 0
                        
                        segments.append({
                            'text': full_text.strip(),
                            'start_time': round(float(start), 3),
                            'end_time': round(float(end), 3),
                        })
            
            # Fall back to just text if no segments created
            if not segments and text:
                segments.append({
                    'text': text.strip() if isinstance(text, str) else str(text),
                    'start_time': 0.0,
                    'end_time': 0.0,
                })
            
            # If still no segments, create empty one
            if not segments:
                segments.append({
                    'text': '',
                    'start_time': 0.0,
                    'end_time': 0.0,
                })
            
            responses.append(segments)
        
        return responses
    
    @torch.no_grad()
    def transcribe(
        self,
        audio_files: List[str],
        lang_codes: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        initial_prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Transcribe audio files.
        
        Note: Parakeet is English-only, so lang_codes is ignored.
        Tasks other than 'transcribe' are not supported.
        
        Args:
            audio_files: List of paths to audio files
            lang_codes: Ignored (Parakeet is English-only)
            tasks: Ignored (only transcription supported)
            initial_prompts: Ignored (Parakeet doesn't support prompts)
            batch_size: Batch size for inference
            **kwargs: Additional arguments
            
        Returns:
            List of transcription results per file
        """
        # Warn about unsupported features
        if lang_codes and any(lc != 'en' for lc in lang_codes if lc):
            print("Warning: Parakeet only supports English. Language codes ignored.")
        
        if tasks and any(t != 'transcribe' for t in tasks if t):
            print("Warning: Parakeet only supports transcription, not translation.")
        
        if initial_prompts and any(p for p in initial_prompts if p):
            print("Warning: Parakeet doesn't support initial prompts. Prompts ignored.")
        
        # Transcribe with timestamps
        outputs = self.model.transcribe(
            audio_files,
            batch_size=batch_size,
            timestamps=True,
        )
        
        # Debug output (can be removed once working)
        if outputs:
            first = outputs[0] if isinstance(outputs, list) else outputs
            if hasattr(first, 'text'):
                print(f"[Parakeet] Transcribed: '{first.text}'")
        
        # Handle single file case (NeMo may return single result instead of list)
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        return self._convert_output_format(outputs, include_timestamps=True)
    
    @torch.no_grad()
    def transcribe_with_vad(
        self,
        audio_files: List[str],
        lang_codes: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        initial_prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        progress_bar: bool = True,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Transcribe audio files with Voice Activity Detection.
        
        Parakeet models have built-in silence handling, so this method
        provides the same functionality as transcribe() but maintains
        API compatibility with Whisper backends.
        
        Args:
            audio_files: List of paths to audio files
            lang_codes: Ignored (Parakeet is English-only)
            tasks: Ignored (only transcription supported)
            initial_prompts: Ignored (Parakeet doesn't support prompts)
            batch_size: Batch size for inference
            progress_bar: Whether to show progress (handled by NeMo)
            **kwargs: Additional arguments
            
        Returns:
            List of transcription results per file
        """
        # Parakeet handles VAD internally through its preprocessing
        # Just call the standard transcribe method
        return self.transcribe(
            audio_files=audio_files,
            lang_codes=lang_codes,
            tasks=tasks,
            initial_prompts=initial_prompts,
            batch_size=batch_size,
            **kwargs
        )
