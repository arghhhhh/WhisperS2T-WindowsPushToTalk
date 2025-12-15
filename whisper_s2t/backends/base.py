"""
Base ASR Model Interface

Provides a minimal abstract interface that all ASR backends (Whisper, Parakeet, etc.) 
should implement for compatibility with the WhisperS2T transcription system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict


class ASRModelBase(ABC):
    """
    Abstract base class for ASR (Automatic Speech Recognition) models.
    
    All ASR backends should inherit from this class and implement the required methods
    to ensure compatibility with the transcription pipeline and hotkey application.
    """
    
    @abstractmethod
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
        Transcribe audio files without Voice Activity Detection.
        
        Args:
            audio_files: List of paths to audio files to transcribe
            lang_codes: Language codes for each file (e.g., ['en', 'es'])
            tasks: Task type for each file ('transcribe' or 'translate')
            initial_prompts: Optional prompts to guide transcription
            batch_size: Number of segments to process in parallel
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of transcription results, one per audio file.
            Each result is a list of segment dictionaries containing:
                - 'text': Transcribed text
                - 'start_time': Segment start time in seconds
                - 'end_time': Segment end time in seconds
                - Additional model-specific fields (avg_logprob, no_speech_prob, etc.)
        """
        pass
    
    @abstractmethod
    def transcribe_with_vad(
        self, 
        audio_files: List[str], 
        lang_codes: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        initial_prompts: Optional[List[str]] = None,
        batch_size: int = 8,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Transcribe audio files with Voice Activity Detection.
        
        Uses VAD to identify speech segments before transcription, which can
        improve accuracy and reduce hallucinations on audio with silence.
        
        Args:
            audio_files: List of paths to audio files to transcribe
            lang_codes: Language codes for each file (e.g., ['en', 'es'])
            tasks: Task type for each file ('transcribe' or 'translate')
            initial_prompts: Optional prompts to guide transcription
            batch_size: Number of segments to process in parallel
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of transcription results, one per audio file.
            Each result is a list of segment dictionaries containing:
                - 'text': Transcribed text
                - 'start_time': Segment start time in seconds
                - 'end_time': Segment end time in seconds
                - Additional model-specific fields
        """
        pass
