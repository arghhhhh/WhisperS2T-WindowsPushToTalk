#!/usr/bin/env python3
"""
WhisperS2T Hotkey Configuration Module

Loads configuration from .env file with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class WhisperHotkeyConfig:
    """Configuration for WhisperS2T Hotkey application."""
    
    # Audio settings
    mic_device: Optional[int]
    sample_rate: int
    
    # Model settings
    model: str
    backend: str
    language: str
    
    # Recording settings
    chunk_duration: int
    chunk_overlap: int
    
    # Hotkey settings
    hotkey: str
    
    # Stop mode settings
    auto_stop_enabled: bool
    silence_threshold: float
    silence_rms_threshold: int
    
    # Output settings
    copy_to_clipboard: bool
    show_progress: bool
    print_transcription: bool
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║              WhisperS2T Hotkey Configuration                 ║
╠══════════════════════════════════════════════════════════════╣
║ Audio Settings                                               ║
║   Mic Device:        {str(self.mic_device):>38} ║
║   Sample Rate:       {self.sample_rate:>38} Hz ║
╠══════════════════════════════════════════════════════════════╣
║ Model Settings                                               ║
║   Model:             {self.model:>38} ║
║   Backend:           {self.backend:>38} ║
║   Language:          {self.language:>38} ║
╠══════════════════════════════════════════════════════════════╣
║ Recording Settings                                           ║
║   Chunk Duration:    {self.chunk_duration:>37}s ║
║   Chunk Overlap:     {self.chunk_overlap:>37}s ║
╠══════════════════════════════════════════════════════════════╣
║ Hotkey Settings                                              ║
║   Hotkey:            {self.hotkey:>38} ║
║   Mode:              {'Auto-stop on silence' if self.auto_stop_enabled else 'Manual (release to stop)':>38} ║
║   Silence Threshold: {self.silence_threshold:>37}s ║
╠══════════════════════════════════════════════════════════════╣
║ Output Settings                                              ║
║   Copy to Clipboard: {str(self.copy_to_clipboard):>38} ║
║   Show Progress:     {str(self.show_progress):>38} ║
║   Print Result:      {str(self.print_transcription):>38} ║
╚══════════════════════════════════════════════════════════════╝
"""


def load_config(env_path: Optional[str] = None) -> WhisperHotkeyConfig:
    """
    Load configuration from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks in current directory
                  and script directory.
    
    Returns:
        WhisperHotkeyConfig with loaded settings.
    """
    # Find .env file
    if env_path is None:
        # Check current directory first
        if Path('.env').exists():
            env_path = '.env'
        else:
            # Check script directory
            script_dir = Path(__file__).parent
            if (script_dir / '.env').exists():
                env_path = str(script_dir / '.env')
    
    # Load .env file if dotenv is available and file exists
    if DOTENV_AVAILABLE and env_path and Path(env_path).exists():
        load_dotenv(env_path)
        print(f"✓ Loaded config from: {env_path}")
    elif env_path and Path(env_path).exists():
        # Manual parsing if dotenv not available
        print(f"⚠ python-dotenv not installed, parsing .env manually")
        _manual_load_env(env_path)
    else:
        print("⚠ No .env file found, using defaults")
    
    # Parse configuration with defaults
    config = WhisperHotkeyConfig(
        # Audio settings
        mic_device=_get_int_or_none('MIC_DEVICE', None),
        sample_rate=_get_int('SAMPLE_RATE', 16000),
        
        # Model settings
        model=os.getenv('MODEL', 'large-v3'),
        backend=os.getenv('BACKEND', 'CTranslate2'),
        language=os.getenv('LANGUAGE', 'en'),
        
        # Recording settings
        chunk_duration=_get_int('CHUNK_DURATION', 10),
        chunk_overlap=_get_int('CHUNK_OVERLAP', 2),
        
        # Hotkey settings
        hotkey=os.getenv('HOTKEY', 'ctrl+shift+space'),
        
        # Stop mode settings
        auto_stop_enabled=_get_bool('AUTO_STOP_ENABLED', False),
        silence_threshold=_get_float('SILENCE_THRESHOLD', 2.0),
        silence_rms_threshold=_get_int('SILENCE_RMS_THRESHOLD', 500),
        
        # Output settings
        copy_to_clipboard=_get_bool('COPY_TO_CLIPBOARD', True),
        show_progress=_get_bool('SHOW_PROGRESS', True),
        print_transcription=_get_bool('PRINT_TRANSCRIPTION', True),
    )
    
    return config


def _manual_load_env(env_path: str) -> None:
    """Manually parse .env file if python-dotenv is not available."""
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Parse key=value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                os.environ[key] = value


def _get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠ Invalid integer for {key}: {value}, using default: {default}")
        return default


def _get_int_or_none(key: str, default: Optional[int]) -> Optional[int]:
    """Get integer or None from environment variable."""
    value = os.getenv(key)
    if value is None or value.lower() in ('none', 'null', ''):
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠ Invalid integer for {key}: {value}, using default: {default}")
        return default


def _get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"⚠ Invalid float for {key}: {value}, using default: {default}")
        return default


def _get_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


# Convenience function to test config loading
if __name__ == '__main__':
    config = load_config()
    print(config)
