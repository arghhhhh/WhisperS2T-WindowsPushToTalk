#!/usr/bin/env python3
"""
WhisperS2T Push-to-Talk Transcription

A keyboard-activated speech-to-text system with parallel recording/transcription.

Usage:
    python whisper_hotkey.py

Configuration:
    Edit .env file to customize settings (see .env.example)

Controls:
    - Hold configured hotkey to record (default: ctrl+,+.+/)
    - Release hotkey to stop and get transcription
    - Press Ctrl+C to exit the application
"""

import os
import sys
import time
from pathlib import Path
import wave
import tempfile
import threading
import queue
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import struct

import numpy as np
import pyaudio

# Import our config module
from whisper_hotkey_config import load_config, WhisperHotkeyConfig


# =============================================================================
# CONSTANTS
# =============================================================================

CHUNK = 1024  # Audio buffer size
FORMAT = pyaudio.paInt16
CHANNELS = 1


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AudioChunk:
    """Represents a chunk of audio data for processing."""
    data: np.ndarray  # Audio samples
    index: int        # Sequence number
    is_final: bool    # True if this is the last chunk
    timestamp: float  # When this chunk was created


class AppState(Enum):
    """Application state machine states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


# =============================================================================
# RECORDING THREAD
# =============================================================================

class RecordingThread(threading.Thread):
    """
    Producer thread that continuously records audio and pushes chunks to queue.
    
    Records in chunks of `chunk_duration` seconds with `chunk_overlap` overlap.
    Monitors for silence if auto-stop is enabled.
    """
    
    def __init__(
        self,
        chunk_queue: queue.Queue,
        config: WhisperHotkeyConfig,
        stop_event: threading.Event,
        auto_stopped_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.config = config
        self.stop_event = stop_event
        self.auto_stopped_event = auto_stopped_event
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.chunk_index = 0
        
        # Calculate buffer sizes
        self.samples_per_chunk = int(config.chunk_duration * config.sample_rate)
        self.overlap_samples = int(config.chunk_overlap * config.sample_rate)
        
        # Silence detection
        self.silence_samples = int(config.silence_threshold * config.sample_rate)
        self.consecutive_silence = 0
        
    def _calculate_rms(self, audio_data: bytes) -> float:
        """Calculate RMS (root mean square) amplitude of audio data."""
        # Convert bytes to int16 array
        count = len(audio_data) // 2
        shorts = struct.unpack(f'{count}h', audio_data)
        # Calculate RMS
        sum_squares = sum(s * s for s in shorts)
        return (sum_squares / count) ** 0.5 if count > 0 else 0
        
    def run(self):
        """Main recording loop."""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.mic_device,
                frames_per_buffer=CHUNK
            )
            
            print("🎙️  Recording started...")
            
            # Buffer for current chunk
            current_buffer = []
            samples_collected = 0
            
            # Keep overlap from previous chunk
            overlap_buffer = []
            
            while not self.stop_event.is_set():
                try:
                    # Read audio data
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    current_buffer.append(data)
                    samples_collected += CHUNK
                    
                    # Silence detection for auto-stop
                    if self.config.auto_stop_enabled:
                        rms = self._calculate_rms(data)
                        if rms < self.config.silence_rms_threshold:
                            self.consecutive_silence += CHUNK
                            if self.consecutive_silence >= self.silence_samples:
                                print("\n🔇 Silence detected, auto-stopping...")
                                self.auto_stopped_event.set()
                                self.stop_event.set()
                                break
                        else:
                            self.consecutive_silence = 0
                    
                    # Show recording indicator
                    if self.config.show_progress:
                        elapsed = samples_collected / self.config.sample_rate
                        print(f"\r🔴 Recording: {elapsed:.1f}s", end="", flush=True)
                    
                    # Check if we have a full chunk
                    if samples_collected >= self.samples_per_chunk:
                        # Create chunk
                        audio_bytes = b''.join(overlap_buffer + current_buffer)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        chunk = AudioChunk(
                            data=audio_array,
                            index=self.chunk_index,
                            is_final=False,
                            timestamp=time.time()
                        )
                        
                        self.chunk_queue.put(chunk)
                        self.chunk_index += 1
                        
                        if self.config.show_progress:
                            print(f"\n📦 Chunk {self.chunk_index} queued ({len(audio_array)/self.config.sample_rate:.1f}s)")
                        
                        # Save overlap for next chunk
                        overlap_buffer = current_buffer[-int(self.overlap_samples / CHUNK * 1.5):]
                        current_buffer = []
                        samples_collected = len(overlap_buffer) * CHUNK
                        
                except IOError as e:
                    # Handle buffer overflow gracefully
                    print(f"\n⚠️  Audio buffer overflow, continuing...")
                    continue
            
            # Process any remaining audio as final chunk
            if current_buffer:
                audio_bytes = b''.join(overlap_buffer + current_buffer)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Only create chunk if we have meaningful audio (at least 0.5 seconds)
                if len(audio_array) >= self.config.sample_rate * 0.5:
                    chunk = AudioChunk(
                        data=audio_array,
                        index=self.chunk_index,
                        is_final=True,
                        timestamp=time.time()
                    )
                    self.chunk_queue.put(chunk)
                    
                    if self.config.show_progress:
                        print(f"\n📦 Final chunk {self.chunk_index + 1} queued ({len(audio_array)/self.config.sample_rate:.1f}s)")
                
                # Always put sentinel after final chunk to signal end
                self.chunk_queue.put(None)
            else:
                # Put a sentinel to signal end
                self.chunk_queue.put(None)
                
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
            self.chunk_queue.put(None)  # Signal end
        finally:
            self._cleanup()
            print("\n🎙️  Recording stopped.")
    
    def _cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        try:
            self.audio.terminate()
        except:
            pass


# =============================================================================
# TRANSCRIPTION THREAD
# =============================================================================

class TranscriptionThread(threading.Thread):
    """
    Consumer thread that processes audio chunks and transcribes them.
    
    Accumulates transcriptions and handles chunk stitching.
    """
    
    def __init__(
        self,
        chunk_queue: queue.Queue,
        config: WhisperHotkeyConfig,
        model,  # WhisperS2T model
        result_callback,
    ):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.config = config
        self.model = model
        self.result_callback = result_callback
        
        self.transcriptions: List[str] = []
        self.total_chunks_processed = 0
        
    def run(self):
        """Main transcription loop."""
        print("🔄 Transcription thread ready...")
        
        while True:
            try:
                # Get chunk from queue (with timeout to allow checking for exit)
                try:
                    chunk = self.chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Check for sentinel (end of recording)
                if chunk is None:
                    break
                
                # Transcribe the chunk
                if self.config.show_progress:
                    print(f"📝 Transcribing chunk {chunk.index + 1}...")
                
                start_time = time.time()
                transcription = self._transcribe_chunk(chunk)
                elapsed = time.time() - start_time
                
                if transcription:
                    self.transcriptions.append(transcription)
                    self.total_chunks_processed += 1
                    
                    if self.config.show_progress:
                        print(f"✅ Chunk {chunk.index + 1} done ({elapsed:.2f}s): \"{transcription[:50]}{'...' if len(transcription) > 50 else ''}\"")
                else:
                    if self.config.show_progress:
                        print(f"⚠️  Chunk {chunk.index + 1}: No speech detected")
                
                self.chunk_queue.task_done()
                
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                import traceback
                traceback.print_exc()
        
        # All chunks processed, combine results
        final_text = self._stitch_transcriptions()
        self.result_callback(final_text)
    
    def _transcribe_chunk(self, chunk: AudioChunk) -> str:
        """Transcribe a single audio chunk."""
        try:
            # Save chunk to temporary file (WhisperS2T expects file paths)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            # Write WAV file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                # Convert float32 back to int16
                audio_int16 = (chunk.data * 32768).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe
            out = self.model.transcribe_with_vad(
                audio_files=[temp_path],
                lang_codes=[self.config.language],
                tasks=['transcribe'],
                initial_prompts=[None],
                batch_size=1
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Extract text
            if out and len(out) > 0 and len(out[0]) > 0:
                return out[0][0]['text'].strip()
            return ""
            
        except Exception as e:
            print(f"❌ Chunk transcription failed: {e}")
            return ""
    
    def _stitch_transcriptions(self) -> str:
        """
        Combine transcriptions from multiple chunks.

        Uses intelligent overlap detection to handle chunk boundaries.
        Handles two common issues:
        1. Chunk 1 ends mid-word/sentence (prefers chunk 2's version of overlap)
        2. Chunk 2 starts with hallucinated words like "And", "So", etc.
        """
        if not self.transcriptions:
            return ""

        if len(self.transcriptions) == 1:
            return self.transcriptions[0]

        def normalize_word(word: str) -> str:
            """Normalize word for comparison (lowercase, remove punctuation)."""
            return ''.join(c.lower() for c in word if c.isalnum())
        
        def find_overlap(words1: List[str], words2: List[str], min_match: int = 3, max_search: int = 30) -> tuple:
            """
            Find overlapping word sequence between end of words1 and start of words2.
            
            Also handles cases where words2 starts with hallucinated filler words.
            
            Returns: (words_to_remove_from_end_of_words1, words_to_skip_from_start_of_words2)
            """
            if not words1 or not words2:
                return 0, 0
            
            # Look at the last max_search words of chunk 1
            search_end = words1[-max_search:] if len(words1) > max_search else words1
            # Look at the first max_search words of chunk 2
            search_start = words2[:max_search] if len(words2) > max_search else words2
            
            best_match_len = 0
            best_remove_from_1 = 0
            best_skip_from_2 = 0
            
            # Try different starting positions in chunk 2 (to skip hallucinated prefix words)
            # Common hallucinations at chunk start: "And", "So", "But", "Well", "Now", "The", "I"
            max_skip = min(5, len(search_start) - min_match)  # Don't skip too many words
            
            for skip in range(max_skip + 1):
                search_start_offset = search_start[skip:]
                
                # Try different starting positions in the end of chunk 1
                for i in range(len(search_end)):
                    # Compare search_end[i:] with search_start_offset
                    match_count = 0
                    
                    for j in range(min(len(search_end) - i, len(search_start_offset))):
                        w1 = normalize_word(search_end[i + j])
                        w2 = normalize_word(search_start_offset[j])
                        
                        if w1 == w2:
                            match_count += 1
                        else:
                            # Allow one mismatch in the middle if surrounded by matches
                            # This handles cases where one chunk misheard a single word
                            if match_count >= 2 and j + 1 < len(search_start_offset) and i + j + 1 < len(search_end):
                                next_w1 = normalize_word(search_end[i + j + 1])
                                next_w2 = normalize_word(search_start_offset[j + 1])
                                if next_w1 == next_w2:
                                    # Skip this mismatch and continue
                                    match_count += 1
                                    continue
                            break
                    
                    # If we found a better overlap, record it
                    # Prefer longer matches, and for equal length, prefer less skipping
                    if match_count >= min_match:
                        if match_count > best_match_len or (match_count == best_match_len and skip < best_skip_from_2):
                            best_match_len = match_count
                            best_remove_from_1 = len(search_end) - i
                            best_skip_from_2 = skip
            
            return best_remove_from_1, best_skip_from_2
        
        # Build result by stitching chunks with overlap removal
        result_words = self.transcriptions[0].split()
        
        if self.config.show_progress:
            print(f"   📝 Chunk 1: \"{' '.join(result_words[-10:])}...\"")
        
        for i in range(1, len(self.transcriptions)):
            next_words = self.transcriptions[i].split()
            
            if not next_words:
                continue
            
            if self.config.show_progress:
                print(f"   📝 Chunk {i+1}: \"{' '.join(next_words[:10])}...\"")
            
            # Find overlap between current result and next chunk
            remove_from_result, skip_from_next = find_overlap(result_words, next_words)
            
            if remove_from_result > 0 or skip_from_next > 0:
                if self.config.show_progress:
                    if remove_from_result > 0:
                        removed_words = result_words[-remove_from_result:]
                        print(f"   🔗 Removing {remove_from_result} words from chunk {i}: \"{' '.join(removed_words)}\"")
                    if skip_from_next > 0:
                        skipped_words = next_words[:skip_from_next]
                        print(f"   🔗 Skipping {skip_from_next} hallucinated words from chunk {i+1}: \"{' '.join(skipped_words)}\"")
                
                # Remove overlapping words from result
                if remove_from_result > 0:
                    result_words = result_words[:-remove_from_result]
                
                # Skip hallucinated prefix from next chunk
                if skip_from_next > 0:
                    next_words = next_words[skip_from_next:]
            
            # Append next chunk's words
            result_words.extend(next_words)
        
        # Join and clean up
        result = " ".join(result_words)
        result = " ".join(result.split())  # Normalize whitespace
        
        return result


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class WhisperHotkeyApp:
    """
    Main application that coordinates hotkey listening, recording, and transcription.
    """
    
    def __init__(self, config: WhisperHotkeyConfig):
        self.config = config
        self.state = AppState.IDLE
        self.model = None
        
        # Threading primitives
        self.recording_thread: Optional[RecordingThread] = None
        self.transcription_thread: Optional[TranscriptionThread] = None
        self.chunk_queue: Optional[queue.Queue] = None
        self.stop_event: Optional[threading.Event] = None
        self.auto_stopped_event: Optional[threading.Event] = None
        
        # Results
        self.final_transcription = ""
        self.transcription_ready = threading.Event()
        
    def load_model(self):
        """Load the WhisperS2T model."""
        import torch
        import whisper_s2t
        
        print(f"\n🔄 Loading WhisperS2T model: {self.config.model} ({self.config.backend})...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
        
        self.model = whisper_s2t.load_model(
            model_identifier=self.config.model,
            backend=self.config.backend,
            device=device
        )
        
        print("✅ Model loaded successfully!\n")
        
    def start_recording(self):
        """Start the recording and transcription threads."""
        if self.state != AppState.IDLE:
            return
        
        self.state = AppState.RECORDING
        
        # Reset state
        self.final_transcription = ""
        self.transcription_ready.clear()
        
        # Create queue and events
        self.chunk_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.auto_stopped_event = threading.Event()
        
        # Start transcription thread first
        self.transcription_thread = TranscriptionThread(
            chunk_queue=self.chunk_queue,
            config=self.config,
            model=self.model,
            result_callback=self._on_transcription_complete
        )
        self.transcription_thread.start()
        
        # Start recording thread
        self.recording_thread = RecordingThread(
            chunk_queue=self.chunk_queue,
            config=self.config,
            stop_event=self.stop_event,
            auto_stopped_event=self.auto_stopped_event
        )
        self.recording_thread.start()
        
        # Play sound to indicate recording started
        self._play_sound("start")

    def stop_recording(self):
        """Stop recording and wait for transcription to complete."""
        if self.state != AppState.RECORDING:
            return
        
        self.state = AppState.PROCESSING
        print("\n\n⏹️  Stopping recording...")
        
        # Signal recording to stop
        self.stop_event.set()
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        
        # Wait for transcription to complete
        print("⏳ Processing remaining audio...")
        self.transcription_ready.wait(timeout=60.0)  # 60 second timeout
        
        # Copy to clipboard
        if self.config.copy_to_clipboard and self.final_transcription:
            self._copy_to_clipboard(self.final_transcription)
        
        # Print result
        if self.config.print_transcription:
            print("\n" + "=" * 60)
            print("📋 TRANSCRIPTION:")
            print("=" * 60)
            print(self.final_transcription)
            print("=" * 60 + "\n")
        
        self.state = AppState.IDLE
        
    def _on_transcription_complete(self, text: str):
        """Callback when transcription is finished."""
        self.final_transcription = text
        self.transcription_ready.set()
        
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            print("📋 Transcription copied to clipboard!")
        except ImportError:
            print("⚠️  pyperclip not installed. Install with: pip install pyperclip")
        except Exception as e:
            print(f"⚠️  Could not copy to clipboard: {e}")

    def _play_sound(self, sound_type: str = "start"):
        """Play a notification sound using Windows built-in winsound (no extra deps)."""
        try:
            # Find the sound file
            script_dir = Path(__file__).parent
            sound_file = script_dir / "files" / "pop.wav"
            
            if not sound_file.exists():
                return
            
            # Use winsound (built into Python on Windows)
            import winsound
            # SND_ASYNC plays without blocking, SND_FILENAME plays from file
            winsound.PlaySound(str(sound_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
                
        except Exception as e:
            # Silently fail - sound is not critical
            pass

    def run(self):
        """Main application loop with hotkey listener."""
        try:
            import keyboard
        except ImportError:
            print("❌ 'keyboard' library not installed. Install with: pip install keyboard")
            print("   Note: On Windows, you may need to run as Administrator.")
            sys.exit(1)
        
        # Load model
        self.load_model()
        
        # Print instructions
        print("=" * 60)
        print("🎤 WhisperS2T Push-to-Talk Ready!")
        print("=" * 60)
        print(f"   Hotkey: {self.config.hotkey}")
        print(f"   Mode: Push-to-talk (hold to record, release to stop)")
        print(f"   Model: {self.config.model}")
        print(f"   Auto-stop on silence: {self.config.auto_stop_enabled}")
        print("=" * 60)
        print("\n💡 Hold the hotkey to start recording, release to transcribe.")
        print("   Press Ctrl+C to exit.\n")
        
        # Track key state
        hotkey_pressed = False
        recording_lock = threading.Lock()
        
        def on_hotkey_press():
            nonlocal hotkey_pressed
            with recording_lock:
                if not hotkey_pressed and self.state == AppState.IDLE:
                    hotkey_pressed = True
                    print(f"\n🔑 Hotkey pressed - starting recording...")
                    self.start_recording()
        
        def on_hotkey_release():
            nonlocal hotkey_pressed
            with recording_lock:
                if hotkey_pressed:
                    hotkey_pressed = False
                    if self.state == AppState.RECORDING:
                        self.stop_recording()
        
        # Register hotkey handlers
        hotkey = self.config.hotkey
        registered_hotkey = None
        
        # List of recommended hotkeys to try
        fallback_hotkeys = ['ctrl+shift+r', 'ctrl+alt+r', 'ctrl+shift+space', 'f9']
        
        # Debug mode - set to True to see all key events
        DEBUG_KEYS = False
        
        def try_register_hotkey(hk):
            """Try to register a hotkey, return True if successful."""
            nonlocal registered_hotkey
            try:
                # For push-to-talk, we need both press and release
                parts = hk.lower().split('+')
                trigger_key = parts[-1]  # Last key is the trigger
                modifiers = parts[:-1]   # Everything else is a modifier
                
                # Track all keys in the combo (with aliases)
                all_keys = set(parts)
                # Add common aliases
                if 'ctrl' in all_keys or 'control' in all_keys:
                    all_keys.add('ctrl')
                    all_keys.add('control')
                    all_keys.add('left ctrl')
                    all_keys.add('right ctrl')
                if 'shift' in all_keys:
                    all_keys.add('left shift')
                    all_keys.add('right shift')
                if 'alt' in all_keys:
                    all_keys.add('left alt')
                    all_keys.add('right alt')
                    all_keys.add('alt gr')
                
                print(f"   Tracking keys: {all_keys}")
                print(f"   Trigger key: '{trigger_key}'")
                
                def check_modifiers():
                    """Check if all modifier keys are pressed."""
                    for mod in modifiers:
                        if mod in ('ctrl', 'control'):
                            if not keyboard.is_pressed('ctrl'):
                                return False
                        elif mod == 'shift':
                            if not keyboard.is_pressed('shift'):
                                return False
                        elif mod == 'alt':
                            if not keyboard.is_pressed('alt'):
                                return False
                    return True
                
                def key_event_handler(event):
                    """Unified handler for both press and release events."""
                    nonlocal hotkey_pressed
                    
                    key_name = event.name.lower() if event.name else ""
                    event_type = event.event_type  # 'down' or 'up'
                    
                    if event_type == 'down':
                        # Key press
                        if DEBUG_KEYS:
                            print(f"   [DEBUG] Press: '{key_name}'", end="")
                        
                        # Check for trigger key (handle 'space' variations)
                        is_trigger = (
                            key_name == trigger_key or
                            (trigger_key == 'space' and key_name in ('space', ' '))
                        )
                        
                        if is_trigger and check_modifiers():
                            if DEBUG_KEYS:
                                print(" -> HOTKEY ACTIVATED!")
                            on_hotkey_press()
                        elif DEBUG_KEYS:
                            print("")
                    
                    elif event_type == 'up':
                        # Key release
                        if DEBUG_KEYS and hotkey_pressed:
                            print(f"   [DEBUG] Release: '{key_name}'", end="")
                        
                        # Check if released key is part of our hotkey combo
                        is_combo_key = key_name in all_keys
                        
                        # Also check trigger key aliases
                        if trigger_key == 'space' and key_name in ('space', ' '):
                            is_combo_key = True
                        
                        if is_combo_key and hotkey_pressed:
                            if DEBUG_KEYS:
                                print(" -> STOPPING!")
                            on_hotkey_release()
                        elif DEBUG_KEYS and hotkey_pressed:
                            print(f" (not in combo)")
                
                # Use hook to capture ALL keyboard events (both press and release)
                keyboard.hook(key_event_handler, suppress=False)
                
                registered_hotkey = hk
                return True
                
            except Exception as e:
                print(f"⚠️  Could not register hotkey '{hk}': {e}")
                return False
        
        # Try the configured hotkey first
        if not try_register_hotkey(hotkey):
            print(f"   Trying fallback hotkeys...")
            for fallback in fallback_hotkeys:
                if try_register_hotkey(fallback):
                    break
        
        if registered_hotkey:
            print(f"✅ Hotkey '{registered_hotkey}' registered successfully!")
            print(f"   (Hold to record, release ANY key in combo to stop)")
            if registered_hotkey != hotkey:
                print(f"   (Update your .env file: HOTKEY={registered_hotkey})")
        else:
            print("❌ Could not register any hotkey. Please check your configuration.")
            return
        
        # Main loop
        try:
            while True:
                # Check for auto-stop
                if self.auto_stopped_event and self.auto_stopped_event.is_set():
                    self.stop_recording()
                    self.auto_stopped_event.clear()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            
        finally:
            keyboard.unhook_all()


# =============================================================================
# SIMPLE MODE (for testing without hotkey)
# =============================================================================

def run_simple_mode(config: WhisperHotkeyConfig, duration: Optional[float] = None):
    """
    Run a simple one-shot recording without hotkey.
    Useful for testing.
    """
    import torch
    import whisper_s2t
    
    print(f"\n🔄 Loading WhisperS2T model: {config.model}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper_s2t.load_model(
        model_identifier=config.model,
        backend=config.backend,
        device=device
    )
    
    print("✅ Model loaded!\n")
    
    # Use duration from args or default
    record_duration = duration or config.chunk_duration
    
    print(f"🎙️  Recording for {record_duration} seconds...")
    print("   Press Ctrl+C to stop early.\n")
    
    # Record audio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=config.sample_rate,
        input=True,
        input_device_index=config.mic_device,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < record_duration:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            elapsed = time.time() - start_time
            print(f"\r🔴 Recording: {elapsed:.1f}s / {record_duration}s", end="", flush=True)
    except KeyboardInterrupt:
        print("\n⏹️  Recording stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    print("\n\n📝 Transcribing...")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
    
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(config.sample_rate)
        wf.writeframes(b''.join(frames))
    
    # Transcribe
    start_time = time.time()
    out = model.transcribe_with_vad(
        audio_files=[temp_path],
        lang_codes=[config.language],
        tasks=['transcribe'],
        initial_prompts=[None],
        batch_size=1
    )
    elapsed = time.time() - start_time
    
    # Clean up
    os.unlink(temp_path)
    
    # Extract result
    if out and len(out) > 0 and len(out[0]) > 0:
        transcription = out[0][0]['text'].strip()
    else:
        transcription = ""
    
    print(f"✅ Transcription complete ({elapsed:.2f}s)\n")
    print("=" * 60)
    print("📋 TRANSCRIPTION:")
    print("=" * 60)
    print(transcription)
    print("=" * 60)
    
    # Copy to clipboard
    if config.copy_to_clipboard and transcription:
        try:
            import pyperclip
            pyperclip.copy(transcription)
            print("\n📋 Copied to clipboard!")
        except:
            pass
    
    return transcription


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="WhisperS2T Push-to-Talk Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python whisper_hotkey.py                  # Run with hotkey mode
  python whisper_hotkey.py --simple         # Simple one-shot mode (no hotkey)
  python whisper_hotkey.py --simple --duration 10  # Record for 10 seconds
  python whisper_hotkey.py --config         # Show current configuration
        """
    )
    
    parser.add_argument('--simple', action='store_true',
                       help='Run in simple mode (one-shot recording, no hotkey)')
    parser.add_argument('--duration', type=float,
                       help='Recording duration in seconds (for simple mode)')
    parser.add_argument('--config', action='store_true',
                       help='Show current configuration and exit')
    parser.add_argument('--env', type=str, default=None,
                       help='Path to .env file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.env)
    
    if args.config:
        print(config)
        return 0
    
    if args.simple:
        run_simple_mode(config, args.duration)
        return 0
    
    # Run main application with hotkey
    app = WhisperHotkeyApp(config)
    app.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
