# WhisperS2T Hotkey Transcription - Planning & Implementation Document

> **Purpose**: This document tracks the complete planning and implementation of a push-to-talk speech-to-text solution using WhisperS2T. It serves as context handoff documentation if implementation needs to continue in a new session.

---

## 📋 Project Overview

**Goal**: Create a hotkey-activated speech-to-text system that:

1. Activates via keyboard shortcut (push-to-talk)
2. Records audio continuously while hotkey is held
3. Transcribes using WhisperS2T (large-v3 model)
4. Copies transcription to clipboard automatically
5. Uses multithreading for parallel recording/transcription of chunks

**User Hardware**:

- Windows 10/11
- NVIDIA RTX GPU with CUDA
- WhisperS2T already set up and working
- Confirmed: 10-second chunks with large-v3 work well

---

## 🎯 Requirements (Confirmed with User)

| Requirement   | Decision                                                   |
| ------------- | ---------------------------------------------------------- |
| Hotkey        | `ctrl+,+.+/` (Ctrl + comma + period + forward slash)       |
| Mode          | **Push-to-talk** (hold key while talking)                  |
| Stop Mode     | Manual stop (default) with option for auto-stop on silence |
| Configuration | `.env` file for defaults                                   |
| Output        | Auto-copy to clipboard                                     |
| Threading     | Parallel recording/transcription with overlapping chunks   |

---

## 🏗️ Architecture

### High-Level Flow

```
[User holds hotkey]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    MAIN THREAD                                │
│  - Hotkey listener (keyboard library)                        │
│  - Coordinates start/stop of recording                       │
│  - Manages application state                                 │
└──────────────────────────────────────────────────────────────┘
       │
       │ On hotkey press
       ▼
┌──────────────────────────────────────────────────────────────┐
│              RECORDING THREAD (Producer)                      │
│                                                               │
│  Audio Stream ──▶ Chunk Buffer (10s) ──▶ Queue               │
│       │                                      │                │
│       └── VAD monitors for silence ──────────┘                │
│           (optional auto-stop)                                │
└──────────────────────────────────────────────────────────────┘
       │
       │ Chunks pushed to thread-safe queue
       ▼
┌──────────────────────────────────────────────────────────────┐
│            TRANSCRIPTION THREAD (Consumer)                    │
│                                                               │
│  Queue ──▶ WhisperS2T ──▶ Text Accumulator ──▶ Clipboard     │
│                                                               │
│  - Processes chunks as they arrive                           │
│  - Stitches text from multiple chunks                        │
│  - Handles overlap deduplication                             │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
[User releases hotkey]
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    FINALIZATION                               │
│  - Wait for remaining chunks to process                      │
│  - Copy final text to clipboard                              │
│  - Play completion sound (optional)                          │
│  - Print transcription to console                            │
└──────────────────────────────────────────────────────────────┘
```

### Threading Model

```
Main Thread
    │
    ├── Hotkey Listener (keyboard callbacks)
    │
    ├── Recording Thread (daemon)
    │       └── Writes audio chunks to Queue
    │
    └── Transcription Thread (daemon)
            └── Reads from Queue, transcribes, accumulates text
```

### Chunk Overlap Strategy

To prevent word cutoffs at chunk boundaries:

```
Time:     0s        10s       20s       30s
          │         │         │         │
Chunk 1:  [=========]
Chunk 2:        [=========]
Chunk 3:              [=========]

Overlap:       ^^^       ^^^
              ~2s       ~2s
```

- Each chunk is ~10 seconds
- Overlap of ~2 seconds between chunks
- Stitching logic removes duplicate words at boundaries

---

## 📁 File Structure

```
WhisperS2T/
├── .env                      # User configuration (gitignored)
├── .env.example              # Template for .env
├── whisper_hotkey.py         # Main application
├── whisper_hotkey_config.py  # Configuration loader
├── WHISPER_HOTKEY_PLAN.md    # This document
└── (existing files...)
```

---

## ⚙️ Configuration Options (.env)

```env
# Audio Settings
MIC_DEVICE=5                    # Microphone device index
SAMPLE_RATE=16000               # Audio sample rate (16kHz for Whisper)

# Model Settings
MODEL=large-v3                  # Whisper model size
BACKEND=CTranslate2             # Backend (CTranslate2, HuggingFace, OpenAI)
LANGUAGE=en                     # Language code

# Recording Settings
CHUNK_DURATION=10               # Seconds per chunk
CHUNK_OVERLAP=2                 # Overlap between chunks (seconds)

# Hotkey Settings
HOTKEY=ctrl+,+.+/               # Push-to-talk hotkey combination

# Stop Mode Settings
AUTO_STOP_ENABLED=false         # Auto-stop on silence (default: false = manual)
SILENCE_THRESHOLD=2.0           # Seconds of silence before auto-stop
SILENCE_AMPLITUDE=500           # Amplitude threshold for silence detection

# Output Settings
COPY_TO_CLIPBOARD=true          # Auto-copy transcription to clipboard
PLAY_SOUND_ON_COMPLETE=false    # Play sound when done
```

---

## 🔧 Implementation Details

### Dependencies (to add)

```
keyboard          # Global hotkey capture (requires admin on some systems)
pyperclip         # Clipboard access
python-dotenv     # .env file loading
```

### Key Classes/Functions

#### 1. `WhisperHotkeyConfig` (whisper_hotkey_config.py)

- Loads settings from .env
- Provides defaults for missing values
- Validates configuration

#### 2. `AudioChunk` (dataclass)

- `data`: numpy array of audio
- `index`: chunk sequence number
- `is_final`: boolean flag for last chunk

#### 3. `RecordingThread`

- Inherits from `threading.Thread`
- Manages PyAudio stream
- Buffers audio into chunks
- Pushes chunks to queue
- Monitors for silence (if auto-stop enabled)

#### 4. `TranscriptionThread`

- Inherits from `threading.Thread`
- Pulls chunks from queue
- Runs WhisperS2T transcription
- Accumulates text with deduplication
- Signals completion

#### 5. `WhisperHotkeyApp` (main class)

- Initializes WhisperS2T model (once at startup)
- Sets up hotkey listener
- Coordinates threads
- Handles clipboard output

### State Machine

```
        ┌─────────────────────────────────────┐
        │                                     │
        ▼                                     │
    ┌───────┐     hotkey_pressed         ┌────────────┐
    │ IDLE  │ ──────────────────────────▶│ RECORDING  │
    └───────┘                            └────────────┘
        ▲                                     │
        │                                     │ hotkey_released
        │         ┌──────────────┐            │ (or silence timeout)
        │         │              │            │
        └─────────│ TRANSCRIBING │◀───────────┘
                  │              │
                  └──────────────┘
                         │
                         │ all chunks processed
                         ▼
                  ┌──────────────┐
                  │   COMPLETE   │──▶ Copy to clipboard
                  └──────────────┘         │
                         │                 │
                         └─────────────────┘
```

---

## 🚧 Implementation Progress

### Phase 1: Core Infrastructure ✅

- [x] Create `.env.example` with all options
- [x] Create `whisper_hotkey_config.py`
- [x] Test config loading

### Phase 2: Threading Framework ✅

- [x] Implement `AudioChunk` dataclass
- [x] Implement `RecordingThread`
- [x] Implement `TranscriptionThread`
- [x] Test thread communication

### Phase 3: Main Application ✅

- [x] Implement `WhisperHotkeyApp`
- [x] Add hotkey listener (using keyboard.hook for reliable release detection)
- [x] Add clipboard integration
- [x] Add state management

### Phase 4: Polish & Testing 🔄

- [x] Test push-to-talk flow
- [ ] Test chunk stitching (longer recordings)
- [ ] Test auto-stop mode
- [x] Error handling
- [ ] Documentation

---

## 🐛 Known Issues / Considerations

1. **Keyboard library on Windows**: May require running as administrator for global hotkeys
2. **Hotkey combination**: `ctrl+,+.+/` is unusual - need to verify keyboard library supports it
3. **Chunk boundary words**: Overlap deduplication needs careful testing
4. **GPU memory**: Model stays loaded in memory for fast response

---

## 📝 Session Notes

### Session 1 (December 11, 2025)

- **Status**: ✅ Core implementation complete and working
- **Decisions Made**:
  - Push-to-talk mode (hold to record)
  - Manual stop as default, auto-stop as option
  - Hotkey: User chose `ctrl+alt+shift+space`
  - 10-second chunks with 2-second overlap
- **Key Technical Findings**:
  - `keyboard.on_release()` does NOT work reliably on Windows
  - **Solution**: Use `keyboard.hook()` which captures ALL events with `event.event_type` ('down' or 'up')
  - Complex hotkeys like `ctrl+,+.+/` don't parse correctly - use standard keys
- **Files Created**:
  - `whisper_hotkey.py` - Main application
  - `whisper_hotkey_config.py` - Configuration loader
  - `.env` / `.env.example` - Configuration files
  - `requirements_hotkey.txt` - Additional dependencies
- **Working Commands**:
  ```cmd
  conda activate whisper
  pip install keyboard pyperclip python-dotenv
  python whisper_hotkey.py
  ```

---

## 🔄 Context Handoff Notes

If continuing in a new session, here's what's needed:

1. **Read this document first** - contains all architectural decisions
2. **Check implementation progress** - see checkboxes above
3. **Existing code to reference**:
   - `mic_transcribe.py` - current recording/transcription logic
   - `whisper_s2t/speech_segmenter/frame_vad.py` - VAD implementation
4. **Key insight**: User confirmed 10s chunks with large-v3 work well on their hardware
5. **Critical requirement**: Push-to-talk (not toggle), manual stop default

---

_Last Updated: December 11, 2025_
