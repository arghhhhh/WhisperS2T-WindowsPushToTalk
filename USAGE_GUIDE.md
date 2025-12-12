# WhisperS2T Push-to-Talk Usage Guide

## 🚀 Quick Start

```bash
conda activate whisper
python whisper_hotkey.py
```

Then **hold your hotkey** to record, **release** to transcribe. The transcription is automatically copied to your clipboard!

---

## 🎤 How It Works

1. **Press & Hold** your configured hotkey (default: `ctrl+alt+shift+space`)
2. **Speak** - you'll hear a pop sound when recording starts
3. **Release** the hotkey when done
4. **Transcription** appears and is automatically copied to clipboard
5. **Paste** anywhere with `Ctrl+V`

---

## ⚙️ Configuration

All settings are in the `.env` file. Edit to customize:

```env
# =============================================================================
# AUDIO SETTINGS
# =============================================================================
MIC_DEVICE=5                    # Your microphone device index
SAMPLE_RATE=16000               # Keep at 16000 for Whisper

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODEL=large-v3                  # Whisper model (see options below)
BACKEND=CTranslate2             # Fastest backend
LANGUAGE=en                     # Language code

# =============================================================================
# RECORDING SETTINGS
# =============================================================================
CHUNK_DURATION=10               # Seconds per chunk (10 recommended)
CHUNK_OVERLAP=2                 # Overlap between chunks

# =============================================================================
# HOTKEY SETTINGS
# =============================================================================
HOTKEY=ctrl+alt+shift+space     # Your push-to-talk hotkey

# =============================================================================
# STOP MODE SETTINGS
# =============================================================================
AUTO_STOP_ENABLED=false         # Set to true for auto-stop on silence
SILENCE_THRESHOLD=2.0           # Seconds of silence before auto-stop

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
COPY_TO_CLIPBOARD=true          # Auto-copy transcription
SHOW_PROGRESS=true              # Show recording progress
PRINT_TRANSCRIPTION=true        # Print final result to console
```

### Finding Your Microphone Device Index

```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]; p.terminate()"
```

---

## 📋 Available Options

### Models

| Model      | Size    | Speed        | Accuracy  | Recommended For      |
| ---------- | ------- | ------------ | --------- | -------------------- |
| `tiny`     | ~39MB   | ⚡ Fastest   | Basic     | Testing only         |
| `base`     | ~74MB   | ⚡ Very Fast | Good      | Quick notes          |
| `small`    | ~244MB  | 🚀 Fast      | Better    | Daily use            |
| `medium`   | ~769MB  | 🐌 Slower    | Very Good | Important recordings |
| `large-v2` | ~1550MB | 🐌 Slowest   | Best      | Maximum accuracy     |
| `large-v3` | ~1550MB | 🐌 Slowest   | Best      | **Recommended**      |

### Hotkey Examples

```env
HOTKEY=ctrl+alt+shift+space     # 4-key combo
HOTKEY=ctrl+shift+r             # 3-key combo
HOTKEY=f9                       # Single function key
```

### Languages

Common codes: `en`, `es`, `fr`, `de`, `it`, `pt`, `ru`, `ja`, `zh`

---

## 🔧 Command Line Options

```bash
# Normal push-to-talk mode
python whisper_hotkey.py

# Show current configuration
python whisper_hotkey.py --config

# Simple one-shot mode (no hotkey, just record for X seconds)
python whisper_hotkey.py --simple --duration 10

# Use a different .env file
python whisper_hotkey.py --env /path/to/custom.env
```

---

## 🎯 Tips for Best Results

1. **Model Selection**: `large-v3` provides the best accuracy. With an RTX 4080, it processes faster than real-time.

2. **Chunk Duration**: 10 seconds works well. The app automatically handles longer recordings by chunking and stitching.

3. **Speak Naturally**: The intelligent stitching algorithm handles sentence boundaries well. Don't worry about pausing between chunks.

4. **Wait for the Pop**: The audio notification confirms recording has started. Speak after you hear it.

5. **Clean Release**: Release the hotkey cleanly after you finish speaking. The transcription starts immediately.

---

## 🔧 Troubleshooting

### Hotkey Not Working

- **Run as Administrator**: The `keyboard` library may need admin privileges for global hotkeys
- **Try a simpler hotkey**: Change to `f9` or `ctrl+shift+r` in `.env`
- **Check for conflicts**: Another app might be using the same hotkey

### No Sound on Recording Start

- Ensure `files/pop.wav` exists
- Check Windows sound settings

### Transcription Not Copying to Clipboard

- Verify `pyperclip` is installed: `pip install pyperclip`
- Check `COPY_TO_CLIPBOARD=true` in `.env`

### Audio Issues

- Verify microphone index with the command above
- Check Windows sound settings → Recording devices
- Ensure no other apps are using the microphone

### CUDA/GPU Issues

- Run `python verify_setup.py` to check GPU status
- Ensure NVIDIA drivers are up to date
- Check that PyTorch sees your GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📊 Performance

With an RTX 4080 and `large-v3` model:

- **10-second chunk**: ~1.5s transcription time
- **Real-time factor**: ~6-7x faster than real-time
- **Memory usage**: ~3-4GB VRAM

---

## 🎉 You're Ready!

```bash
conda activate whisper
python whisper_hotkey.py
```

Hold your hotkey, speak, release, paste! 🎤✨
