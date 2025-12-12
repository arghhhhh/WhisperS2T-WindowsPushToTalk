# WhisperS2T Push-to-Talk Setup Guide

This guide will help you set up WhisperS2T for GPU-accelerated push-to-talk speech-to-text on Windows.

## 📋 System Requirements

- **GPU**: NVIDIA RTX series (recommended: RTX 30-series or newer with 8GB+ VRAM)
- **CUDA**: Version 12.0 or higher
- **RAM**: 16GB minimum
- **Storage**: 10GB free space for models and dependencies
- **OS**: Windows 10/11

---

## 🚀 Step-by-Step Setup

### Step 1: Install Miniconda

1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
2. Install with default settings
3. Restart your terminal/command prompt

Verify installation:

```bash
conda --version
```

### Step 2: Install NVIDIA CUDA Toolkit

Verify your CUDA installation:

```bash
nvidia-smi
```

This should show your GPU and CUDA version.

### Step 3: Install CuDNN

1. Download CuDNN from: https://developer.nvidia.com/cudnn
2. Choose the version matching your CUDA (e.g., cuDNN v8.9.7 for CUDA 12.x)
3. Extract to: `C:\Program Files\NVIDIA\CUDNN\v8.x.x\`
4. Add the `bin` folder to your system PATH

### Step 4: Install FFmpeg

1. Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract to a folder (e.g., `C:\ffmpeg\`)
3. Add to system PATH:
   - Search for "Environment Variables" in Windows
   - Edit "Path" in System Variables
   - Add: `C:\ffmpeg\bin`

### Step 5: Create Conda Environment

```bash
# Create new environment
conda create -n whisper python=3.10 -y

# Activate environment
conda activate whisper
```

### Step 6: Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA installation:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 7: Install WhisperS2T and Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install PyAudio for microphone support
pip install pyaudio

# Install WhisperS2T from current directory
pip install -e .
```

### Step 8: Verify Setup

```bash
python verify_setup.py
```

This checks:

- ✅ Python version
- ✅ CUDA availability
- ✅ CuDNN installation
- ✅ FFmpeg installation
- ✅ WhisperS2T package
- ✅ PyAudio (microphone support)
- ✅ Available microphones

### Step 9: Configure Your Settings

Copy the example configuration and customize:

```bash
copy .env.example .env
```

Edit `.env` with your settings:

```env
# Find your microphone device index first (see step below)
MIC_DEVICE=5

# Choose your model (large-v3 recommended for accuracy)
MODEL=large-v3

# Set your preferred hotkey
HOTKEY=ctrl+alt+shift+space
```

**Find your microphone device index:**

```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]; p.terminate()"
```

### Step 10: First Test

```bash
python whisper_hotkey.py
```

1. Wait for the model to load (first run downloads ~3GB for large-v3)
2. Press and hold your hotkey
3. Speak into your microphone
4. Release the hotkey
5. The transcription appears and is copied to your clipboard!

---

## 🎯 Quick Usage

After setup, the daily workflow is simple:

```bash
# Open a command prompt
conda activate whisper
python whisper_hotkey.py
```

Then:

- **Hold hotkey** → Record (you'll hear a pop sound)
- **Release hotkey** → Transcribe & copy to clipboard
- **Paste** anywhere with `Ctrl+V`

---

## 🔧 Troubleshooting

### "keyboard" Module Requires Admin

Run Command Prompt as Administrator, then:

```bash
conda activate whisper
python whisper_hotkey.py
```

### CUDA Not Found

```bash
# Verify PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`:

- Update NVIDIA drivers
- Reinstall PyTorch with CUDA support

### Microphone Not Working

1. Check Windows Sound Settings → Recording devices
2. Verify the device index in `.env` is correct
3. Ensure no other apps are using the microphone

### Model Download Slow

The first run downloads the model (~3GB for large-v3). Be patient or use a smaller model:

```env
MODEL=base  # ~74MB, faster download
```

---

## 📊 Model Recommendations

| Model      | Size    | Speed        | Accuracy  | Use Case             |
| ---------- | ------- | ------------ | --------- | -------------------- |
| `tiny`     | ~39MB   | ⚡ Fastest   | Basic     | Testing only         |
| `base`     | ~74MB   | ⚡ Very Fast | Good      | Quick notes          |
| `small`    | ~244MB  | 🚀 Fast      | Better    | Daily use            |
| `medium`   | ~769MB  | 🐌 Slower    | Very Good | Important recordings |
| `large-v3` | ~1550MB | 🐌 Slowest   | **Best**  | **Recommended**      |

---

## 🎉 You're Done!

Your push-to-talk speech-to-text is ready!

```bash
conda activate whisper
python whisper_hotkey.py
```

Hold your hotkey, speak, release, and paste! 🎤✨

See `USAGE_GUIDE.md` for detailed configuration options.
