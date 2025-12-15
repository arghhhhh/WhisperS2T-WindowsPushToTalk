# WhisperS2T Push-to-Talk Setup Guide

This guide will help you set up WhisperS2T for GPU-accelerated push-to-talk speech-to-text on Windows, with support for both **Whisper** and **Parakeet** models.

## 📋 System Requirements

- **GPU**: NVIDIA RTX series (recommended: RTX 30-series or newer with 8GB+ VRAM)
- **CUDA**: Version 12.0 or higher
- **RAM**: 16GB minimum
- **Storage**: 10GB free space for models and dependencies
- **OS**: Windows 10/11

---

## 🎯 Choose Your Backend

| Backend                   | Best For                    | Install Complexity          |
| ------------------------- | --------------------------- | --------------------------- |
| **Parakeet**              | English-only, best accuracy | Medium (NeMo has many deps) |
| **Whisper (CTranslate2)** | Multilingual, good balance  | Easy                        |
| **Whisper (TensorRT)**    | Maximum speed               | Complex                     |

**Recommendation:**

- For **English**: Use Parakeet
- For **multilingual**: Use Whisper with CTranslate2

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

---

## 📦 Backend-Specific Installation

### Option A: Whisper Only (Recommended for Multilingual)

```bash
# Install Whisper dependencies
pip install -r requirements-whisper.txt

# Install package
pip install -e .
```

### Option B: Parakeet Only (Recommended for English)

```bash
# Install Parakeet dependencies (NeMo)
pip install -r requirements-parakeet.txt

# Install package
pip install -e .

# Download or place your Parakeet model
# Option 1: Download from NGC (happens automatically on first use)
# Option 2: Place .nemo file in models/ folder
```

### Option C: Both Backends (May Have Conflicts)

```bash
# Install Whisper first
pip install -r requirements-whisper.txt

# Then add NeMo (may upgrade some packages)
pip install nemo_toolkit[asr]

# Install package
pip install -e .
```

**Note:** NeMo may upgrade some packages that Whisper depends on. In practice, this usually works fine, but if you encounter issues, use separate conda environments.

---

## ✅ Verify Setup

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

---

## ⚙️ Configure Your Settings

Copy the example configuration and customize:

```bash
copy .env.example .env
```

Edit `.env` with your settings:

### For Whisper:

```env
MIC_DEVICE=5                    # Your microphone index
MODEL=large-v3                  # Whisper model
BACKEND=CTranslate2             # Whisper backend
LANGUAGE=en                     # Language code
HOTKEY=ctrl+alt+shift+space     # Push-to-talk hotkey
```

### For Parakeet:

```env
MIC_DEVICE=5                                # Your microphone index
MODEL=models/parakeet-tdt-0.6b-v2.nemo      # Path to .nemo file
BACKEND=Parakeet                            # Parakeet backend
LANGUAGE=en                                 # English only
HOTKEY=ctrl+alt+shift+space                 # Push-to-talk hotkey
```

**Find your microphone device index:**

```bash
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]; p.terminate()"
```

---

## 🧪 First Test

```bash
python whisper_hotkey.py
```

1. Wait for the model to load
   - Whisper: First run downloads ~3GB for large-v3
   - Parakeet: Uses your local .nemo file or downloads from NGC
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

### Parakeet Returns Empty Results

- Verify you're using the correct microphone device
- Check that the .nemo file path is correct
- Ensure audio is actually being recorded (check file size)

### NeMo Import Errors

If NeMo fails to import:

```bash
# Try reinstalling in a fresh environment
conda create -n parakeet python=3.10 -y
conda activate parakeet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nemo_toolkit[asr]
```

### Model Download Slow

The first run downloads the model. Be patient or use a smaller model:

```env
# Whisper
MODEL=base  # ~74MB, faster download

# Parakeet - use local file to avoid download
MODEL=models/parakeet-tdt-0.6b-v2.nemo
```

---

## 📊 Model Recommendations

### Whisper Models (Multilingual)

| Model      | Size    | Speed        | Accuracy  | Use Case             |
| ---------- | ------- | ------------ | --------- | -------------------- |
| `tiny`     | ~39MB   | ⚡ Fastest   | Basic     | Testing only         |
| `base`     | ~74MB   | ⚡ Very Fast | Good      | Quick notes          |
| `small`    | ~244MB  | 🚀 Fast      | Better    | Daily use            |
| `medium`   | ~769MB  | 🐌 Slower    | Very Good | Important recordings |
| `large-v3` | ~1550MB | 🐌 Slowest   | **Best**  | Maximum accuracy     |

### Parakeet Models (English Only)

| Model                  | Size   | Speed   | Accuracy | Notes                       |
| ---------------------- | ------ | ------- | -------- | --------------------------- |
| `parakeet-tdt-0.6b-v2` | ~600MB | 🚀 Fast | **Best** | **Recommended for English** |
| `parakeet-tdt-1.1b`    | ~1.1GB | 🚀 Fast | **Best** | Larger, slightly better     |

---

## 🎉 You're Done!

Your push-to-talk speech-to-text is ready!

```bash
conda activate whisper
python whisper_hotkey.py
```

Hold your hotkey, speak, release, and paste! 🎤✨

See `USAGE_GUIDE.md` for detailed configuration options.
