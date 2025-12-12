# WhisperS2T Setup Guide

This guide will help you set up WhisperS2T for GPU-accelerated speech-to-text transcription on Windows.

## 📋 System Requirements

- **GPU**: NVIDIA RTX series (recommended: RTX 30-series or newer with 8GB+ VRAM)
- **CUDA**: Version 12.0 or higher (13.0 recommended)
- **RAM**: 16GB minimum
- **Storage**: 10GB free space for models and dependencies
- **OS**: Windows 10/11

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

Your system already has CUDA 13.0 installed. Verify:

```bash
nvidia-smi
```

This should show CUDA Version: 13.0

### Step 3: Install CuDNN

1. Download CuDNN from: https://developer.nvidia.com/cudnn
2. Choose "Download cuDNN v8.9.7 (November 28th, 2023), for CUDA 12.x"
3. Extract to: `C:\Program Files\NVIDIA\CUDNN\v8.9.7.29\`

The path should be: `C:\Program Files\NVIDIA\CUDNN\v8.9.7.29\bin`

### Step 4: Install FFmpeg

Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

1. Extract to a folder (e.g., `C:\ffmpeg\`)
2. Add to system PATH:
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
# Install PyTorch with CUDA 12.1 support (works with CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA installation:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 7: Install WhisperS2T Dependencies

```bash
# Install required packages
pip install tqdm rich numpy platformdirs ctranslate2 tokenizers huggingface-hub accelerate optimum transformers openai-whisper nvidia-ml-py

# Install PyAudio for microphone support
pip install pyaudio
```

### Step 8: Install WhisperS2T

```bash
# Install from current directory
pip install -e .
```

### Step 9: Verify Setup

Run the comprehensive verification script:

```bash
python verify_setup.py
```

This will check:

- ✅ Python version
- ✅ Conda environment
- ✅ CUDA availability
- ✅ CuDNN installation
- ✅ FFmpeg installation
- ✅ WhisperS2T package
- ✅ PyAudio (microphone support)
- ✅ Available microphones

If everything passes, your setup is complete!

### Step 10: First Test

```bash
# Quick microphone test
python demo_mic.py --device 1 --duration 3
```

Speak into your microphone and you should see real-time transcription!

## 🎯 Usage Examples

### Basic Microphone Transcription

```bash
# Activate environment (run this first)
conda activate whisper

# Record and transcribe for 3 seconds
python demo_mic.py --device 1

# Use different microphone (find device numbers with: python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if p.get_device_info_by_index(i)['maxInputChannels'] > 0]; p.terminate()")
python demo_mic.py --device 5

# Use better model for accuracy
python demo_mic.py --model small --device 1
```

### Advanced Usage

```bash
# Continuous transcription mode
python mic_transcribe.py --continuous --device 1 --duration 3

# Different model sizes
python mic_transcribe.py --model base --device 1
python mic_transcribe.py --model small --device 1

# Compare model performance
python compare_models.py
```

## 🔧 Troubleshooting

### "Command not found" errors

- Ensure you're in the correct conda environment: `conda activate whisper`
- Check if conda is in your PATH

### CUDA Issues

If you get "Could not locate cudnn_ops_infer64_8.dll":

- Ensure CuDNN is installed in: `C:\Program Files\NVIDIA\CUDNN\v8.9.7.29\`
- The scripts automatically set the PATH before CUDA operations
- Try running: `python verify_setup.py` to check CUDA status

### Microphone Issues

If microphone recording fails:

- Check Windows sound settings → Recording devices
- Try different device indices (use `python verify_setup.py` to see available mics)
- Ensure no other apps are using the microphone
- Try restarting your computer

### Import Errors

If you get "Module not found" errors:

- Ensure you're in the conda environment: `conda activate whisper`
- Try reinstalling: `pip install -e .`
- Check that all dependencies were installed: `pip list | grep torch`

### FFmpeg Issues

If you get FFmpeg errors:

- Verify FFmpeg is in your system PATH
- Test: `ffmpeg -version` in command prompt
- Reinstall FFmpeg and add to PATH

### Memory Issues

For large models or long audio:

- Use smaller batch sizes: `--batch_size 1`
- Try smaller models first: `tiny` or `base`
- Close other GPU-intensive applications

### Verification Script Issues

If `verify_setup.py` fails:

- Run individual checks manually
- Check conda environment: `conda info --envs`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## 📊 Model Recommendations

| Model         | Size    | Speed        | Accuracy  | Use Case                   |
| ------------- | ------- | ------------ | --------- | -------------------------- |
| `tiny`        | ~39MB   | ⚡ Fastest   | Basic     | Testing only               |
| `base`        | ~74MB   | ⚡ Very Fast | Good      | **Default choice**         |
| `small`       | ~244MB  | 🚀 Fast      | Better    | **Best for conversations** |
| `medium`      | ~769MB  | 🐌 Slower    | Very Good | Important recordings       |
| `large-v2/v3` | ~1550MB | 🐌 Slowest   | Best      | Maximum accuracy           |

## 🎉 You're Done!

Your WhisperS2T setup is now complete. You can transcribe speech in real-time using your RTX 4080's GPU acceleration!

**Quick test:**

```bash
conda activate whisper
python demo_mic.py --device 1
```

Say something into your microphone and watch the transcription appear instantly! 🎤✨
