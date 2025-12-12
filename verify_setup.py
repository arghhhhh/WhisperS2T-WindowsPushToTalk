#!/usr/bin/env python3
"""
Setup verification script for WhisperS2T
"""

import sys
import os

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.10+")
        return False

def check_conda_env():
    """Check if we're in a conda environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda environment: {conda_env}")
        return True
    else:
        print("⚠️  Not in a conda environment (optional but recommended)")
        return True

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            return True
        else:
            print("❌ CUDA not available - GPU acceleration disabled")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_cudnn():
    """Check CuDNN availability."""
    try:
        import torch
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version:
            print(f"✅ CuDNN version: {cudnn_version}")
            return True
        else:
            print("❌ CuDNN not available")
            return False
    except:
        print("❌ CuDNN check failed")
        return False

def check_whisper_s2t():
    """Check WhisperS2T installation."""
    try:
        import whisper_s2t
        print("✅ WhisperS2T installed")
        return True
    except ImportError:
        print("❌ WhisperS2T not installed")
        return False

def check_pyaudio():
    """Check PyAudio installation."""
    try:
        import pyaudio
        print("✅ PyAudio installed")
        return True
    except ImportError:
        print("❌ PyAudio not installed - microphone recording disabled")
        return False

def check_microphones():
    """Check available microphones."""
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        input_count = 0

        for i in range(audio.get_device_count()):
            if audio.get_device_info_by_index(i).get('maxInputChannels', 0) > 0:
                input_count += 1

        audio.terminate()

        if input_count > 0:
            print(f"✅ Found {input_count} microphone(s)")
            return True
        else:
            print("❌ No microphones found")
            return False
    except:
        print("❌ Could not check microphones")
        return False

def check_ffmpeg():
    """Check FFmpeg availability."""
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg available: {version_line}")
            return True
        else:
            print("❌ FFmpeg not found in PATH")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found in PATH")
        return False

def main():
    """Run all checks."""
    print("🔍 WhisperS2T Setup Verification")
    print("=" * 40)

    checks = [
        ("Python Version", check_python_version),
        ("Conda Environment", check_conda_env),
        ("CUDA GPU", check_cuda),
        ("CuDNN", check_cudnn),
        ("FFmpeg", check_ffmpeg),
        ("WhisperS2T", check_whisper_s2t),
        ("PyAudio", check_pyaudio),
        ("Microphones", check_microphones),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        result = check_func()
        results.append(result)

    # Summary
    print("\n" + "=" * 40)
    print("📊 SUMMARY:")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\n🚀 Your WhisperS2T setup is ready!")
        print("Try: python demo_mic.py --device 1")
    elif passed >= total - 1:  # Allow 1 failure
        print(f"✅ Setup mostly ready ({passed}/{total})")
        print("Some optional components missing but core functionality should work.")
    else:
        print(f"❌ Setup incomplete ({passed}/{total})")
        print("Please check the SETUP.md guide for missing components.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)