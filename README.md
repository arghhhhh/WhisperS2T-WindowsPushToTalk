<h1 align="center"> WhisperS2T ⚡ </h1>
<p align="center"><b>Fast Speech-to-Text Pipeline Supporting Multiple ASR Backends: Whisper + Parakeet</b></p>
<p align="center">
    <a href="https://www.pepy.tech/projects/whisper-s2t">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/whisper-s2t?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads" />
    </a>
    <a href="https://pepy.tech/project/whisper-s2t">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/shashikg/WhisperS2T" />
    </a>
    <a href="https://badge.fury.io/py/whisper-s2t">
        <img alt="PyPi Release Version" src="https://badge.fury.io/py/whisper-s2t.svg" />
    </a>
    <a href="https://github.com/shashikg/WhisperS2T/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/shashikg/WhisperS2T?color=0088ff" />
    </a>
</p>
<hr><br>

WhisperS2T is an optimized, lightning-fast **Speech-to-Text** (ASR) pipeline supporting multiple model backends:

| Backend          | Model                       | Languages     | Best For                              |
| ---------------- | --------------------------- | ------------- | ------------------------------------- |
| **Parakeet**     | NVIDIA Parakeet TDT 0.6B v2 | English only  | **State-of-the-art English accuracy** |
| **CTranslate2**  | OpenAI Whisper              | 99+ languages | Fast multilingual transcription       |
| **TensorRT-LLM** | OpenAI Whisper              | 99+ languages | Maximum speed on NVIDIA GPUs          |
| **HuggingFace**  | OpenAI Whisper              | 99+ languages | Flexibility, Distil models            |

The pipeline provides **2.3X speed improvement over WhisperX** and **3X boost over HuggingFace Pipeline** with FlashAttention 2.

---

## 🎤 Push-to-Talk Mode

This fork includes a **push-to-talk hotkey application** for instant speech-to-text with automatic clipboard copy.

### Quick Start

```bash
conda activate whisper
python whisper_hotkey.py
```

- **Hold hotkey** → Record (hear a pop sound)
- **Release** → Transcribe & auto-copy to clipboard
- **Paste** anywhere with `Ctrl+V`

### Features

- ⌨️ **Configurable hotkey** (default: `ctrl+alt+shift+space`)
- 📋 **Auto-copy to clipboard** - paste transcriptions anywhere instantly
- 🔊 **Audio notification** when recording starts
- 🧵 **Multi-threaded** - records and transcribes in parallel for long recordings
- 🔗 **Intelligent stitching** - handles chunk boundaries with smart overlap detection
- ⚙️ **`.env` configuration** - easily customize model, mic, hotkey, and more
- 🦜 **Parakeet support** - use NVIDIA's state-of-the-art English ASR model

### Configuration

Edit `.env` to choose your backend:

```env
# ============ Option 1: Whisper (multilingual) ============
MODEL=large-v3
BACKEND=CTranslate2
LANGUAGE=en

# ============ Option 2: Parakeet (English, best accuracy) ============
# MODEL=models/parakeet-tdt-0.6b-v2.nemo
# BACKEND=Parakeet
# LANGUAGE=en
```

See [SETUP.md](SETUP.md) for installation and [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed options.

---

## 🦜 Parakeet Backend (New!)

NVIDIA's **Parakeet TDT 0.6B v2** is the current state-of-the-art for English speech recognition, outperforming Whisper large-v3 on accuracy benchmarks.

### Quick Setup

```bash
# Install NeMo (in a fresh conda env recommended)
pip install nemo_toolkit[asr]

# Or use the Parakeet-specific requirements
pip install -r requirements-parakeet.txt
```

### Usage

```python
import whisper_s2t

# Load Parakeet model
model = whisper_s2t.load_model(
    "models/parakeet-tdt-0.6b-v2.nemo",  # or "nvidia/parakeet-tdt-0.6b-v2"
    backend="Parakeet"
)

# Transcribe
result = model.transcribe_with_vad(["audio.wav"])
print(result[0][0]['text'])
```

### Parakeet vs Whisper

| Feature              | Parakeet TDT | Whisper large-v3   |
| -------------------- | ------------ | ------------------ |
| **English Accuracy** | Best         | Very Good          |
| **Languages**        | English only | 99+ languages      |
| **Speed**            | Fast         | Depends on backend |
| **Model Size**       | ~600MB       | ~1.5GB             |
| **Timestamps**       | Built-in     | Via alignment      |

**Recommendation:**

- For **English**: Use Parakeet
- For **multilingual**: Use Whisper with CTranslate2

---

## Release Notes

- [Dec 15, 2025]: Added NVIDIA Parakeet TDT backend for state-of-the-art English ASR
- [Feb 25, 2024]: Added prebuilt docker images and transcript exporter to `txt, json, tsv, srt, vtt`.
- [Jan 28, 2024]: Added support for TensorRT-LLM backend.
- [Dec 23, 2023]: Added support for word alignment for CTranslate2 backend.
- [Dec 19, 2023]: Added support for Whisper-Large-V3 and Distil-Whisper-Large-V2.
- [Dec 17, 2023]: Released WhisperS2T!

## Quickstart

Checkout the Google Colab notebooks provided here: [notebooks](notebooks)

## Features

- 🔄 **Multi-Backend Support:** Whisper (CTranslate2, HuggingFace, TensorRT-LLM, OpenAI) + Parakeet (NeMo)
- 🦜 **State-of-the-Art English:** NVIDIA Parakeet TDT achieves best-in-class English accuracy
- 🎙️ **Easy Integration of Custom VAD Models:** Seamlessly add custom Voice Activity Detection models
- 🎧 **Effortless Handling of Audio Files:** Intelligently batch smaller speech segments
- ⏳ **Streamlined Processing:** Asynchronously loads large audio files while transcribing
- 🌐 **Batching Support:** Decode multiple languages or tasks in a single batch
- 🧠 **Reduction in Hallucination:** Optimized parameters to decrease repeated text
- ⏱️ **Dynamic Time Length Support:** Process variable-length inputs (CTranslate2)

## Getting Started

### Requirements Files

| File                        | Use Case                                 |
| --------------------------- | ---------------------------------------- |
| `requirements.txt`          | Full reference (all backends documented) |
| `requirements-whisper.txt`  | Whisper-only (lighter install)           |
| `requirements-parakeet.txt` | Parakeet-only (NeMo)                     |

### Local Installation

Install audio packages required for resampling and loading audio files.

#### For Ubuntu

```sh
apt-get install -y libsndfile1 ffmpeg
```

#### For MAC

```sh
brew install ffmpeg
```

#### For Windows/Any with Conda

```sh
conda install conda-forge::ffmpeg
```

### Install WhisperS2T

```sh
# For Whisper backends
pip install -r requirements-whisper.txt
pip install -e .

# For Parakeet backend (fresh env recommended)
pip install -r requirements-parakeet.txt
pip install -e .
```

### Usage

#### Whisper (CTranslate2 Backend)

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v2", backend='CTranslate2')

files = ['audio.wav']
out = model.transcribe_with_vad(files, lang_codes=['en'], tasks=['transcribe'])

print(out[0][0]['text'])
```

#### Parakeet Backend

```py
import whisper_s2t

model = whisper_s2t.load_model(
    model_identifier="nvidia/parakeet-tdt-0.6b-v2",
    backend='Parakeet'
)

files = ['audio.wav']
out = model.transcribe_with_vad(files)

print(out[0][0]['text'])
```

#### TensorRT-LLM Backend

```py
import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v2", backend='TensorRT-LLM')

files = ['audio.wav']
out = model.transcribe_with_vad(files, lang_codes=['en'], tasks=['transcribe'])

print(out[0][0]['text'])
```

Check [docs.md](docs.md) for more details.

## Acknowledgements

- [**OpenAI Whisper Team**](https://github.com/openai/whisper): Thanks for open-sourcing the whisper model.
- [**NVIDIA NeMo Team**](https://github.com/NVIDIA/NeMo): Thanks for the Parakeet TDT models and VAD model.
- [**HuggingFace Team**](https://huggingface.co/docs/transformers/model_doc/whisper): Thanks for FlashAttention2 integration.
- [**CTranslate2 Team**](https://github.com/OpenNMT/CTranslate2/): Thanks for the faster inference engine.
- [**NVIDIA TensorRT-LLM Team**](https://github.com/NVIDIA/TensorRT-LLM/): Thanks for LLM inference optimizations.

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.
