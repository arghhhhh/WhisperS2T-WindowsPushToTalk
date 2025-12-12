# WhisperS2T Setup and Usage Guide

## 🚀 Quick Start

Your WhisperS2T environment is now ready! Here's how to use it:

### 1. Activate the Environment

**Option 1: Manual activation**

```bash
conda activate whisper
```

### 2. Record and Transcribe from Microphone

**Recommended for best accuracy:**

```bash
# Use base model (better accuracy than tiny, still fast)
python demo_mic.py --device 1

# Use small model (even better accuracy, still real-time on RTX 4080)
python demo_mic.py --model small --device 1

# Use specific microphone (examples with your available devices)
python demo_mic.py --device 5 --duration 5
```

**Compare model performance:**

```bash
python compare_models.py  # Shows speed vs accuracy trade-offs
```

### 3. Advanced Microphone Usage

```bash
# Record for 5 seconds with base model
python mic_transcribe.py --duration 5 --model base

# Continuous mode (record and transcribe repeatedly)
python mic_transcribe.py --continuous --duration 3

# Use specific microphone with continuous mode
python mic_transcribe.py --continuous --device 5 --duration 3

# Use different model size
python mic_transcribe.py --model small --device 1

# Use different backend
python mic_transcribe.py --backend HuggingFace --device 52
```

## 📋 Available Options

### Models

- `tiny` - Fastest, least accurate (~39 MB)
- `base` - Good balance (~74 MB)
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB)
- `large-v2` - Best accuracy (~1550 MB)
- `large-v3` - Latest best accuracy (~1550 MB)

### Backends

- `CTranslate2` - Fastest, recommended
- `HuggingFace` - Compatible with all models
- `OpenAI` - Original OpenAI implementation

### Languages

- `en` - English (default)
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `zh` - Chinese
- And many more...

## 🎯 Real-time Usage Tips

1. **Model Selection**: For best accuracy with your RTX 4080, use `base` or `small` models. They provide excellent accuracy while remaining real-time.

2. **Recording Duration**: 3-5 seconds works well for most speech. Longer recordings may capture more context but increase latency.

3. **Continuous Mode**: Use `--continuous` for ongoing conversations. Each segment is processed independently.

4. **GPU Acceleration**: Your RTX 4080 provides significant speedup - expect 10-50x faster processing compared to CPU.

## 🔧 Troubleshooting

### CUDA Issues

If you get CUDA-related errors:

1. Make sure your NVIDIA drivers are up to date
2. PyTorch CUDA 12.1 is compatible with your CUDA drivers
3. If you still get CuDNN errors, ensure the PATH is set before any CUDA operations

### Audio Issues

If microphone recording fails:

1. Check that your microphone is enabled in Windows settings
2. Try different audio devices
3. Make sure no other applications are using the microphone
4. Specify device index with `--device X` in recording scripts

### Memory Issues

For large models on long audio:

1. Use smaller batch sizes: `--batch_size 1`
2. Process audio in smaller chunks
3. Consider using CPU for very large models if GPU memory is limited

## 📊 Performance Expectations

With your RTX 4080, expect:

- **Tiny model**: ~50-100x real-time speed
- **Base model**: ~30-70x real-time speed
- **Small model**: ~20-50x real-time speed
- **Medium model**: ~10-30x real-time speed
- **Large models**: ~5-15x real-time speed

_Real-time speed means processing audio faster than it takes to speak it._

## 🎉 What's Working

✅ GPU acceleration with CUDA
✅ Multiple model sizes and backends
✅ Real-time microphone transcription
✅ Multi-language support
✅ Voice Activity Detection (VAD)
✅ Batch processing capabilities
