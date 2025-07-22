# Quick Usage Guide

## 🚀 Basic Usage (with defaults)

```bash
# Simple: just specify input file, output will be auto-generated
python3 podcast_translator_cli.py my_russian_podcast.mp3

# Output will be: translated_en_my_russian_podcast.mp3
```

## 📝 Command Options

```bash
# Full syntax
python3 podcast_translator_cli.py [input] [output] [options]

# Examples:
python3 podcast_translator_cli.py                                    # Uses input.mp3
python3 podcast_translator_cli.py russian_pod.mp3                    # Auto output: translated_en_russian_pod.mp3
python3 podcast_translator_cli.py russian_pod.mp3 english_pod.mp3    # Custom output
python3 podcast_translator_cli.py russian_pod.mp3 --no-voice-cloning # Disable voice cloning
python3 podcast_translator_cli.py russian_pod.mp3 --model llama2:7b  # Different model
```

## ⚙️ Options

- `input` - Input MP3 file (default: `input.mp3`)
- `output` - Output MP3 file (default: auto-generated with `translated_en_` prefix)
- `--no-voice-cloning` - Use standard TTS instead of voice cloning
- `--model MODEL` - Ollama model to use (default: `gemma3n:latest`)

## 🎭 Voice Cloning

Voice cloning creates much better results by cloning the original speaker's voice:

```bash
# With voice cloning (default)
python3 podcast_translator_cli.py my_podcast.mp3

# Without voice cloning (faster, generic voice)
python3 podcast_translator_cli.py my_podcast.mp3 --no-voice-cloning
```

## 🔧 Fix Voice Cloning Issues

If voice cloning isn't working:

```bash
# Run the fix script
chmod +x fix_voice_cloning.sh
./fix_voice_cloning.sh

# Or install manually
pip uninstall -y bangla
pip install "bangla==0.0.2"
pip install TTS
```

## 📋 Requirements

1. **Ollama running** with `gemma3n:latest` model
2. **Python packages**: `whisper`, `gtts`, `pydub`, `requests`
3. **Optional**: `TTS` for voice cloning
4. **FFmpeg** installed and in PATH

## 🎯 What it does

1. **📁 Loads** your Russian MP3 podcast
2. **🎤 Transcribes** speech to Russian text (using Whisper)
3. **🔄 Translates** Russian → English (using Ollama)
4. **🎭 Generates** English audio with cloned voice (using TTS)
5. **💾 Saves** the translated podcast

## ⏱️ Performance

- **Small podcast (5-10 min)**: ~2-5 minutes
- **Medium podcast (30 min)**: ~10-15 minutes  
- **Large podcast (1 hour)**: ~20-30 minutes

Voice cloning adds ~50% more time but much better quality.

## 🚨 Troubleshooting

**"Input file not found"**
```bash
ls -la  # Check your files
python3 podcast_translator_cli.py /full/path/to/file.mp3
```

**"Ollama server not running"**
```bash
ollama serve  # Start in another terminal
```

**"Voice cloning not available"**
```bash
./fix_voice_cloning.sh  # Run fix script
# Or use: --no-voice-cloning
```

**"Model not found"**
```bash
ollama pull gemma3n:latest
# Or use different model: --model llama2:7b
``` 