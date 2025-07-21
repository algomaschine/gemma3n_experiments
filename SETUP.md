# Podcast Translator Setup Guide

This guide will help you set up the Russian to English Podcast Translator with optional voice cloning capabilities.

## Prerequisites

1. **Python 3.9+** (recommended: Python 3.9)
2. **Ollama** - Download from [https://ollama.com/](https://ollama.com/)
3. **FFmpeg** - Required for audio processing

## Installation Options

### Option 1: Conda Environment (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate environment
conda activate podcast_translator

# Install Ollama model
ollama pull gemma3n:latest
```

### Option 2: pip with Virtual Environment

```bash
# Create virtual environment
python -m venv podcast_translator_env

# Activate virtual environment
# Windows:
podcast_translator_env\Scripts\activate
# Linux/Mac:
source podcast_translator_env/bin/activate

# Install dependencies
pip install -r requirements-minimal.txt

# For voice cloning support (optional):
pip install TTS>=0.20.0
```

### Option 3: Manual Installation

```bash
# Core dependencies only
pip install -r requirements-minimal.txt

# Install Ollama model
ollama pull gemma3n:latest
```

## FFmpeg Installation

### Windows
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract and add to your system PATH
3. OR install via conda: `conda install ffmpeg`

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### macOS
```bash
brew install ffmpeg
```

## Voice Cloning Setup (Optional)

Voice cloning provides much better audio quality by cloning the original speaker's voice:

### Easy Installation (if pip worked)
```bash
pip install TTS>=0.20.0
```

### Alternative Installation (if compilation fails)
```bash
# Install pre-compiled wheels
pip install --find-links https://download.pytorch.org/whl/torch_stable.html TTS

# Or try the development version
pip install git+https://github.com/coqui-ai/TTS.git
```

### Windows Compilation Issues
If you encounter compilation errors on Windows:

1. Install Visual Studio Build Tools
2. OR use conda instead of pip:
   ```bash
   conda install -c conda-forge coqui-tts
   ```
3. OR disable voice cloning by commenting out TTS lines in requirements.txt

## Verification

Test your installation:

```python
# Test basic functionality
python -c "import whisper, gtts, pydub; print('Basic setup OK')"

# Test voice cloning (optional)
python -c "from TTS.api import TTS; print('Voice cloning available')"
```

## Running the Application

```bash
python podcast_translator.py
```

## Troubleshooting

### Common Issues

1. **"TTS library not installed"** - Voice cloning disabled, but basic translation works
2. **"Cannot import 'tarfile' from 'backports'"** - Run: `pip install --upgrade backports.tarfile`
3. **FFmpeg not found** - Install FFmpeg and add to PATH
4. **Ollama connection error** - Start Ollama server: `ollama serve`

### Windows-Specific Issues

1. **Compilation errors** - Use conda or pre-compiled wheels
2. **Path issues** - Use forward slashes or raw strings in file paths
3. **Permission errors** - Run as administrator or install in user directory

### Voice Cloning Issues

1. **Model download fails** - Check internet connection, ~1.8GB download
2. **GPU memory error** - Use CPU-only mode or reduce chunk size
3. **Audio quality poor** - Ensure good quality voice sample (10+ seconds)

## File Structure

After setup, your project should look like:

```
podcast_translator/
├── podcast_translator.py      # Main application
├── requirements.txt           # Full dependencies
├── requirements-minimal.txt   # Basic dependencies
├── environment.yml           # Conda environment
├── SETUP.md                 # This file
├── config.json              # Auto-generated config
├── error_log.txt           # Auto-generated error log
└── temp files/             # Temporary audio files (auto-cleaned)
```

## Performance Tips

1. **GPU Support** - PyTorch with CUDA for faster processing
2. **Audio Quality** - Higher quality input = better voice cloning
3. **Chunk Size** - Adjust text chunk sizes for memory management
4. **Model Selection** - Use appropriate Whisper model size for your hardware

## Support

- **Ollama Issues** - Check [Ollama Documentation](https://ollama.com/docs)
- **Voice Cloning** - See [Coqui TTS Repository](https://github.com/coqui-ai/TTS)
- **Audio Processing** - Check FFmpeg installation and PATH

## Development

For development setup:

```bash
# Install development dependencies
pip install pytest black isort

# Run tests
pytest

# Format code
black podcast_translator.py
isort podcast_translator.py
``` 