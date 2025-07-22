#!/usr/bin/env python3
"""
Command-line version of the Russian to English Podcast Translator with Voice Cloning
Works in WSL and headless environments.
"""

import argparse
import os
import sys
import traceback
import time
import json
import requests
import subprocess
import tempfile
import whisper
from gtts import gTTS
from pydub import AudioSegment

# Voice cloning support with better error handling
VOICE_CLONING_AVAILABLE = False
try:
    # Fix for Python 3.9 compatibility
    import sys
    if sys.version_info >= (3, 10):
        from TTS.api import TTS
        VOICE_CLONING_AVAILABLE = True
    else:
        # Try to import with monkey patch for older Python
        import types
        import builtins
        
        # Monkey patch for union operator compatibility
        if not hasattr(types, 'UnionType'):
            types.UnionType = type(int | str) if sys.version_info >= (3, 10) else type(None)
        
        # Try importing with compatibility patches
        try:
            from TTS.api import TTS
            VOICE_CLONING_AVAILABLE = True
            print("✅ Voice cloning available!")
        except Exception as e:
            print(f"❌ Voice cloning not available: {e}")
            print("📝 Using standard text-to-speech instead")
except ImportError as e:
    print(f"❌ TTS library not installed: {e}")
    print("📝 Install with: pip install TTS")
except Exception as e:
    print(f"❌ Voice cloning initialization failed: {e}")

# Configuration
CONFIG_PATH = "config.json"
ERROR_LOG_PATH = "error_log.txt"
LLM_MODEL = "gemma3n:latest"

config = {
    "OLLAMA_HOST": "http://localhost:11434",
    "OLLAMA_MODEL": LLM_MODEL,
    "MAX_RETRIES": 3,
    "RETRY_DELAY_SECONDS": 60
}

def load_config():
    """Loads configuration from JSON file."""
    global config
    if not os.path.exists(CONFIG_PATH):
        print(f"Creating default config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return config
    try:
        with open(CONFIG_PATH, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading config: {e}. Using defaults.")
        return config

def log_error(error_message):
    """Appends error to log file."""
    log_entry = (
        f"--- ERROR ---\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Error: {error_message}\n"
        f"{'='*50}\n"
    )
    
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(log_entry)

def is_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get(config["OLLAMA_HOST"], timeout=5)
        return True
    except:
        return False

def start_ollama():
    """Start Ollama server if not running."""
    if is_ollama_running():
        print("✅ Ollama server is running")
        return True
    
    print("🚀 Starting Ollama server...")
    try:
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        for i in range(20):
            if is_ollama_running():
                print("✅ Ollama server started")
                return True
            time.sleep(1)
            
        print("❌ Ollama server failed to start in time")
        return False
    except FileNotFoundError:
        print("❌ 'ollama' command not found. Please install Ollama.")
        return False
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False

def ensure_ollama_model(model_name):
    """Ensure the specified model is available."""
    try:
        resp = requests.get(f"{config['OLLAMA_HOST']}/api/tags")
        resp.raise_for_status()
        models = [m['name'] for m in resp.json().get('models', [])]
        
        if model_name not in models:
            print(f"📥 Downloading model '{model_name}'...")
            with requests.post(f"{config['OLLAMA_HOST']}/api/pull", 
                             json={"name": model_name}, stream=True) as pull_resp:
                pull_resp.raise_for_status()
                for line in pull_resp.iter_lines():
                    if line:
                        progress = json.loads(line.decode())
                        if 'total' in progress and 'completed' in progress:
                            percent = (progress['completed'] / progress['total']) * 100
                            print(f"\r📥 Downloading: {percent:.1f}%", end="", flush=True)
            print(f"\n✅ Model '{model_name}' ready!")
        else:
            print(f"✅ Model '{model_name}' available")
        return True
    except Exception as e:
        print(f"❌ Error with model {model_name}: {e}")
        return False

def translate_text_with_ollama(russian_text):
    """Translate Russian text to English using Ollama."""
    if not russian_text.strip():
        return "No text to translate."

    # Split into chunks
    chunks = []
    sentences = russian_text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 3000:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    translated_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"🔄 Translating chunk {i+1}/{len(chunks)}...")
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the Russian text to English. Provide only the English translation, maintaining natural flow and meaning."
            },
            {
                "role": "user", 
                "content": f"Translate this Russian text to English: {chunk}"
            }
        ]

        payload = {
            "model": config["OLLAMA_MODEL"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3}
        }

        for attempt in range(config["MAX_RETRIES"]):
            try:
                response = requests.post(f"{config['OLLAMA_HOST']}/api/chat", 
                                       json=payload, timeout=120)
                response.raise_for_status()
                response_data = response.json()
                translated_chunk = response_data.get("message", {}).get("content", "")
                translated_chunks.append(translated_chunk)
                break
            except Exception as e:
                if attempt < config["MAX_RETRIES"] - 1:
                    print(f"⚠️  Translation failed (attempt {attempt + 1}), retrying...")
                    time.sleep(config["RETRY_DELAY_SECONDS"])
                else:
                    error_msg = f"Translation failed after {config['MAX_RETRIES']} attempts: {e}"
                    log_error(error_msg)
                    return f"ERROR: {error_msg}"

    return " ".join(translated_chunks)

def extract_audio_from_mp3(mp3_path):
    """Load audio from MP3 file."""
    try:
        return AudioSegment.from_mp3(mp3_path)
    except Exception as e:
        log_error(f"Error loading MP3: {e}")
        return None

def speech_to_text(audio_path, language="ru"):
    """Convert speech to text using Whisper."""
    try:
        print("🎤 Loading Whisper model...")
        model = whisper.load_model("base")
        print("🎤 Transcribing audio...")
        result = model.transcribe(audio_path, language=language)
        return result["text"]
    except Exception as e:
        log_error(f"Speech-to-text error: {e}")
        return None

def text_to_speech(text, output_path, language="en"):
    """Convert text to speech using gTTS."""
    try:
        print("🔊 Converting text to speech...")
        
        # Split text into chunks
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 5000:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate audio for each chunk
        audio_segments = []
        temp_files = []
        
        for i, chunk in enumerate(chunks):
            temp_file = f"temp_chunk_{i}.mp3"
            temp_files.append(temp_file)
            
            print(f"🔊 Processing chunk {i+1}/{len(chunks)}...")
            tts = gTTS(text=chunk, lang=language, slow=False)
            tts.save(temp_file)
            
            audio_segment = AudioSegment.from_mp3(temp_file)
            audio_segments.append(audio_segment)
        
        # Combine audio segments
        if audio_segments:
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            combined_audio.export(output_path, format="mp3")
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
                
        return True
    except Exception as e:
        log_error(f"Text-to-speech error: {e}")
        return False

def extract_voice_sample(audio_path, sample_duration=10):
    """Extract voice sample for cloning."""
    try:
        audio = AudioSegment.from_file(audio_path)
        sample = audio[:sample_duration * 1000]
        
        sample_path = "voice_sample.wav"
        sample.export(sample_path, format="wav")
        return sample_path
    except Exception as e:
        log_error(f"Voice sample extraction error: {e}")
        return None

def text_to_speech_with_cloning(text, output_path, speaker_wav_path, language="en"):
    """Convert text to speech with voice cloning."""
    if not VOICE_CLONING_AVAILABLE:
        print("⚠️  Voice cloning not available, using standard TTS...")
        return text_to_speech(text, output_path, language)
    
    try:
        print("🎭 Loading XTTS model for voice cloning...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Split text into chunks
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate cloned audio
        audio_segments = []
        temp_files = []
        
        for i, chunk in enumerate(chunks):
            temp_file = f"temp_cloned_chunk_{i}.wav"
            temp_files.append(temp_file)
            
            print(f"🎭 Generating cloned voice {i+1}/{len(chunks)}...")
            
            tts.tts_to_file(
                text=chunk,
                speaker_wav=speaker_wav_path,
                language=language,
                file_path=temp_file
            )
            
            audio_segment = AudioSegment.from_wav(temp_file)
            audio_segments.append(audio_segment)
        
        # Combine segments
        if audio_segments:
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            combined_audio.export(output_path, format="mp3")
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
                
        return True
    except Exception as e:
        log_error(f"Voice cloning error: {e}")
        print(f"⚠️  Voice cloning failed: {e}")
        print("🔊 Falling back to standard TTS...")
        return text_to_speech(text, output_path, language)

def translate_podcast(input_path, output_path, use_voice_cloning=True):
    """Main translation function."""
    try:
        print(f"🎬 Starting podcast translation...")
        print(f"📂 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        
        # Load audio
        print("📁 Loading audio file...")
        audio = extract_audio_from_mp3(input_path)
        if audio is None:
            return False, "Failed to load audio file"

        # Extract voice sample for cloning
        voice_sample_path = None
        if use_voice_cloning and VOICE_CLONING_AVAILABLE:
            print("🎭 Extracting voice sample for cloning...")
            voice_sample_path = extract_voice_sample(input_path)

        # Create temp WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            audio.export(temp_wav_path, format="wav")

        # Speech to text
        print("🎤 Converting speech to text...")
        russian_text = speech_to_text(temp_wav_path, language="ru")
        if russian_text is None:
            os.unlink(temp_wav_path)
            return False, "Failed to convert speech to text"

        print(f"📝 Extracted text preview: {russian_text[:200]}...")

        # Translate text
        print("🔄 Translating text...")
        english_text = translate_text_with_ollama(russian_text)
        if english_text.startswith("ERROR:"):
            os.unlink(temp_wav_path)
            return False, english_text

        print(f"📝 Translated text preview: {english_text[:200]}...")

        # Convert to speech
        if use_voice_cloning and voice_sample_path and VOICE_CLONING_AVAILABLE:
            print("🎭 Converting to speech with voice cloning...")
            success = text_to_speech_with_cloning(english_text, output_path, voice_sample_path, language="en")
        else:
            print("🔊 Converting to speech with standard TTS...")
            success = text_to_speech(english_text, output_path, language="en")
        
        # Cleanup
        os.unlink(temp_wav_path)
        if voice_sample_path and os.path.exists(voice_sample_path):
            os.unlink(voice_sample_path)
        
        if success:
            print("🎉 Translation completed successfully!")
            return True, "Translation completed successfully!"
        else:
            return False, "Failed to convert translated text to speech"

    except Exception as e:
        error_msg = f"Translation process error: {e}"
        log_error(error_msg)
        print(f"❌ {error_msg}")
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="Russian to English Podcast Translator with Voice Cloning")
    parser.add_argument("input", nargs='?', default="podcast.mp3", 
                       help="Input MP3 file path (default: input.mp3)")
    parser.add_argument("output", nargs='?', 
                       help="Output MP3 file path (default: auto-generated from input)")
    parser.add_argument("--no-voice-cloning", action="store_true", 
                       help="Disable voice cloning, use standard TTS")
    parser.add_argument("--model", default=LLM_MODEL, 
                       help=f"Ollama model to use (default: {LLM_MODEL})")
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not provided
    if args.output is None:
        input_path = args.input
        # Get directory, filename without extension, and extension
        input_dir = os.path.dirname(input_path)
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        input_ext = os.path.splitext(input_path)[1]
        
        # Create output filename with 'translated_en' prefix
        output_filename = f"translated_en_{input_name}{input_ext}"
        args.output = os.path.join(input_dir, output_filename)
        
        print(f"📝 Auto-generated output filename: {args.output}")
    
    # Load config
    config = load_config()
    config["OLLAMA_MODEL"] = args.model
    
    print("🚀 Russian to English Podcast Translator with Voice Cloning")
    print("=" * 60)
    print(f"📂 Input file: {args.input}")
    print(f"📂 Output file: {args.output}")
    print(f"🤖 Model: {args.model}")
    
    # Check prerequisites
    if not start_ollama():
        print("❌ Failed to start Ollama server")
        sys.exit(1)
    
    if not ensure_ollama_model(config["OLLAMA_MODEL"]):
        print(f"❌ Failed to load model {config['OLLAMA_MODEL']}")
        sys.exit(1)
    
    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        print(f"💡 Make sure your input file exists or specify a different path")
        print(f"💡 Usage examples:")
        print(f"   python3 podcast_translator_cli.py my_podcast.mp3")
        print(f"   python3 podcast_translator_cli.py my_podcast.mp3 my_output.mp3")
        sys.exit(1)
    
    # Check voice cloning status
    use_voice_cloning = not args.no_voice_cloning
    if use_voice_cloning and not VOICE_CLONING_AVAILABLE:
        print("⚠️  Voice cloning requested but not available")
        print("🔊 Will use standard text-to-speech instead")
        use_voice_cloning = False
    
    print(f"🎭 Voice cloning: {'✅ Enabled' if use_voice_cloning else '❌ Disabled'}")
    print("=" * 60)
    
    # Run translation
    start_time = time.time()
    success, message = translate_podcast(args.input, args.output, use_voice_cloning)
    end_time = time.time()
    
    print("=" * 60)
    if success:
        print(f"✅ SUCCESS: {message}")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        print(f"📂 Output saved to: {args.output}")
        print(f"🎉 Translation complete! Your translated podcast is ready.")
    else:
        print(f"❌ FAILED: {message}")
        print(f"📋 Check error log: {ERROR_LOG_PATH}")
        sys.exit(1)

if __name__ == "__main__":
    main() 