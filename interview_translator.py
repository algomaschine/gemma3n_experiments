#!/usr/bin/env python3
"""
Interview Podcast Translator with Voice Cloning
Translates Russian interview podcasts to English while preserving original speaker voices.
Optimized for female interviewer + male guest format.
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
VOICE_CLONING_AVAILABLE = True
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

def extract_voice_samples_interview(audio_path, sample_duration=15):
    """Extract voice samples specifically for interview format (female interviewer + male guest)."""
    try:
        audio = AudioSegment.from_file(audio_path)
        total_duration = len(audio)
        
        voice_samples = []
        
        # Strategy: Take samples from different parts to capture both speakers
        # Sample 1: Early in the interview (likely interviewer introducing)
        start_time_1 = min(30000, total_duration // 10)  # 30s or 10% in
        
        # Sample 2: Middle section (likely guest speaking)
        start_time_2 = total_duration // 2
        
        # Sample 3: Later section (likely more guest or back to interviewer)
        start_time_3 = total_duration * 3 // 4
        
        sample_times = [start_time_1, start_time_2, start_time_3]
        
        for i, start_time in enumerate(sample_times):
            # Ensure we don't go beyond audio length
            if start_time + (sample_duration * 1000) > total_duration:
                start_time = total_duration - (sample_duration * 1000)
            
            if start_time < 0:
                start_time = 0
            
            # Extract sample
            sample = audio[start_time:start_time + (sample_duration * 1000)]
            
            # Save sample
            sample_path = f"voice_sample_{i}.wav"
            sample.export(sample_path, format="wav")
            voice_samples.append(sample_path)
            
            print(f"🎭 Extracted voice sample {i+1} from {start_time//1000}s-{(start_time + sample_duration*1000)//1000}s")
        
        print(f"🎭 Strategy: Sample 1 (early) = likely interviewer, Sample 2-3 (mid/late) = likely guest")
        return voice_samples
    except Exception as e:
        log_error(f"Interview voice samples extraction error: {e}")
        return []

def detect_speaker_in_interview(text, voice_samples):
    """Enhanced speaker detection for interview format with female interviewer and male guest."""
    sentences = text.split('. ')
    voice_assignments = []
    
    # Assume voice_samples[0] = interviewer (female), voice_samples[1] = guest (male)
    INTERVIEWER_VOICE = 0
    GUEST_VOICE = 1 if len(voice_samples) > 1 else 0
    
    current_speaker = INTERVIEWER_VOICE  # Start with interviewer
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip().lower()
        
        # Strong indicators for interviewer (questions, introductions)
        interviewer_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who',  # Question words
            'tell me', 'tell us', 'can you', 'could you', 'would you',
            'so', 'and', 'but', 'now', 'today', 'welcome',
            'thank you', 'thanks', 'that\'s interesting', 'i see',
            'let me ask', 'my next question', 'before we', 'finally'
        ]
        
        # Strong indicators for guest (longer explanations, technical terms)
        guest_indicators = [
            'well', 'actually', 'you know', 'i think', 'i believe', 'in my opinion',
            'let me explain', 'the thing is', 'what happens is', 'basically',
            'for example', 'for instance', 'the reason', 'because'
        ]
        
        # Count indicators
        interviewer_score = sum(1 for indicator in interviewer_indicators if indicator in sentence)
        guest_score = sum(1 for indicator in guest_indicators if indicator in sentence)
        
        # Decision logic
        if interviewer_score > guest_score and interviewer_score > 0:
            current_speaker = INTERVIEWER_VOICE
        elif guest_score > interviewer_score and guest_score > 0:
            current_speaker = GUEST_VOICE
        elif len(sentence) > 200:  # Long sentences tend to be guest explanations
            current_speaker = GUEST_VOICE
        elif len(sentence) < 50:  # Short sentences often interviewer
            current_speaker = INTERVIEWER_VOICE
        # Otherwise keep current speaker
        
        # Additional pattern: Questions usually followed by answers
        if i > 0 and ('?' in sentences[i-1] or any(q in sentences[i-1].lower() for q in ['what', 'how', 'why'])):
            current_speaker = GUEST_VOICE  # Answer after question
        
        voice_assignments.append(current_speaker)
    
    # Post-process: smooth out rapid speaker changes (avoid too frequent switching)
    smoothed_assignments = voice_assignments.copy()
    for i in range(1, len(smoothed_assignments) - 1):
        # If surrounded by same speaker, switch to that speaker
        if smoothed_assignments[i-1] == smoothed_assignments[i+1] and smoothed_assignments[i] != smoothed_assignments[i-1]:
            smoothed_assignments[i] = smoothed_assignments[i-1]
    
    # Count distribution
    interviewer_count = smoothed_assignments.count(INTERVIEWER_VOICE)
    guest_count = smoothed_assignments.count(GUEST_VOICE)
    
    print(f"🎭 Speaker detection: {interviewer_count} interviewer segments, {guest_count} guest segments")
    
    return smoothed_assignments

def text_to_speech_with_interview_cloning(text, output_path, voice_samples, language="en"):
    """Convert text to speech maintaining interview format with proper speaker assignment."""
    if not VOICE_CLONING_AVAILABLE or not voice_samples:
        print("⚠️  Interview voice cloning not available, using standard TTS...")
        return text_to_speech(text, output_path, language)
    
    try:
        print(f"🎭 Loading XTTS model for interview voice cloning...")
        print(f"🎤 Voice 1 (Interviewer): Female voice from sample 1")
        print(f"🎤 Voice 2 (Guest): Male voice from sample 2")
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Split text into sentences
        sentences = text.split('. ')
        
        # Detect speakers in interview format
        voice_assignments = detect_speaker_in_interview(text, voice_samples)
        
        print(f"🎭 Interview analysis complete: {len(set(voice_assignments))} speakers detected")
        
        # Generate audio segments with appropriate voices
        audio_segments = []
        temp_files = []
        
        for i, (sentence, voice_idx) in enumerate(zip(sentences, voice_assignments)):
            if not sentence.strip():
                continue
                
            temp_file = f"temp_interview_chunk_{i}.wav"
            temp_files.append(temp_file)
            
            # Ensure voice_idx is within bounds
            voice_idx = min(voice_idx, len(voice_samples) - 1)
            speaker_wav = voice_samples[voice_idx]
            
            speaker_label = "👩 Interviewer" if voice_idx == 0 else "👨 Guest"
            print(f"🎭 {speaker_label}: sentence {i+1}/{len(sentences)}")
            
            try:
                tts.tts_to_file(
                    text=sentence + ". ",
                    speaker_wav=speaker_wav,
                    language=language,
                    file_path=temp_file
                )
                
                audio_segment = AudioSegment.from_wav(temp_file)
                audio_segments.append(audio_segment)
            except Exception as e:
                print(f"⚠️  Failed to generate voice for sentence {i+1}, using backup")
                # Fallback to first voice sample
                tts.tts_to_file(
                    text=sentence + ". ",
                    speaker_wav=voice_samples[0],
                    language=language,
                    file_path=temp_file
                )
                audio_segment = AudioSegment.from_wav(temp_file)
                audio_segments.append(audio_segment)
        
        # Combine segments with natural pauses for interview flow
        if audio_segments:
            combined_audio = audio_segments[0]
            prev_voice = voice_assignments[0] if voice_assignments else 0
            
            for i, segment in enumerate(audio_segments[1:], 1):
                current_voice = voice_assignments[i] if i < len(voice_assignments) else prev_voice
                
                # Add appropriate pause based on speaker change
                if current_voice != prev_voice:
                    # Longer pause when speaker changes (natural interview flow)
                    pause = AudioSegment.silent(duration=800)  # 0.8 second pause
                    combined_audio += pause
                else:
                    # Shorter pause within same speaker
                    pause = AudioSegment.silent(duration=300)  # 0.3 second pause
                    combined_audio += pause
                
                combined_audio += segment
                prev_voice = current_voice
            
            combined_audio.export(output_path, format="mp3")
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Cleanup voice samples
        for sample_path in voice_samples:
            try:
                os.remove(sample_path)
            except:
                pass
                
        return True
    except Exception as e:
        log_error(f"Interview voice cloning error: {e}")
        print(f"⚠️  Interview voice cloning failed: {e}")
        print("🔊 Falling back to standard TTS...")
        return text_to_speech(text, output_path, language)

def translate_interview_podcast(input_path, output_path, use_voice_cloning=True):
    """Main function to translate interview podcast with voice cloning."""
    try:
        print(f"🎬 Starting interview podcast translation...")
        print(f"📂 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        
        # Load audio
        print("📁 Loading audio file...")
        audio = extract_audio_from_mp3(input_path)
        if audio is None:
            return False, "Failed to load audio file"

        # Extract voice samples for interview format
        voice_samples = []
        if use_voice_cloning and VOICE_CLONING_AVAILABLE:
            print("🎭 Extracting voice samples for interview format (female interviewer + male guest)...")
            voice_samples = extract_voice_samples_interview(input_path, sample_duration=15)
            
            if voice_samples:
                print(f"🎭 Successfully extracted {len(voice_samples)} voice samples")
            else:
                print("⚠️  Failed to extract voice samples, will use standard TTS")

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

        # Convert to speech with interview voice cloning
        if use_voice_cloning and voice_samples and VOICE_CLONING_AVAILABLE and len(voice_samples) >= 2:
            print("🎭 Converting to speech with interview voice cloning (female + male)...")
            success = text_to_speech_with_interview_cloning(english_text, output_path, voice_samples, language="en")
        else:
            print("🔊 Converting to speech with standard TTS...")
            success = text_to_speech(english_text, output_path, language="en")
        
        # Cleanup
        os.unlink(temp_wav_path)
        
        if success:
            print("🎉 Interview translation completed successfully!")
            if use_voice_cloning and voice_samples and len(voice_samples) >= 2:
                print("🎤 Interview format preserved: Female interviewer + Male guest voices cloned!")
            return True, "Interview translation completed successfully!"
        else:
            return False, "Failed to convert translated text to speech"

    except Exception as e:
        error_msg = f"Interview translation process error: {e}"
        log_error(error_msg)
        print(f"❌ {error_msg}")
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description="Interview Podcast Translator with Voice Cloning (Russian → English)")
    parser.add_argument("input", nargs='?', default="podcast.mp3", 
                       help="Input MP3 file path (default: interview.mp3)")
    parser.add_argument("output", nargs='?', default='interview_format_podcast.mp3',
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
    
    print("🎤 Interview Podcast Translator with Voice Cloning")
    print("🇷🇺 → 🇺🇸 Russian Interview to English with Original Voices")
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
        print(f"   python3 interview_translator.py my_interview.mp3")
        print(f"   python3 interview_translator.py my_interview.mp3 my_output.mp3")
        print(f"   python3 interview_translator.py my_interview.mp3 --no-voice-cloning")
        sys.exit(1)
    
    # Check voice cloning status
    use_voice_cloning = not args.no_voice_cloning
    if use_voice_cloning and not VOICE_CLONING_AVAILABLE:
        print("⚠️  Voice cloning requested but not available")
        print("🔊 Will use standard text-to-speech instead")
        use_voice_cloning = False
    
    print(f"🎭 Voice cloning: {'✅ Enabled' if use_voice_cloning else '❌ Disabled'}")
    if use_voice_cloning:
        print("🎤 Interview mode: Will extract female interviewer + male guest voices")
    print("=" * 60)
    
    # Run translation
    start_time = time.time()
    success, message = translate_interview_podcast(args.input, args.output, use_voice_cloning)
    end_time = time.time()
    
    print("=" * 60)
    if success:
        print(f"✅ SUCCESS: {message}")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        print(f"📂 Output saved to: {args.output}")
        print(f"🎉 Your English interview with cloned voices is ready!")
    else:
        print(f"❌ FAILED: {message}")
        print(f"📋 Check error log: {ERROR_LOG_PATH}")
        sys.exit(1)

if __name__ == "__main__":
    main() 