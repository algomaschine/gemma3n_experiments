import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, font, ttk
import threading
import re
import os
import traceback
import sys
import requests
import json
import time
import socket
import subprocess
import whisper
from gtts import gTTS
from pydub import AudioSegment
import tempfile

# Additional import for voice cloning
try:
    from TTS.api import TTS
    VOICE_CLONING_AVAILABLE = True
except ImportError:
    VOICE_CLONING_AVAILABLE = False
    print("TTS library not installed. Voice cloning will not be available.")

# --- Environment Setup Instructions ---
# 1. Install and run Ollama from https://ollama.com/
# 2. In your terminal, run: `ollama pull gemma3n:latest` to get the specific model.
# 3. Create a clean conda environment:
#    conda create -n podcast_translator python=3.9 -y
#    conda activate podcast_translator
# 4. Install the required packages for this script:
#    pip install requests openai-whisper gtts pydub
#    
#    For voice cloning support (optional but recommended):
#    pip install TTS
#    
#    Note: TTS library is ~1.8GB and requires additional dependencies
# 5. Install FFmpeg for audio processing:
#    - Windows: Download from https://ffmpeg.org/ and add to PATH
#    - Or use conda: conda install ffmpeg
# 6. Run this script:
#    python podcast_translator.py

# --- Configuration ---
CONFIG_PATH = "config.json"
ERROR_LOG_PATH = "error_log.txt"
LLM_MODEL = "gemma3n:latest"

# Load configuration
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
        print(f"Configuration file not found. Creating a default '{CONFIG_PATH}'.")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return config
    try:
        with open(CONFIG_PATH, 'r') as f:
            loaded_config = json.load(f)
            config.update(loaded_config)
            return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading config file: {e}. Using default values.")
        return config

# Load config and set globals
config = load_config()
OLLAMA_HOST = config["OLLAMA_HOST"]
OLLAMA_MODEL = config["OLLAMA_MODEL"]
MAX_RETRIES = config["MAX_RETRIES"]
RETRY_DELAY_SECONDS = config["RETRY_DELAY_SECONDS"]
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"

# --- Helper Functions ---

def log_error(error_message):
    """Appends a detailed error message to the log file."""
    log_entry = (
        f"--- ERROR ---\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Error: {error_message}\n"
        f"{'='*50}\n"
    )
    
    with open(ERROR_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(log_entry)

# --- Ollama Interaction ---

def is_ollama_running():
    """Checks if the Ollama server is running and accessible."""
    try:
        requests.get(OLLAMA_HOST, timeout=5)
        return True
    except requests.exceptions.ConnectionError:
        return False

def start_ollama():
    """Starts the Ollama server if it's not already running."""
    if is_ollama_running():
        print("Ollama server is already running.")
        return True

    print("Ollama server not running. Attempting to start it...")
    try:
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):  # Wait up to 20 seconds
            if is_ollama_running():
                print("Ollama server started successfully.")
                return True
            time.sleep(1)
        print("Error: Ollama server did not start in time.")
        return False
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and its directory is in your system's PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while starting Ollama: {e}")
        return False

def ensure_ollama_model(model_name):
    """Checks if the specified model is available locally and pulls it if not."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        models = [m['name'] for m in resp.json().get('models', [])]
        if model_name not in models:
            print(f"Model '{model_name}' not found locally. Pulling it now (this may take a while)...")
            with requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model_name}, stream=True) as pull_resp:
                pull_resp.raise_for_status()
                for line in pull_resp.iter_lines():
                    if line:
                        progress = json.loads(line.decode())
                        if 'total' in progress and 'completed' in progress:
                            percent = (progress['completed'] / progress['total']) * 100
                            print(f"\rDownloading {model_name}: {percent:.2f}%", end="")
            print(f"\nModel '{model_name}' pulled successfully.")
        else:
            print(f"Model '{model_name}' is already available locally.")
        return True
    except Exception as e:
        print(f"Error checking/pulling Ollama model: {e}")
        return False

def translate_text_with_ollama(russian_text):
    """Translates Russian text to English using the Ollama model."""
    if not russian_text.strip():
        return "No text to translate."

    # Split long text into chunks to avoid token limits
    chunks = []
    sentences = russian_text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 3000:  # Keep chunks under 3000 chars
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    translated_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i+1}/{len(chunks)}...")
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate the following Russian text to English. Provide only the English translation, nothing else. Maintain the natural flow and meaning of the original text."
            },
            {
                "role": "user", 
                "content": f"Translate this Russian text to English: {chunk}"
            }
        ]

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.3  # Lower temperature for more consistent translations
            }
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=120)
                response.raise_for_status()
                response_data = response.json()
                translated_chunk = response_data.get("message", {}).get("content", "")
                translated_chunks.append(translated_chunk)
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\nTranslation failed (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    error_msg = f"Translation failed after {MAX_RETRIES} attempts: {e}"
                    log_error(error_msg)
                    return f"ERROR: {error_msg}"
            except Exception as e:
                error_msg = traceback.format_exc()
                log_error(error_msg)
                return f"ERROR: An unexpected error occurred. See {ERROR_LOG_PATH} for details."

    return " ".join(translated_chunks)

# --- Audio Processing Functions ---

def extract_audio_from_mp3(mp3_path):
    """Loads audio from MP3 file."""
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        return audio
    except Exception as e:
        error_msg = f"Error loading MP3 file: {e}"
        log_error(error_msg)
        return None

def speech_to_text(audio_path, language="ru"):
    """Converts speech to text using OpenAI Whisper."""
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Transcribing audio...")
        result = model.transcribe(audio_path, language=language)
        return result["text"]
    except Exception as e:
        error_msg = f"Error in speech-to-text conversion: {e}"
        log_error(error_msg)
        return None

def text_to_speech(text, output_path, language="en"):
    """Converts text to speech using gTTS."""
    try:
        print("Converting text to speech...")
        
        # Split text into chunks for gTTS (it has a character limit)
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 5000:  # gTTS limit
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Create audio segments for each chunk
        audio_segments = []
        temp_files = []
        
        for i, chunk in enumerate(chunks):
            temp_file = f"temp_chunk_{i}.mp3"
            temp_files.append(temp_file)
            
            tts = gTTS(text=chunk, lang=language, slow=False)
            tts.save(temp_file)
            
            audio_segment = AudioSegment.from_mp3(temp_file)
            audio_segments.append(audio_segment)
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            combined_audio.export(output_path, format="mp3")
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
                
        return True
    except Exception as e:
        error_msg = f"Error in text-to-speech conversion: {e}"
        log_error(error_msg)
        return False

def extract_voice_sample(audio_path, sample_duration=10):
    """Extract a sample of the original voice for cloning."""
    try:
        audio = AudioSegment.from_file(audio_path)
        # Take first 10 seconds as voice sample
        sample = audio[:sample_duration * 1000]
        
        # Save sample as WAV for voice cloning
        sample_path = "voice_sample.wav"
        sample.export(sample_path, format="wav")
        return sample_path
    except Exception as e:
        error_msg = f"Error extracting voice sample: {e}"
        log_error(error_msg)
        return None

def text_to_speech_with_cloning(text, output_path, speaker_wav_path, language="en"):
    """Converts text to speech using voice cloning with Coqui XTTS."""
    if not VOICE_CLONING_AVAILABLE:
        print("Voice cloning not available, falling back to gTTS...")
        return text_to_speech(text, output_path, language)
    
    try:
        print("Loading XTTS model for voice cloning...")
        # Initialize XTTS model (this will download ~1.8GB on first use)
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Split text into chunks for XTTS (it has limits too)
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:  # XTTS works better with shorter chunks
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
            temp_file = f"temp_cloned_chunk_{i}.wav"
            temp_files.append(temp_file)
            
            print(f"Generating cloned voice for chunk {i+1}/{len(chunks)}...")
            
            tts.tts_to_file(
                text=chunk,
                speaker_wav=speaker_wav_path,
                language=language,
                file_path=temp_file
            )
            
            audio_segment = AudioSegment.from_wav(temp_file)
            audio_segments.append(audio_segment)
        
        # Combine all audio segments
        if audio_segments:
            combined_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                combined_audio += segment
            
            combined_audio.export(output_path, format="mp3")
        
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
                
        return True
    except Exception as e:
        error_msg = f"Error in voice cloning TTS: {e}"
        log_error(error_msg)
        print("Voice cloning failed, falling back to gTTS...")
        return text_to_speech(text, output_path, language)

def translate_podcast(input_mp3_path, output_mp3_path, progress_callback=None, use_voice_cloning=True):
    """Main function to translate a Russian podcast to English with optional voice cloning."""
    try:
        if progress_callback:
            progress_callback("Starting translation process...")

        # Step 1: Load and prepare audio
        if progress_callback:
            progress_callback("Loading audio file...")
        
        audio = extract_audio_from_mp3(input_mp3_path)
        if audio is None:
            return False, "Failed to load audio file"

        # Extract voice sample for cloning if enabled
        voice_sample_path = None
        if use_voice_cloning and VOICE_CLONING_AVAILABLE:
            if progress_callback:
                progress_callback("Extracting voice sample for cloning...")
            voice_sample_path = extract_voice_sample(input_mp3_path)

        # Create temporary WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            audio.export(temp_wav_path, format="wav")

        # Step 2: Convert speech to text
        if progress_callback:
            progress_callback("Converting speech to text...")
        
        russian_text = speech_to_text(temp_wav_path, language="ru")
        if russian_text is None:
            os.unlink(temp_wav_path)
            return False, "Failed to convert speech to text"

        print(f"Extracted text: {russian_text[:200]}...")  # Show first 200 chars

        # Step 3: Translate text
        if progress_callback:
            progress_callback("Translating text...")
        
        english_text = translate_text_with_ollama(russian_text)
        if english_text.startswith("ERROR:"):
            os.unlink(temp_wav_path)
            return False, english_text

        print(f"Translated text: {english_text[:200]}...")  # Show first 200 chars

        # Step 4: Convert translated text to speech (with or without voice cloning)
        if use_voice_cloning and voice_sample_path and VOICE_CLONING_AVAILABLE:
            if progress_callback:
                progress_callback("Converting translated text to speech with voice cloning...")
            success = text_to_speech_with_cloning(english_text, output_mp3_path, voice_sample_path, language="en")
        else:
            if progress_callback:
                progress_callback("Converting translated text to speech...")
            success = text_to_speech(english_text, output_mp3_path, language="en")
        
        # Cleanup
        os.unlink(temp_wav_path)
        if voice_sample_path and os.path.exists(voice_sample_path):
            os.unlink(voice_sample_path)
        
        if success:
            if progress_callback:
                progress_callback("Translation completed successfully!")
            return True, "Translation completed successfully!"
        else:
            return False, "Failed to convert translated text to speech"

    except Exception as e:
        error_msg = f"Error in translation process: {e}"
        log_error(error_msg)
        if progress_callback:
            progress_callback(f"Error: {error_msg}")
        return False, error_msg

# --- GUI Application ---

class PodcastTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Russian to English Podcast Translator")
        self.root.geometry("800x600")
        
        self.input_file = ""
        self.output_file = ""
        
        self.create_widgets()
        
        # Check Ollama on startup
        self.check_ollama_status()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Russian to English Podcast Translator", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        ttk.Label(main_frame, text="Input MP3 file:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_label = ttk.Label(main_frame, text="No file selected", foreground="gray")
        self.input_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_input_file).grid(row=1, column=2, padx=5)
        
        # Output file selection
        ttk.Label(main_frame, text="Output MP3 file:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_label = ttk.Label(main_frame, text="No file selected", foreground="gray")
        self.output_label.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.select_output_file).grid(row=2, column=2, padx=5)
        
        # Voice cloning option
        self.voice_cloning_var = tk.BooleanVar(value=VOICE_CLONING_AVAILABLE)
        self.voice_cloning_checkbox = ttk.Checkbutton(
            main_frame, 
            text="Use Voice Cloning (clone original speaker's voice)", 
            variable=self.voice_cloning_var,
            state="normal" if VOICE_CLONING_AVAILABLE else "disabled"
        )
        self.voice_cloning_checkbox.grid(row=3, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        if not VOICE_CLONING_AVAILABLE:
            ttk.Label(main_frame, text="Note: Install 'TTS' package for voice cloning support", 
                     foreground="orange", font=('Arial', 8)).grid(row=4, column=0, columnspan=3, sticky=tk.W)
        
        # Translate button
        self.translate_button = ttk.Button(main_frame, text="Translate Podcast", 
                                         command=self.start_translation, state="disabled")
        self.translate_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Status text
        self.status_text = scrolledtext.ScrolledText(main_frame, height=20, wrap=tk.WORD)
        self.status_text.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(7, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def log_message(self, message):
        """Add message to status text widget."""
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def check_ollama_status(self):
        """Check if Ollama is running and the model is available."""
        self.log_message("Checking Ollama server status...")
        
        if not start_ollama():
            self.log_message("❌ Failed to start Ollama server. Please start it manually.")
            return
        
        if not ensure_ollama_model(OLLAMA_MODEL):
            self.log_message(f"❌ Failed to ensure model {OLLAMA_MODEL} is available.")
            return
        
        self.log_message("✅ Ollama server is running and model is ready!")

    def select_input_file(self):
        """Select input MP3 file."""
        file_path = filedialog.askopenfilename(
            title="Select Russian Podcast MP3 File",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        if file_path:
            self.input_file = file_path
            self.input_label.config(text=os.path.basename(file_path), foreground="black")
            self.update_translate_button()

    def select_output_file(self):
        """Select output MP3 file."""
        file_path = filedialog.asksaveasfilename(
            title="Save Translated Podcast As",
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3"), ("All files", "*.*")]
        )
        if file_path:
            self.output_file = file_path
            self.output_label.config(text=os.path.basename(file_path), foreground="black")
            self.update_translate_button()

    def update_translate_button(self):
        """Enable translate button if both files are selected."""
        if self.input_file and self.output_file:
            self.translate_button.config(state="normal")

    def progress_callback(self, message):
        """Callback function for translation progress."""
        self.log_message(message)

    def start_translation(self):
        """Start the translation process in a separate thread."""
        if not self.input_file or not self.output_file:
            messagebox.showerror("Error", "Please select both input and output files.")
            return

        self.translate_button.config(state="disabled")
        self.progress.start()
        
        # Start translation in a separate thread
        thread = threading.Thread(target=self.translate_thread)
        thread.daemon = True
        thread.start()

    def translate_thread(self):
        """Translation thread function."""
        try:
            self.log_message("🚀 Starting podcast translation...")
            use_voice_cloning = self.voice_cloning_var.get()
            
            if use_voice_cloning and VOICE_CLONING_AVAILABLE:
                self.log_message("🎤 Voice cloning enabled - will clone original speaker's voice")
            else:
                self.log_message("🔊 Using standard text-to-speech")
            
            success, message = translate_podcast(
                self.input_file, 
                self.output_file, 
                self.progress_callback,
                use_voice_cloning=use_voice_cloning
            )
            
            if success:
                self.log_message("🎉 Translation completed successfully!")
                messagebox.showinfo("Success", "Podcast translation completed!")
            else:
                self.log_message(f"❌ Translation failed: {message}")
                messagebox.showerror("Error", f"Translation failed: {message}")
                
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
        finally:
            self.progress.stop()
            self.translate_button.config(state="normal")

def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = PodcastTranslatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 