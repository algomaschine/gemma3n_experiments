import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, font
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

# --- Environment Setup Instructions ---
# 1. Install and run Ollama from https://ollama.com/
# 2. In your terminal, run: `ollama pull phi3:mini` to get the specific model.
# 3. Create a clean conda environment:
#    conda create -n chatapp_env python=3.9 -y
#    conda activate chatapp_env
# 4. Install the required packages for this script:
#    pip install requests
# 5. Run this script:
#    python .\local_chat.py

# --- Configuration ---
CONFIG_PATH = "config.json"
ERROR_LOG_PATH = "error_log.txt"
LLM_MODEL = "gemma3n:latest"

def load_config():
    """Loads configuration from JSON file."""
    if not os.path.exists(CONFIG_PATH):
        print(f"Configuration file not found. Creating a default '{CONFIG_PATH}'.")
        default_config = {
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_MODEL": LLM_MODEL,
            "MAX_RETRIES": 3,
            "RETRY_DELAY_SECONDS": 600
        }
        with open(CONFIG_PATH, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading config file: {e}. Using default values.")
        return {
            "OLLAMA_HOST": "http://localhost:11434",
            "OLLAMA_MODEL": LLM_MODEL,
            "MAX_RETRIES": 3,
            "RETRY_DELAY_SECONDS": 60
        }

# Load configuration at startup
config = load_config()
OLLAMA_HOST = config.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = LLM_MODEL # config.get("OLLAMA_MODEL", "phi3:mini")
MAX_RETRIES = config.get("MAX_RETRIES", 3)
RETRY_DELAY_SECONDS = config.get("RETRY_DELAY_SECONDS", 60)
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"



def log_error(error_message):
    """Appends a detailed error message to the log file."""
    log_entry = (
        f"--- ERROR ---\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Error: {error_message}\n"
        f"-------------\n\n"
    )
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)


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
        # This command works if 'ollama' is in the system's PATH.
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for the server to start
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
            # Using stream=True to show download progress
            with requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model_name}, stream=True) as pull_resp:
                pull_resp.raise_for_status()
                for line in pull_resp.iter_lines():
                    if line:
                        # Process and print the streaming progress
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


def get_response_from_ollama(messages):
    """
    Sends a list of messages to the local Ollama server and gets the response.
    Includes a retry mechanism for transient network errors.
    """
    if not messages:
        return "Conversation history is empty."

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)  # Longer timeout for generation
            response.raise_for_status()

            response_data = response.json()
            response_text = response_data.get("message", {}).get("content", "")
            return response_text
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\nRequest failed (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY_SECONDS}s...")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            else:
                error_msg = f"Ollama request failed after {MAX_RETRIES} attempts: {e}\nIs Ollama running?"
                log_error(error_msg)
                return f"ERROR: {error_msg}"
        except Exception as e:
            error_msg = traceback.format_exc()
            log_error(error_msg)
            return f"ERROR: An unexpected error occurred. See {ERROR_LOG_PATH} for details."

    return "ERROR: Failed to get a response after multiple retries."


class OllamaChatApp:
    def __init__(self, root):
        self.root = root
        root.title("Local Chat")
        root.geometry("800x600")
        self.conversation_history = []

        # Top frame for input
        input_frame = tk.Frame(root, pady=10)
        input_frame.pack(fill=tk.X, padx=10)

        tk.Label(input_frame, text="Your Request:").pack(side=tk.LEFT)

        self.input_text = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 5))

        # Middle frame for buttons
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.send_button = tk.Button(button_frame, text="Send", command=self.send_request_thread)
        self.send_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(button_frame, text="Clear History", command=self.clear_history)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.export_button = tk.Button(button_frame, text="Export Response", command=self.export_response)
        self.export_button.pack(side=tk.RIGHT)

        # Bottom frame for output
        output_frame = tk.Frame(root, pady=10)
        output_frame.pack(fill=tk.X, padx=10)

        tk.Label(output_frame, text="Ollama's Response:").pack(side=tk.LEFT)

        self.output_text = scrolledtext.ScrolledText(root, height=15, wrap=tk.WORD, state='disabled')
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Configure tags for rich text
        default_font = font.nametofont("TkDefaultFont")
        bold_font = font.Font(family=default_font.cget("family"), size=default_font.cget("size"), weight="bold")
        self.output_text.tag_configure("bold", font=bold_font)


        self.status_bar = tk.Label(root, text="Ready.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def send_request_thread(self):
        self.send_button.config(state='disabled')
        self.status_bar.config(text="Sending request to Ollama...")
        self.set_output_text("Waiting for response...")
        # Run in a separate thread to keep the GUI responsive
        thread = threading.Thread(target=self.send_request)
        thread.start()

    def send_request(self):
        try:
            prompt = self.input_text.get("1.0", tk.END).strip()
            if not prompt:
                messagebox.showwarning("Empty Prompt", "Please enter a message.")
                return

            self.conversation_history.append({"role": "user", "content": prompt})
            
            response = get_response_from_ollama(self.conversation_history)

            self.conversation_history.append({"role": "assistant", "content": response})

            self.root.after(0, self.set_output_text, response)
            self.root.after(0, self.status_bar.config, {'text': "Ready."})
            self.root.after(0, self.input_text.delete, "1.0", tk.END)
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            self.root.after(0, self.set_output_text, error_msg)
            self.root.after(0, messagebox.showerror, "Error", error_msg)
            self.root.after(0, self.status_bar.config, {'text': "Error."})
        finally:
            self.root.after(0, self.send_button.config, {'state': 'normal'})

    def clear_history(self):
        self.conversation_history = []
        self.set_output_text("")
        self.status_bar.config(text="History cleared. Ready.")
        messagebox.showinfo("History Cleared", "The conversation history has been cleared.")

    def set_output_text(self, text):
        self.output_text.config(state='normal')
        self.output_text.delete("1.0", tk.END)

        for line in text.split('\n'):
            is_list_item = line.strip().startswith('*')
            if is_list_item:
                # Insert bullet point and trim the leading '*' and spaces
                self.output_text.insert(tk.END, "• ")
                line_content = line.strip().lstrip('*').lstrip()
            else:
                line_content = line

            # Handle bold text within the line content
            parts = re.split(r'(\*\*.*?\*\*)', line_content)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    self.output_text.insert(tk.END, part[2:-2], 'bold')
                else:
                    self.output_text.insert(tk.END, part)
            
            self.output_text.insert(tk.END, "\n")

        self.output_text.config(state='disabled')

    def export_response(self):
        response_text = self.output_text.get("1.0", tk.END).strip()
        if not response_text:
            messagebox.showwarning("Export Empty", "There is no response to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Response As"
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(response_text)
                messagebox.showinfo("Export Successful", f"Response saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save file: {e}")


def main():
    """
    Main function to orchestrate the application startup.
    """
    # Create a hidden root window for pre-start checks
    hidden_root = tk.Tk()
    hidden_root.withdraw()

    print("Checking Ollama server...")
    if not start_ollama():
        messagebox.showerror("Ollama Error", "Ollama server could not be started. Please start it manually and restart the application.")
        print("Exiting script because Ollama server could not be started.")
        hidden_root.destroy()
        return

    if not ensure_ollama_model(OLLAMA_MODEL):
        messagebox.showerror("Ollama Model Error", f"Model '{OLLAMA_MODEL}' could not be pulled. Please check your connection or pull it manually.")
        print(f"Exiting script because model '{OLLAMA_MODEL}' could not be pulled.")
        hidden_root.destroy()
        return

    hidden_root.destroy()
    
    # Create the main application window
    root = tk.Tk()
    app = OllamaChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
