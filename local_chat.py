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

# --- Helper Functions ---

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


# --- UI/UX Enhancements ---

def log_error(error_message):
    """Appends a detailed error message to the log file."""
    log_entry = (
        f"--- ERROR ---\n"
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        