# config.py
# Shared settings and the Azure OpenAI client.
# Put your endpoint and key in a .env file (copy .env.example to .env).
# Never paste the key directly into this file.

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load settings from a local .env file (never committed to GitHub).
load_dotenv()

# --- Azure OpenAI resource -------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")   # e.g. https://<name>.openai.azure.com/
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

# --- Deployment names (the names YOU gave the models in the lab) ------------
WHISPER_DEPLOYMENT = os.getenv("WHISPER_DEPLOYMENT", "whisper")
DALLE_DEPLOYMENT = os.getenv("DALLE_DEPLOYMENT", "gpt-image-1-mini")
GPT_DEPLOYMENT = os.getenv("GPT_DEPLOYMENT", "gpt-4o")

# --- API versions (lab defaults; change only if the lab says otherwise) ---------
WHISPER_API_VERSION = os.getenv("WHISPER_API_VERSION", "2024-06-01")
DALLE_API_VERSION = os.getenv("DALLE_API_VERSION", "2025-04-01-preview")
GPT_API_VERSION = os.getenv("GPT_API_VERSION", "2024-10-21")

# --- Project paths --------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(PROJECT_DIR, "audio")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CATEGORIES_PATH = os.path.join(PROJECT_DIR, "categories.json")


def get_client(api_version):
    """Return an AzureOpenAI client for the given API version."""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in config.py "
            "or as environment variables before running."
        )
    return AzureOpenAI(
        api_version=api_version,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg")


def find_audio_files():
    """
    Return every complaint recording under audio/ as (complaint_id, path) pairs.

    Supports audio/customer1/audio.mp3 style folders (the id is the folder name)
    and loose files like audio/complaint.mp3 (the id is the file name).
    """
    found = []
    for root, _dirs, files in os.walk(AUDIO_DIR):
        for name in sorted(files):
            if name.lower().endswith(AUDIO_EXTENSIONS):
                path = os.path.join(root, name)
                if root == AUDIO_DIR:
                    complaint_id = os.path.splitext(name)[0]
                else:
                    complaint_id = os.path.relpath(root, AUDIO_DIR).replace(os.sep, "_")
                found.append((complaint_id, path))
    if not found:
        raise FileNotFoundError(f"No audio files found in {AUDIO_DIR}. Add the complaint recordings there.")
    return sorted(found)


def find_audio_file():
    """Return the path of the first complaint recording found under audio/."""
    return find_audio_files()[0][1]
