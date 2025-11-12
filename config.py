import os
from dotenv import load_dotenv

load_dotenv()

# APP CONFIGURATION
APP_NAME = os.getenv("APP_NAME", "F5-TTS-INDO-LOCAL")
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")

# DEVICE CONFIGURATION
DEVICE = os.getenv("DEVICE", "cpu")

# TTS MODEL CONFIGURATION
MODEL_PATH = os.getenv(
    "MODEL_PATH", "/Users/marsudi/PycharmProjects/tts/F5-TTS-INDO-FINETUNE-V2"
)

MODEL_FILE = os.path.join(MODEL_PATH, "f5_tts_indo_v2.pt")
VOCAB_FILE = os.path.join(MODEL_PATH, "vocab.txt")
REFERENSI_AUDIO = os.getenv("REFERENSI_AUDIO", None)
