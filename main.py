import torch
import soundfile as sf
import os
import sys
import subprocess

# ---------------------------------
# 🧠 KONFIGURASI
# ---------------------------------
MODEL_PATH = "/Users/marsudi/PycharmProjects/tts/F5-TTS-INDO-FINETUNE-V2"
MODEL_FILE = os.path.join(MODEL_PATH, "f5_tts_indo_v2.pt")
VOCAB_FILE = os.path.join(MODEL_PATH, "vocab.txt")
OUTPUT_FILE = "output_f5_tts_indo_local.wav"
TEXT = "Halo, ini adalah suara Bahasa Indonesia dari model F5 TTS lokal di Mac M1."

# Device
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 NVIDIA GPU terdeteksi: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Menggunakan Apple Metal (MPS)")
else:
    device = "cpu"
    print("🚀 Tidak ada GPU, fallback ke CPU")

print(f"🚀 Device: {device}")

# ---------------------------------
# 🔄 PREPARE REFERENCE AUDIO (WAV)
# ---------------------------------
print("\n📁 Preparing reference audio...")

# Cek apakah ada WAV file yang sudah ada


REF_AUDIO = (
    "/Users/marsudi/PycharmProjects/tts/F5-TTS-INDO-FINETUNE-V2/ref_reporter.mp3"
)


# ---------------------------------
# ✅ CEK FILES
# ---------------------------------
print(f"\n📋 Configuration:")
print(f"  Model: {os.path.basename(MODEL_FILE)}")
print(f"  Vocab: {os.path.basename(VOCAB_FILE)}")
print(f"  Reference: {os.path.basename(REF_AUDIO)}")

for f in [MODEL_FILE, VOCAB_FILE, REF_AUDIO]:
    if not os.path.exists(f):
        print(f"❌ File not found: {f}")
        sys.exit(1)

# ---------------------------------
# 🔍 IMPORT & LOAD MODEL
# ---------------------------------
print("\n🌀 Loading F5TTS model...")

try:
    from f5_tts.api import F5TTS

    tts = F5TTS(ckpt_file=MODEL_FILE, vocab_file=VOCAB_FILE, device=device)
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ---------------------------------
# 🎙️ GENERATE AUDIO
# ---------------------------------
print("\n🔊 Generating speech...")
print(f"📝 Text: '{TEXT}'")

try:
    # Reference text (sesuaikan dengan isi audio reference Anda)
    ref_text = "Selamat pagi saudara-saudara sebangsa dan setanah air"

    # Generate audio
    print("⏳ Processing...")
    wav, sr, spect = tts.infer(
        ref_file=REF_AUDIO,
        ref_text=ref_text,
        gen_text=TEXT,
        target_rms=0.1,
        cross_fade_duration=0.15,
        nfe_step=32,  # number of function evaluations
        cfg_strength=2.0,  # classifier-free guidance
        sway_sampling_coef=-1.0,
        speed=1.0,
    )

    # Convert to numpy if tensor
    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()

    # Save output
    sf.write(OUTPUT_FILE, wav, sr)

    # Success message
    print(f"\n🎉 SUCCESS!")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"📊 Sample rate: {sr} Hz")
    print(f"📏 Duration: {len(wav)/sr:.2f} seconds")
    print(f"🔊 Shape: {wav.shape}")

    # Play audio (optional)
    print(f"\n💡 To play: afplay {OUTPUT_FILE}")

except Exception as e:
    print(f"\n❌ Error during inference:")
    print(f"{type(e).__name__}: {e}")

    # Debug info
    print("\n🔍 Debug information:")
    try:
        import inspect

        print(f"tts.infer signature: {inspect.signature(tts.infer)}")
    except:
        pass

    import traceback

    traceback.print_exc()
    sys.exit(1)
