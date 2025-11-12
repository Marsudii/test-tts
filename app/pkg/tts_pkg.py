import torch
import soundfile as sf
import os
import traceback
import io


class TtsLibrary:
    def __init__(self, config):
        """
        config harus memiliki atribut:
        - MODEL_PATH
        - MODEL_FILE
        - VOCAB_FILE
        - DEVICE
        - REF_AUDIO
        - OUTPUT_FILE
        """
        self.MODEL_PATH = config.MODEL_PATH
        self.MODEL_FILE = config.MODEL_FILE
        self.VOCAB_FILE = config.VOCAB_FILE
        self.DEVICE = config.DEVICE
        self.REF_AUDIO = config.REFERENSI_AUDIO

        # Validasi file penting
        for f in [self.MODEL_FILE, self.VOCAB_FILE]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing required file: {f}")

    # -------------------------------
    # 🔧 Setup Device
    # -------------------------------
    def _set_device(self):
        if self.DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
        elif self.DEVICE == "mps" and torch.backends.mps.is_available():
            print("🚀 Using Apple Metal (MPS)")
        else:
            self.DEVICE = "cpu"
            torch.set_num_threads(1)
            print("🚀 Using CPU")

    # -------------------------------
    # 🎙️ Load Model
    # -------------------------------
    def _load_model(self):
        from f5_tts.api import F5TTS

        print("🌀 Loading F5-TTS model...")
        tts = F5TTS(
            ckpt_file=self.MODEL_FILE,
            vocab_file=self.VOCAB_FILE,
            device=self.DEVICE,
        )
        print("✅ Model loaded successfully!")
        return tts

    # -------------------------------
    # 🔊 Generate Audio
    # -------------------------------
    def generate_audio(
        self,
        text: str,
        ref_text: str = "halo",
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
    ):
        """Generate audio from text."""
        try:
            self._set_device()
            tts = self._load_model()

            if not self.REF_AUDIO or not os.path.exists(self.REF_AUDIO):
                raise FileNotFoundError("Reference audio (REF_AUDIO) not found.")

            print(f"🔊 Generating speech for: '{text}'")
            wav, sr, spect = tts.infer(
                ref_file=self.REF_AUDIO,
                ref_text=ref_text,
                gen_text=text,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
            )

            # Convert tensor to numpy
            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()

            # Simpan ke file
            sf.write(self.OUTPUT_FILE, wav, sr)
            print(f"✅ Saved: {self.OUTPUT_FILE} ({len(wav)/sr:.2f}s @ {sr}Hz)")

            # Return BytesIO buffer
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV")
            buf.seek(0)
            return buf, sr

        except Exception as e:
            print("\n❌ Error during TTS inference:")
            print(f"{type(e).__name__}: {e}")
            traceback.print_exc()
            # ✅ RAISE the exception instead of returning None
            raise
