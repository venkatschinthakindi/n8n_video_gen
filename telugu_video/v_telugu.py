from TTS.config.shared_configs import load_config
from TTS.tts.models import setup_model
import torch, soundfile as sf

FASTPITCH_CONFIG = "/app/models/fastpitch/config.json"
FASTPITCH_MODEL  = "/app/models/fastpitch/model_file.pth"
HIFIGAN_CONFIG   = "/app/models/hifigan/config.json"
HIFIGAN_MODEL    = "/app/models/hifigan/model_file.pth"

fp_cfg = load_config(FASTPITCH_CONFIG)
fp = setup_model(fp_cfg)
fp.load_checkpoint(fp_cfg, FASTPITCH_MODEL, eval=True)

hg_cfg = load_config(HIFIGAN_CONFIG)
hg = setup_model(hg_cfg)
hg.load_checkpoint(hg_cfg, HIFIGAN_MODEL, eval=True)

text = "ఈ రోజు ముఖ్యమైన వార్తలు ఇవే."
with torch.no_grad():
    out = fp.tts(text)
    mel = torch.tensor(out["mel_postnet_spec"])
    wav = hg.forward(mel)

sf.write("/app/output/output_telugu.wav", wav.squeeze().cpu().numpy(), 22050)
print("✅ Telugu TTS generated.")
