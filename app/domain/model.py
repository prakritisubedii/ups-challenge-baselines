import base64
import io
import sys
from pathlib import Path

import torch
import torchaudio

# Add UPS baseline and vendor paths for local imports.
ZIP_ROOT = Path(__file__).resolve().parents[2]
UPS_BASELINES = ZIP_ROOT / "ups_challenge_baselines"
VENDOR_ROOT = ZIP_ROOT / "vendor"
for path in (UPS_BASELINES, VENDOR_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from scripts.train_mel_msm_bimamba2 import BiMambaMSM


class ModelController:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Bi-Mamba2 inference. "
                "torch.cuda.is_available() returned False."
            )

        self.device = torch.device("cuda")
        ckpt_path = ZIP_ROOT / "app" / "resources" / "ckpt_step_11000_infer.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        if isinstance(cfg, dict):
            d_model = cfg.get("d_model")
        else:
            d_model = getattr(cfg, "d_model", None)
        if d_model is None:
            raise ValueError("Checkpoint cfg is missing d_model.")

        self.model = BiMambaMSM(d_model=d_model)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.sample_rate = 16000
        self.chunk_sec = 10.0
        self.chunk_samples = int(self.sample_rate * self.chunk_sec)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80,
        )

    def _prepare_waveform(self, wav_b64: str, sr: int) -> torch.Tensor:
        wav_bytes = base64.b64decode(wav_b64)
        waveform, wav_sr = torchaudio.load(io.BytesIO(wav_bytes))

        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if wav_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, wav_sr, self.sample_rate)

        if waveform.size(1) < self.chunk_samples:
            pad_amount = self.chunk_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.size(1) > self.chunk_samples:
            waveform = waveform[:, : self.chunk_samples]

        return waveform

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1).contiguous()
        return mel

    def single_evaluation(self, payload: dict):
        wav_b64 = payload["wav_b64"]
        sr = payload.get("sr", self.sample_rate)

        waveform = self._prepare_waveform(wav_b64, sr)
        mel = self._extract_features(waveform)

        x = mel.unsqueeze(0).to(self.device, dtype=torch.float32)
        reps_holder = []

        def hook(_module, _inputs, output):
            reps_holder.append(output)

        handle = self.model.backbone.register_forward_hook(hook)
        with torch.no_grad():
            _ = self.model(x)
        handle.remove()

        if not reps_holder:
            raise RuntimeError("Backbone hook did not capture representations.")

        reps = reps_holder[0]
        if reps.dim() != 3:
            raise RuntimeError("Unexpected backbone output shape.")

        t_dim = reps.shape[1]
        embedding = reps[0].detach().cpu().tolist()
        return {"embedding": embedding, "shape": [t_dim, reps.shape[2]]}

    def batch_evaluation(self, payload: dict):
        items = payload.get("items", [])
        results = [self.single_evaluation(item) for item in items]
        return {"results": results}
