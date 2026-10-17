"""A fixed, checksum-verified two-speaker model in the optional heavy service."""

import hashlib
from pathlib import Path


REPOSITORY = "speechbrain/sepformer-libri2mix"
REVISION = "eb43c5bfbb2aa654630adbf849373bcec0a20ed4"
MODEL_RATE = 8000
ASSETS = {
    "encoder.ckpt": (17272, "740d082e75b92370bc12f176d5b5554f6383cf8bc607da8ac8a550255987ed4a"),
    "decoder.ckpt": (17272, "05340634088efb00ec448f548af107d5d932057ea214dd64497f34106247e184"),
    "masknet.ckpt": (113112646, "f54762647082a8fe75a1cacc6c3d463b967223fc77c836cc4fb8ca2a272d9a75"),
}


def verify_assets(directory):
    for name, (size, digest) in ASSETS.items():
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("The fixed separation model is missing or damaged.")
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("The fixed separation model checksum did not match.")


class Separator:
    def __init__(self, directory, device="cpu"):
        verify_assets(directory)
        import torch
        from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder, SBTransformerBlock

        self.torch, self.device = torch, device
        torch.set_num_threads(2)
        encoder = Encoder(kernel_size=16, out_channels=256)
        intra = SBTransformerBlock(num_layers=8, d_model=256, nhead=8, d_ffn=1024,
                                   dropout=0, use_positional_encoding=True, norm_before=True)
        inter = SBTransformerBlock(num_layers=8, d_model=256, nhead=8, d_ffn=1024,
                                   dropout=0, use_positional_encoding=True, norm_before=True)
        masknet = Dual_Path_Model(in_channels=256, out_channels=256, num_spks=2, num_layers=2,
                                  K=250, intra_model=intra, inter_model=inter, norm="ln",
                                  linear_layer_after_inter_intra=False, skip_around_intra=True)
        decoder = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
        self.modules = {"encoder": encoder, "masknet": masknet, "decoder": decoder}
        # No remote YAML, arbitrary classes, or user-supplied checkpoints are loaded.
        for name, module in self.modules.items():
            state = torch.load(Path(directory) / (name + ".ckpt"), map_location="cpu", weights_only=True)
            module.load_state_dict(state, strict=True)
            module.to(device).eval()

    def separate(self, samples):
        import numpy as np

        torch = self.torch
        with torch.inference_mode():
            x = torch.as_tensor(samples, dtype=torch.float32, device=self.device).unsqueeze(0)
            encoded = self.modules["encoder"](x)
            masked = self.modules["masknet"](encoded) * torch.stack([encoded, encoded])
            outputs = torch.stack([self.modules["decoder"](masked[index])[0] for index in range(2)], dim=-1)
            if len(outputs) < len(samples):
                outputs = torch.nn.functional.pad(outputs, (0, 0, 0, len(samples) - len(outputs)))
            result = outputs[:len(samples)].cpu().numpy()
        if result.shape != (len(samples), 2) or not np.all(np.isfinite(result)):
            raise ValueError("The separation model returned invalid audio.")
        # Use one gain for both sources, preserving their relative loudness.
        peak = float(np.max(np.abs(result)))
        return (result / max(1, peak / 0.98)).astype(np.float32)
