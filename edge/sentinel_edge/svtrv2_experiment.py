"""Cached-only official SVTRv2-B CTC reader for isolated comparisons."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

REVISION = "1ccfc6ee6161f7133b192e16af3a9265273997ba"
WEIGHTS_SHA256 = "b65dd4675ef061a116390ba1590f538b69ab18795a55fd30a14bd98af1338f49"
CONFIG_SHA256 = "9b3918ec96a1b7f3a6686906826207fbe37d4380d130f608779e2aa29245a873"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_size(width, height, *, policy="publisher_cli"):
    """Pinned publisher ratio resize, including its CLI's truthy ceil tuple.

    The current CLI assigns ``self.ceil = cfg.get('ceil', False),``; therefore
    its default rounds down and adds one. ``dataset_round`` is the separate
    publisher dataset-transform policy, never a silently substituted default.
    """
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError("Positive native crop dimensions required")
    if policy not in {"publisher_cli", "dataset_round"}:
        raise ValueError("Unknown SVTRv2 preprocessing policy")
    ratio = int(width / height) + 1 if policy == "publisher_cli" else max(1, round(width / height))
    ratio = min(ratio, 12)
    return ((64, 64), (96, 48), (112, 40), (128, 32))[ratio - 1] if ratio <= 4 else (32 * ratio, 32)


def input_tensor(image, *, policy="publisher_cli"):
    import numpy as np
    from PIL import Image
    from torchvision import transforms as T
    from torchvision.transforms import functional as F
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Nonempty native BGR uint8 crop required")
    height, width = image.shape[:2]
    out_width, out_height = input_size(width, height, policy=policy)
    rgb = Image.fromarray(image[:, :, ::-1])
    resized = F.resize(rgb, (out_height, out_width), interpolation=T.InterpolationMode.BICUBIC)
    return T.Normalize(0.5, 0.5)(T.ToTensor()(resized)).unsqueeze(0)


class SVTRv2Experiment:
    def __init__(self, root: Path, *, device="gpu", policy="publisher_cli"):
        import torch
        import yaml
        if device not in {"cpu", "gpu"}:
            raise ValueError("Explicit CPU/GPU device required")
        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("SVTRv2 GPU unavailable; no silent CPU fallback")
        if policy not in {"publisher_cli", "dataset_round"}:
            raise ValueError("Unknown SVTRv2 preprocessing policy")
        manifest = json.loads((root / "assets.json").read_text(encoding="utf-8"))
        if manifest["source_revision"] != REVISION:
            raise ValueError("Unpinned OpenOCR source")
        if sha256(root / "best.pth") != WEIGHTS_SHA256 or sha256(root / "config.yml") != CONFIG_SHA256:
            raise ValueError("Modified SVTRv2 checkpoint or configuration")
        for item in manifest.get("source_files", []):
            filename = (root / "OpenOCR" / item["path"]).resolve()
            if not filename.is_relative_to((root / "OpenOCR").resolve()) or sha256(filename) != item["sha256"]:
                raise ValueError("Modified OpenOCR source file")
        if not manifest.get("source_files"):
            raise ValueError("OpenOCR source integrity receipt required")
        self.torch, self.device, self.policy = torch, device, policy
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config = yaml.safe_load((root / "config.yml").read_text(encoding="utf-8"))
        self.tokens = ("", *(root / "OpenOCR/tools/utils/EN_symbol_dict.txt").read_text(encoding="utf-8").splitlines())
        if len(self.tokens) != 95 or len(set(self.tokens)) != 95:
            raise ValueError("Unexpected SVTRv2 CTC dictionary")
        architecture = copy.deepcopy(config["Architecture"])
        if architecture["Encoder"]["dims"] != [128, 256, 384] or architecture["Encoder"]["depths"] != [6, 6, 6]:
            raise ValueError("Expected official SVTRv2-B architecture")
        # The published inference config uses the CTC branch. Its auxiliary
        # training/SMTR head is deliberately omitted, never randomly initialised.
        architecture["Decoder"]["infer_gtc"] = False
        architecture["Decoder"]["out_channels"] = len(self.tokens)
        sys.path.insert(0, str(root / "OpenOCR"))
        from openrec.modeling import build_model
        self.model = build_model(architecture)
        checkpoint = torch.load(root / "best.pth", map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        self.omitted_auxiliary_keys = [key for key in state if key.startswith("decoder.gtc_decoder.")]
        inference_state = {key: value for key, value in state.items() if key not in self.omitted_auxiliary_keys}
        self.model.load_state_dict(inference_state, strict=True)
        self.model.eval().to("cuda:0" if device == "gpu" else "cpu")
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def synchronize(self):
        if self.device == "gpu":
            self.torch.cuda.synchronize()

    def probabilities(self, image):
        import numpy as np
        tensor = input_tensor(image, policy=self.policy).to("cuda:0" if self.device == "gpu" else "cpu")
        with self.torch.inference_mode():
            output = self.model(tensor)
        rows = output.detach().cpu().numpy()[0]
        if rows.ndim != 2 or rows.shape[1] != len(self.tokens) or not np.isfinite(rows).all() or (rows < 0).any() or not np.allclose(rows.sum(axis=1), 1, atol=1e-4):
            raise ValueError("Unexpected SVTRv2 CTC posterior")
        return rows.tolist()
