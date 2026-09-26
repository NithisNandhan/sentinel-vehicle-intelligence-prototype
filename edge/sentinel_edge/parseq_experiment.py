"""Opt-in research OCR adapter. Default live workers do not load it.

Pinned official PARSeq code/weights; no labels enter inference. All output
probability mass is retained, including unsupported punctuation as '?'.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

REVISION = "1902db043c029a7e03a3818c616c06600af574be"
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_?"


def map_probabilities(rows, tokens, *, slots=13):
    """Case-normalize distributions without deleting unsupported-token mass."""
    import math
    if not rows or len(rows) > slots:
        raise ValueError("Unexpected PARSeq output length")
    result = []
    ended = False
    for source in rows:
        if (not source or len(source) > len(tokens) or
                any(not math.isfinite(v) or v < 0 for v in source) or abs(sum(source) - 1) > 1e-4):
            raise ValueError("Invalid PARSeq distribution")
        row = [0.0] * len(ALPHABET)
        if ended:
            row[ALPHABET.index("_")] = 1.0
        else:
            for token, value in zip(tokens, source):
                normalized = "_" if token == "[E]" else token.upper()
                destination = normalized if len(normalized) == 1 and normalized in ALPHABET[:-2] else "?"
                if token == "[E]":
                    destination = "_"
                row[ALPHABET.index(destination)] += value
            # Tokens after the first greedy EOS have no text semantics. The EOS
            # position itself retains its real score and repeated-support gate.
            ended = tokens[max(range(len(source)), key=source.__getitem__)] == "[E]"
        result.append(row)
    if len(result) < slots and not ended:
        raise ValueError("PARSeq stopped without an end token")
    while len(result) < slots:
        row = [0.0] * len(ALPHABET)
        row[ALPHABET.index("_")] = 1.0
        result.append(row)
    return result


class PARSeqExperiment:
    def __init__(self, cache: Path, *, cached_only=False):
        import hashlib
        import torch
        from torchvision import transforms
        self.torch = torch
        torch.hub.set_dir(str(cache / "torch-hub"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cached_only:
            repository = cache / 'torch-hub' / f'baudm_parseq_{REVISION}'
            weights = cache / 'torch-hub' / 'checkpoints' / 'parseq-bb5792a6.pt'
            if not (repository / 'hubconf.py').is_file() or not weights.is_file():
                raise FileNotFoundError('Cache the pinned PARSeq source and weights before enabling hybrid mode')
            if not hashlib.sha256(weights.read_bytes()).hexdigest().startswith('bb5792a6'):
                raise ValueError('Cached PARSeq weights do not match the publisher hash')
            model = torch.hub.load(str(repository), 'parseq', source='local', pretrained=True)
        else:
            model = torch.hub.load(f"baudm/parseq:{REVISION}", "parseq", pretrained=True,
                                   trust_repo=True, skip_validation=True)
        self.model = model.eval().to(self.device)
        self.tokens = self.model.tokenizer._itos
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.model.hparams.img_size), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(0.5, 0.5),
        ])
        self.config = SimpleNamespace(alphabet=ALPHABET, pad_char="_", max_plate_slots=13)

    def probabilities(self, image):
        from PIL import Image
        tensor = self.transform(Image.fromarray(image[:, :, ::-1].copy())).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            # Fixed capacity, never derived from a target label. Up to 12 plate
            # characters plus EOS; pretrained model capacity is 25 characters.
            rows = self.model(tensor, max_length=12).softmax(-1)[0].cpu().tolist()
        return map_probabilities(rows, self.tokens)
