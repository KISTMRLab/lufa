import json
import wave
from pathlib import Path
import numpy as np
from .retrieval import CHANNELS, resample


def manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    speakers, ids = {}, set()
    for r in rows:
        if r["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if not isinstance(r["speaker"], str) or not isinstance(r["text"], str) or not r["text"].strip():
            raise ValueError("Speaker string and nonempty crop-aligned transcript required")
        if r["id"] in ids:
            raise ValueError("Duplicate clip id")
        ids.add(r["id"])
        if speakers.setdefault(r["speaker"], r["split"]) != r["split"]:
            raise ValueError("Speaker overlap across splits")
        for k in ("wav", "motion"):
            r[k] = str((path.parent / r[k]).resolve())
    return rows


def waveform(path):
    with wave.open(str(path), "rb") as f:
        if f.getframerate() != 16000 or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError("Expected16-kHz mono16-bit PCM WAV")
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if len(x) < 1600:
        raise ValueError("At least100ms speech required")
    return (x - x.mean()) / max(float(x.std()), 1e-5)


def motion(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        names = obj["names"]
        values = np.asarray([r["weights"] for r in obj["frames"]], dtype=np.float32)
        x = values[:, [names.index(n) for n in CHANNELS]]
    else:
        x = np.load(path, allow_pickle=False)
    return resample(x, 90)


def batch(rows, tokenizer, device):
    import torch
    waves = [waveform(r["wav"]) for r in rows]
    # Contracts require crop-aligned clips; longer whole recordings would make supervision wrong.
    if any(not 47000 <= len(w) <= 49000 for w in waves):
        raise ValueError("Training/bank records must be approximately3s, 47000..49000 samples")
    samples = torch.zeros(len(rows), max(map(len, waves)))
    masks = torch.zeros_like(samples, dtype=torch.long)
    for i, w in enumerate(waves):
        samples[i, :len(w)] = torch.from_numpy(w)
        masks[i, :len(w)] = 1
    text = tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=128, return_tensors="pt")
    target = torch.tensor(np.stack([motion(r["motion"]) for r in rows]))
    return samples.to(device), masks.to(device), {k: v.to(device) for k, v in text.items()}, target.to(device)
