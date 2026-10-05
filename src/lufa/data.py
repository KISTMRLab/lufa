import json
from pathlib import Path
import numpy as np
from .beat import (FACE_FPS, MAX_CLIP_SAMPLES, MAX_CLIP_SECONDS, MIN_CLIP_SAMPLES, clip_unit_range,
                   normalize_text, read_face, read_wav, span_seconds)
from .retrieval import resample


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
    """Standardized mono 16-kHz samples. PCM or IEEE-float WAV; other rates/channels are converted with a warning."""
    x = read_wav(path) / 32768
    if len(x) < 1600:
        raise ValueError("At least100ms speech required")
    return (x - x.mean()) / max(float(x.std()), 1e-5)


def face_motion(path, start=None, end=None, max_seconds=MAX_CLIP_SECONDS, allow_resample=False, fps=FACE_FPS):
    """Nine-channel [T,9] motion in [0,1] from BEAT named-channel JSON or a [T,9] .npy, optionally cropped.

    The JSON ``time`` span (or ``fps`` for .npy/untimed JSON) is checked: inputs longer than
    ``max_seconds`` are whole takes, and squeezing them into one 90-frame clip would compress time.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        times, x = read_face(path, fps)
    else:
        x = np.load(path, allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != 9:
            raise ValueError(f"{path}: expected [T,9] motion")
        times = np.arange(len(x)) / fps
    if start is not None or end is not None:
        lo = -np.inf if start is None else float(start)
        hi = np.inf if end is None else float(end)
        keep = (times >= lo) & (times < hi)
        if keep.sum() < 2:
            raise ValueError(f"{path}: crop {start}..{end} s contains fewer than two frames")
        times, x = times[keep], x[keep]
    span = span_seconds(times)
    if span > max_seconds and not allow_resample:
        raise ValueError(f"{path}: {span:.2f} s of facial motion exceeds {max_seconds} s. This looks like a whole "
                         "take; crop it (--start/--end or `lufa prepare-beat`) or pass --allow-resample to "
                         "deliberately time-compress it into one 90-frame clip")
    return clip_unit_range(x, str(path))


def motion(path):
    return resample(face_motion(path), 90)


def group_batches(groups, batch_size, rng):
    """Shuffled batches in which no two records share a transcript group (avoids in-batch false negatives).

    Records that would repeat a group are deferred to later batches, so every record is used once per epoch.
    """
    order = [int(i) for i in rng.permutation(len(groups))]
    batches = []
    while order:
        chosen, seen, rest = [], set(), []
        for i in order:
            if len(chosen) < batch_size and groups[i] not in seen:
                chosen.append(i)
                seen.add(groups[i])
            else:
                rest.append(i)
        batches.append(chosen)
        order = rest
    return batches


def transcript_group(row):
    return row.get("text_group") or normalize_text(row["text"])


def batch(rows, tokenizer, device):
    import torch
    waves = [waveform(r["wav"]) for r in rows]
    # Contracts require crop-aligned clips; longer whole recordings would make supervision wrong.
    if any(not MIN_CLIP_SAMPLES <= len(w) <= MAX_CLIP_SAMPLES for w in waves):
        raise ValueError("Training/bank records must be approximately3s, 47000..49000 samples")
    samples = torch.zeros(len(rows), max(map(len, waves)))
    masks = torch.zeros_like(samples, dtype=torch.long)
    for i, w in enumerate(waves):
        samples[i, :len(w)] = torch.from_numpy(w)
        masks[i, :len(w)] = 1
    text = tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=128, return_tensors="pt")
    target = torch.tensor(np.stack([motion(r["motion"]) for r in rows]))
    return samples.to(device), masks.to(device), {k: v.to(device) for k, v in text.items()}, target.to(device)
