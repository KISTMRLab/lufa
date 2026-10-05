"""BEAT preparation: word-timed TextGrid windows -> aligned ~3 s WAV/face/transcript clips and a manifest.

Expected input is the official BEAT English layout (``beat_english_v0.2.1/<speaker>/<take>.{wav,json,TextGrid}``);
any folder tree containing same-stem WAV, ARKit face JSON and TextGrid files works.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import warnings
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from .retrieval import CHANNELS

SAMPLE_RATE = 16000
FACE_FPS = 60.0
# Training/bank contract (data.batch): 47000..49000 samples, i.e. approximately three seconds.
MIN_CLIP_SAMPLES, MAX_CLIP_SAMPLES = 47000, 49000
# Face inputs longer than this are whole takes, not crops; converting them would compress time.
MAX_CLIP_SECONDS = 3.5
BEAT_BASE = "https://huggingface.co/datasets/H-Liu1997/BEAT/resolve/main/beat_english_v0.2.1/beat_english_v0.2.1"
NON_WORDS = {"", "sil", "sp", "spn", "<sil>", "<sp>", "<unk>", "{lg}", "{ns}"}


# ---------------------------------------------------------------------------------------------- parsing
def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1].replace('""', '"')
    return value


def parse_textgrid(path) -> dict[str, list[tuple[float, float, str]]]:
    """Parse a long-format Praat TextGrid into {tier name: [(xmin, xmax, text), ...]} for interval tiers."""
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    current, interval = None, None
    text = _decode(Path(path).read_bytes())
    if "IntervalTier" not in text:
        raise ValueError(f"{path}: no interval tiers (long-format TextGrid expected)")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("item [") and not s.startswith("item []"):
            current, interval = None, None
            continue
        if s.startswith("intervals ["):
            interval = {}
            continue
        if s.startswith("points ["):
            interval = None  # point tiers carry no word intervals
            continue
        match = re.match(r"(\w+)\s*=\s*(.*)$", s)
        if not match:
            continue
        key, value = match.groups()
        if key == "name" and interval is None:
            current = tiers.setdefault(_unquote(value), [])
        elif interval is not None and current is not None and key in ("xmin", "xmax"):
            interval[key] = float(value)
        elif interval is not None and current is not None and key == "text":
            current.append((interval["xmin"], interval["xmax"], _unquote(value)))
            interval = None
    if not tiers:
        raise ValueError(f"{path}: no tiers parsed")
    return tiers


def word_intervals(tiers) -> list[tuple[float, float, str]]:
    """The word tier (``words`` in BEAT; first tier otherwise) without silences/noise markers."""
    name = next((n for n in tiers if n.lower() in ("words", "word")), next(iter(tiers)))
    return [(a, b, w.strip()) for a, b, w in tiers[name] if w.strip().lower() not in NON_WORDS and b > a]


def clip_unit_range(values, source="motion"):
    """Clip facial weights to [0,1]; warn when real values (not float noise) were out of range."""
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: non-finite facial weights")
    outside = int(((values < -1e-6) | (values > 1 + 1e-6)).sum())
    if outside:
        warnings.warn(f"{source}: clipped {outside} facial weight(s) outside [0,1] "
                      f"(min {float(values.min()):.3f}, max {float(values.max()):.3f})", stacklevel=2)
    return np.clip(values, 0., 1.)


def read_face(path, fps: float = FACE_FPS):
    """Return (times [T], weights [T,9]) from BEAT named-channel JSON (uses each frame's ``time`` when present)."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    names = list(obj["names"])
    missing = [n for n in CHANNELS if n not in names]
    if missing:
        raise ValueError(f"{path}: missing facial channels {missing}")
    if "frames" in obj:
        frames = obj["frames"]
        rows = [f["weights"] for f in frames]
        if frames and all("time" in f for f in frames):
            times = np.asarray([float(f["time"]) for f in frames])
        else:
            times = np.arange(len(rows)) / float(obj.get("fps", fps))
    else:  # prepared named rows {"names", "weights": [[...]], "fps"}
        rows = obj["weights"]
        times = np.arange(len(rows)) / float(obj.get("fps", fps))
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError(f"{path}: expected at least two facial frames")
    return times, values[:, [names.index(n) for n in CHANNELS]]


def span_seconds(times) -> float:
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        return 0.
    step = float(np.median(np.diff(times)))
    return float(times[-1] - times[0] + step)


def _riff_samples(path):
    """(rate, mono float32 samples in int16 scale) for PCM 16/24/32-bit or IEEE-float WAV (BEAT ships both)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    fmt, payload, pos = None, None, 12
    while pos + 8 <= len(data):
        tag, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if tag == b"fmt ":
            fmt = body
        elif tag == b"data":
            payload = body
        pos += 8 + size + (size & 1)
    if fmt is None or payload is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    code, channels, rate = (int.from_bytes(fmt[a:b], "little") for a, b in ((0, 2), (2, 4), (4, 8)))
    bits = int.from_bytes(fmt[14:16], "little")
    if code == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: sub-format GUID starts with the real code
        code = int.from_bytes(fmt[24:26], "little")
    width = bits // 8
    usable = len(payload) - len(payload) % (width * channels)
    raw = payload[:usable]
    if code == 3 and bits in (32, 64):
        x = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float32) * 32768
    elif code == 1 and bits == 16:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    elif code == 1 and bits == 32:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 65536
    elif code == 1 and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v) / 256).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported WAV encoding (format {code}, {bits} bit)")
    return rate, x.reshape(-1, channels).mean(1)


def read_wav(path):
    """Mono float samples (int16 scale) at 16 kHz; other rates are linearly resampled with a warning."""
    rate, x = _riff_samples(path)
    if rate != SAMPLE_RATE:
        warnings.warn(f"{path}: resampling {rate} Hz to {SAMPLE_RATE} Hz by linear interpolation", stacklevel=2)
        target = int(round(len(x) * SAMPLE_RATE / rate))
        x = np.interp(np.arange(target) * rate / SAMPLE_RATE, np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.round(samples), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, SAMPLE_RATE, len(data), "NONE", ""))
        f.writeframes(data.tobytes())


# --------------------------------------------------------------------------------------------- windows
def windows(words, duration, length=3.0, hop=3.0, snap=True, min_words=1):
    """Fixed-length windows with transcripts; a word belongs to the window that contains its midpoint.

    ``snap`` starts every window at a word onset (the first onset at or after the hop cursor).
    Windows with fewer than ``min_words`` words are skipped and counted.
    """
    if length <= 0 or hop <= 0:
        raise ValueError("Window length and hop must be positive")
    words = sorted(words)
    out, skipped, cursor = [], 0, 0.
    while True:
        if snap:
            start = next((a for a, _, _ in words if a >= cursor - 1e-9), None)
            if start is None:
                break
        else:
            start = cursor
        end = start + length
        if end > duration + 1e-9:
            break
        inside = [w for a, b, w in words if start <= (a + b) / 2 < end]
        if len(inside) >= min_words:
            out.append((round(start, 4), round(end, 4), " ".join(inside)))
        else:
            skipped += 1
        cursor = start + hop
    return out, skipped


# ------------------------------------------------------------------------------------------- discovery
def take_parts(stem: str):
    """BEAT stems look like ``<speaker id>_<name>_<type>_<a>_<b>``; returns (speaker id, name, script key)."""
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[1], "_".join(parts[2:]) or stem
    return stem, stem, stem


def discover(root, speakers=None):
    """All takes under ``root`` with same-stem .wav, .json face and .TextGrid files, filtered by speaker id/name."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"BEAT root not found: {root}")
    wanted = {str(s).lower() for s in speakers} if speakers else None
    takes = []
    for grid in sorted(root.rglob("*.TextGrid")):
        wav, face = grid.with_suffix(".wav"), grid.with_suffix(".json")
        if not (wav.is_file() and face.is_file()):
            continue
        speaker, name, script = take_parts(grid.stem)
        if wanted and speaker.lower() not in wanted and name.lower() not in wanted:
            continue
        takes.append({"take": grid.stem, "speaker": speaker, "name": name, "script": script,
                      "wav": wav, "face": face, "textgrid": grid})
    return takes


# ----------------------------------------------------------------------------------- splits and dedupe
def _speaker_key(s):
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def assign_splits(speakers, validation=None, test=None, fractions=(.1, .1), seed=42):
    """Speaker-disjoint split. Explicit lists win; otherwise a seeded permutation reserves ~10 %/10 %."""
    speakers = sorted(set(speakers), key=_speaker_key)
    validation, test = set(map(str, validation or [])), set(map(str, test or []))
    if validation & test:
        raise ValueError("Validation and test speakers overlap")
    if validation or test:
        unknown = (validation | test) - set(speakers)
        if unknown:
            raise ValueError(f"Split speakers not found in prepared takes: {sorted(unknown)}")
        result = {s: "validation" if s in validation else "test" if s in test else "train" for s in speakers}
    else:
        n = len(speakers)
        n_test = max(1, round(n * fractions[1])) if n >= 2 else 0
        n_val = max(1, round(n * fractions[0])) if n >= 3 else 0
        order = [speakers[i] for i in np.random.default_rng(seed).permutation(n)]
        result = {s: "test" for s in order[:n_test]}
        result.update({s: "validation" for s in order[n_test:n_test + n_val]})
        result.update({s: "train" for s in order[n_test + n_val:]})
    if "train" not in result.values():
        raise ValueError("At least one training speaker is required")
    return result


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def text_groups(texts, threshold=0.6, max_bucket=2000):
    """Group repeated or near-repeated transcripts (e.g. BEAT's scripted readings shared across speakers).

    Exact normalized matches always group; otherwise two windows group when their word-set Jaccard
    similarity reaches ``threshold`` (candidates come from shared word trigrams). Returns group ids.
    """
    norm = [normalize_text(t) for t in texts]
    tokens = [n.split() for n in norm]
    sets = [set(t) for t in tokens]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    buckets = defaultdict(list)
    for i, words in enumerate(tokens):
        keys = {" ".join(words[k:k + 3]) for k in range(len(words) - 2)} or {" ".join(words)}
        for key in keys:
            buckets[key].append(i)
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        for x, i in enumerate(members):
            for j in members[x + 1:]:
                if find(i) == find(j):
                    continue
                inter = len(sets[i] & sets[j])
                if norm[i] == norm[j] or inter / max(len(sets[i] | sets[j]), 1) >= threshold:
                    union(i, j)
    ids = {}
    groups = []
    for i in range(len(texts)):
        root = find(i)
        ids.setdefault(root, "g" + hashlib.sha1(norm[root].encode()).hexdigest()[:10] + f"-{root}")
        groups.append(ids[root])
    return groups


# ------------------------------------------------------------------------------------------- prepare
def prepare(root, output, speakers=None, window=3.0, hop=3.0, snap=True, min_words=1,
            validation_speakers=None, test_speakers=None, duplicates="flag", dup_threshold=0.6,
            max_takes=None, seed=42, log=print):
    samples = int(round(window * SAMPLE_RATE))
    if not MIN_CLIP_SAMPLES <= samples <= MAX_CLIP_SAMPLES:
        raise ValueError(f"--window {window} s gives {samples} samples; the training contract needs "
                         f"{MIN_CLIP_SAMPLES}..{MAX_CLIP_SAMPLES} (about 3 s)")
    if duplicates not in {"flag", "drop"}:
        raise ValueError("duplicates must be 'flag' or 'drop'")
    takes = discover(root, speakers)
    if max_takes:
        counts, limited = defaultdict(int), []
        for t in takes:
            counts[t["speaker"]] += 1
            if counts[t["speaker"]] <= max_takes:
                limited.append(t)
        takes = limited
    if not takes:
        raise ValueError(f"No takes with .wav/.json/.TextGrid under {root}" + (f" for speakers {speakers}" if speakers else ""))
    output = Path(output)
    clips = output / "clips"
    split_of = assign_splits([t["speaker"] for t in takes], validation_speakers, test_speakers, seed=seed)
    rows, stats = [], {"takes": len(takes), "windows_without_words": 0, "windows_missing_face": 0, "clipped_weights": 0}
    for take in takes:
        words = word_intervals(parse_textgrid(take["textgrid"]))
        audio = read_wav(take["wav"])
        times, face = read_face(take["face"])
        if not np.isfinite(face).all():
            raise ValueError(f"{take['face']}: non-finite facial weights")
        stats["clipped_weights"] += int(((face < -1e-6) | (face > 1 + 1e-6)).sum())
        face = np.clip(face, 0., 1.)
        duration = min(len(audio) / SAMPLE_RATE, float(times[-1]) + 1e-6)
        spans, skipped = windows(words, duration, window, hop, snap, min_words)
        stats["windows_without_words"] += skipped
        for start, end, text in spans:
            keep = (times >= start) & (times < end)
            if keep.sum() < 0.9 * window * FACE_FPS:
                stats["windows_missing_face"] += 1
                continue
            s0 = int(round(start * SAMPLE_RATE))
            crop = audio[s0:s0 + samples]
            if len(crop) != samples:
                continue
            clip_id = f"beat-{take['take']}-{int(round(start * 1000)):07d}"
            wav_path = clips / take["speaker"] / f"{clip_id}.wav"
            motion_path = clips / take["speaker"] / f"{clip_id}-face.npy"
            write_wav(wav_path, crop)
            np.save(motion_path, face[keep].astype(np.float32))
            rows.append({"id": clip_id, "speaker": take["speaker"], "split": split_of[take["speaker"]], "text": text,
                         "wav": wav_path.relative_to(output).as_posix(),
                         "motion": motion_path.relative_to(output).as_posix(),
                         "take": take["take"], "script": take["script"], "start": start, "end": end})
    if not rows:
        raise ValueError("No windows with words were produced")
    groups = text_groups([r["text"] for r in rows], dup_threshold)
    sizes = defaultdict(int)
    for g in groups:
        sizes[g] += 1
    for r, g in zip(rows, groups):
        r["text_group"] = g
        r["repeated_text"] = sizes[g] > 1
    stats["repeated_text_windows"] = sum(r["repeated_text"] for r in rows)
    if duplicates == "drop":
        seen, kept = set(), []
        for r in rows:
            key = (r["split"], r["text_group"])
            if key not in seen:
                seen.add(key)
                kept.append(r)
            else:
                for k in ("wav", "motion"):
                    (output / r[k]).unlink(missing_ok=True)
        stats["dropped_repeats"] = len(rows) - len(kept)
        rows = kept
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    stats.update({"clips": len(rows), "window_seconds": window, "hop_seconds": hop, "snap_to_words": snap,
                  "duplicates": duplicates, "speakers": {s: split_of[s] for s in sorted(split_of, key=_speaker_key)},
                  "clips_per_split": {k: sum(r["split"] == k for r in rows) for k in ("train", "validation", "test")}})
    (output / "prepare-summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if stats["clipped_weights"]:
        warnings.warn(f"Clipped {stats['clipped_weights']} facial weight(s) outside [0,1]")
    log(json.dumps(stats))
    return rows, stats


# --------------------------------------------------------------------------------------------- fetch
def fetch(speaker: str, takes, output, max_bytes=25_000_000, extensions=("wav", "json", "TextGrid")):
    """Download named official BEAT takes (WAV, face JSON, TextGrid) into ``output/<speaker>/``."""
    records = []
    for take in takes:
        if not take.startswith(f"{speaker}_") or not re.fullmatch(r"[A-Za-z0-9_]+", take):
            raise ValueError(f"Invalid take {take!r} for speaker {speaker}")
        for ext in extensions:
            target = Path(output) / str(speaker) / f"{take}.{ext}"
            if target.is_file():
                records.append({"path": str(target), "bytes": target.stat().st_size, "cached": True})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{BEAT_BASE}/{speaker}/{take}.{ext}"
            request = urllib.request.Request(url, headers={"User-Agent": "lufa-beat-prepare/1.0"})
            size, partial = 0, target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise ValueError(f"{url} exceeds {max_bytes} bytes")
                while chunk := response.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        stream.close()
                        partial.unlink(missing_ok=True)
                        raise ValueError(f"{url} exceeds {max_bytes} bytes")
                    stream.write(chunk)
            partial.replace(target)
            records.append({"url": url, "path": str(target), "bytes": size})
    return records
