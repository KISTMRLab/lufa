"""Local LUFA nine-channel viewer and optional trained-bank query API."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import math
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .avatar_http import serve_avatar_asset

import numpy as np

from .retrieval import CHANNELS, nearest, resample


def validate_motion(values, names=CHANNELS, fps=30):
    if list(names) != CHANNELS:
        raise ValueError("Motion channels must match the documented nine-channel order")
    if int(fps) <= 0 or int(fps) > 120:
        raise ValueError("FPS must be 1–120")
    motion = np.asarray(values, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 9 or not 2 <= len(motion) <= 10000:
        raise ValueError("Motion must be [T,9] with 2–10000 frames")
    if not np.isfinite(motion).all() or (motion < 0).any() or (motion > 1).any():
        raise ValueError("Motion channels must be finite in [0,1]")
    return {"motion": motion.tolist(), "names": CHANNELS, "fps": int(fps)}


def parse_motion(body: bytes, filename: str):
    if filename.lower().endswith(".npz"):
        with np.load(io.BytesIO(body), allow_pickle=False) as obj:
            values = obj["motion"]
            names = obj["names"].tolist() if "names" in obj else CHANNELS
            fps = int(obj["fps"]) if "fps" in obj else 30
    elif filename.lower().endswith(".json"):
        obj = json.loads(body)
        values = obj.get("motion", obj.get("weights", obj.get("frames")))
        names = obj.get("names", CHANNELS)
        fps = obj.get("fps", 30)
        if values and isinstance(values[0], dict):
            values = [frame["weights"] for frame in values]
        if list(names) != CHANNELS:
            values = np.asarray(values, dtype=np.float32)[:, [list(names).index(n) for n in CHANNELS]]
            names = CHANNELS
    else:
        raise ValueError("Upload a .json or .npz motion file")
    return validate_motion(values, names, fps)


def authored_controller():
    """A visibly labeled channel-controller fixture, not recorded motion."""
    frames = []
    for i in range(90):
        phase = 2 * math.pi * i / 90
        brow = .12 + .55 * (1 + math.sin(phase)) / 2
        squint = .1 + .35 * (1 + math.sin(phase + 1)) / 2
        frames.append([brow, brow * .8, brow * .8, .05, .05, squint, squint, .08, .08])
    return {**validate_motion(frames), "origin": "authored-controller-example", "note": "Synthetic channel controls; no recorded clip or learned retrieval."}


DEFAULT_DEMO_TEXT = "I really like to talk about the things that make me happy"


def model_provenance(model_dir: Path | None):
    """Optional note written next to a checkpoint (e.g. by start_demo --train-small)."""
    path = Path(model_dir) / "demo-provenance.json" if model_dir else None
    if path and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def create_handler(model_dir: Path | None = None, bank_path: Path | None = None, device="cpu",
                   demo_text: str | None = DEFAULT_DEMO_TEXT):
    static = Path(__file__).parents[2] / "static"
    model_state = {}
    demo_text = demo_text or DEFAULT_DEMO_TEXT

    def query_bank(text: str, wav_bytes: bytes | None, top_k: int, frames: int | None):
        if not model_dir or not bank_path:
            raise ValueError("Start server with --model and --bank for trained retrieval")
        if "model" not in model_state:
            from .cli import load_model, read_bank
            model_state["model"], model_state["tokenizer"], _ = load_model(model_dir, device)
            model_state["bank"] = read_bank(bank_path)
        from .cli import query
        bank = model_state["bank"]
        wav_path = None
        try:
            if wav_bytes:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                    temp.write(wav_bytes); wav_path = temp.name
            embedding = query(model_state["model"], model_state["tokenizer"], text, wav_path, device)
            indices, scores = nearest(embedding, bank["embeddings"], min(top_k, len(bank["ids"])))
            selected = bank["motions"][indices[0]]
            if frames: selected = resample(selected, frames)
            texts = bank.get("texts")
            matches = [{"id": str(bank["ids"][i]), "speaker": str(bank["speakers"][i]), "cosine": float(s),
                        **({"text": str(texts[i])} if texts is not None else {})} for i, s in zip(indices, scores)]
            return {**validate_motion(selected, CHANNELS, 30), "origin": "trained-bank-retrieval", "matches": matches}
        finally:
            if wav_path: Path(wav_path).unlink(missing_ok=True)

    def first_clip():
        """Learned retrieval for a default text query when a model and bank are configured; authored otherwise."""
        if not (model_dir and bank_path):
            return authored_controller()
        try:
            return {**query_bank(demo_text, None, 3, None), "query": demo_text}
        except Exception as exc:  # keep the viewer usable; the fallback is labelled
            return {**authored_controller(), "note": f"Learned retrieval failed ({exc}); showing the authored controller example."}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if serve_avatar_asset(self, Path(__file__).resolve().parents[2] / "static"): return
            if self.path == "/api/demo": self._json(first_clip()); return
            if self.path == "/api/authored": self._json(authored_controller()); return
            if self.path == "/api/status":
                self._json({"retrieval_configured": bool(model_dir and bank_path), "channels": CHANNELS,
                            "models_bundled": False, "model_note": model_provenance(model_dir)}); return
            assets = {"/": (static / "index.html", "text/html; charset=utf-8"),
                      "/static/avatar.js": (static / "avatar.js", "text/javascript; charset=utf-8"),
                      "/static/vendor/three.module.js": (static / "vendor/three.module.js", "text/javascript; charset=utf-8"),
                      "/static/vendor/three.core.js": (static / "vendor/three.core.js", "text/javascript; charset=utf-8")}
            if self.path not in assets or not assets[self.path][0].is_file(): self.send_error(404); return
            data = assets[self.path][0].read_bytes(); self.send_response(200)
            self.send_header("Content-Type", assets[self.path][1]); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000_000: raise ValueError("Request must be 1–20 MB")
                body = self.rfile.read(length)
                if self.path == "/api/upload":
                    filename = self.headers.get("X-Filename", "")
                    result = parse_motion(body, filename)
                    result["origin"] = "user-uploaded-recorded-motion"
                    self._json(result)
                elif self.path == "/api/query":
                    payload = json.loads(body)
                    mode = payload.get("mode")
                    text = str(payload.get("text") or "") if mode in {"text", "fused"} else ""
                    encoded = payload.get("wav_base64") if mode in {"audio", "fused"} else None
                    if mode not in {"audio", "text", "fused"} or (mode in {"text", "fused"} and not text.strip()) or (mode in {"audio", "fused"} and not encoded):
                        raise ValueError("Choose audio, text, or fused input and supply its required fields")
                    wav = base64.b64decode(encoded, validate=True) if encoded else None
                    top_k = int(payload.get("top_k", 3))
                    if not 1 <= top_k <= 20: raise ValueError("top_k must be 1–20")
                    frames = payload.get("frames")
                    if frames is not None:
                        frames = int(frames)
                        if not 2 <= frames <= 10000: raise ValueError("frames must be 2–10000")
                    self._json(query_bank(text, wav, top_k, frames))
                else: self.send_error(404)
            except (ValueError, KeyError, TypeError, IndexError, binascii.Error, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:
                self._json({"error": f"query failed: {exc}"}, 502)

        def _json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local LUFA upper-face viewer")
    parser.add_argument("--model", type=Path, help="user-trained checkpoint directory")
    parser.add_argument("--bank", type=Path, help="user-built recorded-motion bank")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--demo-text", default=DEFAULT_DEMO_TEXT)
    args = parser.parse_args()
    if bool(args.model) != bool(args.bank): parser.error("provide --model and --bank together")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(args.model, args.bank, args.device, args.demo_text))
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__": main()
