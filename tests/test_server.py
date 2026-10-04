import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lufa.retrieval import CHANNELS
from lufa.server import authored_controller, create_handler, parse_motion, validate_motion


class ViewerTests(unittest.TestCase):
    def test_all_nine_channels_and_order_are_enforced(self):
        demo = authored_controller()
        self.assertEqual(demo["names"], CHANNELS)
        self.assertEqual(np.asarray(demo["motion"]).shape, (90, 9))
        with self.assertRaises(ValueError):
            validate_motion(np.zeros((90, 8)))
        with self.assertRaises(ValueError):
            validate_motion(np.ones((90, 9)) * 1.2)
        with self.assertRaises(ValueError):
            validate_motion(np.zeros((90, 9)), CHANNELS[::-1])

    def test_recorded_npz_and_named_json_import(self):
        motion = np.linspace(0, 1, 90 * 9).reshape(90, 9)
        stream = io.BytesIO()
        np.savez(stream, motion=motion, names=np.asarray(CHANNELS), fps=30)
        loaded = parse_motion(stream.getvalue(), "clip.npz")
        np.testing.assert_allclose(loaded["motion"], motion, atol=1e-6)
        reverse = CHANNELS[::-1]
        data = {"names": reverse, "frames": [{"weights": row[::-1]} for row in motion.tolist()], "fps": 30}
        loaded_json = parse_motion(json.dumps(data).encode(), "clip.json")
        np.testing.assert_allclose(loaded_json["motion"], motion, atol=1e-6)

    def test_api_query_uses_trained_bank_path_for_text_audio_and_fused(self):
        bank = {"embeddings": np.asarray([[1., 0.], [0., 1.]]),
                "motions": np.stack([np.zeros((90, 9)), np.ones((90, 9))]),
                "ids": np.asarray(["record-a", "record-b"]), "speakers": np.asarray(["speaker-a", "speaker-b"])}
        with patch("lufa.cli.load_model", return_value=(object(), object(), {})), \
             patch("lufa.cli.read_bank", return_value=bank), \
             patch("lufa.cli.query", return_value=np.asarray([1., 0.])) as query:
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(Path("local-model"), Path("local-bank")))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                for mode in ("text", "audio", "fused"):
                    payload = {"mode": mode, "text": "hello" if mode != "audio" else "",
                               "wav_base64": base64.b64encode(b"dummy wav").decode() if mode != "text" else None,
                               "top_k": 2}
                    request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/query",
                                                     json.dumps(payload).encode(), {"Content-Type": "application/json"})
                    with urllib.request.urlopen(request) as response: result = json.load(response)
                    self.assertEqual(result["origin"], "trained-bank-retrieval")
                    self.assertEqual(result["matches"][0]["id"], "record-a")
                    self.assertEqual(result["names"], CHANNELS)
                self.assertEqual(query.call_count, 3)
            finally:
                server.shutdown(); server.server_close(); thread.join()
