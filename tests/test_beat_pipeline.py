"""BEAT preparation, convert-motion guard, Wav2Vec2 masking, group-aware batches and learned demo launch."""
import importlib.util
import json
import struct
import tempfile
import threading
import unittest
import urllib.request
import warnings
import wave
from argparse import Namespace
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

from lufa.beat import (assign_splits, parse_textgrid, prepare, read_face, read_wav, text_groups, windows,
                       word_intervals)
from lufa.data import face_motion, group_batches, manifest, motion, transcript_group
from lufa.retrieval import CHANNELS, resample

REPO = Path(__file__).resolve().parents[1]
SCRIPT = ("the first thing i like to do on weekends is").split()
TAIL = "relaxing and going shopping with friends".split()
EXTRA = ["cheekPuff", "jawOpen"]


def textgrid(words, xmax):
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0.0", f"xmax = {xmax}",
             "tiers? <exists>", "size = 2", "item []:"]
    for index, (name, intervals) in enumerate((("words", words), ("phones", [(0.0, xmax, "AH0")])), 1):
        lines += [f"\titem [{index}]:", '\t\tclass = "IntervalTier"', f'\t\tname = "{name}"', "\t\txmin = 0.0",
                  f"\t\txmax = {xmax}", f"\t\tintervals: size = {len(intervals)}"]
        for i, (a, b, w) in enumerate(intervals, 1):
            lines += [f"\t\t\tintervals [{i}]:", f"\t\t\t\txmin = {a}", f"\t\t\t\txmax = {b}", f'\t\t\t\ttext = "{w}"']
    return "\n".join(lines) + "\n"


def write_float_wav(path, samples, rate=16000):
    data = (np.asarray(samples, dtype=np.float32) / 32768).astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def make_take(folder, stem, seed, float_wav=False, out_of_range=False, duration=12.0):
    """12 s take: words 1.0-4.0 s, silence 4-9.2 s, words 9.2-11.6 s; 60-fps face JSON with frame times."""
    folder.mkdir(parents=True, exist_ok=True)
    words = [(0.0, 1.0, "")]
    words += [(round(1.0 + .3 * i, 2), round(1.3 + .3 * i, 2), w) for i, w in enumerate(SCRIPT)]
    words += [(4.0, 9.2, "")]
    words += [(round(9.2 + .4 * i, 2), round(9.6 + .4 * i, 2), w) for i, w in enumerate(TAIL)]
    (folder / f"{stem}.TextGrid").write_text(textgrid(words, duration), encoding="utf-8")
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, 2000, int(duration * 16000)).clip(-32000, 32000)
    if float_wav:
        write_float_wav(folder / f"{stem}.wav", samples)
    else:
        with wave.open(str(folder / f"{stem}.wav"), "wb") as f:
            f.setparams((1, 2, 16000, len(samples), "NONE", ""))
            f.writeframes(samples.astype("<i2").tobytes())
    names = EXTRA[:1] + CHANNELS[::-1] + EXTRA[1:]
    frames = []
    for i in range(int(duration * 60)):
        t = (i + 1) / 60
        values = {n: .5 + .4 * np.sin(t + k + seed) for k, n in enumerate(names)}
        if out_of_range and i == 100:
            values["browInnerUp"] = 1.2
        frames.append({"time": t, "weights": [float(values[n]) for n in names], "rotation": [0, 0, 0]})
    (folder / f"{stem}.json").write_text(json.dumps({"names": names, "frames": frames}), encoding="utf-8")


def make_beat(root):
    make_take(root / "1", "1_wayne_0_1_1", 1)
    make_take(root / "2", "2_scott_0_1_1", 2, out_of_range=True)
    make_take(root / "3", "3_solomon_0_1_1", 3, float_wav=True)
    (root / "3" / "3_solomon_0_9_9.wav").write_bytes(b"")  # incomplete take (no json/TextGrid) is ignored


def local_sample_dir():
    for candidate in (REPO / "outputs" / "beat-raw" / "1", REPO.parents[1] / "tmp" / "beat-demo" / "source"):
        if (candidate / "1_wayne_0_1_1.TextGrid").is_file() and (candidate / "1_wayne_0_1_1.json").is_file():
            return candidate
    return None


class TextGridAndWindowTests(unittest.TestCase):
    def test_parse_textgrid_word_tier_drops_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_take(Path(tmp), "1_wayne_0_1_1", 1)
            tiers = parse_textgrid(Path(tmp) / "1_wayne_0_1_1.TextGrid")
            self.assertEqual(set(tiers), {"words", "phones"})
            words = word_intervals(tiers)
            self.assertEqual([w for _, _, w in words], SCRIPT + TAIL)
            self.assertEqual(words[0][:2], (1.0, 1.3))

    def test_windows_snap_to_word_onsets_and_skip_wordless(self):
        words = word_intervals({"words": [(1.0, 1.5, "hello"), (1.5, 2.0, "there"), (8.0, 8.5, "again")]})
        snapped, skipped = windows(words, 12.0, 3.0, 3.0, snap=True)
        self.assertEqual([s for s, _, _ in snapped], [1.0, 8.0])
        self.assertEqual(snapped[0][2], "hello there")
        self.assertEqual(skipped, 0)
        fixed, skipped = windows(words, 12.0, 3.0, 3.0, snap=False)
        self.assertEqual([(s, t) for s, _, t in fixed], [(0.0, "hello there"), (6.0, "again")])
        self.assertEqual(skipped, 2)  # 3-6 s and 9-12 s contain no word midpoints

    def test_speaker_disjoint_splits(self):
        auto = assign_splits(["1", "2", "3", "10"])
        self.assertEqual(sorted(auto.values()).count("test"), 1)
        self.assertEqual(sorted(auto.values()).count("validation"), 1)
        self.assertEqual(assign_splits(["1"]), {"1": "train"})
        explicit = assign_splits(["1", "2", "3"], validation=["2"], test=["3"])
        self.assertEqual(explicit, {"1": "train", "2": "validation", "3": "test"})
        with self.assertRaises(ValueError):
            assign_splits(["1", "2"], validation=["1"], test=["2"])  # no training speaker left
        with self.assertRaises(ValueError):
            assign_splits(["1", "2"], test=["9"])

    def test_repeated_and_near_repeated_transcripts_share_a_group(self):
        groups = text_groups(["The first thing I like to do on weekends", "the first thing i like to do on weekends",
                              "first thing i'd like to do on weekends", "a completely different sentence here"])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])


class PrepareBeatTests(unittest.TestCase):
    def test_prepare_synthetic_official_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "beat_english_v0.2.1", Path(tmp) / "data"
            make_beat(root)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rows, stats = prepare(root, out, snap=False, hop=3.0, log=lambda *_: None)
            self.assertTrue(any("outside [0,1]" in str(w.message) for w in caught))
            self.assertEqual(stats["takes"], 3)
            self.assertEqual(stats["windows_without_words"], 3)  # the 6-9 s silence of every take
            self.assertGreater(stats["clipped_weights"], 0)
            loaded = manifest(out / "manifest.jsonl")  # production contract: splits, ids, speaker disjointness
            self.assertEqual(len(loaded), 9)
            self.assertEqual({r["split"] for r in loaded}, {"train", "validation", "test"})
            by_speaker = {}
            for r in loaded:
                by_speaker.setdefault(r["speaker"], set()).add(r["split"])
            self.assertTrue(all(len(s) == 1 for s in by_speaker.values()))
            for r in loaded:
                with wave.open(r["wav"]) as f:
                    self.assertEqual((f.getframerate(), f.getnchannels(), f.getnframes()), (16000, 1, 48000))
                face = np.load(r["motion"])
                self.assertIn(face.shape, ((179, 9), (180, 9)))  # BEAT frame times start at 1/60 s
                self.assertTrue((face >= 0).all() and (face <= 1).all())
                self.assertEqual(motion(r["motion"]).shape, (90, 9))
            first = next(r for r in loaded if r["speaker"] == "1" and r["start"] == 0.0)
            self.assertEqual(first["text"], "the first thing i like to do")  # word midpoints inside 0-3 s
            # Same scripted sentence read by every speaker is flagged as one transcript group.
            same = [r for r in loaded if r["start"] == 0.0]
            self.assertEqual(len({r["text_group"] for r in same}), 1)
            self.assertTrue(all(r["repeated_text"] for r in same))
            # Audio crop alignment, including the IEEE-float source WAV of speaker 3.
            source = read_wav(root / "3" / "3_solomon_0_1_1.wav")
            crop = next(r for r in loaded if r["speaker"] == "3" and r["start"] == 9.0)
            with wave.open(crop["wav"]) as f:
                written = np.frombuffer(f.readframes(48000), "<i2").astype(np.float32)
            np.testing.assert_allclose(written, np.round(source[144000:192000]), atol=1)
            # Face crop alignment: first cropped frame is the first JSON frame at/after the window start.
            times, values = read_face(root / "1" / "1_wayne_0_1_1.json")
            start = next(r for r in loaded if r["speaker"] == "1" and r["start"] == 3.0)
            np.testing.assert_allclose(np.load(start["motion"])[0], values[np.argmax(times >= 3.0)], atol=1e-6)

    def test_drop_mode_keeps_one_record_per_group_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "beat", Path(tmp) / "data"
            make_beat(root)
            rows, stats = prepare(root, out, speakers=["wayne", "2"], test_speakers=["2"], snap=False,
                                  duplicates="drop", log=lambda *_: None)
            self.assertEqual({r["speaker"] for r in rows}, {"1", "2"})
            keys = [(r["split"], r["text_group"]) for r in rows]
            self.assertEqual(len(keys), len(set(keys)))
            for r in rows:
                self.assertTrue((out / r["wav"]).is_file())

    def test_window_outside_training_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_beat(Path(tmp) / "beat")
            with self.assertRaises(ValueError):
                prepare(Path(tmp) / "beat", Path(tmp) / "data", window=2.0, log=lambda *_: None)

    @unittest.skipUnless(local_sample_dir(), "local BEAT sample (1_wayne_0_1_1) not present")
    def test_local_real_beat_sample(self):
        source = local_sample_dir()
        tiers = parse_textgrid(source / "1_wayne_0_1_1.TextGrid")
        self.assertEqual(set(tiers), {"words", "phones"})
        self.assertEqual(word_intervals(tiers)[0], (1.35, 1.46, "the"))
        with tempfile.TemporaryDirectory() as tmp:
            rows, stats = prepare(source, Path(tmp), speakers=["1"], log=lambda *_: None)
            loaded = manifest(Path(tmp) / "manifest.jsonl")
            self.assertGreater(len(loaded), 15)
            self.assertEqual(loaded[0]["start"], 1.35)
            self.assertTrue(loaded[0]["text"].startswith("the first thing i like to do on weekends"))
            self.assertEqual(stats["clipped_weights"], 0)  # real BEAT weights stay inside [0,1]
            times, values = read_face(source / "1_wayne_0_1_1.json")
            face = np.load(loaded[0]["motion"])
            self.assertIn(len(face), (179, 180, 181))
            np.testing.assert_allclose(face[0], values[np.argmax(times >= 1.35)], atol=1e-6)


class ConvertMotionGuardTests(unittest.TestCase):
    def test_whole_take_is_refused_unless_resampling_is_explicit(self):
        from lufa.cli import convert
        with tempfile.TemporaryDirectory() as tmp:
            make_take(Path(tmp), "1_wayne_0_1_1", 1)
            face = Path(tmp) / "1_wayne_0_1_1.json"
            with self.assertRaisesRegex(ValueError, "exceeds 3.5 s"):
                convert(Namespace(input=str(face), output=str(Path(tmp) / "x.npy"), start=None, end=None,
                                  max_seconds=3.5, allow_resample=False))
            with self.assertRaises(ValueError):
                motion(face)  # manifests cannot smuggle whole takes in either
            convert(Namespace(input=str(face), output=str(Path(tmp) / "crop.npy"), start=1.0, end=4.0,
                              max_seconds=3.5, allow_resample=False))
            crop = np.load(Path(tmp) / "crop.npy")
            self.assertEqual(crop.shape, (90, 9))
            times, values = read_face(face)
            np.testing.assert_allclose(crop[0], values[np.argmax(times >= 1.0)], atol=1e-6)
            convert(Namespace(input=str(face), output=str(Path(tmp) / "whole.npy"), start=None, end=None,
                              max_seconds=3.5, allow_resample=True))
            self.assertEqual(np.load(Path(tmp) / "whole.npy").shape, (90, 9))

    def test_out_of_range_values_are_clipped_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_take(Path(tmp), "2_scott_0_1_1", 2, out_of_range=True)
            with self.assertWarnsRegex(UserWarning, "outside"):
                values = face_motion(Path(tmp) / "2_scott_0_1_1.json", 1.0, 3.0)
            self.assertLessEqual(float(values.max()), 1.0)
        with self.assertWarns(UserWarning):
            result = resample(np.full((4, 9), 1.3), 6)
        self.assertTrue(np.allclose(result, 1.0))
        with self.assertRaises(ValueError):
            resample(np.full((4, 9), np.nan), 6)


class TrainingTests(unittest.TestCase):
    def test_group_batches_never_repeat_a_transcript_group(self):
        groups = ["a", "a", "a", "b", "b", "c", "d", "e"]
        batches = group_batches(groups, 4, np.random.default_rng(0))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(groups))))
        for b in batches:
            self.assertEqual(len({groups[i] for i in b}), len(b))
            self.assertLessEqual(len(b), 4)
        self.assertEqual(transcript_group({"text": "Hello, World"}), transcript_group({"text": "hello world"}))

    def test_wav2vec2_attention_mask_policy(self):
        import torch
        from unittest.mock import patch
        from transformers import BertConfig, BertModel, Wav2Vec2Config, Wav2Vec2Model
        from lufa.model import RetrievalModel
        torch.manual_seed(0)
        text = BertModel(BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                    intermediate_size=32))
        common = dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                      conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=4,
                      num_conv_pos_embedding_groups=2, mask_time_prob=0.)
        samples = torch.randn(2, 4000)
        mask = torch.ones(2, 4000, dtype=torch.long)
        mask[1, 3000:] = 0
        dirty = samples.clone()
        dirty[1, 3000:] = 5.0  # garbage padding must be zeroed before the encoder sees it
        clean = samples * mask
        for norm, expect_mask in (("group", False), ("layer", True)):
            audio = Wav2Vec2Model(Wav2Vec2Config(feat_extract_norm=norm, do_stable_layer_norm=norm == "layer", **common))
            model = RetrievalModel(audio, text, 8).eval()
            with patch.object(model.audio, "forward", wraps=model.audio.forward) as forward:
                with torch.no_grad():
                    a = model.encode_audio(dirty, mask)
                    b = model.encode_audio(clean, mask)
            kwargs = forward.call_args.kwargs
            self.assertEqual(kwargs["attention_mask"] is not None, expect_mask, norm)
            torch.testing.assert_close(forward.call_args.args[0], clean)
            torch.testing.assert_close(a, b)

    def test_from_scratch_corpus_vocab_training_bank_and_learned_demo(self):
        from lufa.cli import build_bank, read_bank, train
        from lufa.server import create_handler
        with tempfile.TemporaryDirectory() as tmp:
            root, data = Path(tmp) / "beat", Path(tmp) / "data"
            make_beat(root)
            prepare(root, data, snap=False, validation_speakers=["2"], test_speakers=["3"], log=lambda *_: None)
            model_dir, bank_path = Path(tmp) / "model", Path(tmp) / "bank.npz"
            train(Namespace(manifest=str(data / "manifest.jsonl"), audio_model=None, text_model=None, from_scratch=True,
                            corpus_vocab=True, sampler="group", epochs=1, batch_size=4, lr=1e-4,
                            output=str(model_dir), seed=1, device="cpu"))
            self.assertTrue((model_dir / "tokenizer" / "vocab.txt").is_file())
            build_bank(Namespace(model=str(model_dir), manifest=str(data / "manifest.jsonl"), output=str(bank_path),
                                 device="cpu"))
            bank = read_bank(bank_path)
            self.assertEqual(set(bank["speakers"].tolist()), {"1"})
            self.assertIn("weekends", " ".join(bank["texts"].tolist()))
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(model_dir, bank_path, "cpu", "shopping with friends"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/demo") as response:
                    demo = json.load(response)
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/authored") as response:
                    authored = json.load(response)
            finally:
                server.shutdown(); server.server_close(); thread.join()
            self.assertEqual(demo["origin"], "trained-bank-retrieval")
            self.assertEqual(demo["query"], "shopping with friends")
            self.assertTrue(demo["matches"][0]["id"].startswith("beat-1_wayne"))
            self.assertIn("text", demo["matches"][0])
            self.assertEqual(authored["origin"], "authored-controller-example")


class StartDemoTests(unittest.TestCase):
    def load_script(self):
        spec = importlib.util.spec_from_file_location("lufa_start_demo", REPO / "scripts" / "start_demo.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_learned_retrieval_only_when_model_and_bank_exist(self):
        script = self.load_script()
        with tempfile.TemporaryDirectory() as tmp:
            model, bank = Path(tmp) / "model", Path(tmp) / "bank.npz"
            command, learned = script.serve_command(model, bank, 8080)
            self.assertFalse(learned)
            self.assertNotIn("--model", command)
            model.mkdir()
            (model / "model.pt").write_bytes(b"x")
            bank.write_bytes(b"x")
            command, learned = script.serve_command(model, bank, 8080)
            self.assertTrue(learned)
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertEqual(command[command.index("--bank") + 1], str(bank))
            self.assertFalse(script.serve_command(model, bank, 8080, authored=True)[1])
        defaults = script.parse_args([])
        self.assertEqual((defaults.model, defaults.bank), ("outputs/lufa/model", "outputs/lufa/bank.npz"))

    def test_viewer_without_model_serves_labelled_authored_example(self):
        from lufa.server import create_handler
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/demo") as response:
                demo = json.load(response)
        finally:
            server.shutdown(); server.server_close(); thread.join()
        self.assertEqual(demo["origin"], "authored-controller-example")


if __name__ == "__main__":
    unittest.main()
