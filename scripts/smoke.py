"""Exercise LUFA data loading, one real optimization step, and retrieval offline."""
import json
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, Wav2Vec2Config, Wav2Vec2Model

from lufa.data import batch, manifest
from lufa.cli import build_bank, load_model, retrieve, save_model
from lufa.model import RetrievalModel


def main():
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    root = Path("outputs/smoke").resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(2):
        wav = root / f"clip-{index}.wav"
        samples = (rng.normal(0, 800, 48000)).clip(-32768, 32767).astype("<i2")
        with wave.open(str(wav), "wb") as handle:
            handle.setparams((1, 2, 16000, len(samples), "NONE", "")); handle.writeframes(samples.tobytes())
        motion = root / f"clip-{index}.npy"
        np.save(motion, rng.uniform(0, 1, (90, 9)).astype("float32"))
        rows.append({"id": f"demo-{index}", "speaker": f"speaker-{index}", "split": "train",
                     "text": f"demo utterance {index}", "wav": wav.name, "motion": motion.name})
    path = root / "manifest.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "demo", "utterance", "0", "1"]
    tokenizer_dir = root / "tokenizer-source"; tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(tokenizer_dir / "vocab.txt"), do_lower_case=True)
    records = manifest(path)
    samples, mask, tokens, motion = batch(records, tokenizer, "cpu")
    audio = Wav2Vec2Model(Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                intermediate_size=64, conv_dim=(8, 8, 8), conv_kernel=(10, 3, 3), conv_stride=(5, 2, 2),
                num_conv_pos_embedding_groups=4, mask_time_prob=0.0))
    text = BertModel(BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                                num_attention_heads=4, intermediate_size=64))
    model = RetrievalModel(audio, text, latent=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, reconstruction, contrastive = model.objective(samples, mask, tokens, motion)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    model_dir = root / "model"
    save_model(model, tokenizer, model_dir, {r["speaker"] for r in records}, seed=11)
    reloaded, _, _ = load_model(model_dir, "cpu")
    assert sum(p.numel() for p in reloaded.parameters()) == sum(p.numel() for p in model.parameters())
    bank_path = root / "bank.npz"
    build_bank(Namespace(model=str(model_dir), manifest=str(path), output=str(bank_path), device="cpu"))
    output = root / "retrieved-face.npz"
    retrieve(Namespace(model=str(model_dir), bank=str(bank_path), text="demo utterance 0", wav=None,
                       top_k=2, frames=None, output=str(output), device="cpu"))
    assert output.exists() and Path(str(output) + ".json").exists()
    print(f"smoke passed: loss={loss.item():.4f}, checkpoint reload, bank, retrieval -> {output}")


if __name__ == "__main__":
    main()
