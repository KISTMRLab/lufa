import argparse
import json
from pathlib import Path
import numpy as np
from .data import manifest, batch, waveform, motion
from .retrieval import CHANNELS, normalize, nearest, resample


def save_model(model, tokenizer, directory, train_speakers, seed):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    torch.save({"state_dict": model.state_dict(), "audio_config": model.audio.config.to_dict(),
                "text_config": model.text.config.to_dict(), "latent": 128,
                "train_speakers": sorted(train_speakers), "seed": seed}, directory / "model.pt")


def load_model(directory, device):
    import torch
    from transformers import BertConfig, BertModel, Wav2Vec2Config, Wav2Vec2Model, BertTokenizerFast
    from .model import RetrievalModel
    directory = Path(directory)
    ck = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model = RetrievalModel(Wav2Vec2Model(Wav2Vec2Config.from_dict(ck["audio_config"])),
                           BertModel(BertConfig.from_dict(ck["text_config"])), ck["latent"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(directory / "tokenizer", local_files_only=True)
    return model, tokenizer, ck


def train(args):
    import torch
    from transformers import BertTokenizerFast
    from .model import RetrievalModel
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    all_rows = manifest(args.manifest)
    rows = [r for r in all_rows if r["split"] == "train"]
    if len(rows) < 2 or args.batch_size < 2 or args.epochs < 1:
        raise ValueError("Need >=2 training records, batch size>=2, epochs>=1")
    tokenizer = BertTokenizerFast.from_pretrained(args.text_model, local_files_only=True)
    if not args.from_scratch and not args.audio_model:
        raise ValueError("Local --audio-model is required unless --from-scratch")
    model = RetrievalModel.initialize(args.audio_model, args.text_model, args.from_scratch, len(tokenizer)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(rows))
        losses = []
        for start in range(0, len(order), args.batch_size):
            chosen = [rows[i] for i in order[start:start + args.batch_size]]
            if len(chosen) < 2:
                continue  # singleton has no contrastive negative; rotated each epoch by shuffling
            inputs = batch(chosen, tokenizer, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, rec, nce = model.objective(*inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append([float(loss.detach()), float(rec.detach()), float(nce.detach())])
        record = dict(zip(["loss", "reconstruction", "contrastive"], np.mean(losses, axis=0).tolist()))
        record["epoch"] = epoch + 1
        history.append(record)
        print(json.dumps(record))
        save_model(model, tokenizer, args.output, {r["speaker"] for r in rows}, args.seed)
    (Path(args.output) / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def paired_embedding(model, tokenizer, row, device):
    import torch
    samples, mask, tokens, _ = batch([row], tokenizer, device)
    with torch.no_grad():
        a, t = model.encode_audio(samples, mask), model.encode_text(tokens)
    return normalize(a[0].cpu().numpy() + t[0].cpu().numpy())


def build_bank(args):
    model, tokenizer, ck = load_model(args.model, args.device)
    rows = [r for r in manifest(args.manifest) if r["split"] == "train"]
    if not rows:
        raise ValueError("No training bank records")
    if not {r["speaker"] for r in rows}.issubset(set(ck["train_speakers"])):
        raise ValueError("Bank manifest must use this checkpoint's training speakers")
    embeddings = np.stack([paired_embedding(model, tokenizer, r, args.device) for r in rows])
    motions = np.stack([motion(r["motion"]) for r in rows])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=embeddings, motions=motions,
                        ids=np.asarray([r["id"] for r in rows]), speakers=np.asarray([r["speaker"] for r in rows]),
                        channels=np.asarray(CHANNELS), fps=np.asarray(30))
    # Bank is generated locally; it is never a bundled distribution artifact.
    print(f"Indexed {len(rows)} recorded clips")


def query(model, tokenizer, text, wav, device):
    import torch
    vectors = []
    with torch.no_grad():
        if text:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            vectors.append(model.encode_text({k: v.to(device) for k, v in tokens.items()})[0].cpu().numpy())
        if wav:
            x = torch.tensor(waveform(wav)[None], device=device)
            vectors.append(model.encode_audio(x, torch.ones_like(x, dtype=torch.long))[0].cpu().numpy())
    if not vectors:
        raise ValueError("Supply --text and/or --wav")
    return normalize(sum(vectors))


def read_bank(path):
    with np.load(path, allow_pickle=False) as obj:
        bank = {k: obj[k].copy() for k in obj.files}
    if bank["channels"].tolist() != CHANNELS or int(bank["fps"]) != 30:
        raise ValueError("Incompatible bank channels/FPS")
    if len(bank["embeddings"]) != len(bank["motions"]) or bank["motions"].shape[1:] != (90, 9):
        raise ValueError("Invalid bank dimensions")
    return bank


def retrieve(args):
    model, tokenizer, _ = load_model(args.model, args.device)
    bank = read_bank(args.bank)
    embedding = query(model, tokenizer, args.text, args.wav, args.device)
    indices, scores = nearest(embedding, bank["embeddings"], args.top_k)
    selected = bank["motions"][indices[0]]
    if args.frames:
        selected = resample(selected, args.frames)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, motion=selected, names=np.asarray(CHANNELS), fps=np.asarray(30))
    result = [{"id": str(bank["ids"][i]), "cosine": float(s)} for i, s in zip(indices, scores)]
    Path(str(output) + ".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def evaluate(args):
    model, tokenizer, ck = load_model(args.model, args.device)
    bank = read_bank(args.bank)
    rows = [r for r in manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        raise ValueError("No held-out records")
    held = {r["speaker"] for r in rows}
    if held & (set(ck["train_speakers"]) | set(bank["speakers"].tolist())):
        raise ValueError("Held-out speakers overlap training or bank speakers")
    errors, oracle_errors = [], []
    for r in rows:
        embedding = paired_embedding(model, tokenizer, r, args.device)
        indices, _ = nearest(embedding, bank["embeddings"])
        target = motion(r["motion"])
        errors.append(float(np.abs(bank["motions"][indices[0]] - target).mean()))
        oracle_errors.append(float(np.abs(bank["motions"] - target).mean(axis=(1, 2)).min()))
    print(json.dumps({"clips": len(rows), "retrieved_motion_mae": float(np.mean(errors)),
                      "bank_oracle_mae": float(np.mean(oracle_errors)),
                      "note": "Reference-specific errors do not establish perceptual quality or the paper's scores."}, indent=2))


def convert(args):
    # Named-channel BEAT JSON to normalized nine-channel motion. No downloads.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, motion(args.input))


def serve(args):
    from http.server import ThreadingHTTPServer
    from .server import create_handler
    if bool(args.model) != bool(args.bank):
        raise ValueError("Provide --model and --bank together")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(Path(args.model) if args.model else None,
                                                                 Path(args.bank) if args.bank else None, args.device))
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="LUFA-inspired audio/text retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert-motion")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=convert)
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-model")
    p.add_argument("--text-model", required=True, help="Local BERT tokenizer/model directory")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=train)
    p = sub.add_parser("bank")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=build_bank)
    p = sub.add_parser("retrieve")
    p.add_argument("--model", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--text")
    p.add_argument("--wav")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--frames", type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=retrieve)
    p = sub.add_parser("evaluate")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.set_defaults(run=evaluate)
    p = sub.add_parser("serve")
    p.add_argument("--model", help="user-trained checkpoint directory")
    p.add_argument("--bank", help="user-built recorded-motion bank")
    p.add_argument("--device", default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.set_defaults(run=serve)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
