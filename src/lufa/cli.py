import argparse
import json
from pathlib import Path
import numpy as np
from .data import manifest, batch, waveform, motion
from .retrieval import CHANNELS, normalize, nearest, resample
from .server import DEFAULT_DEMO_TEXT


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


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def corpus_tokenizer(texts, directory):
    """Small lowercase WordPiece vocabulary from training transcripts, with character fallback pieces.

    Only for ``--from-scratch`` runs without a downloaded BERT tokenizer; unseen words split into characters.
    """
    from transformers import BertTokenizerFast
    import re
    import string
    words = sorted({w for t in texts for w in re.findall(r"[a-z0-9]+|[^\sa-z0-9]", t.lower())})
    chars = list(string.ascii_lowercase + string.digits)
    punctuation = list(string.punctuation)
    vocab = list(dict.fromkeys(SPECIAL_TOKENS + punctuation + chars + ["##" + c for c in chars] + words))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(directory / "vocab.txt"), do_lower_case=True)


def train(args):
    import torch
    from transformers import BertTokenizerFast
    from .data import group_batches, transcript_group
    from .model import RetrievalModel
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    all_rows = manifest(args.manifest)
    rows = [r for r in all_rows if r["split"] == "train"]
    if len(rows) < 2 or args.batch_size < 2 or args.epochs < 1:
        raise ValueError("Need >=2 training records, batch size>=2, epochs>=1")
    if args.corpus_vocab:
        if not args.from_scratch:
            raise ValueError("--corpus-vocab only applies to --from-scratch encoders")
        tokenizer = corpus_tokenizer([r["text"] for r in rows], Path(args.output) / "tokenizer-source")
    elif args.text_model:
        tokenizer = BertTokenizerFast.from_pretrained(args.text_model, local_files_only=True)
    else:
        raise ValueError("Supply a local --text-model, or --from-scratch --corpus-vocab")
    if not args.from_scratch and not args.audio_model:
        raise ValueError("Local --audio-model is required unless --from-scratch")
    model = RetrievalModel.initialize(args.audio_model, args.text_model, args.from_scratch, len(tokenizer)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    groups = [transcript_group(r) for r in rows] if args.sampler == "group" else list(range(len(rows)))
    repeated = len(groups) - len(set(groups))
    if repeated:
        print(json.dumps({"repeated_transcripts": repeated,
                          "note": "group-aware batches keep repeated transcripts out of the same contrastive batch"}))
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for indices in group_batches(groups, args.batch_size, rng):
            chosen = [rows[i] for i in indices]
            if len(chosen) < 2:
                continue  # singleton has no contrastive negative; rotated each epoch by shuffling
            inputs = batch(chosen, tokenizer, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, rec, nce = model.objective(*inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append([float(loss.detach()), float(rec.detach()), float(nce.detach())])
        if not losses:
            raise ValueError("No batch contained two distinct transcripts; add records or use --sampler random")
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
                        texts=np.asarray([r["text"] for r in rows]),
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
    # Named-channel BEAT JSON (or [T,9] .npy) to a 90-frame nine-channel clip. No downloads.
    from .data import face_motion
    values = face_motion(args.input, args.start, args.end, args.max_seconds, args.allow_resample)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, resample(values, 90))


def prepare_beat(args):
    from .beat import prepare
    prepare(args.root, args.output, args.speakers, args.window, args.hop, not args.no_snap, args.min_words,
            args.validation_speakers, args.test_speakers, args.duplicates, args.dup_threshold, args.max_takes, args.seed)


def fetch_beat(args):
    from .beat import fetch
    print(json.dumps(fetch(args.speaker, args.takes, args.output, args.max_bytes), indent=2))


def serve(args):
    from http.server import ThreadingHTTPServer
    from .server import create_handler
    if bool(args.model) != bool(args.bank):
        raise ValueError("Provide --model and --bank together")
    server = ThreadingHTTPServer((args.host, args.port), create_handler(Path(args.model) if args.model else None,
                                                                 Path(args.bank) if args.bank else None, args.device,
                                                                 args.demo_text))
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="LUFA-inspired audio/text retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert-motion", help="Crop-aligned BEAT face JSON or [T,9] .npy to a 90-frame clip")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, help="Crop start in seconds (JSON frame time)")
    p.add_argument("--end", type=float, help="Crop end in seconds (exclusive)")
    p.add_argument("--max-seconds", type=float, default=3.5, help="Refuse longer inputs (whole takes)")
    p.add_argument("--allow-resample", action="store_true",
                   help="Deliberately time-compress an input longer than --max-seconds into 90 frames")
    p.set_defaults(run=convert)
    p = sub.add_parser("prepare-beat", help="Official BEAT layout -> aligned ~3 s clips + manifest.jsonl")
    p.add_argument("--root", required=True, help="beat_english_v0.2.1 directory (or any tree of BEAT takes)")
    p.add_argument("--output", required=True, help="Directory for manifest.jsonl and clips/")
    p.add_argument("--speakers", nargs="+", help="Speaker ids or names to include (default: all)")
    p.add_argument("--window", type=float, default=3.0, help="Window length in seconds (training contract: ~3 s)")
    p.add_argument("--hop", type=float, default=3.0, help="Hop between window starts in seconds")
    p.add_argument("--no-snap", action="store_true", help="Do not snap window starts to word onsets")
    p.add_argument("--min-words", type=int, default=1, help="Skip windows with fewer words")
    p.add_argument("--validation-speakers", nargs="+")
    p.add_argument("--test-speakers", nargs="+")
    p.add_argument("--duplicates", choices=["flag", "drop"], default="flag",
                   help="flag: keep repeated transcripts with a shared text_group; drop: keep one per group and split")
    p.add_argument("--dup-threshold", type=float, default=0.6, help="Word-set Jaccard for near-repeated transcripts")
    p.add_argument("--max-takes", type=int, help="Limit takes per speaker")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(run=prepare_beat)
    p = sub.add_parser("fetch-beat", help="Download named official BEAT takes (WAV, face JSON, TextGrid)")
    p.add_argument("--speaker", required=True)
    p.add_argument("--takes", nargs="+", required=True, help="e.g. 2_scott_0_1_1")
    p.add_argument("--output", default="outputs/beat-raw")
    p.add_argument("--max-bytes", type=int, default=25_000_000)
    p.set_defaults(run=fetch_beat)
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio-model")
    p.add_argument("--text-model", help="Local BERT tokenizer/model directory")
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--corpus-vocab", action="store_true",
                   help="With --from-scratch: build a small tokenizer from training transcripts instead of --text-model")
    p.add_argument("--sampler", choices=["group", "random"], default="group",
                   help="group: never batch repeated transcripts together (default)")
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
    p.add_argument("--demo-text", default=DEFAULT_DEMO_TEXT, help="Text query for the first learned clip")
    p.set_defaults(run=serve)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
