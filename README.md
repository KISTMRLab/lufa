# LUFA: Lightweight Upper-Face Animation for VR/MR Avatars

**Hwang Youn Kim, Ghazanfar Ali, Jae-In Hwang**

**IEEE ISMAR-Adjunct · 2025** · Published

[Paper / publisher](https://doi.org/10.1109/ismar-adjunct68609.2025.00217) · [Project page](https://ghazanfarali.com/research/lufa/) · [BibTeX](citation.bib) · [Requirements](REQUIREMENTS.md) · [Code & setup](#implementation-and-usage)

> Lightweight upper-face animation for VR and MR avatars.

![Scientific method schematic for lufa](paper-assets/method.svg)

*New scientific schematic based on the LUFA conference abstract. This is not an extracted paper figure; pooling, loss wiring and bank organization are implementation assumptions documented in REQUIREMENTS.md.*

## Why this research

Upper-face movement contributes to a conversational avatar's expressiveness, but large generative models can be costly. LUFA explores retrieving existing facial motion through representations aligned with voice and text.

LUFA encodes voice and text with fine-tuned Wav2Vec2.0 and BERT models. Reconstruction loss and contrastive learning align latent representations, which are used to retrieve facial-animation sequences for VR and MR avatars. The related journal manuscript studies a later emotion–liveness approach and is in final review.

## Method at a glance

**Voice + text** → **Aligned latent representations** → **Facial animation retrieval**

| | Research system |
|---|---|
| Input | Voice and text |
| Method | Wav2Vec2.0 and BERT encoders; reconstruction loss; contrastive latent alignment and retrieval |
| Output | Retrieved upper-face facial-animation sequences |

## Evidence and scope

Consult the full conference paper for evaluation details

**Attribution:** These findings describe the paper or manuscript, not results obtained with this repository's code.

**Study context:** Conference abstract describes the representation and retrieval framework.

**Limitations:** Retrieval selects existing animation sequences. The later emotion–liveness manuscript’s parameter counts and study results do not describe LUFA.

## Explore the implementation

Audio/text fine-tuning, reconstruction and contrastive objectives, bank building and recorded facial-clip retrieval. Architecture details absent from the accessible abstract are explicitly assumed.

This repository contains independently written research code. The institute's original source, datasets and trained models are not distributed. Public-data preparation, commands, assumptions and checks are documented below and in [REQUIREMENTS.md](REQUIREMENTS.md).

## Resources and citation

Read the paper through its [publisher record](https://doi.org/10.1109/ismar-adjunct68609.2025.00217). PDFs are hosted by publishers or preprint archives rather than stored in this repository.

Please cite the research paper when using its ideas; [download the BibTeX citation](citation.bib). The implementation has its own documented scope.

## Implementation and usage

<!-- implementation-guide -->

Hwang Youn Kim, Ghazanfar Ali and Jae-In Hwang, **IEEE ISMAR-Adjunct 2025**, pp. 841–842. [Paper DOI](https://doi.org/10.1109/ISMAR-Adjunct68609.2025.00217) · [Paper page](https://ghazanfarali.com/research/lufa/).

This independent research baseline fine-tunes **Wav2Vec2.0 and BERT** using motion reconstruction and contrastive audio/text alignment, then retrieves an existing upper-face animation from a local bank. The reconstruction heads supervise training; they do not synthesize motion at inference.

The indexed conference abstract was available; the full PDF was not supplied locally or accessible during implementation. [REQUIREMENTS.md](REQUIREMENTS.md) distinguishes abstract-supported components from our choices: pooled 128-dimensional latents, symmetric InfoNCE,90 × 9 reconstruction heads, BEAT data, and equal-weight multimodal retrieval. This does **not** reuse or claim the later emotion–liveness manuscript's architecture, metrics or timing.

### Setup and local encoders

Python 3.10+. Install a suitable [PyTorch build](https://pytorch.org/get-started/locally/), then:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
lufa --help
```

On Linux/macOS activate with `source .venv/bin/activate`. Obtain Wav2Vec2 and BERT model/tokenizer files yourself, following [Wav2Vec2 documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2) and [BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert), respecting each checkpoint's terms. Place them in local directories. The code uses `local_files_only=True` and performs no implicit downloads. Supply a BERT tokenizer consistent with your text language and checkpoint vocabulary.

`--from-scratch` constructs compact random four-layer Wav2Vec2/BERT encoders (an explicit baseline alternative to pretrained fine-tuning). It still requires a **local BERT tokenizer**; no pretrained encoder weights are needed. No model files are bundled here.

### Synthetic quickstart

After installation, run `python scripts/smoke.py`. It creates two aligned three-second WAV/motion records, loads them through the production manifest and batching code, performs one CPU optimization step with small randomly initialized Wav2Vec2/BERT configurations, saves and reloads a production-compatible checkpoint, builds a bank, and retrieves a recorded motion. Inspect the model, bank, and `retrieved-face.npz` under `outputs/smoke/`. It needs no network or model download. For real training, retain the manifest, WAV, transcript, and `[T,9]` motion contracts below, then point the CLI at local pretrained encoders or use `--from-scratch` with your local tokenizer.

### Public dataset preparation

Obtain synchronized speech, transcript and ARKit facial motion from [BEAT](https://github.com/srymurphy/BEAT) using its official access instructions. Versions, access conditions and channel formats vary. Do not redistribute recordings or generated banks merely because the source repository is public. MEAD could be an alternative after AU extraction/retargeting, but BEAT's recorded blendshapes avoid that extra estimation step.

Prepare approximately 3-second **aligned** clips: 16-kHz mono signed 16-bit PCM WAV (47000..49000 samples), transcript of exactly that window, and `[T,9]` float32 motion 0..1 in this order:

```text
browInnerUp, browOuterUpLeft, browOuterUpRight, browDownLeft, browDownRight,
eyeSquintLeft, eyeSquintRight, eyeWideLeft, eyeWideRight
```

Use FFmpeg for waveform conversion (`ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le output.wav`) and timed transcripts to choose windows. Do not pair a whole recording's transcript with a short motion crop. Motion is resampled to 90 frames; this preserves the chosen crop's endpoints but does not perform timestamp alignment. Audio/motion/transcript synchronization is your preparation responsibility.

An adapter accepts BEAT named-channel JSON `{"names":[...],"frames":[{"weights":[...]}]}` and exports nine-channel motion. **Pass a previously cropped facial JSON**, not a full recording to time-compress:

```powershell
lufa convert-motion --input data/clips/clip-face.json --output data/clips/clip-face.npy
```

Write `data/manifest.jsonl` (paths relative to manifest; one record per line):

```json
{"id":"beat-a-001","speaker":"speaker-a","split":"train","text":"the transcript of this clip","wav":"clips/a-001.wav","motion":"clips/a-001-face.npy"}
{"id":"beat-b-001","speaker":"speaker-b","split":"test","text":"another aligned transcript","wav":"clips/b-001.wav","motion":"clips/b-001-face.npy"}
```

These examples describe the schema; files are not included. Use at least two training records for contrastive negatives. Keep speakers disjoint across train/validation/test; the loader rejects overlap and duplicate clip ids. WAV is standardized per clip; tokenizer padding and Wav2Vec2 convolution output masks are excluded from pooled representations.

### Train, build a bank, retrieve

```powershell
lufa train --manifest data/manifest.jsonl --audio-model data/encoders/wav2vec2 --text-model data/encoders/bert --epochs 20 --batch-size 8 --output runs/lufa --device cuda
# Optional alternative: random compact encoders plus a local tokenizer
lufa train --manifest data/manifest.jsonl --text-model data/encoders/bert --from-scratch --output runs/lufa-random --device cuda
lufa bank --manifest data/manifest.jsonl --model runs/lufa --output runs/bank.npz --device cpu
lufa retrieve --model runs/lufa --bank runs/bank.npz --text "I am pleased to see you" --output outputs/face.npz
lufa retrieve --model runs/lufa --bank runs/bank.npz --wav data/query.wav --text "I am pleased to see you" --top-k 3 --frames 120 --output outputs/face-long.npz
lufa evaluate --manifest data/manifest.jsonl --model runs/lufa --bank runs/bank.npz --split test
```

Training defaults (20 epochs, 2e-5 AdamW, temperature .07) are implementation assumptions. All encoder parameters are trainable. Singleton final batches are skipped because they have no contrastive negatives. CPU execution is supported as a code path; full fine-tuning can be expensive.

The bank contains normalized fused embeddings, recorded 90 × 9 motions, clip ids and speaker ids from **training records only**. Query can use audio, text or their normalized average. Retrieval copies the highest-scoring clip; `--frames` optionally interpolates to caller length. Output NPZ includes motion/names/fps, and adjacent JSON records top-k ids and cosine similarities. This preserves recorded motion rather than decoding an average expression.

Evaluation rejects held-out speaker overlap with checkpoint/bank speakers and reports motion MAE plus the best possible bank MAE against each reference. These are diagnostics, not the paper's reported evaluation or evidence of perceived naturalness. A semantically valid retrieved expression can differ from the single reference.
