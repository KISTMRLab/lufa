# Requirements: LUFA retrieval

## Evidence and limits

Read the indexed abstract for *LUFA: Lightweight Upper-Face Animation for VR/MR Avatars*, IEEE ISMAR-Adjunct 2025, DOI 10.1109/ISMAR-Adjunct68609.2025.00217. The local inventory contains no full conference PDF; the laboratory link points to the publisher. Only the abstract's claims are treated as paper requirements. Later emotion–liveness manuscript architecture and results are **not LUFA**.

## Paper requirements

Voice and text input; fine-tuned Wav2Vec2.0 and BERT encoders, reconstruction supervision, contrastive latent alignment, retrieval of existing facial-animation sequences. No motion synthesis decoder at inference.

## Assumptions needed for an implementable research baseline

Use pooled audio and text representations projected into a shared normalized 128-dimensional space. During training only, two small heads reconstruct a resampled 90×9 upper-face clip. Optimize average branch L1 reconstruction plus symmetric audio/text InfoNCE (temperature .07). These objective details, temporal pooling, latent width, upper-face subset and retrieval weighting are explicit assumptions, not recovered from full text. Build a bank with fused audio/text embeddings; query audio, text or their normalized average; return exact stored motion (optionally resampled for caller duration). No claim about paper latency or quality.

Use BEAT's synchronized speech, transcripts and ARKit facial motion as the primary public substitute. User prepares local 16-kHz mono WAV and nine-channel NumPy motion; manifest names speaker, text, split and paths. Reject speaker overlap across splits. Train encoders from local user-provided pretrained directories or from standard random configurations via `--from-scratch`; no implicit downloads. Mask audio/text padding for pooling. Fixed crop records require transcripts aligned to that crop by the user, rather than attaching whole-recording text to short motion.

## Deliverables and acceptance

Standalone package with train, bank, retrieve and evaluate commands; data-preparation helper for BEAT facial JSON; README with source/citation/access/licensing, exact contracts, commands and assumptions. No dataset, model or checkpoint distribution. Small procedural retrieval checks and syntax validation; full pretrained fine-tuning is outside verification in this workspace.
