# Requirements: LUFA retrieval

## Evidence and limits

Read the indexed abstract for *LUFA: Lightweight Upper-Face Animation for VR/MR Avatars*, IEEE ISMAR-Adjunct 2025, DOI 10.1109/ISMAR-Adjunct68609.2025.00217. The local inventory contains no full conference PDF; the laboratory link points to the publisher. Only the abstract's claims are treated as paper requirements. Later emotion–liveness manuscript architecture and results are **not LUFA**.

## Paper requirements

Voice and text input; fine-tuned Wav2Vec2.0 and BERT encoders, reconstruction supervision, contrastive latent alignment, retrieval of existing facial-animation sequences. No motion synthesis decoder at inference.

## Assumptions needed for an implementable research baseline

Use pooled audio and text representations projected into a shared normalized 128-dimensional space. During training only, two small heads reconstruct a resampled 90×9 upper-face clip. Optimize average branch L1 reconstruction plus symmetric audio/text InfoNCE (temperature .07). These objective details, temporal pooling, latent width, upper-face subset and retrieval weighting are explicit assumptions, not recovered from full text. Build a bank with fused audio/text embeddings; query audio, text or their normalized average; return exact stored motion (optionally resampled for caller duration). No claim about paper latency or quality.

Use BEAT's synchronized speech, transcripts and ARKit facial motion as the primary public substitute. `lufa prepare-beat` walks the official BEAT English layout. It cuts 3-second windows from TextGrid word timings and writes 16-kHz mono WAV crops, nine-channel NumPy face crops selected by frame time, crop-aligned transcripts and a speaker-disjoint manifest naming speaker, text, split and paths. Other data follows the same manifest contract. Reject speaker overlap across splits. Repeated scripted transcripts are grouped and kept out of the same contrastive batch, or dropped on request. Train encoders from local user-provided pretrained directories or from standard random configurations via `--from-scratch`. Downloads happen only through the explicit `fetch-beat` command. Mask audio/text padding for pooling. Pass the Wav2Vec2 attention mask only to layer-norm feature extractors, and zero-pad for group-norm ones. Facial conversion refuses whole takes rather than time-compressing them.

## Deliverables and acceptance

Standalone package with prepare-beat, train, bank, retrieve and evaluate commands; a guarded converter for BEAT facial JSON; a demo launcher that uses learned retrieval when a local checkpoint and bank exist; README with source/citation/access/licensing, exact contracts, commands and assumptions. A local browser viewer can inspect all nine channels from an explicitly authored controller example, user-uploaded recorded motion, or the actual trained model/bank retrieval path. The authored example is not presented as learned output. No dataset, model or checkpoint distribution. Procedural retrieval, API/channel checks and syntax validation; full pretrained fine-tuning is outside verification in this workspace.

## Bundled fictional avatar substitution

Two newly generated fictional CC0 humanoids replace the original avatar assets in the browser demo. They provide a 53-bone rig and named ARKit/viseme targets. Motion retargeting adapts source joints to their bind pose; speaking envelopes approximate mouth motion rather than phoneme alignment. The optional recorded BEAT companion inspects public motion, face and audio files prepared locally, independently of the paper's learned algorithm. No dataset recordings or trained weights are bundled.
