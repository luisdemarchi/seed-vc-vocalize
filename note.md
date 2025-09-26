# Seed-VC V2 — Practical Notes (EN)

This note focuses ONLY on V2. It covers what you need to train for specific voices (e.g., 4 target speakers), how to prepare data, how to run training, and how to use the trained checkpoints for inference (CLI and Web UI).
 
> Scope: This project will focus 100% on V2.

References in repo:
- Config: `configs/v2/vc_wrapper.yaml`
- Wrapper (pipeline): `modules/v2/vc_wrapper.py`
- Inference script: `inference_v2.py`
- Training script: `train_v2.py`

---

## 1) What V2 Is

V2 is a full pipeline that uses ASTRAL-Quantization content extractors (narrow + wide), an AR model (for style/emotion/accent when enabled), a CFM (diffusion) model for mel generation, a style encoder (CAMPPlus), and a vocoder (e.g., BigVGAN). It supports:
- Timbre conversion (default, AR off)
- Style/accent/emotion conversion (AR on via `--convert-style true`)
- Anonymization (`--anonymization-only true`)
- Streaming/chunked inference

Check `modules/v2/vc_wrapper.py` for how content features, AR, CFM, style, and vocoder interact.

---

## 2) Hardware & Environment

- Strongly recommended: NVIDIA GPU (>= 8 GB VRAM). CPU works but will be slow, especially for training.
- Python 3.10.
- Install dependencies:
  - Windows/Linux: `pip install -r requirements.txt`
  - Mac Apple Silicon: `pip install -r requirements-mac.txt`

---

## 3) Data Preparation (for 4 specific speakers)

- Audio quality: clean speech, no BGM, low noise, decent mic.
- Formats accepted: `.wav`, `.flac`, `.mp3`, `.m4a`, `.opus`, `.ogg`.
- Recommended per speaker: 5–10 minutes clean speech. Minimum works with less (even ~1–2 minutes), but more is better.
- Clip length: 5–30 seconds per file. Avoid very long files.
- Suggested directory structure (multi-speaker):
  ```
  data/my_4_voices/
    speaker_a/
      a_001.wav
      a_002.wav
      ...
    speaker_b/
      b_001.wav
      ...
    speaker_c/
      ...
    speaker_d/
      ...
  ```
  Labels are not strictly required by the code, but per-folder organization helps.

Tip: 16 kHz, mono, 16-bit WAV is a good baseline. The pipeline resamples internally when needed, but consistent input helps reproducibility.

---

## 4) Training Options (V2)

You can either:
- Train one multi-speaker model including all 4 speakers (single run, easier to manage);
- Or train four separate runs (one per speaker) for maximum similarity per voice.

### 4.1 Launch training (single multi-speaker run)

`train_v2.py` supports multi-GPU via Accelerate:
```bash
accelerate launch train_v2.py \
  --dataset-dir data/my_4_voices \
  --run-name vc_v2_my4voices \
  --batch-size 2 \
  --max-steps 1000 \
  --max-epochs 1000 \
  --save-every 500 \
  --num-workers 0 \
  --train-cfm
```
Notes:
- Start small (e.g., 300–500 steps) to validate the setup, then extend to 1000+ for quality.
- Outputs are saved under `./runs/<run-name>/`.

### 4.2 One run per speaker (optional)
Run four times, changing dataset dir and run name each time, e.g.:
```bash
accelerate launch train_v2.py \
  --dataset-dir data/speaker_a \
  --run-name vc_v2_speaker_a \
  --batch-size 2 --max-steps 1000 --save-every 500 --num-workers 0 --train-cfm
```
Repeat for b/c/d. You will get four checkpoints.

### 4.3 Training time (rough guidance)
- On a mid-range GPU (e.g., RTX 3060), 1000 steps can take considerably longer than V1. If time/VRAM is tight, try fewer steps first.

---

## 5) What gets produced

- Checkpoints and logs under `./runs/<run-name>/`.
- For V2, the wrapper expects AR and CFM checkpoints. If your run outputs a single `ft_model.pth`, reuse it for both AR and CFM args at inference unless you explicitly split.

---

## 6) Inference with V2 (CLI)

Base script: `inference_v2.py`. It loads V2 via `configs/v2/vc_wrapper.yaml` and `modules/v2/vc_wrapper.py`.

### 6.1 Timbre-only (AR off)
```bash
python inference_v2.py \
  --source examples/source/source_s1.wav \
  --target examples/reference/s1p1.wav \
  --output output \
  --diffusion-steps 30 \
  --length-adjust 1.0 \
  --intelligibility-cfg-rate 0.7 \
  --similarity-cfg-rate 0.7 \
  --ar-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth \
  --cfm-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth
```

### 6.2 Style/accent/emotion conversion (AR on)
```bash
python inference_v2.py \
  --source examples/source/source_s1.wav \
  --target examples/reference/s1p1.wav \
  --output output \
  --convert-style true \
  --top-p 0.9 --temperature 1.0 --repetition-penalty 1.2 \
  --diffusion-steps 30 --length-adjust 1.0 \
  --intelligibility-cfg-rate 0.7 --similarity-cfg-rate 0.7 \
  --ar-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth \
  --cfm-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth
```

### 6.3 Anonymization
```bash
python inference_v2.py \
  --source examples/source/source_s1.wav \
  --target examples/reference/s1p1.wav \
  --output output \
  --anonymization-only true \
  --diffusion-steps 30 --length-adjust 1.0 \
  --intelligibility-cfg-rate 0.7 --similarity-cfg-rate 0.7 \
  --ar-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth \
  --cfm-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth
```

Tips:
- `--intelligibility-cfg-rate` vs `--similarity-cfg-rate` trades clarity vs timbre similarity.
- `--top-p`, `--temperature`, `--repetition-penalty` control AR sampling when `--convert-style true`.
- Use `--compile true` to try Inductor optimizations (see `inference_v2.py`).

---

## 7) Web UI (V2)

`app_vc_v2.py` loads V2 models and starts a Gradio app.
```bash
python app_vc_v2.py \
  --cfm-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth \
  --ar-checkpoint-path ./runs/vc_v2_my4voices/ft_model.pth
```
Then open `http://localhost:7860/`.

- Enable style/accent/emotion conversion via the UI’s checkbox equivalent of `--convert-style`.
- Use shorter `target` (<= 25s) for best stability (see `vc_wrapper.py`).

---

## 8) Recommended Workflow for 4 Speakers

1. Prepare `data/my_4_voices/` with 5–10 minutes of clean speech per speaker, split into 5–30 sec clips.
2. Run a single multi-speaker training with `train_v2.py` (or 4 separate runs if you need per-speaker max similarity).
3. At inference time, always pass a `--target` clip of the desired speaker (the reference voice).
4. Tune CFGs and (if needed) AR sampling parameters to balance clarity vs. similarity and achieve desired style/accent effects.

---

## 9) Troubleshooting

- If VRAM is insufficient, reduce batch size, diffusion steps, or target clip length.
- If output is unstable with long inputs, use streaming/chunking methods exposed in `vc_wrapper.py` (the default UI already chunks).
- If downloads from Hugging Face fail, try a mirror: `HF_ENDPOINT=https://hf-mirror.com` before commands.

---

## 10) Key Files & Knobs

- `configs/v2/vc_wrapper.yaml`: model assembly via Hydra/omegaconf.
- `modules/v2/vc_wrapper.py`: end-to-end logic (style encoder, content extractors, AR, CFM, vocoder, streaming).
- `inference_v2.py`: CLI flags (CFG rates, AR sampling, anonymization, compile).
- `train_v2.py`: training entrypoint. Use `accelerate` for multi-GPU.

This is all you need to focus on V2 training and usage for your 4 speakers. Adjust steps upward and iterate on data quality for best results.

---

## 11) Python 3.13 Setup (V2 only)

These steps were validated to run V2 on Python 3.13 using a dedicated environment and minor adjustments.

### 11.1 Create and activate the environment
```bash
python3.13 -m venv .venv313
.venv313/bin/pip install --upgrade pip
```

### 11.2 Install dependencies for 3.13
Use the dedicated file `requirements-py313.txt` (keeps the 3.12 path intact):
```bash
.venv313/bin/pip install -r requirements-py313.txt
```
Notes:
- PyTorch stable wheels for 3.13 may be limited; the file uses nightly CPU by default.
- If you have CUDA, switch to the CUDA index in `requirements-py313.txt` and comment out the nightly CPU lines.
- Included important pins: `numpy>=2.1`, `scipy>=1.14`, `librosa==0.10.2.post1`, `munch==4.0.0`, `matplotlib`.
- We avoid `audioread` issues on 3.13 by using `soundfile` for loading in V2.

### 11.3 Code adjustment for audio loading
File: `modules/v2/vc_wrapper.py`
- Added `VoiceConversionWrapper._load_wave()` that loads audio via `soundfile` and resamples via `librosa.resample`.
- Replaced `librosa.load(...)[0]` with `self._load_wave(...)` in:
  - `convert_timbre()`
  - `convert_voice()`
  - `convert_voice_with_streaming()`

### 11.4 Quick validation
```bash
.venv313/bin/python inference_v2.py \
  --source examples/source/source_s1.wav \
  --target examples/reference/s1p1.wav \
  --output output \
  --diffusion-steps 10 \
  --length-adjust 1.0 \
  --intelligibility-cfg-rate 0.7 \
  --similarity-cfg-rate 0.7
```
This should produce a WAV under `output/`.
