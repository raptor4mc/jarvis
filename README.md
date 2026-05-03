# Ferris

Ferris is an offline Rust-focused chatbot + scaffold assistant that trains on local `.txt` files (including Rust source exported to text).

## What it does
- Trains a small local model on text corpora under configured roots.
- Retrieves matching snippets from corpus files at prompt time.
- Uses hybrid retrieval (keyword frequency + lightweight trigram similarity) for better code-context matching.
- Supports `/draft` and `/scaffold` commands to generate Rust-oriented project layouts.

## Data format
Ferris currently trains on `.txt` files only.

Recommended corpus shape:
- Convert `.rs` files to `.txt` (preserving paths if possible).
- Keep one logical source per file.
- Avoid binary dumps disguised as text.

## Key environment variables
- `RINGTAIL_DATA_ROOTS`: comma-separated roots to scan (default: `knowledge,.`).
- `RINGTAIL_MAX_FILES`: maximum loaded `.txt` files.
- `RINGTAIL_MAX_FILE_BYTES`: maximum bytes read per file.
- `RINGTAIL_DATA_INCLUDE`: optional comma-separated path substrings that must match.
- `RINGTAIL_DATA_EXCLUDE`: optional comma-separated path substrings to ignore.
- `RINGTAIL_MAX_TOKENS`: cap total training tokens.
- `RINGTAIL_REPLY_TOKENS_MIN` / `RINGTAIL_REPLY_TOKENS_MAX`: generation bounds for normal chat.
- `RINGTAIL_BASE_MODEL_CMD`: optional stronger inference command. Ferris passes the full prompt in `RINGTAIL_PROMPT` and uses command stdout as the model answer (fallback is local model).

## Training behavior
- Uses micro-batches + grad accumulation.
- Reports both train and validation loss per epoch.
- Uses corpus fingerprinting based on path + content hash to detect dataset changes.

## CPU training profile (under ~7 minutes)
Use this profile when you want fast CPU retraining without changing architecture.

### Exact environment variables
```bash
export RINGTAIL_DATA_ROOTS="knowledge,."
export RINGTAIL_DATA_INCLUDE="rust-lang/compiler"
export RINGTAIL_DATA_EXCLUDE="target,node_modules,.git"
export RINGTAIL_MAX_FILES=96
export RINGTAIL_MAX_FILE_BYTES=120000
export RINGTAIL_MAX_TOKENS=110000

export RINGTAIL_MODEL_DIM=96
export RINGTAIL_SEQ_LEN=96
export RINGTAIL_EPOCHS=2
export RINGTAIL_SAMPLE_STRIDE=1
export RINGTAIL_BATCH_SIZE=16
export RINGTAIL_LEARN_RATE=0.001

export RINGTAIL_FORCE_TRAIN=1
```

### Minimal `train.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

export RINGTAIL_DATA_ROOTS="knowledge,."
export RINGTAIL_DATA_INCLUDE="rust-lang/compiler"
export RINGTAIL_DATA_EXCLUDE="target,node_modules,.git"
export RINGTAIL_MAX_FILES=96
export RINGTAIL_MAX_FILE_BYTES=120000
export RINGTAIL_MAX_TOKENS=110000
export RINGTAIL_MODEL_DIM=96
export RINGTAIL_SEQ_LEN=96
export RINGTAIL_EPOCHS=2
export RINGTAIL_SAMPLE_STRIDE=1
export RINGTAIL_BATCH_SIZE=16
export RINGTAIL_LEARN_RATE=0.001
export RINGTAIL_FORCE_TRAIN=1

/usr/bin/time -f "elapsed=%E" cargo run --release
```

### Minimal Colab cell (CPU only)
```python
import os, subprocess

env = {
    "RINGTAIL_DATA_ROOTS": "knowledge,.",
    "RINGTAIL_DATA_INCLUDE": "rust-lang/compiler",
    "RINGTAIL_DATA_EXCLUDE": "target,node_modules,.git",
    "RINGTAIL_MAX_FILES": "96",
    "RINGTAIL_MAX_FILE_BYTES": "120000",
    "RINGTAIL_MAX_TOKENS": "110000",
    "RINGTAIL_MODEL_DIM": "96",
    "RINGTAIL_SEQ_LEN": "96",
    "RINGTAIL_EPOCHS": "2",
    "RINGTAIL_SAMPLE_STRIDE": "1",
    "RINGTAIL_BATCH_SIZE": "16",
    "RINGTAIL_LEARN_RATE": "0.001",
    "RINGTAIL_FORCE_TRAIN": "1",
}
os.environ.update(env)
subprocess.run(["/usr/bin/time", "-f", "elapsed=%E", "cargo", "run", "--release"], check=True)
```

### Verification checklist (expected output)
- `Loaded files:` should be between **50 and 150**.
- `tokens:` should be **<= 120000**.
- `model_dim:` should print **96** (or another value in 64..128 if you adjust).
- `seq_len:` should print **96** (or another value in 64..128 if you adjust).
- `stride:` should print **1**.
- Training log should show **Epoch 1/** and **Epoch 2/** (or up to 3 if you raise epochs).
- `/usr/bin/time` should report `elapsed=` under about **07:00**.

### Fallback config (if training exceeds 7 minutes)
Keep architecture unchanged and lower only corpus/steps:

```bash
export RINGTAIL_MAX_FILES=64
export RINGTAIL_MAX_TOKENS=90000
export RINGTAIL_EPOCHS=1
# keep these unchanged
export RINGTAIL_MODEL_DIM=96
export RINGTAIL_SEQ_LEN=96
export RINGTAIL_SAMPLE_STRIDE=1
export RINGTAIL_BATCH_SIZE=16
```
## Known limitations
- `.txt`-only ingestion (no direct `.rs` parser).
- Single-head architecture and CPU-oriented training loop.
- Retrieval is keyword-match based, not embedding/vector DB based.
- Retrieval is improved but still not full embedding/vector DB semantic search.
