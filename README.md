# Ferris

Ferris is an offline Rust-focused chatbot + scaffold assistant that trains on local `.txt` files (including Rust source exported to text).

## What it does
- Trains a small local model on text corpora under configured roots.
- Retrieves matching snippets from corpus files at prompt time.
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

## Training behavior
- Uses micro-batches + grad accumulation.
- Reports both train and validation loss per epoch.
- Uses corpus fingerprinting based on path + content hash to detect dataset changes.

## Known limitations
- `.txt`-only ingestion (no direct `.rs` parser).
- Single-head architecture and CPU-oriented training loop.
- Retrieval is keyword-match based, not embedding/vector DB based.
