// main.rs
// Port of main.cpp — trains the model then runs the chat loop.
// Usage:
//   cargo run --release
//
// Environment variables (same as C++ version):
//   RINGTAIL_EPOCHS        override epoch count
//   RINGTAIL_BATCH_SIZE    override micro-batch size
//   RINGTAIL_SAMPLE_STRIDE override sample stride
//   RINGTAIL_MAX_TOKENS    cap training tokens
//
// Extra corpus/model knobs:
//   RINGTAIL_DATA_ROOTS      comma-separated roots to scan (default: "knowledge,.")
//   RINGTAIL_MAX_FILES       cap number of loaded .txt files (default: 25000)
//   RINGTAIL_MODEL_DIM       override model dimension (default: 256)
//   RINGTAIL_SEQ_LEN         override context length (default: 128)
//   RINGTAIL_LEARN_RATE      override learning rate (default: 0.001)
//   RINGTAIL_RETRIEVE_TOP_K  files retrieved per prompt (default: 4)
//   RINGTAIL_RETRIEVE_CHARS  max chars per retrieved file snippet (default: 1200)

mod model;
mod tokenizer;

use model::ChatModel;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct CorpusDoc {
    path: String,
    content: String,
    content_lc: String,
}

fn collect_txt_files(root: &Path, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }

    if root.is_file() {
        if root.extension().and_then(|e| e.to_str()) == Some("txt") {
            out.push(root.to_path_buf());
        }
        return;
    }

    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_txt_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                out.push(path);
            }
        }
    }
}

fn normalize_project_path(path: &Path) -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    path.strip_prefix(cwd).unwrap_or(path).display().to_string()
}

fn append_structure_tokens(text: &mut String, docs: &[CorpusDoc]) {
    let mut dirs: HashSet<String> = HashSet::new();
    let mut dir_files: HashMap<String, Vec<String>> = HashMap::new();

    for doc in docs {
        let p = Path::new(&doc.path);
        if let Some(parent) = p.parent() {
            let dir = parent.display().to_string();
            dirs.insert(dir.clone());
            dir_files.entry(dir).or_default().push(
                p.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string(),
            );
        }
    }

    let mut sorted_dirs: Vec<String> = dirs.into_iter().collect();
    sorted_dirs.sort();

    text.push_str("\n\n<project_structure>\n");
    for dir in sorted_dirs {
        text.push_str("<dir>");
        text.push_str(&dir);
        text.push_str("</dir>\n");

        if let Some(files) = dir_files.get_mut(&dir) {
            files.sort();
            files.dedup();
            for file in files.iter() {
                text.push_str("<file_in_dir dir=\"");
                text.push_str(&dir);
                text.push_str("\">");
                text.push_str(file);
                text.push_str("</file_in_dir>\n");
            }
        }
    }
    text.push_str("</project_structure>\n");
}

fn load_corpus(data_roots: &[PathBuf], max_files: usize) -> (String, Vec<CorpusDoc>) {
    let mut text = String::from(
        "hello there i am an offline chatbot created to demonstrate how a small neural network can learn patterns from text \
         i do not use the internet and i do not rely on external data everything i know is written inside this program \
         you can ask simple questions and i will try to respond based on the patterns i learned during training \
         my goal is not to be perfect but to show how a tiny model can generate sentences and imitate conversation \
         the more text i am trained on the better i become at forming coherent replies and continuing your messages \
         this dataset is small but it teaches basic structure like greetings statements and simple explanations \
         feel free to experiment with different prompts and see how the model reacts to your input \
         remember that this chatbot is only a demonstration and not a full language model but it can still be fun to interact with ",
    );

    let mut seen = HashSet::new();
    let mut files = Vec::new();
    for root in data_roots {
        collect_txt_files(root, &mut files);
    }
    files.sort();

    let mut docs = Vec::new();

    for path in files {
        if docs.len() >= max_files {
            break;
        }

        let canonical = path.canonicalize().unwrap_or(path.clone());
        if !seen.insert(canonical.clone()) {
            continue;
        }

        if let Ok(contents) = fs::read_to_string(&canonical) {
            let tag = normalize_project_path(&canonical);
            let doc = CorpusDoc {
                path: tag.clone(),
                content_lc: contents.to_lowercase(),
                content: contents,
            };

            text.push_str("\n\n<file:");
            text.push_str(&tag);
            text.push_str(">\n");
            text.push_str(&doc.content);

            docs.push(doc);
        }
    }
    files.sort();

    append_structure_tokens(&mut text, &docs);

    (text, docs)
}

fn parse_usize_env(name: &str, default: usize, min: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= min)
        .unwrap_or(default)
}

fn parse_f32_env(name: &str, default: f32, min: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| *v >= min)
        .unwrap_or(default)
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != ':')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s.len() >= 3)
        .collect()
}

fn build_prompt_with_retrieval(
    user_input: &str,
    docs: &[CorpusDoc],
    top_k: usize,
    max_chars: usize,
) -> String {
    if docs.is_empty() || user_input.trim().is_empty() {
        return user_input.to_string();
    }

    let terms = query_terms(user_input);
    if terms.is_empty() {
        return user_input.to_string();
    }

    let mut scored: Vec<(usize, usize)> = docs
        .iter()
        .enumerate()
        .filter_map(|(i, doc)| {
            let score = terms
                .iter()
                .map(|t| {
                    doc.content_lc.matches(t).count()
                        + doc.path.to_lowercase().matches(t).count() * 2
                })
                .sum::<usize>();
            if score > 0 {
                Some((i, score))
            } else {
                None
            }
        })
        .collect();

    if scored.is_empty() {
        return user_input.to_string();
    }

    scored.sort_by(|a, b| b.1.cmp(&a.1));

    let mut prompt = String::new();
    prompt.push_str("<retrieved_context>\n");
    for (idx, score) in scored.into_iter().take(top_k.max(1)) {
        let doc = &docs[idx];
        let snippet: String = doc.content.chars().take(max_chars.max(64)).collect();
        prompt.push_str("<context_file path=\"");
        prompt.push_str(&doc.path);
        prompt.push_str("\" score=\"");
        prompt.push_str(&score.to_string());
        prompt.push_str("\">\n");
        prompt.push_str(&snippet);
        prompt.push_str("\n</context_file>\n");
    }
    prompt.push_str("</retrieved_context>\n");
    prompt.push_str("<user_prompt>\n");
    prompt.push_str(user_input);
    prompt.push_str("\n</user_prompt>");
    prompt
}

fn main() {
    // ── data loading ────────────────────────────────────────────────────────
    let roots_env =
        std::env::var("RINGTAIL_DATA_ROOTS").unwrap_or_else(|_| "knowledge,.".to_string());
    let data_roots: Vec<PathBuf> = roots_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();

    let max_files = parse_usize_env("RINGTAIL_MAX_FILES", 25_000, 1);
    let (text, docs) = load_corpus(&data_roots, max_files);
    let mut data = tokenizer::tokenize_bytes(&text);

    // ── hyperparameters ──────────────────────────────────────────────────────
    let vocab = 256;
    let model_dim = parse_usize_env("RINGTAIL_MODEL_DIM", 256, 32);
    let seq_len = parse_usize_env("RINGTAIL_SEQ_LEN", 128, 8);
    let learn_rate = parse_f32_env("RINGTAIL_LEARN_RATE", 0.001f32, 1e-6);
    let retrieve_top_k = parse_usize_env("RINGTAIL_RETRIEVE_TOP_K", 4, 1);
    let retrieve_chars = parse_usize_env("RINGTAIL_RETRIEVE_CHARS", 1200, 64);

    let mut epochs_if_loaded = 1usize;
    let mut epochs_if_fresh = 6usize;
    let mut batch_size = 32usize;
    let mut sample_stride = 2usize;

    // Environment variable overrides
    if let Ok(v) = std::env::var("RINGTAIL_SAMPLE_STRIDE") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                sample_stride = n;
            }
        }
    }
    if let Ok(v) = std::env::var("RINGTAIL_EPOCHS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                epochs_if_loaded = n;
                epochs_if_fresh = n;
            }
        }
    }
    if let Ok(v) = std::env::var("RINGTAIL_BATCH_SIZE") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                batch_size = n;
            }
        }
    }
    if let Ok(v) = std::env::var("RINGTAIL_MAX_TOKENS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 && data.len() > n {
                data = data[data.len() - n..].to_vec();
            }
        }
    }

    if data.len() < 16 {
        println!("Not enough training tokens.");
        return;
    }

    println!("Loaded .txt files: {}, roots: {:?}", docs.len(), data_roots);
    println!(
        "Vocab: {}, tokens: {}, model_dim: {}, seq_len: {}, ff_dim: {}, stride: {}, lr: {}, retrieve_top_k: {}, retrieve_chars: {}",
        vocab,
        data.len(),
        model_dim,
        seq_len,
        model_dim * 4,
        sample_stride,
        learn_rate,
        retrieve_top_k,
        retrieve_chars,
    );

    // ── model init + optional weight load ───────────────────────────────────
    let weights_file = "weights.bin";
    let mut model = ChatModel::new(vocab, model_dim, seq_len);

    let loaded = model.load_weights(weights_file);
    let epochs = if loaded {
        println!(
            "Loaded weights from {}. Continuing training for {} epoch(s).",
            weights_file, epochs_if_loaded
        );
        epochs_if_loaded
    } else {
        println!(
            "No weights found. Training from scratch for {} epoch(s).",
            epochs_if_fresh
        );
        epochs_if_fresh
    };

    // ── train ────────────────────────────────────────────────────────────────
    model.train(&data, epochs, learn_rate, batch_size, sample_stride);

    if model.save_weights(weights_file) {
        println!("Saved weights to {}.", weights_file);
    } else {
        println!("Warning: failed to save weights.");
    }

    // ── chat loop ────────────────────────────────────────────────────────────
    println!("\nChatbot ready. 🦝");
    println!("Type a message and press Enter.");
    println!("Commands:");
    println!("  /temp <value>   set temperature (e.g. /temp 0.7)");
    println!("  /det on|off     toggle deterministic mode");
    println!("  quit            exit\n");

    let stdin = io::stdin();
    let mut temperature = 1.0f64;
    let mut deterministic = false;

    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }
        let line = line
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string();

        if line == "quit" {
            break;
        }

        // /temp command
        if line.starts_with("/temp") {
            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                match parts[1].trim().parse::<f64>() {
                    Ok(t) if t > 0.0 => {
                        temperature = t;
                        println!("Bot: temperature set to {}", temperature);
                    }
                    _ => println!("Bot: invalid temperature. Use /temp <positive number>"),
                }
            }
            continue;
        }

        // /det command
        if line == "/det on" {
            deterministic = true;
            println!("Bot: deterministic mode ON");
            continue;
        }
        if line == "/det off" {
            deterministic = false;
            println!("Bot: deterministic mode OFF");
            continue;
        }

        // Generate reply with retrieval context from .txt corpus on every prompt.
        let prompt = build_prompt_with_retrieval(&line, &docs, retrieve_top_k, retrieve_chars);
        let mut ctx = tokenizer::tokenize_bytes(&prompt);
        if ctx.is_empty() {
            ctx.push(0);
        }

        let reply = model.generate(&ctx, 15, temperature, deterministic);
        println!("Bot: {}", reply);
    }
}
