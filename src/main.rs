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

mod model;
mod tokenizer;

use model::ChatModel;
use std::io::{self, BufRead, Write};
use std::fs;

fn main() {
    // ── training text ────────────────────────────────────────────────────────
    let mut text = String::from(
        "hello there i am an offline chatbot created to demonstrate how a small neural network can learn patterns from text \
         i do not use the internet and i do not rely on external data everything i know is written inside this program \
         you can ask simple questions and i will try to respond based on the patterns i learned during training \
         my goal is not to be perfect but to show how a tiny model can generate sentences and imitate conversation \
         the more text i am trained on the better i become at forming coherent replies and continuing your messages \
         this dataset is small but it teaches basic structure like greetings statements and simple explanations \
         feel free to experiment with different prompts and see how the model reacts to your input \
         remember that this chatbot is only a demonstration and not a full language model but it can still be fun to interact with "
    );

    // Load .txt files from current directory (equivalent to wikipedia.txt, greeting.txt, general.txt)
    let training_files = ["wikipedia.txt", "greeting.txt", "general.txt"];
    for path in &training_files {
        if let Ok(contents) = fs::read_to_string(path) {
            text.push('\n');
            text.push_str(&contents);
        }
    }

    // Also load any .txt file found in current directory automatically
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Skip the ones we already loaded
                if !training_files.contains(&name) {
                    if let Ok(contents) = fs::read_to_string(&path) {
                        println!("Loading training file: {}", name);
                        text.push('\n');
                        text.push_str(&contents);
                    }
                }
            }
        }
    }

    let mut data = tokenizer::tokenize_bytes(&text);

    // ── hyperparameters ──────────────────────────────────────────────────────
    let vocab      = 256;
    let model_dim  = 128;
    let seq_len    = 32;
    let learn_rate = 0.001f32;

    let mut epochs_if_loaded = 1usize;
    let mut epochs_if_fresh  = 6usize;
    let mut batch_size       = 32usize;
    let mut sample_stride    = 2usize;

    // Environment variable overrides
    if let Ok(v) = std::env::var("RINGTAIL_SAMPLE_STRIDE") {
        if let Ok(n) = v.parse::<usize>() { if n > 0 { sample_stride = n; } }
    }
    if let Ok(v) = std::env::var("RINGTAIL_EPOCHS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 { epochs_if_loaded = n; epochs_if_fresh = n; }
        }
    }
    if let Ok(v) = std::env::var("RINGTAIL_BATCH_SIZE") {
        if let Ok(n) = v.parse::<usize>() { if n > 0 { batch_size = n; } }
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

    println!(
        "Vocab: {}, tokens: {}, model_dim: {}, seq_len: {}, ff_dim: {}, stride: {}",
        vocab, data.len(), model_dim, seq_len, model_dim * 4, sample_stride
    );

    // ── model init + optional weight load ───────────────────────────────────
    let weights_file = "weights.bin";
    let mut model = ChatModel::new(vocab, model_dim, seq_len);

    let loaded = model.load_weights(weights_file);
    let epochs = if loaded {
        println!("Loaded weights from {}. Continuing training for {} epoch(s).", weights_file, epochs_if_loaded);
        epochs_if_loaded
    } else {
        println!("No weights found. Training from scratch for {} epoch(s).", epochs_if_fresh);
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
        if stdin.lock().read_line(&mut line).is_err() { break; }
        let line = line.trim_end_matches('\n').trim_end_matches('\r').to_string();

        if line == "quit" { break; }

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

        // Generate reply
        let mut ctx = tokenizer::tokenize_bytes(&line);
        if ctx.is_empty() { ctx.push(0); }

        let reply = model.generate(&ctx, 15, temperature, deterministic);
        println!("Bot: {}", reply);
    }
}
