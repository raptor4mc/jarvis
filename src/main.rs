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
//   RINGTAIL_MODEL_DIM       override model dimension (default: 600)
//   RINGTAIL_SEQ_LEN         override context length (default: 1600)
//   RINGTAIL_LEARN_RATE      override learning rate (default: 0.001)
//   RINGTAIL_RETRIEVE_TOP_K  files retrieved per prompt (default: 4)
//   RINGTAIL_RETRIEVE_CHARS  max chars per retrieved file snippet (default: 1200)
//   RINGTAIL_SCAFFOLD_TOKENS generation tokens for /draft and /scaffold (default: 900)
//   RINGTAIL_REPLY_TOKENS_MIN min generation tokens for normal chat (default: 24)
//   RINGTAIL_REPLY_TOKENS_MAX max generation tokens for normal chat (default: 192)
//   RINGTAIL_MAX_FILE_BYTES    max bytes to read per .txt file (default: 1_000_000)
//   RINGTAIL_DATA_INCLUDE      optional comma-separated substrings a path must include
//   RINGTAIL_DATA_EXCLUDE      optional comma-separated substrings a path must NOT include
//   RINGTAIL_BASE_MODEL_CMD    optional command for stronger inference engine (reads $RINGTAIL_PROMPT)

mod model;
mod tokenizer;

use model::ChatModel;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

#[derive(Clone)]
struct ExtractedZip {
    zip_path: PathBuf,
    extract_dir: PathBuf,
}

fn should_skip_zip_walk_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|name| name == ".jarvis_unzipped" || name == "target" || name == ".git")
        .unwrap_or(false)
}

fn collect_zip_files(root: &Path, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }

    if root.is_file() {
        if root.extension().and_then(|e| e.to_str()) == Some("zip") {
            out.push(root.to_path_buf());
        }
        return;
    }

    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if should_skip_zip_walk_dir(&path) {
                    continue;
                }
                collect_zip_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("zip") {
                out.push(path);
            }
        }
    }
}

fn prepare_zip_sources(data_roots: &mut Vec<PathBuf>) -> Vec<ExtractedZip> {
    let mut zip_files = Vec::new();
    for root in data_roots.iter() {
        collect_zip_files(root, &mut zip_files);
    }
    let mut unique_canonical = HashSet::new();
    zip_files = zip_files
        .into_iter()
        .filter(|p| unique_canonical.insert(p.canonicalize().unwrap_or(p.clone())))
        .collect();
    zip_files.sort();

    if zip_files.is_empty() {
        return Vec::new();
    }

    let base = PathBuf::from(".jarvis_unzipped");
    let _ = fs::create_dir_all(&base);

    let mut extracted = Vec::new();
    for (idx, zip_path) in zip_files.into_iter().enumerate() {
        let canonical = zip_path.canonicalize().unwrap_or(zip_path.clone());
        let extract_dir = base.join(format!("zip_{}", idx));
        let _ = fs::create_dir_all(&extract_dir);

        let stamp_file = extract_dir.join(".jarvis_zip_stamp");
        let zip_mtime = fs::metadata(&canonical)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let stamp_mtime = fs::metadata(&stamp_file)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let needs_extract = !extract_dir.exists() || zip_mtime > stamp_mtime;

        if !needs_extract {
            println!(
                "Zip unchanged, reusing extracted dir {}",
                extract_dir.display()
            );
            data_roots.push(extract_dir.clone());
            extracted.push(ExtractedZip {
                zip_path: canonical,
                extract_dir,
            });
            continue;
        }

        let status = Command::new("unzip")
            .arg("-o")
            .arg(&canonical)
            .arg("-d")
            .arg(&extract_dir)
            .status();

        match status {
            Ok(s) if s.success() => {
                if needs_extract {
                    let _ = fs::write(&stamp_file, b"ok");
                }
                println!(
                    "Unzipped {} -> {}",
                    canonical.display(),
                    extract_dir.display()
                );
                data_roots.push(extract_dir.clone());
                extracted.push(ExtractedZip {
                    zip_path: canonical,
                    extract_dir,
                });
            }
            Ok(_) => {
                println!("Warning: failed to unzip {}", canonical.display());
            }
            Err(e) => {
                println!(
                    "Warning: unzip command error for {}: {}",
                    canonical.display(),
                    e
                );
            }
        }
    }

    extracted
}

fn rezip_sources(extracted: &[ExtractedZip]) {
    for entry in extracted {
        let stamp_file = entry.extract_dir.join(".jarvis_zip_stamp");
        let extract_mtime = fs::metadata(&entry.extract_dir)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let zip_mtime = fs::metadata(&entry.zip_path)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if zip_mtime >= extract_mtime {
            continue;
        }
        let status = Command::new("zip")
            .arg("-qr")
            .arg(&entry.zip_path)
            .arg(".")
            .current_dir(&entry.extract_dir)
            .status();

        match status {
            Ok(s) if s.success() => {
                let _ = fs::write(&stamp_file, b"ok");
                println!(
                    "Rezipped {} from {}",
                    entry.zip_path.display(),
                    entry.extract_dir.display()
                )
            }
            Ok(_) => println!("Warning: failed to rezip {}", entry.zip_path.display()),
            Err(e) => println!(
                "Warning: zip command error for {}: {}",
                entry.zip_path.display(),
                e
            ),
        }
    }
}

#[derive(Debug)]
struct ToolRunResult {
    name: &'static str,
    ok: bool,
    status: i32,
    stdout: String,
    stderr: String,
}

fn run_tool_command(name: &'static str, args: &[&str]) -> ToolRunResult {
    match Command::new("cargo").args(args).output() {
        Ok(out) => ToolRunResult {
            name,
            ok: out.status.success(),
            status: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        },
        Err(e) => ToolRunResult {
            name,
            ok: false,
            status: -1,
            stdout: String::new(),
            stderr: format!("failed to run cargo {}: {}", args.join(" "), e),
        },
    }
}

fn run_dev_loop(iterations: usize) {
    println!("Bot: starting dev loop (max {} iteration(s))", iterations);
    for i in 1..=iterations {
        println!("Bot: iteration {}", i);
        let steps = [
            run_tool_command("fmt", &["fmt", "--all"]),
            run_tool_command("check", &["check", "--all-targets"]),
            run_tool_command("test", &["test", "--all-targets"]),
            run_tool_command(
                "clippy",
                &["clippy", "--all-targets", "--", "-D", "warnings"],
            ),
        ];

        let mut all_ok = true;
        for step in &steps {
            if step.ok {
                println!("  ✅ {}", step.name);
            } else {
                all_ok = false;
                println!("  ❌ {} (exit code: {})", step.name, step.status);
                if !step.stdout.trim().is_empty() {
                    println!("  stdout:\n{}", step.stdout);
                }
                if !step.stderr.trim().is_empty() {
                    println!("  stderr:\n{}", step.stderr);
                }
                break;
            }
        }

        if all_ok {
            println!("Bot: dev loop is green.");
            return;
        }
    }
    println!("Bot: dev loop reached max iterations without going green.");
}

fn run_foundation_model(prompt: &str) -> Option<String> {
    let cmd = std::env::var("RINGTAIL_BASE_MODEL_CMD").ok()?;
    if cmd.trim().is_empty() {
        return None;
    }
    let out = Command::new("sh")
        .arg("-lc")
        .arg(cmd)
        .env("RINGTAIL_PROMPT", prompt)
        .output()
        .ok()?;
    if !out.status.success() {
        eprintln!(
            "Warning: foundation model command failed with status {:?}",
            out.status.code()
        );
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

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

fn parse_csv_env(name: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn path_allowed(path: &Path, includes: &[String], excludes: &[String]) -> bool {
    let p = path.to_string_lossy().to_lowercase();
    if !includes.is_empty() && !includes.iter().any(|needle| p.contains(needle)) {
        return false;
    }
    !excludes.iter().any(|needle| p.contains(needle))
}

fn looks_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let sample = &bytes[..bytes.len().min(4096)];
    let bad = sample
        .iter()
        .filter(|&&b| b == 0 || (b < 0x09) || (b > 0x0D && b < 0x20))
        .count();
    bad * 100 / sample.len() > 5
}

fn load_corpus(
    data_roots: &[PathBuf],
    max_files: usize,
    max_file_bytes: usize,
) -> (String, Vec<CorpusDoc>) {
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
    let mut seen_content = HashSet::new();
    let includes = parse_csv_env("RINGTAIL_DATA_INCLUDE");
    let excludes = parse_csv_env("RINGTAIL_DATA_EXCLUDE");
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

        if !path_allowed(&canonical, &includes, &excludes) {
            continue;
        }

        if let Ok(raw) = fs::read(&canonical) {
            if raw.is_empty() || looks_binary(&raw) {
                continue;
            }
            let clipped = if raw.len() > max_file_bytes {
                &raw[..max_file_bytes]
            } else {
                &raw
            };
            let contents = String::from_utf8_lossy(clipped).to_string();
            let mut content_hasher = DefaultHasher::new();
            contents.hash(&mut content_hasher);
            if !seen_content.insert(content_hasher.finish()) {
                continue;
            }
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

    append_structure_tokens(&mut text, &docs);

    (text, docs)
}

fn corpus_fingerprint(docs: &[CorpusDoc]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for doc in docs {
        doc.path.hash(&mut hasher);
        doc.content.hash(&mut hasher);
    }
    hasher.finish()
}

fn read_corpus_fingerprint(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

fn write_corpus_fingerprint(path: &Path, fp: u64) {
    let _ = fs::write(path, fp.to_string());
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

fn parse_bool_env(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != ':')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s.len() >= 3)
        .collect()
}

fn char_trigrams(s: &str) -> HashSet<String> {
    let compact: String = s
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_ascii_whitespace() || *c == '_')
        .collect();
    let chars: Vec<char> = compact.chars().collect();
    if chars.len() < 3 {
        return HashSet::new();
    }
    let mut grams = HashSet::new();
    for i in 0..=chars.len() - 3 {
        grams.insert(chars[i..i + 3].iter().collect());
    }
    grams
}

fn jaccard_score(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
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

    #[derive(Clone)]
    struct RetrievalChunk {
        path: String,
        boundary: String,
        content: String,
        content_lc: String,
    }

    fn chunk_doc(doc: &CorpusDoc, max_chars: usize) -> Vec<RetrievalChunk> {
        let mut starts = vec![0usize];
        for (idx, _) in doc.content.match_indices("\nfn ") {
            starts.push(idx + 1);
        }
        for (idx, _) in doc.content.match_indices("\nimpl ") {
            starts.push(idx + 1);
        }
        for (idx, _) in doc.content.match_indices("\nmod ") {
            starts.push(idx + 1);
        }
        starts.sort_unstable();
        starts.dedup();

        let mut chunks = Vec::new();
        for w in starts.windows(2) {
            let s = w[0];
            let e = w[1];
            if s >= e || s >= doc.content.len() {
                continue;
            }
            let raw = &doc.content[s..e];
            let snippet: String = raw.chars().take(max_chars.max(96)).collect();
            let boundary = raw.lines().next().unwrap_or("").trim().to_string();
            chunks.push(RetrievalChunk {
                path: doc.path.clone(),
                boundary,
                content_lc: snippet.to_lowercase(),
                content: snippet,
            });
        }
        if let Some(&s) = starts.last() {
            let raw = &doc.content[s..];
            let snippet: String = raw.chars().take(max_chars.max(96)).collect();
            chunks.push(RetrievalChunk {
                path: doc.path.clone(),
                boundary: raw.lines().next().unwrap_or("").trim().to_string(),
                content_lc: snippet.to_lowercase(),
                content: snippet,
            });
        }
        if chunks.is_empty() {
            chunks.push(RetrievalChunk {
                path: doc.path.clone(),
                boundary: "file".to_string(),
                content: doc.content.chars().take(max_chars.max(96)).collect(),
                content_lc: doc.content_lc.chars().take(max_chars.max(96)).collect(),
            });
        }
        chunks
    }

    let query_grams = char_trigrams(user_input);
    let all_chunks: Vec<RetrievalChunk> = docs
        .iter()
        .flat_map(|d| chunk_doc(d, max_chars.saturating_mul(2)))
        .collect();
    let mut scored: Vec<(usize, f32)> = all_chunks
        .iter()
        .enumerate()
        .filter_map(|(i, ch)| {
            let path_lc = ch.path.to_lowercase();
            let lexical = terms
                .iter()
                .map(|t| {
                    ch.content_lc.matches(t).count()
                        + path_lc.matches(t).count() * 2
                        + ch.boundary.to_lowercase().matches(t).count() * 3
                })
                .sum::<usize>();
            let grams = char_trigrams(&(ch.path.clone() + " " + &ch.content_lc));
            let semantic = jaccard_score(&query_grams, &grams);
            let score = lexical as f32 + (semantic * 30.0);
            (score > 0.0).then_some((i, score))
        })
        .collect();

    if scored.is_empty() {
        return user_input.to_string();
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut prompt = String::new();
    prompt.push_str("<retrieved_context>\n");
    for (idx, score) in scored.into_iter().take(top_k.max(1)) {
        let ch = &all_chunks[idx];
        prompt.push_str("<context_file path=\"");
        prompt.push_str(&ch.path);
        prompt.push_str("\" boundary=\"");
        prompt.push_str(&ch.boundary);
        prompt.push_str("\" score=\"");
        prompt.push_str(&format!("{:.3}", score));
        prompt.push_str("\">\n");
        prompt.push_str(&ch.content);
        prompt.push_str("\n</context_file>\n");
    }
    prompt.push_str("</retrieved_context>\n");
    prompt.push_str("<user_prompt>\n");
    prompt.push_str(user_input);
    prompt.push_str("\n</user_prompt>");
    prompt
}

fn build_scaffold_prompt(task: &str, docs: &[CorpusDoc], top_k: usize, max_chars: usize) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are ferris, a Rust coding agent. Output ONLY this format:\n[[DIR:path]]\n[[FILE:path]]\n<rust code or text>\n[[END_FILE]]\nAlways generate compiling Rust code with Cargo.toml, src/main.rs or src/lib.rs, and required modules.\n",
    );
    prompt.push_str("Task:\n");
    prompt.push_str(task);
    prompt.push_str("\n");
    prompt.push_str(&build_prompt_with_retrieval(task, docs, top_k, max_chars));
    prompt
}

fn sanitize_rel_path(raw: &str) -> Option<PathBuf> {
    let candidate = Path::new(raw.trim());
    if candidate.as_os_str().is_empty() || candidate.is_absolute() {
        return None;
    }

    let mut out = PathBuf::new();
    for c in candidate.components() {
        match c {
            Component::Normal(part) => out.push(part),
            Component::CurDir => {}
            _ => return None,
        }
    }
    if out.as_os_str().is_empty() {
        None
    } else {
        Some(out)
    }
}

fn parse_scaffold_output(spec: &str) -> (Vec<PathBuf>, Vec<(PathBuf, String)>) {
    let mut dirs = Vec::new();
    let mut files = Vec::new();

    let mut current_file: Option<PathBuf> = None;
    let mut current_content = String::new();

    for line in spec.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("[[DIR:") && trimmed.ends_with("]]") {
            if let Some(path) = sanitize_rel_path(&trimmed[6..trimmed.len() - 2]) {
                dirs.push(path);
            }
            continue;
        }

        if trimmed.starts_with("[[FILE:") && trimmed.ends_with("]]") {
            if let Some(path) = current_file.take() {
                let content = current_content.trim().to_string();
                if !content.is_empty() {
                    files.push((path, content));
                }
                current_content.clear();
            }
            current_file = sanitize_rel_path(&trimmed[7..trimmed.len() - 2]);
            continue;
        }

        if trimmed == "[[END_FILE]]" {
            if let Some(path) = current_file.take() {
                let content = current_content.trim().to_string();
                if !content.is_empty() {
                    files.push((path, content));
                }
                current_content.clear();
            }
            continue;
        }

        if current_file.is_some() {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    if let Some(path) = current_file.take() {
        let content = current_content.trim().to_string();
        if !content.is_empty() {
            files.push((path, content));
        }
    }

    (dirs, files)
}

fn apply_scaffold(spec: &str, out_root: &Path) -> io::Result<(usize, usize, usize)> {
    let (dirs, files) = parse_scaffold_output(spec);

    fs::create_dir_all(out_root)?;

    let mut created_dirs = 0usize;
    for dir in dirs {
        let full = out_root.join(dir);
        fs::create_dir_all(&full)?;
        created_dirs += 1;
    }

    let mut created_files = 0usize;
    let mut skipped_files = 0usize;
    for (rel_path, content) in files {
        if rel_path.extension().is_none() {
            skipped_files += 1;
            continue;
        }
        let full = out_root.join(&rel_path);
        if full.exists() && full.is_dir() {
            skipped_files += 1;
            continue;
        }
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&full, content)?;
        created_files += 1;
    }

    Ok((created_dirs, created_files, skipped_files))
}

fn main() {
    // ── data loading ────────────────────────────────────────────────────────
    let roots_env =
        std::env::var("RINGTAIL_DATA_ROOTS").unwrap_or_else(|_| "knowledge,.".to_string());
    let mut data_roots: Vec<PathBuf> = roots_env
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();

    let extracted_zip_sources = prepare_zip_sources(&mut data_roots);

    let max_files = parse_usize_env("RINGTAIL_MAX_FILES", 25_000, 1);
    let max_file_bytes = parse_usize_env("RINGTAIL_MAX_FILE_BYTES", 1_000_000, 1024);
    let (text, docs) = load_corpus(&data_roots, max_files, max_file_bytes);
    let mut data = tokenizer::tokenize_bytes(&text);

    // ── hyperparameters ──────────────────────────────────────────────────────
    let vocab = tokenizer::vocab_size();
    let model_dim = parse_usize_env("RINGTAIL_MODEL_DIM", 600, 32);
    let seq_len = parse_usize_env("RINGTAIL_SEQ_LEN", 1600, 8);
    let learn_rate = parse_f32_env("RINGTAIL_LEARN_RATE", 0.001f32, 1e-6);
    let retrieve_top_k = parse_usize_env("RINGTAIL_RETRIEVE_TOP_K", 4, 1);
    let retrieve_chars = parse_usize_env("RINGTAIL_RETRIEVE_CHARS", 1200, 64);
    let scaffold_tokens = parse_usize_env("RINGTAIL_SCAFFOLD_TOKENS", 900, 64);
    let reply_tokens_min = parse_usize_env("RINGTAIL_REPLY_TOKENS_MIN", 24, 8);
    let reply_tokens_max = parse_usize_env("RINGTAIL_REPLY_TOKENS_MAX", 192, reply_tokens_min);

    let mut epochs_if_loaded = 3usize;
    let mut epochs_if_fresh = 24usize;
    let mut batch_size = 32usize;
    let mut sample_stride = 1usize;

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
    // Auto-scale epochs for larger runs unless explicitly overridden.
    if std::env::var("RINGTAIL_EPOCHS").is_err() {
        let token_scale = (data.len() / 100_000).max(1);
        let adaptive = (token_scale * 8).clamp(8, 80);
        epochs_if_fresh = epochs_if_fresh.max(adaptive);
        epochs_if_loaded = epochs_if_loaded.max((adaptive / 4).max(2));
    }

    if data.len() < 16 {
        println!("Not enough training tokens.");
        return;
    }

    println!("Loaded .txt files: {}, roots: {:?}", docs.len(), data_roots);
    println!(
        "Vocab: {}, tokens: {}, model_dim: {}, seq_len: {}, ff_dim: {}, stride: {}, lr: {}, retrieve_top_k: {}, retrieve_chars: {}, scaffold_tokens: {}",
        vocab,
        data.len(),
        model_dim,
        seq_len,
        model_dim * 4,
        sample_stride,
        learn_rate,
        retrieve_top_k,
        retrieve_chars,
        scaffold_tokens,
    );

    // ── model init + optional weight load ───────────────────────────────────
    let weights_file = "weights.bin";
    let mut model = ChatModel::new(vocab, model_dim, seq_len);

    let loaded = model.load_weights(weights_file);
    let weights_meta = PathBuf::from("weights.meta");
    let current_fingerprint = corpus_fingerprint(&docs);
    let last_fingerprint = read_corpus_fingerprint(&weights_meta);
    let corpus_changed = last_fingerprint != Some(current_fingerprint);
    let force_train = parse_bool_env("RINGTAIL_FORCE_TRAIN", false);

    let epochs = if loaded && !corpus_changed && !force_train {
        println!(
            "Weights loaded and corpus unchanged (fingerprint {}). Skipping retraining for faster startup.",
            current_fingerprint
        );
        0
    } else if loaded {
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
    if epochs > 0 {
        model.train(&data, epochs, learn_rate, batch_size, sample_stride);
    } else {
        println!("Skipping training (0 epochs requested).");
    }

    rezip_sources(&extracted_zip_sources);

    if model.save_weights(weights_file) {
        println!("Saved weights to {}.", weights_file);
        write_corpus_fingerprint(&weights_meta, current_fingerprint);
    } else {
        println!("Warning: failed to save weights.");
    }

    // ── chat loop ────────────────────────────────────────────────────────────
    println!("\nChatbot ready. 🦝");
    println!("Type a message and press Enter.");
    println!("Commands:");
    println!("  /temp <value>      set temperature (e.g. /temp 0.7)");
    println!("  /det on|off        toggle deterministic mode");
    println!("  /draft <task>      generate project layout + files (text only)");
    println!("  /scaffold <task>   generate and write files to generated/");
    println!("  /devloop [n]       run cargo fmt/check/test/clippy loop");
    println!("  quit               exit\n");

    let stdin = io::stdin();
    let mut temperature = 0.85f64;
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

        if line.starts_with("/devloop") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let iterations = if parts.len() > 1 {
                parts[1]
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n > 0)
                    .unwrap_or(1)
            } else {
                1
            };
            run_dev_loop(iterations);
            continue;
        }

        if let Some(task) = line.strip_prefix("/draft ") {
            let prompt = build_scaffold_prompt(task.trim(), &docs, retrieve_top_k, retrieve_chars);
            let draft = run_foundation_model(&prompt).unwrap_or_else(|| {
                let mut ctx = tokenizer::tokenize_bytes(&prompt);
                if ctx.is_empty() {
                    ctx.push(0);
                }
                model.generate(&ctx, scaffold_tokens, temperature, deterministic)
            });
            println!("Bot draft:\n{}", draft);
            continue;
        }

        if let Some(task) = line.strip_prefix("/scaffold ") {
            let prompt = build_scaffold_prompt(task.trim(), &docs, retrieve_top_k, retrieve_chars);
            let draft = run_foundation_model(&prompt).unwrap_or_else(|| {
                let mut ctx = tokenizer::tokenize_bytes(&prompt);
                if ctx.is_empty() {
                    ctx.push(0);
                }
                model.generate(&ctx, scaffold_tokens, temperature, deterministic)
            });
            match apply_scaffold(&draft, Path::new("generated")) {
                Ok((d, f, skipped)) => {
                    println!(
                        "Bot: wrote scaffold to ./generated (dirs: {}, files: {}, skipped: {})",
                        d, f, skipped
                    );
                }
                Err(e) => {
                    println!("Bot: failed to write scaffold: {}", e);
                }
            }
            continue;
        }

        // Generate reply with retrieval context from .txt corpus on every prompt.
        let prompt = build_prompt_with_retrieval(&line, &docs, retrieve_top_k, retrieve_chars);
        let reply = run_foundation_model(&prompt).unwrap_or_else(|| {
            let mut ctx = tokenizer::tokenize_bytes(&prompt);
            if ctx.is_empty() {
                ctx.push(0);
            }

            let dynamic_tokens =
                ((line.len() / 8).clamp(reply_tokens_min, reply_tokens_max)) as usize;
            model.generate(&ctx, dynamic_tokens, temperature, deterministic)
        });
        println!("Bot: {}", reply);
    }
}
