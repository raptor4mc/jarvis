// tokenizer.rs
// Hybrid tokenizer tuned for Rust/code-heavy corpora.
// - Base vocab: byte tokens 0..255
// - Extra vocab: common Rust/code lexemes as single tokens

const BASE_VOCAB: usize = 256;

const RUST_TOKENS: [&str; 64] = [
    "fn ", "let ", "mut ", "pub ", "impl ", "struct ", "enum ", "trait ", "use ", "mod ",
    "crate::", "self", "Self", "->", "=>", "::", "&&", "||", "==", "!=", "<=", ">=", "//", "/*",
    "*/", "{", "}", "(", ")", "[", "]", "<", ">", "=", ";", ",", ".", ":", "\n", "\n\n",
    "String", "Vec", "Result", "Option", "Some", "None", "Ok", "Err", "match ", "if ", "else ",
    "for ", "while ", "loop ", "return ", "where ", "as ", "&str", "usize", "i32", "f32", "bool", "true", "false",
];

pub fn vocab_size() -> usize {
    BASE_VOCAB + RUST_TOKENS.len()
}

pub fn tokenize_bytes(s: &str) -> Vec<i32> {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let mut best: Option<(usize, usize)> = None; // (idx, len)
        for (idx, tok) in RUST_TOKENS.iter().enumerate() {
            let tb = tok.as_bytes();
            if i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb {
                if best.map(|(_, l)| tb.len() > l).unwrap_or(true) {
                    best = Some((idx, tb.len()));
                }
            }
        }

        if let Some((idx, len)) = best {
            out.push((BASE_VOCAB + idx) as i32);
            i += len;
        } else {
            out.push(bytes[i] as i32);
            i += 1;
        }
    }

    out
}

pub fn detokenize_bytes(tokens: &[i32]) -> String {
    let mut out = String::new();
    for &t in tokens {
        if (0..BASE_VOCAB as i32).contains(&t) {
            out.push(t as u8 as char);
        } else {
            let idx = (t as usize).saturating_sub(BASE_VOCAB);
            if idx < RUST_TOKENS.len() {
                out.push_str(RUST_TOKENS[idx]);
            }
        }
    }
    out
}
