// tokenizer.rs
// Direct port of tokenizer.h / tokenizer.cpp
// Byte-level tokenizer: each byte becomes one token (vocab size = 256)

pub fn tokenize_bytes(s: &str) -> Vec<i32> {
    s.bytes().map(|b| b as i32).collect()
}

pub fn detokenize_bytes(tokens: &[i32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .filter(|&&t| t >= 0 && t <= 255)
        .map(|&t| t as u8)
        .collect();
    // Use lossy conversion so invalid UTF-8 sequences don't panic
    String::from_utf8_lossy(&bytes).into_owned()
}
