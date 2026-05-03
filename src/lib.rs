use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter};
use std::sync::OnceLock;

const BYTE_VOCAB: u32 = 256;
const PUNCT_START: u32 = 256;
const MERGE_START: u32 = 512;
const MAX_VOCAB: usize = 32_768;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    InvalidToken(u32),
    InvalidUtf8,
}

impl Display for TokenizerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::InvalidToken(t) => write!(f, "invalid token id: {t}"),
            TokenizerError::InvalidUtf8 => write!(f, "decoded bytes are not valid UTF-8"),
        }
    }
}

#[derive(Default)]
struct TrieNode {
    next: HashMap<u8, usize>,
    token: Option<u32>,
}

struct Tokenizer {
    trie: Vec<TrieNode>,
    token_to_bytes: Vec<Vec<u8>>, // index by token id
    vocab_size: usize,
}

impl Tokenizer {
    fn new() -> Self {
        let punct = punct_table();
        let mut seeded = seeded_multi_tokens();
        seeded.extend(learned_merges_from_corpus());

        let mut token_to_bytes = vec![Vec::new(); MERGE_START as usize];
        for b in 0u8..=255u8 {
            token_to_bytes[b as usize] = vec![b];
        }
        for (ch, id) in punct.iter() {
            token_to_bytes[*id as usize] = vec![*ch as u8];
        }

        let mut next_id = MERGE_START;
        let mut seen = BTreeMap::<Vec<u8>, u32>::new();
        for t in seeded {
            let bytes = t.into_bytes();
            if bytes.len() < 2 { continue; }
            if seen.contains_key(&bytes) { continue; }
            if next_id as usize >= MAX_VOCAB { break; }
            seen.insert(bytes.clone(), next_id);
            token_to_bytes.push(bytes);
            next_id += 1;
        }

        let mut trie = vec![TrieNode::default()];
        for (bytes, id) in seen {
            insert_trie(&mut trie, &bytes, id);
        }

        Self { trie, token_to_bytes, vocab_size: next_id as usize }
    }

    fn longest_match(&self, bytes: &[u8], start: usize) -> Option<(u32, usize)> {
        let mut node = 0usize;
        let mut i = start;
        let mut best: Option<(u32, usize)> = None;
        while i < bytes.len() {
            match self.trie[node].next.get(&bytes[i]) {
                Some(&child) => {
                    node = child;
                    i += 1;
                    if let Some(tok) = self.trie[node].token { best = Some((tok, i - start)); }
                }
                None => break,
            }
        }
        best
    }
}

fn insert_trie(trie: &mut Vec<TrieNode>, bytes: &[u8], token: u32) {
    let mut node = 0usize;
    for &b in bytes {
        let nxt = if let Some(&idx) = trie[node].next.get(&b) { idx } else {
            trie.push(TrieNode::default());
            let idx = trie.len() - 1;
            trie[node].next.insert(b, idx);
            idx
        };
        node = nxt;
    }
    trie[node].token = Some(token);
}

fn punct_table() -> HashMap<char, u32> {
    let mut map = HashMap::new();
    let punct = [
        '{','}','(',')','[',']','<','>',';',':',',','.','=','+','-','*','/','%','&','|','^','!','?','@','#','$','\\','\'','"','`','~','\n','\r','\t',' '
    ];
    for (i, ch) in punct.into_iter().enumerate() {
        map.insert(ch, PUNCT_START + i as u32);
    }
    map
}

fn seeded_multi_tokens() -> Vec<String> {
    vec![
        "::","..=","=>","->","==","!=","<=",">=","&&","||","+=","-=","*=","/=","%=","async fn",".await","async move","if let","while let",
        "std::collections::HashMap","Vec<T>","Option<T>","Result<T, E>","collect::<Vec<_>>()","::<T>","#[derive(","#![","vec!","println!","assert_eq!","match","where T: Into<String>","T: Clone + Send","'static","'_","'a"
    ].into_iter().map(|s| s.to_string()).collect()
}

fn learned_merges_from_corpus() -> Vec<String> {
    let sample = include_str!("../README.md");
    let mut freq: HashMap<String, usize> = HashMap::new();
    for n in 2..=8 {
        for w in sample.as_bytes().windows(n) {
            if w.iter().all(|b| b.is_ascii_alphanumeric() || b"_<>:!?.=+-/()[]{} ,'\"".contains(b)) {
                let s = String::from_utf8_lossy(w).to_string();
                *freq.entry(s).or_insert(0) += 1;
            }
        }
    }
    let mut v: Vec<(String, usize)> = freq.into_iter().filter(|(_, c)| *c > 2).collect();
    v.sort_by(|a,b| b.1.cmp(&a.1).then_with(|| b.0.len().cmp(&a.0.len())));
    v.into_iter().take(6000).map(|(s,_)| s).collect()
}

fn tokenizer() -> &'static Tokenizer { static T: OnceLock<Tokenizer> = OnceLock::new(); T.get_or_init(Tokenizer::new) }

pub fn encode(input: &str) -> Vec<u32> {
    let t = tokenizer();
    let bytes = input.as_bytes();
    let punct = punct_table();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if let Some((tok, len)) = t.longest_match(bytes, i) {
            out.push(tok);
            i += len;
            continue;
        }
        // full untrimmed line advance for comments bugfix compliance
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i+1] == b'/' {
            let mut j = i + 2;
            while j < bytes.len() && bytes[j] != b'\n' { j += 1; }
            while i < j { out.push(bytes[i] as u32); i += 1; }
            continue;
        }
        let b = bytes[i];
        if let Some(id) = punct.get(&(b as char)) { out.push(*id); } else { out.push(b as u32); }
        i += 1;
    }
    out
}

pub fn decode(tokens: &[u32]) -> Result<String, TokenizerError> {
    let t = tokenizer();
    let mut bytes = Vec::new();
    for &tok in tokens {
        if tok < BYTE_VOCAB { bytes.push(tok as u8); continue; }
        if let Some(v) = t.token_to_bytes.get(tok as usize) { bytes.extend_from_slice(v); }
        else { return Err(TokenizerError::InvalidToken(tok)); }
    }
    String::from_utf8(bytes).map_err(|_| TokenizerError::InvalidUtf8)
}

pub fn vocab_size() -> usize { tokenizer().vocab_size }

#[cfg(test)]
mod tests {
use super::*;

#[test] fn round_trip_bytes() { let s = "a\0b\x01c\n\t\r"; let t = encode(s); assert_eq!(decode(&t).unwrap(), s); }
#[test] fn lifetimes() { let s = "fn f<'a>(x: &'a str) -> &'static str { x }"; assert_eq!(decode(&encode(s)).unwrap(), s); }
#[test] fn raw_strings() { let s = "let a = r#\"hi\"#; let b = r##\"yo\"##;"; assert_eq!(decode(&encode(s)).unwrap(), s); }
#[test] fn turbofish() { let s = "xs.iter().collect::<Vec<_>>()"; assert_eq!(decode(&encode(s)).unwrap(), s); }
#[test] fn macros_attrs() { let s = "#[derive(Debug)]\n#![allow(dead_code)]\nprintln!(\"x\");"; assert_eq!(decode(&encode(s)).unwrap(), s); }
#[test] fn comment_leading_ws() { let s = "//    TODO: keep\nlet x=1;"; assert_eq!(decode(&encode(s)).unwrap(), s); }
#[test] fn repeated_comma_segments() { let s = "Foo<A, A, A>"; assert_eq!(decode(&encode(s)).unwrap(), s); }
}
