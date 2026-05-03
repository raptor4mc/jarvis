// tokenizer.rs
// Hybrid tokenizer tuned for Rust/code-heavy corpora.
// - Base vocab: byte tokens 0..255
// - Extra vocab: common Rust/code lexemes as single tokens

const BASE_VOCAB: usize = 256;

const RUST_TOKENS: [&str; 202] = [
    "fn ",
    "let ",
    "mut ",
    "pub ",
    "impl ",
    "struct ",
    "enum ",
    "trait ",
    "use ",
    "mod ",
    "crate::",
    "tokio::",
    "std::",
    "core::",
    "alloc::",
    "futures::",
    "serde::",
    "hyper::",
    "self",
    "Self",
    "->",
    "=>",
    "::",
    "&&",
    "||",
    "==",
    "!=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "<=",
    ">=",
    "//",
    "///",
    "//!",
    "/**",
    "/*",
    "*/",
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    "<",
    ">",
    "=",
    ";",
    ",",
    ".",
    ":",
    "\n",
    "\n\n",
    "String",
    "Vec",
    "Result",
    "Option",
    "Some",
    "None",
    "Ok",
    "Err",
    "match ",
    "if ",
    "else ",
    "for ",
    "while ",
    "loop ",
    "return ",
    "where ",
    "as ",
    "&str",
    "&mut",
    "&self",
    "&mut self",
    "usize",
    "0usize",
    "1usize",
    "i32",
    "f32",
    "0_f32",
    "bool",
    "true",
    "false",
    "u8",
    "u16",
    "u32",
    "0u32",
    "u64",
    "1u64",
    "u128",
    "i8",
    "i16",
    "i64",
    "i128",
    "f64",
    "1_f64",
    "isize",
    "char",
    "str",
    "0.0",
    "1.0",
    "async ",
    "await",
    "move ",
    "const ",
    "static ",
    "type ",
    "dyn ",
    "ref ",
    "unsafe ",
    "extern ",
    "super::",
    "pub(crate)",
    "pub(super)",
    "pub(self)",
    "impl<",
    "Box<dyn",
    "impl Future<Output =",
    "Pin<Box<dyn Future<Output =",
    "trait ",
    "derive",
    "#[",
    "]\n",
    "match",
    " if ",
    " else",
    "Some(",
    "Some::<_>",
    "Ok(",
    "Ok::<_, _>",
    "Err(",
    "Err::<_, _>",
    "None",
    "None::<_>",
    "Result<(), ()>",
    "Option<()>",
    "async move {",
    "match {",
    "match {}",
    "Vec<",
    "String::",
    "format!",
    "println!",
    "vec!",
    "macro_rules!",
    "todo!",
    "unimplemented!",
    "assert!",
    "assert_eq!",
    "assert_ne!",
    "dbg!",
    "Result<",
    "Option<",
    "Box<",
    "Arc<",
    "Rc<",
    "HashMap<",
    "HashSet<",
    "BTreeMap<",
    "BTreeSet<",
    "insert(",
    "push(",
    "len()",
    "iter()",
    "iter_mut()",
    "collect()",
    ".map(",
    ".filter(",
    ".unwrap()",
    ".expect(",
    "use ",
    "mod ",
    "crate",
    "self::",
    "Self::",
    " where ",
    ";\n",
    ",\n",
    ", ",
    " => ",
    " -> ",
    " )",
    "( ",
    " ->\n",
    "<T>",
    "<T, U>",
    "<'a>",
    "<'a, T>",
    "<'a, 'b>",
    "'a",
    "'static",
    "'_",
    "::",
    " ",
    "  ",
    "    ",
    "\t",
];

const IDENT_TOKENS: [&str; 36] = [
    "tokio::sync::mpsc::channel",
    "tokio::sync::mpsc",
    "tokio::spawn",
    "std::collections::HashMap",
    "std::collections::HashSet",
    "std::collections::BTreeMap",
    "std::collections::BTreeSet",
    "std::sync::Arc",
    "std::rc::Rc",
    "my_variable_name",
    "my_function_call",
    "crate::module::submodule::Type",
    "'a",
    "'static",
    "println!",
    "format!",
    "vec!",
    "todo!",
    "debug_assert!",
    "assert_eq!",
    "assert_ne!",
    "mod.rs",
    "Cargo.toml",
    "src/main.rs",
    "src/lib.rs",
    "Result<T, E>",
    "Option<T>",
    "Vec<T>",
    "HashMap<K, V>",
    "HashSet<T>",
    "BTreeMap<K, V>",
    "BTreeSet<T>",
    "String::new",
    "Vec::new",
    "Iterator::collect",
    "Self::new",
];

const IDENT_SUBWORD_TOKENS: [&str; 40] = [
    "my", "super", "long", "variable", "name", "with", "generics", "async", "await", "result",
    "option", "future", "stream", "sender", "receiver", "channel", "token", "model", "train",
    "test", "check", "build", "config", "state", "value", "index", "count", "error", "parse",
    "format", "request", "response", "client", "server", "module", "path", "type", "data", "cache",
    "buffer",
];
const NUM_SUFFIX_TOKENS: [&str; 13] = [
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
];

pub fn vocab_size() -> usize {
    BASE_VOCAB
        + RUST_TOKENS.len()
        + IDENT_TOKENS.len()
        + IDENT_SUBWORD_TOKENS.len()
        + NUM_SUFFIX_TOKENS.len()
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

fn split_identifier_chunks(ident: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = ident.as_bytes();

    for i in 1..bytes.len() {
        let prev = bytes[i - 1];
        let cur = bytes[i];
        let boundary = cur == b'_'
            || prev == b'_'
            || (prev.is_ascii_lowercase() && cur.is_ascii_uppercase())
            || (prev.is_ascii_alphabetic() && cur.is_ascii_digit())
            || (prev.is_ascii_digit() && cur.is_ascii_alphabetic());
        if boundary {
            if start < i {
                out.push(&ident[start..i]);
            }
            start = i;
        }
    }
    if start < ident.len() {
        out.push(&ident[start..]);
    }
    out
}

fn emit_identifier_token(ident: &str, out: &mut Vec<i32>) {
    if let Some(idx) = IDENT_TOKENS.iter().position(|t| *t == ident) {
        out.push((BASE_VOCAB + RUST_TOKENS.len() + idx) as i32);
        return;
    }
    for chunk in split_identifier_chunks(ident) {
        if let Some(idx) = IDENT_SUBWORD_TOKENS.iter().position(|t| *t == chunk) {
            out.push((BASE_VOCAB + RUST_TOKENS.len() + IDENT_TOKENS.len() + idx) as i32);
        } else {
            for b in chunk.as_bytes() {
                out.push(*b as i32);
            }
        }
    }
}

fn rust_token_idx(tok: &str) -> Option<i32> {
    RUST_TOKENS
        .iter()
        .position(|t| *t == tok)
        .map(|idx| (BASE_VOCAB + idx) as i32)
}

fn tokenize_numeric_literal(bytes: &[u8], i: &mut usize, out: &mut Vec<i32>) -> bool {
    if *i >= bytes.len() || !bytes[*i].is_ascii_digit() {
        return false;
    }
    let start = *i;
    *i += 1;
    while *i < bytes.len() {
        let c = bytes[*i];
        if c.is_ascii_alphanumeric() || c == b'_' || c == b'.' || c == b'+' || c == b'-' {
            *i += 1;
        } else {
            break;
        }
    }
    let lit = std::str::from_utf8(&bytes[start..*i]).unwrap_or("");

    // Try to split numeric suffixes into dedicated tokens.
    let mut matched_suffix = false;
    for suf in NUM_SUFFIX_TOKENS {
        if let Some(num) = lit.strip_suffix(suf) {
            for b in num.as_bytes() {
                out.push(*b as i32);
            }
            let idx = NUM_SUFFIX_TOKENS
                .iter()
                .position(|s| *s == suf)
                .unwrap_or(0);
            out.push(
                (BASE_VOCAB
                    + RUST_TOKENS.len()
                    + IDENT_TOKENS.len()
                    + IDENT_SUBWORD_TOKENS.len()
                    + idx) as i32,
            );
            matched_suffix = true;
            break;
        }
    }
    if !matched_suffix {
        for b in lit.as_bytes() {
            out.push(*b as i32);
        }
    }
    true
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
        for (idx, tok) in IDENT_TOKENS.iter().enumerate() {
            let tb = tok.as_bytes();
            if i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb {
                let token_idx = RUST_TOKENS.len() + idx;
                if best.map(|(_, l)| tb.len() > l).unwrap_or(true) {
                    best = Some((token_idx, tb.len()));
                }
            }
        }

        if let Some((idx, len)) = best {
            out.push((BASE_VOCAB + idx) as i32);
            i += len;
        } else if tokenize_numeric_literal(bytes, &mut i, &mut out) {
            continue;
        } else if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            emit_identifier_token(ident, &mut out);

            // Generic module path: IDENT ("::" IDENT)+
            while i + 2 <= bytes.len() && &bytes[i..i + 2] == b"::" {
                if let Some(tok) = rust_token_idx("::") {
                    out.push(tok);
                } else {
                    out.push(b':' as i32);
                    out.push(b':' as i32);
                }
                i += 2;
                let seg_start = i;
                if i < bytes.len() && is_ident_start(bytes[i]) {
                    i += 1;
                    while i < bytes.len() && is_ident_continue(bytes[i]) {
                        i += 1;
                    }
                    let seg = std::str::from_utf8(&bytes[seg_start..i]).unwrap_or("");
                    emit_identifier_token(seg, &mut out);
                } else {
                    break;
                }
            }

            // Generic forms: <IDENT>, <IDENT, IDENT>, <'a>, <'a, IDENT>
            if i < bytes.len() && bytes[i] == b'<' {
                let generic_start = i;
                i += 1;
                while i < bytes.len() && bytes[i] != b'>' {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b'>' {
                    let inner = std::str::from_utf8(&bytes[generic_start + 1..i]).unwrap_or("");
                    if let Some(tok) = rust_token_idx("<") {
                        out.push(tok);
                    } else {
                        out.push(b'<' as i32);
                    }
                    for part in inner.split(',') {
                        let p = part.trim();
                        if p.starts_with('\'') {
                            for b in p.as_bytes() {
                                out.push(*b as i32);
                            }
                        } else if !p.is_empty() {
                            emit_identifier_token(p, &mut out);
                        }
                        if part != inner.split(',').last().unwrap_or("") {
                            if let Some(tok) = rust_token_idx(",") {
                                out.push(tok);
                            } else {
                                out.push(b',' as i32);
                            }
                        }
                    }
                    if let Some(tok) = rust_token_idx(">") {
                        out.push(tok);
                    } else {
                        out.push(b'>' as i32);
                    }
                    i += 1;
                } else {
                    i = generic_start;
                }
            }
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
            } else {
                let ident_idx = idx - RUST_TOKENS.len();
                if ident_idx < IDENT_TOKENS.len() {
                    out.push_str(IDENT_TOKENS[ident_idx]);
                } else {
                    let sub_idx = ident_idx - IDENT_TOKENS.len();
                    if sub_idx < IDENT_SUBWORD_TOKENS.len() {
                        out.push_str(IDENT_SUBWORD_TOKENS[sub_idx]);
                    } else {
                        let num_idx = sub_idx - IDENT_SUBWORD_TOKENS.len();
                        if num_idx < NUM_SUFFIX_TOKENS.len() {
                            out.push_str(NUM_SUFFIX_TOKENS[num_idx]);
                        }
                    }
                }
            }
        }
    }
    out
}
