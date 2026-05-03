// tokenizer.rs
// Hybrid tokenizer tuned for Rust/code-heavy corpora.
// - Base vocab: byte tokens 0..255
// - Extra vocab: common Rust/code lexemes as single tokens

const BASE_VOCAB: usize = 256;

const RUST_TOKENS: [&str; 157] = [
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
    "usize",
    "i32",
    "f32",
    "bool",
    "true",
    "false",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "i8",
    "i16",
    "i64",
    "i128",
    "f64",
    "isize",
    "char",
    "str",
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
    "trait ",
    "derive",
    "#[",
    "]\n",
    "match",
    " if ",
    " else",
    "Some(",
    "Ok(",
    "Err(",
    "None",
    "Vec<",
    "String::",
    "format!",
    "println!",
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

pub fn vocab_size() -> usize {
    BASE_VOCAB + RUST_TOKENS.len() + IDENT_TOKENS.len()
}

fn is_ident_start(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphabetic()
}

fn is_ident_continue(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
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
        } else if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            if let Some(idx) = IDENT_TOKENS.iter().position(|t| *t == ident) {
                out.push((BASE_VOCAB + RUST_TOKENS.len() + idx) as i32);
            } else {
                for b in ident.as_bytes() {
                    out.push(*b as i32);
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
                }
            }
        }
    }
    out
}
