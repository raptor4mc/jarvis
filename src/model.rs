// model.rs
// Direct port of model.h / model.cpp
// Single-layer transformer with:
//   - Token + positional embeddings
//   - Causal self-attention (Q K V O)
//   - Feed-forward network (FF = 4 * D) with ReLU
//   - LayerNorm after attention and FFN
//   - Adam optimizer
//   - Weight save / load (binary, same format as C++ version)

use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// ── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn idx2d(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}

/// Xavier-style random init matching C++ `rand_weight()`
#[inline]
fn rand_weight() -> f32 {
    // Simple LCG so we don't need an external crate.
    // For better randomness you can swap in `rand` crate later.
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(12345678901234567);
    let s = STATE.fetch_add(6364136223846793005, Ordering::Relaxed);
    let f = ((s >> 33) as f32) / (u32::MAX as f32); // 0..1
    (f - 0.5) / 5.0
}

fn softmax(z: &[f32]) -> Vec<f32> {
    let maxv = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut y: Vec<f32> = z.iter().map(|&v| (v - maxv).exp()).collect();
    let sum: f32 = y.iter().sum();
    y.iter_mut().for_each(|v| *v /= sum);
    y
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// y += alpha * x
fn axpy(y: &mut [f32], x: &[f32], alpha: f32) {
    y.iter_mut()
        .zip(x.iter())
        .for_each(|(yi, &xi)| *yi += alpha * xi);
}

/// Row-major matrix-vector: y = A * x   (A is rows×cols)
fn matvec(a: &[f32], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        y[r] = dot(&a[r * cols..(r + 1) * cols], x);
    }
}

fn layer_norm_forward(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(&xi, (&g, &b))| g * (xi - mean) * inv_std + b)
        .collect()
}

/// Returns dx; accumulates dgamma and dbeta
fn layer_norm_backward(
    x: &[f32],
    gamma: &[f32],
    dout: &[f32],
    dgamma: &mut [f32],
    dbeta: &mut [f32],
) -> Vec<f32> {
    let n = x.len();
    let nf = n as f32;
    let mean: f32 = x.iter().sum::<f32>() / nf;
    let var: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / nf;
    let inv_std = 1.0 / (var + 1e-5_f32).sqrt();

    let xhat: Vec<f32> = x.iter().map(|&xi| (xi - mean) * inv_std).collect();
    let dxhat: Vec<f32> = dout
        .iter()
        .zip(gamma.iter())
        .map(|(&d, &g)| d * g)
        .collect();

    for i in 0..n {
        dgamma[i] += dout[i] * xhat[i];
        dbeta[i] += dout[i];
    }

    let sum_dxhat: f32 = dxhat.iter().sum();
    let sum_dxhat_xhat: f32 = dxhat.iter().zip(xhat.iter()).map(|(&a, &b)| a * b).sum();

    (0..n)
        .map(|i| (1.0 / nf) * inv_std * (nf * dxhat[i] - sum_dxhat - xhat[i] * sum_dxhat_xhat))
        .collect()
}

fn snap_batch_size(b: usize) -> usize {
    if b <= 4 {
        4
    } else if b <= 8 {
        8
    } else if b <= 16 {
        16
    } else {
        32
    }
}

// ── Muon optimizer state helper ───────────────────────────────────────────────

struct MuonState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl MuonState {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    fn update(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        lr: f32,
        step: i32,
        momentum: f32,
        eps: f32,
    ) {
        let bias_correction = 1.0 - momentum.powi(step);
        for i in 0..params.len() {
            self.m[i] = momentum * self.m[i] + (1.0 - momentum) * grads[i];
            self.v[i] = 0.99 * self.v[i] + 0.01 * grads[i] * grads[i];
            let m_hat = self.m[i] / bias_correction.max(eps);
            let rms = self.v[i].sqrt() + eps;
            let mut upd = m_hat / rms;
            let mag = upd.abs();
            if mag > 1.0 {
                upd /= mag;
            }
            params[i] -= lr * upd;
        }
    }
}

// ── ChatModel ────────────────────────────────────────────────────────────────

pub struct ChatModel {
    vocab: usize,
    d: usize,  // model_dim
    t: usize,  // seq_len
    ff: usize, // feed-forward dim = 4 * d

    token_emb: Vec<f32>, // vocab × d
    pos_emb: Vec<f32>,   // t × d

    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>, // d × d
    wff1: Vec<f32>,
    bff1: Vec<f32>, // ff × d,  ff
    wff2: Vec<f32>,
    bff2: Vec<f32>, // d × ff,  d

    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>, // d
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>, // d

    wout: Vec<f32>,
    bout: Vec<f32>, // vocab × d,  vocab
}

impl ChatModel {
    pub fn new(vocab: usize, model_dim: usize, seq_len: usize) -> Self {
        let d = model_dim;
        let t = seq_len;
        let ff = model_dim * 4;

        let mut m = ChatModel {
            vocab,
            d,
            t,
            ff,
            token_emb: vec![0.0; vocab * d],
            pos_emb: vec![0.0; t * d],
            wq: vec![0.0; d * d],
            wk: vec![0.0; d * d],
            wv: vec![0.0; d * d],
            wo: vec![0.0; d * d],
            wff1: vec![0.0; ff * d],
            bff1: vec![0.0; ff],
            wff2: vec![0.0; d * ff],
            bff2: vec![0.0; d],
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
            wout: vec![0.0; vocab * d],
            bout: vec![0.0; vocab],
        };

        for v in m.token_emb.iter_mut() {
            *v = rand_weight();
        }
        for v in m.pos_emb.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wq.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wk.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wv.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wo.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wff1.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wff2.iter_mut() {
            *v = rand_weight();
        }
        for v in m.wout.iter_mut() {
            *v = rand_weight();
        }

        m
    }

    // ── weight IO ────────────────────────────────────────────────────────────

    pub fn load_weights(&mut self, path: &str) -> bool {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut r = BufReader::new(file);
        let ok = read_mat(&mut r, &mut self.token_emb)
            && read_mat(&mut r, &mut self.pos_emb)
            && read_mat(&mut r, &mut self.wq)
            && read_mat(&mut r, &mut self.wk)
            && read_mat(&mut r, &mut self.wv)
            && read_mat(&mut r, &mut self.wo)
            && read_mat(&mut r, &mut self.wff1)
            && read_mat(&mut r, &mut self.bff1)
            && read_mat(&mut r, &mut self.wff2)
            && read_mat(&mut r, &mut self.bff2)
            && read_mat(&mut r, &mut self.ln1_gamma)
            && read_mat(&mut r, &mut self.ln1_beta)
            && read_mat(&mut r, &mut self.ln2_gamma)
            && read_mat(&mut r, &mut self.ln2_beta)
            && read_mat(&mut r, &mut self.wout)
            && read_mat(&mut r, &mut self.bout);
        ok
    }

    pub fn save_weights(&self, path: &str) -> bool {
        let file = match File::create(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut w = BufWriter::new(file);
        write_mat(&mut w, &self.token_emb)
            && write_mat(&mut w, &self.pos_emb)
            && write_mat(&mut w, &self.wq)
            && write_mat(&mut w, &self.wk)
            && write_mat(&mut w, &self.wv)
            && write_mat(&mut w, &self.wo)
            && write_mat(&mut w, &self.wff1)
            && write_mat(&mut w, &self.bff1)
            && write_mat(&mut w, &self.wff2)
            && write_mat(&mut w, &self.bff2)
            && write_mat(&mut w, &self.ln1_gamma)
            && write_mat(&mut w, &self.ln1_beta)
            && write_mat(&mut w, &self.ln2_gamma)
            && write_mat(&mut w, &self.ln2_beta)
            && write_mat(&mut w, &self.wout)
            && write_mat(&mut w, &self.bout)
    }

    // ── context normalisation ─────────────────────────────────────────────────

    fn normalize_context(&self, tokens: &[i32]) -> Vec<i32> {
        if tokens.is_empty() {
            return vec![0; self.t];
        }
        if tokens.len() >= self.t {
            tokens[tokens.len() - self.t..].to_vec()
        } else {
            let pad = tokens[0];
            let mut ctx = vec![pad; self.t - tokens.len()];
            ctx.extend_from_slice(tokens);
            ctx
        }
    }

    // ── forward pass (last hidden state) ────────────────────────────────────

    pub fn forward_last_hidden(&self, tokens: &[i32]) -> Vec<f32> {
        let ctx = self.normalize_context(tokens);
        let (d, t) = (self.d, self.t);

        // Embedding lookup: x[t] = token_emb[tok] + pos_emb[t]
        let mut x: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let tok = ctx[i] as usize;
                (0..d)
                    .map(|j| self.token_emb[idx2d(tok, j, d)] + self.pos_emb[idx2d(i, j, d)])
                    .collect()
            })
            .collect();

        // Q K V projections
        let mut q: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        let mut k: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        let mut v: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        for i in 0..t {
            matvec(&self.wq, &x[i], &mut q[i], d, d);
            matvec(&self.wk, &x[i], &mut k[i], d, d);
            matvec(&self.wv, &x[i], &mut v[i], d, d);
        }

        // Causal self-attention
        let scale = 1.0 / (d as f32).sqrt();
        let mut attn_out: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        for ti in 0..t {
            let mut scores = vec![f32::NEG_INFINITY; t];
            for j in 0..=ti {
                scores[j] = dot(&q[ti], &k[j]) * scale;
            }
            let a = softmax(&scores);
            let mut head = vec![0.0f32; d];
            for j in 0..=ti {
                axpy(&mut head, &v[j], a[j]);
            }
            matvec(&self.wo, &head, &mut attn_out[ti], d, d);
        }

        // Residual + LayerNorm 1
        let h1: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let pre: Vec<f32> = x[i]
                    .iter()
                    .zip(attn_out[i].iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                layer_norm_forward(&pre, &self.ln1_gamma, &self.ln1_beta)
            })
            .collect();

        // FFN + Residual + LayerNorm 2
        let h2: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let mut ff = vec![0.0f32; self.ff];
                matvec(&self.wff1, &h1[i], &mut ff, self.ff, d);
                for j in 0..self.ff {
                    let z = ff[j] + self.bff1[j];
                    ff[j] = if z > 0.0 { z } else { 0.0 }; // ReLU
                }
                let mut ff2 = vec![0.0f32; d];
                matvec(&self.wff2, &ff, &mut ff2, d, self.ff);
                let pre2: Vec<f32> = (0..d).map(|j| h1[i][j] + self.bff2[j] + ff2[j]).collect();
                layer_norm_forward(&pre2, &self.ln2_gamma, &self.ln2_beta)
            })
            .collect();

        h2[t - 1].clone()
    }

    // ── token sampling ───────────────────────────────────────────────────────

    fn sample_next_token(&self, logits: &[f32], temperature: f64, deterministic: bool) -> i32 {
        if deterministic {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as i32)
                .unwrap_or(0);
        }
        let temp = (temperature as f32).max(1e-6);
        let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();
        let probs = softmax(&scaled);

        // simple linear search cumulative sample
        let r: f32 = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static R: AtomicU64 = AtomicU64::new(987654321);
            let s = R.fetch_add(2862933555777941757, Ordering::Relaxed);
            ((s >> 33) as f32) / (u32::MAX as f32)
        };
        let mut cum = 0.0f32;
        for (k, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return k as i32;
            }
        }
        (probs.len() - 1) as i32
    }

    // ── generate ─────────────────────────────────────────────────────────────

    pub fn generate(
        &self,
        context: &[i32],
        length: usize,
        temperature: f64,
        deterministic: bool,
    ) -> String {
        let mut ctx = if context.is_empty() {
            vec![0i32]
        } else {
            context.to_vec()
        };

        for _ in 0..length {
            let norm = self.normalize_context(&ctx);
            let h = self.forward_last_hidden(&norm);
            let mut logits = vec![0.0f32; self.vocab];
            matvec(&self.wout, &h, &mut logits, self.vocab, self.d);
            for k in 0..self.vocab {
                logits[k] += self.bout[k];
            }
            let next = self.sample_next_token(&logits, temperature, deterministic);
            ctx.push(next);
        }

        crate::tokenizer::detokenize_bytes(&ctx)
    }

    // ── train ────────────────────────────────────────────────────────────────

    pub fn train(
        &mut self,
        data: &[i32],
        epochs: usize,
        lr: f32,
        batch_size: usize,
        sample_stride: usize,
    ) {
        let (d, t, ff, vocab) = (self.d, self.t, self.ff, self.vocab);
        if data.len() <= t {
            return;
        }

        let batch_size = snap_batch_size(batch_size.max(1));
        let sample_stride = sample_stride.max(1);

        let sample_starts: Vec<usize> = (0..data.len() - t).step_by(sample_stride).collect();
        let sample_count = sample_starts.len();
        if sample_count == 0 {
            return;
        }
        let val_count = (sample_count / 20).max(1);
        let train_count = sample_count.saturating_sub(val_count).max(1);
        let train_starts = &sample_starts[..train_count];
        let val_starts = &sample_starts[train_count..];

        let grad_accum_steps = if batch_size == 4 {
            4
        } else if batch_size == 8 {
            2
        } else {
            1
        };

        println!(
            "Samples: {}, stride: {}, micro-batch: {}, grad_accum: {}, effective batch: {}",
            train_count,
            sample_stride,
            batch_size,
            grad_accum_steps,
            batch_size * grad_accum_steps
        );

        let scale = 1.0f32 / (d as f32).sqrt();
        let last = t - 1;
        let muon_momentum = 0.95f32;
        let muon_eps = 1e-8f32;
        let mut adam_step = 0i32;

        // Adam states — one per parameter tensor (flattened)
        let mut am_token_emb = MuonState::new(vocab * d);
        let mut am_pos_emb = MuonState::new(t * d);
        let mut am_wq = MuonState::new(d * d);
        let mut am_wk = MuonState::new(d * d);
        let mut am_wv = MuonState::new(d * d);
        let mut am_wo = MuonState::new(d * d);
        let mut am_wff1 = MuonState::new(ff * d);
        let mut am_bff1 = MuonState::new(ff);
        let mut am_wff2 = MuonState::new(d * ff);
        let mut am_bff2 = MuonState::new(d);
        let mut am_ln1g = MuonState::new(d);
        let mut am_ln1b = MuonState::new(d);
        let mut am_ln2g = MuonState::new(d);
        let mut am_ln2b = MuonState::new(d);
        let mut am_wout = MuonState::new(vocab * d);
        let mut am_bout = MuonState::new(vocab);

        for ep in 0..epochs {
            let mut total_loss = 0.0f32;

            // Accumulated gradient buffers
            let mut g_token_emb = vec![0.0f32; vocab * d];
            let mut g_pos_emb = vec![0.0f32; t * d];
            let mut g_wq = vec![0.0f32; d * d];
            let mut g_wk = vec![0.0f32; d * d];
            let mut g_wv = vec![0.0f32; d * d];
            let mut g_wo = vec![0.0f32; d * d];
            let mut g_wff1 = vec![0.0f32; ff * d];
            let mut g_bff1 = vec![0.0f32; ff];
            let mut g_wff2 = vec![0.0f32; d * ff];
            let mut g_bff2 = vec![0.0f32; d];
            let mut g_ln1g = vec![0.0f32; d];
            let mut g_ln1b = vec![0.0f32; d];
            let mut g_ln2g = vec![0.0f32; d];
            let mut g_ln2b = vec![0.0f32; d];
            let mut g_wout = vec![0.0f32; vocab * d];
            let mut g_bout = vec![0.0f32; vocab];

            let mut accum_samples = 0usize;
            let mut accum_steps = 0usize;

            let mut batch_start = 0;
            while batch_start < train_starts.len() {
                let batch_end = (batch_start + batch_size).min(train_starts.len());
                let b_count = batch_end - batch_start;

                // ── forward pass for each sample in the micro-batch ──────────

                for bi in 0..b_count {
                    let start = train_starts[batch_start + bi];
                    let ctx_tokens = &data[start..start + t];
                    let target = data[start + t] as usize;

                    // Embeddings
                    let x: Vec<Vec<f32>> = (0..t)
                        .map(|i| {
                            let tok = ctx_tokens[i] as usize;
                            (0..d)
                                .map(|j| {
                                    self.token_emb[idx2d(tok, j, d)] + self.pos_emb[idx2d(i, j, d)]
                                })
                                .collect()
                        })
                        .collect();

                    // Q K V
                    let mut q: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    let mut k_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    let mut v_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for i in 0..t {
                        matvec(&self.wq, &x[i], &mut q[i], d, d);
                        matvec(&self.wk, &x[i], &mut k_all[i], d, d);
                        matvec(&self.wv, &x[i], &mut v_all[i], d, d);
                    }

                    // Causal attention at last position
                    let mut scores = vec![f32::NEG_INFINITY; t];
                    for j in 0..t {
                        scores[j] = dot(&q[last], &k_all[j]) * scale;
                    }
                    let a = softmax(&scores);
                    let mut head = vec![0.0f32; d];
                    for j in 0..t {
                        axpy(&mut head, &v_all[j], a[j]);
                    }

                    let mut attn_out = vec![0.0f32; d];
                    matvec(&self.wo, &head, &mut attn_out, d, d);

                    // Residual + LN1
                    let pre1: Vec<f32> = (0..d).map(|j| x[last][j] + attn_out[j]).collect();
                    let h1 = layer_norm_forward(&pre1, &self.ln1_gamma, &self.ln1_beta);

                    // FFN
                    let mut ff_pre = vec![0.0f32; ff];
                    matvec(&self.wff1, &h1, &mut ff_pre, ff, d);
                    let ff_act: Vec<f32> = (0..ff)
                        .map(|i| {
                            let z = ff_pre[i] + self.bff1[i];
                            ff_pre[i] = z; // store pre-activation for backward
                            if z > 0.0 {
                                z
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    let mut ff2 = vec![0.0f32; d];
                    matvec(&self.wff2, &ff_act, &mut ff2, d, ff);
                    let pre2: Vec<f32> = (0..d).map(|j| h1[j] + self.bff2[j] + ff2[j]).collect();
                    let h2 = layer_norm_forward(&pre2, &self.ln2_gamma, &self.ln2_beta);

                    // Output logits
                    let mut logits = vec![0.0f32; vocab];
                    matvec(&self.wout, &h2, &mut logits, vocab, d);
                    for kk in 0..vocab {
                        logits[kk] += self.bout[kk];
                    }

                    let probs = softmax(&logits);
                    total_loss += -(probs[target] + 1e-12).ln();

                    // ── backward pass ────────────────────────────────────────

                    // dL/dlogits = probs - one_hot(target)
                    let mut dz: Vec<f32> = probs.clone();
                    dz[target] -= 1.0;

                    // LM head backward
                    let mut dh2 = vec![0.0f32; d];
                    for kk in 0..vocab {
                        g_bout[kk] += dz[kk];
                        for j in 0..d {
                            g_wout[idx2d(kk, j, d)] += dz[kk] * h2[j];
                            dh2[j] += self.wout[idx2d(kk, j, d)] * dz[kk];
                        }
                    }

                    // LN2 backward
                    let mut dln2g = vec![0.0f32; d];
                    let mut dln2b = vec![0.0f32; d];
                    let dpre2 =
                        layer_norm_backward(&pre2, &self.ln2_gamma, &dh2, &mut dln2g, &mut dln2b);
                    for j in 0..d {
                        g_ln2g[j] += dln2g[j];
                        g_ln2b[j] += dln2b[j];
                    }

                    // FFN backward
                    let mut dh1 = dpre2.clone();
                    let mut dff2_grad = dpre2.clone();
                    let mut dff_act = vec![0.0f32; ff];
                    for i in 0..d {
                        g_bff2[i] += dff2_grad[i];
                        for j in 0..ff {
                            g_wff2[idx2d(i, j, ff)] += dff2_grad[i] * ff_act[j];
                            dff_act[j] += self.wff2[idx2d(i, j, ff)] * dff2_grad[i];
                        }
                    }
                    // ReLU backward
                    let mut dff_pre_grad = vec![0.0f32; ff];
                    for i in 0..ff {
                        dff_pre_grad[i] = if ff_pre[i] > 0.0 { dff_act[i] } else { 0.0 };
                    }
                    for i in 0..ff {
                        g_bff1[i] += dff_pre_grad[i];
                        for j in 0..d {
                            g_wff1[idx2d(i, j, d)] += dff_pre_grad[i] * h1[j];
                            dh1[j] += self.wff1[idx2d(i, j, d)] * dff_pre_grad[i];
                        }
                    }

                    // LN1 backward
                    let mut dln1g = vec![0.0f32; d];
                    let mut dln1b = vec![0.0f32; d];
                    let dpre1 =
                        layer_norm_backward(&pre1, &self.ln1_gamma, &dh1, &mut dln1g, &mut dln1b);
                    for j in 0..d {
                        g_ln1g[j] += dln1g[j];
                        g_ln1b[j] += dln1b[j];
                    }

                    // Attention output backward
                    let mut dhead = vec![0.0f32; d];
                    for i in 0..d {
                        for j in 0..d {
                            g_wo[idx2d(i, j, d)] += dpre1[i] * head[j];
                            dhead[j] += self.wo[idx2d(i, j, d)] * dpre1[i];
                        }
                    }

                    // Attention backward
                    let mut da = vec![0.0f32; t];
                    let mut dv_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for j in 0..t {
                        for dd in 0..d {
                            dv_all[j][dd] += a[j] * dhead[dd];
                            da[j] += dhead[dd] * v_all[j][dd];
                        }
                    }

                    let sum_da_a: f32 = da.iter().zip(a.iter()).map(|(&x, &y)| x * y).sum();
                    let dscores: Vec<f32> = (0..t).map(|j| a[j] * (da[j] - sum_da_a)).collect();

                    let mut dq_last = vec![0.0f32; d];
                    let mut dk_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for j in 0..t {
                        for dd in 0..d {
                            dq_last[dd] += scale * dscores[j] * k_all[j][dd];
                            dk_all[j][dd] += scale * dscores[j] * q[last][dd];
                        }
                    }

                    // Q projection backward (only last position)
                    let mut dx_last = dpre1.clone();
                    for i in 0..d {
                        for j in 0..d {
                            g_wq[idx2d(i, j, d)] += dq_last[i] * x[last][j];
                            dx_last[j] += self.wq[idx2d(i, j, d)] * dq_last[i];
                        }
                    }

                    // K V projection backward (all positions)
                    let mut dx_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for j in 0..t {
                        for i in 0..d {
                            for dd in 0..d {
                                g_wk[idx2d(i, dd, d)] += dk_all[j][i] * x[j][dd];
                                g_wv[idx2d(i, dd, d)] += dv_all[j][i] * x[j][dd];
                                dx_all[j][dd] += self.wk[idx2d(i, dd, d)] * dk_all[j][i]
                                    + self.wv[idx2d(i, dd, d)] * dv_all[j][i];
                            }
                        }
                    }
                    for dd in 0..d {
                        dx_all[last][dd] += dx_last[dd];
                    }

                    // Embedding backward
                    for ti in 0..t {
                        let tok = ctx_tokens[ti] as usize;
                        for dd in 0..d {
                            g_token_emb[idx2d(tok, dd, d)] += dx_all[ti][dd];
                            g_pos_emb[idx2d(ti, dd, d)] += dx_all[ti][dd];
                        }
                    }
                } // end sample loop

                accum_samples += b_count;
                accum_steps += 1;

                let is_last_batch = batch_end == train_starts.len();
                let should_step = accum_steps >= grad_accum_steps || is_last_batch;

                if should_step {
                    adam_step += 1;
                    let inv = 1.0 / accum_samples as f32;

                    // Scale gradients
                    for v in g_token_emb.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_pos_emb.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wq.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wk.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wv.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wo.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wff1.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_bff1.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wff2.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_bff2.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_ln1g.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_ln1b.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_ln2g.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_ln2b.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_wout.iter_mut() {
                        *v *= inv;
                    }
                    for v in g_bout.iter_mut() {
                        *v *= inv;
                    }

                    // Adam updates
                    am_token_emb.update(
                        &mut self.token_emb,
                        &g_token_emb,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_pos_emb.update(
                        &mut self.pos_emb,
                        &g_pos_emb,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_wq.update(&mut self.wq, &g_wq, lr, adam_step, muon_momentum, muon_eps);
                    am_wk.update(&mut self.wk, &g_wk, lr, adam_step, muon_momentum, muon_eps);
                    am_wv.update(&mut self.wv, &g_wv, lr, adam_step, muon_momentum, muon_eps);
                    am_wo.update(&mut self.wo, &g_wo, lr, adam_step, muon_momentum, muon_eps);
                    am_wff1.update(
                        &mut self.wff1,
                        &g_wff1,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_bff1.update(
                        &mut self.bff1,
                        &g_bff1,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_wff2.update(
                        &mut self.wff2,
                        &g_wff2,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_bff2.update(
                        &mut self.bff2,
                        &g_bff2,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_ln1g.update(
                        &mut self.ln1_gamma,
                        &g_ln1g,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_ln1b.update(
                        &mut self.ln1_beta,
                        &g_ln1b,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_ln2g.update(
                        &mut self.ln2_gamma,
                        &g_ln2g,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_ln2b.update(
                        &mut self.ln2_beta,
                        &g_ln2b,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_wout.update(
                        &mut self.wout,
                        &g_wout,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );
                    am_bout.update(
                        &mut self.bout,
                        &g_bout,
                        lr,
                        adam_step,
                        muon_momentum,
                        muon_eps,
                    );

                    // Zero accumulators
                    g_token_emb.iter_mut().for_each(|v| *v = 0.0);
                    g_pos_emb.iter_mut().for_each(|v| *v = 0.0);
                    g_wq.iter_mut().for_each(|v| *v = 0.0);
                    g_wk.iter_mut().for_each(|v| *v = 0.0);
                    g_wv.iter_mut().for_each(|v| *v = 0.0);
                    g_wo.iter_mut().for_each(|v| *v = 0.0);
                    g_wff1.iter_mut().for_each(|v| *v = 0.0);
                    g_bff1.iter_mut().for_each(|v| *v = 0.0);
                    g_wff2.iter_mut().for_each(|v| *v = 0.0);
                    g_bff2.iter_mut().for_each(|v| *v = 0.0);
                    g_ln1g.iter_mut().for_each(|v| *v = 0.0);
                    g_ln1b.iter_mut().for_each(|v| *v = 0.0);
                    g_ln2g.iter_mut().for_each(|v| *v = 0.0);
                    g_ln2b.iter_mut().for_each(|v| *v = 0.0);
                    g_wout.iter_mut().for_each(|v| *v = 0.0);
                    g_bout.iter_mut().for_each(|v| *v = 0.0);

                    accum_samples = 0;
                    accum_steps = 0;
                }

                batch_start = batch_end;
            }

            let train_loss = total_loss / train_count as f32;
            let mut val_total = 0.0f32;
            for &start in val_starts {
                let ctx_tokens = &data[start..start + t];
                let target = data[start + t] as usize;
                let h = self.forward_last_hidden(ctx_tokens);
                let mut logits = vec![0.0f32; vocab];
                matvec(&self.wout, &h, &mut logits, vocab, d);
                for kk in 0..vocab {
                    logits[kk] += self.bout[kk];
                }
                let probs = softmax(&logits);
                val_total += -(probs[target] + 1e-12).ln();
            }
            let val_loss = val_total / val_starts.len().max(1) as f32;
            println!(
                "Epoch {}/{} train loss: {:.6}, val loss: {:.6}",
                ep + 1,
                epochs,
                train_loss,
                val_loss
            );
        }
    }
}

// ── binary weight IO helpers ─────────────────────────────────────────────────

fn read_mat(r: &mut BufReader<File>, buf: &mut Vec<f32>) -> bool {
    let byte_len = buf.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    match r.read_exact(&mut bytes) {
        Ok(_) => {
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                buf[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            true
        }
        Err(_) => false,
    }
}

fn write_mat(w: &mut BufWriter<File>, buf: &[f32]) -> bool {
    let bytes: Vec<u8> = buf.iter().flat_map(|&f| f.to_le_bytes()).collect();
    w.write_all(&bytes).is_ok()
}
