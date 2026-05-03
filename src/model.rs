// model.rs
// Direct port of model.h / model.cpp
// Single-layer transformer with:
//   - Token + positional embeddings
//   - Causal self-attention (Q K V O)
//   - Feed-forward network (FF = 4 * D) with ReLU
//   - LayerNorm after attention and FFN
//   - Adam optimizer
//   - Weight save / load (binary, same format as C++ version)

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

fn clip_grad_norm(grads: &mut [f32], max_norm: f32) {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        grads.iter_mut().for_each(|g| *g *= scale);
    }
}

struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamState {
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
        beta1: f32,
        beta2: f32,
        eps: f32,
    ) {
        for i in 0..params.len() {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i];
            let m_hat = self.m[i] / (1.0 - beta1.powi(step));
            let v_hat = self.v[i] / (1.0 - beta2.powi(step));
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

struct LayerWeights {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub wff1: Vec<f32>,
    pub bff1: Vec<f32>,
    pub wff2: Vec<f32>,
    pub bff2: Vec<f32>,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
}

impl LayerWeights {
    fn new(d: usize, ff: usize) -> Self {
        let mut wq = vec![0.0; d * d];
        let mut wk = vec![0.0; d * d];
        let mut wv = vec![0.0; d * d];
        let mut wo = vec![0.0; d * d];
        let mut wff1 = vec![0.0; ff * d];
        let mut wff2 = vec![0.0; d * ff];

        wq.iter_mut().for_each(|v| *v = rand_weight());
        wk.iter_mut().for_each(|v| *v = rand_weight());
        wv.iter_mut().for_each(|v| *v = rand_weight());
        wo.iter_mut().for_each(|v| *v = rand_weight());
        wff1.iter_mut().for_each(|v| *v = rand_weight());
        wff2.iter_mut().for_each(|v| *v = rand_weight());

        Self {
            wq,
            wk,
            wv,
            wo,
            wff1,
            bff1: vec![0.0; ff],
            wff2,
            bff2: vec![0.0; d],
            ln1_gamma: vec![1.0; d],
            ln1_beta: vec![0.0; d],
            ln2_gamma: vec![1.0; d],
            ln2_beta: vec![0.0; d],
        }
    }
}

struct LayerGrads {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wff1: Vec<f32>,
    bff1: Vec<f32>,
    wff2: Vec<f32>,
    bff2: Vec<f32>,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
}

impl LayerGrads {
    fn new(d: usize, ff: usize) -> Self {
        Self {
            wq: vec![0.0; d * d],
            wk: vec![0.0; d * d],
            wv: vec![0.0; d * d],
            wo: vec![0.0; d * d],
            wff1: vec![0.0; ff * d],
            bff1: vec![0.0; ff],
            wff2: vec![0.0; d * ff],
            bff2: vec![0.0; d],
            ln1_gamma: vec![0.0; d],
            ln1_beta: vec![0.0; d],
            ln2_gamma: vec![0.0; d],
            ln2_beta: vec![0.0; d],
        }
    }

    fn zero(&mut self) {
        self.wq.fill(0.0);
        self.wk.fill(0.0);
        self.wv.fill(0.0);
        self.wo.fill(0.0);
        self.wff1.fill(0.0);
        self.bff1.fill(0.0);
        self.wff2.fill(0.0);
        self.bff2.fill(0.0);
        self.ln1_gamma.fill(0.0);
        self.ln1_beta.fill(0.0);
        self.ln2_gamma.fill(0.0);
        self.ln2_beta.fill(0.0);
    }
}

struct LayerAdamState {
    wq: AdamState,
    wk: AdamState,
    wv: AdamState,
    wo: AdamState,
    wff1: AdamState,
    bff1: AdamState,
    wff2: AdamState,
    bff2: AdamState,
    ln1_gamma: AdamState,
    ln1_beta: AdamState,
    ln2_gamma: AdamState,
    ln2_beta: AdamState,
}

impl LayerAdamState {
    fn new(d: usize, ff: usize) -> Self {
        Self {
            wq: AdamState::new(d * d),
            wk: AdamState::new(d * d),
            wv: AdamState::new(d * d),
            wo: AdamState::new(d * d),
            wff1: AdamState::new(ff * d),
            bff1: AdamState::new(ff),
            wff2: AdamState::new(d * ff),
            bff2: AdamState::new(d),
            ln1_gamma: AdamState::new(d),
            ln1_beta: AdamState::new(d),
            ln2_gamma: AdamState::new(d),
            ln2_beta: AdamState::new(d),
        }
    }
}

pub struct ChatModel {
    vocab: usize,
    d: usize,
    t: usize,
    ff: usize,
    num_heads: usize,
    num_layers: usize,
    token_emb: Vec<f32>,
    layers: Vec<LayerWeights>,
    wout: Vec<f32>,
    bout: Vec<f32>,
}

impl ChatModel {
    pub fn new(vocab: usize, d: usize, t: usize, num_layers: usize, num_heads: usize) -> Self {
        if d % num_heads != 0 {
            panic!(
                "model_dim ({}) must be divisible by num_heads ({})",
                d, num_heads
            );
        }

        let ff = d * 4;
        let mut token_emb = vec![0.0; vocab * d];
        token_emb.iter_mut().for_each(|v| *v = rand_weight());

        let mut wout = vec![0.0; vocab * d];
        wout.iter_mut().for_each(|v| *v = rand_weight());

        let layers = (0..num_layers).map(|_| LayerWeights::new(d, ff)).collect();

        Self {
            vocab,
            d,
            t,
            ff,
            num_heads,
            num_layers,
            token_emb,
            layers,
            wout,
            bout: vec![0.0; vocab],
        }
    }

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

    pub fn forward_last_hidden(&self, tokens: &[i32]) -> Vec<f32> {
        let ctx = self.normalize_context(tokens);
        let (d, t) = (self.d, self.t);
        let layer = &self.layers[0];

        let x: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let tok = ctx[i] as usize;
                (0..d).map(|j| self.token_emb[idx2d(tok, j, d)]).collect()
            })
            .collect();

        let mut q: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        let mut k: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        let mut v: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
        for i in 0..t {
            matvec(&layer.wq, &x[i], &mut q[i], d, d);
            matvec(&layer.wk, &x[i], &mut k[i], d, d);
            matvec(&layer.wv, &x[i], &mut v[i], d, d);
        }

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
            matvec(&layer.wo, &head, &mut attn_out[ti], d, d);
        }

        let h1: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let pre: Vec<f32> = x[i]
                    .iter()
                    .zip(attn_out[i].iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                layer_norm_forward(&pre, &layer.ln1_gamma, &layer.ln1_beta)
            })
            .collect();

        let h2: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let mut ff = vec![0.0f32; self.ff];
                matvec(&layer.wff1, &h1[i], &mut ff, self.ff, d);
                for j in 0..self.ff {
                    let z = ff[j] + layer.bff1[j];
                    ff[j] = if z > 0.0 { z } else { 0.0 };
                }
                let mut ff2 = vec![0.0f32; d];
                matvec(&layer.wff2, &ff, &mut ff2, d, self.ff);
                let pre2: Vec<f32> = (0..d).map(|j| h1[i][j] + layer.bff2[j] + ff2[j]).collect();
                layer_norm_forward(&pre2, &layer.ln2_gamma, &layer.ln2_beta)
            })
            .collect();

        h2[t - 1].clone()
    }

    pub fn save_weights(&self, path: &str) -> bool {
        let file = match File::create(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut w = BufWriter::new(file);
        if w.write_all(&(self.vocab as u64).to_le_bytes()).is_err()
            || w.write_all(&(self.d as u64).to_le_bytes()).is_err()
            || w.write_all(&(self.num_layers as u64).to_le_bytes()).is_err()
            || w.write_all(&(self.num_heads as u64).to_le_bytes()).is_err()
        {
            return false;
        }

        if !write_mat(&mut w, &self.token_emb) {
            return false;
        }
        for layer in &self.layers {
            if !write_mat(&mut w, &layer.wq)
                || !write_mat(&mut w, &layer.wk)
                || !write_mat(&mut w, &layer.wv)
                || !write_mat(&mut w, &layer.wo)
                || !write_mat(&mut w, &layer.wff1)
                || !write_mat(&mut w, &layer.bff1)
                || !write_mat(&mut w, &layer.wff2)
                || !write_mat(&mut w, &layer.bff2)
                || !write_mat(&mut w, &layer.ln1_gamma)
                || !write_mat(&mut w, &layer.ln1_beta)
                || !write_mat(&mut w, &layer.ln2_gamma)
                || !write_mat(&mut w, &layer.ln2_beta)
            {
                return false;
            }
        }

        write_mat(&mut w, &self.wout) && write_mat(&mut w, &self.bout)
    }

    pub fn load_weights(&mut self, path: &str) -> bool {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut r = BufReader::new(file);

        let mut header = [0u8; 32];
        if r.read_exact(&mut header).is_err() {
            return false;
        }
        let vocab = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let d = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let num_layers = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;
        let num_heads = u64::from_le_bytes(header[24..32].try_into().unwrap()) as usize;

        if vocab != self.vocab
            || d != self.d
            || num_layers != self.num_layers
            || num_heads != self.num_heads
        {
            return false;
        }

        if !read_mat(&mut r, &mut self.token_emb) {
            return false;
        }
        for layer in &mut self.layers {
            if !read_mat(&mut r, &mut layer.wq)
                || !read_mat(&mut r, &mut layer.wk)
                || !read_mat(&mut r, &mut layer.wv)
                || !read_mat(&mut r, &mut layer.wo)
                || !read_mat(&mut r, &mut layer.wff1)
                || !read_mat(&mut r, &mut layer.bff1)
                || !read_mat(&mut r, &mut layer.wff2)
                || !read_mat(&mut r, &mut layer.bff2)
                || !read_mat(&mut r, &mut layer.ln1_gamma)
                || !read_mat(&mut r, &mut layer.ln1_beta)
                || !read_mat(&mut r, &mut layer.ln2_gamma)
                || !read_mat(&mut r, &mut layer.ln2_beta)
            {
                return false;
            }
        }

        read_mat(&mut r, &mut self.wout) && read_mat(&mut r, &mut self.bout)
    }

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

    pub fn generate(
        &self,
        context: &[i32],
        length: usize,
        temperature: f64,
        deterministic: bool,
    ) -> String {
        let mut ctx = if context.is_empty() { vec![0i32] } else { context.to_vec() };
        let prompt_len = ctx.len();

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

        crate::tokenizer::detokenize_bytes(&ctx[prompt_len..])
    }

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
        let adam_beta1 = 0.95f32;
        let adam_beta2 = 0.999f32;
        let adam_eps = 1e-8f32;
        let mut adam_step = 0i32;

        let mut am_token_emb = AdamState::new(vocab * d);
        let mut layer_adam: Vec<LayerAdamState> =
            (0..self.num_layers).map(|_| LayerAdamState::new(d, ff)).collect();
        let mut am_wout = AdamState::new(vocab * d);
        let mut am_bout = AdamState::new(vocab);

        for ep in 0..epochs {
            let mut total_loss = 0.0f32;

            let mut g_token_emb = vec![0.0f32; vocab * d];
            let mut layer_grads: Vec<LayerGrads> =
                (0..self.num_layers).map(|_| LayerGrads::new(d, ff)).collect();
            let mut g_wout = vec![0.0f32; vocab * d];
            let mut g_bout = vec![0.0f32; vocab];

            let mut accum_samples = 0usize;
            let mut accum_steps = 0usize;

            let mut batch_start = 0;
            while batch_start < train_starts.len() {
                let batch_end = (batch_start + batch_size).min(train_starts.len());
                let b_count = batch_end - batch_start;

                for bi in 0..b_count {
                    let start = train_starts[batch_start + bi];
                    let ctx_tokens = &data[start..start + t];
                    let target = data[start + t] as usize;
                    let layer = &self.layers[0];

                    let x: Vec<Vec<f32>> = (0..t)
                        .map(|i| {
                            let tok = ctx_tokens[i] as usize;
                            (0..d).map(|j| self.token_emb[idx2d(tok, j, d)]).collect()
                        })
                        .collect();

                    let mut q: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    let mut k_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    let mut v_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for i in 0..t {
                        matvec(&layer.wq, &x[i], &mut q[i], d, d);
                        matvec(&layer.wk, &x[i], &mut k_all[i], d, d);
                        matvec(&layer.wv, &x[i], &mut v_all[i], d, d);
                    }

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
                    matvec(&layer.wo, &head, &mut attn_out, d, d);

                    let pre1: Vec<f32> = (0..d).map(|j| x[last][j] + attn_out[j]).collect();
                    let h1 = layer_norm_forward(&pre1, &layer.ln1_gamma, &layer.ln1_beta);

                    let mut ff_pre = vec![0.0f32; ff];
                    matvec(&layer.wff1, &h1, &mut ff_pre, ff, d);
                    let ff_act: Vec<f32> = (0..ff)
                        .map(|i| {
                            let z = ff_pre[i] + layer.bff1[i];
                            ff_pre[i] = z;
                            if z > 0.0 { z } else { 0.0 }
                        })
                        .collect();

                    let mut ff2 = vec![0.0f32; d];
                    matvec(&layer.wff2, &ff_act, &mut ff2, d, ff);
                    let pre2: Vec<f32> = (0..d).map(|j| h1[j] + layer.bff2[j] + ff2[j]).collect();
                    let h2 = layer_norm_forward(&pre2, &layer.ln2_gamma, &layer.ln2_beta);

                    let mut logits = vec![0.0f32; vocab];
                    matvec(&self.wout, &h2, &mut logits, vocab, d);
                    for kk in 0..vocab { logits[kk] += self.bout[kk]; }

                    let probs = softmax(&logits);
                    total_loss += -(probs[target] + 1e-12).ln();

                    let mut dz: Vec<f32> = probs.clone();
                    dz[target] -= 1.0;

                    let mut dh2 = vec![0.0f32; d];
                    for kk in 0..vocab {
                        g_bout[kk] += dz[kk];
                        for j in 0..d {
                            g_wout[idx2d(kk, j, d)] += dz[kk] * h2[j];
                            dh2[j] += self.wout[idx2d(kk, j, d)] * dz[kk];
                        }
                    }

                    let mut dln2g = vec![0.0f32; d];
                    let mut dln2b = vec![0.0f32; d];
                    let dpre2 = layer_norm_backward(&pre2, &layer.ln2_gamma, &dh2, &mut dln2g, &mut dln2b);
                    for j in 0..d {
                        layer_grads[0].ln2_gamma[j] += dln2g[j];
                        layer_grads[0].ln2_beta[j] += dln2b[j];
                    }

                    let mut dh1 = dpre2.clone();
                    let dff2_grad = dpre2.clone();
                    let mut dff_act = vec![0.0f32; ff];
                    for i in 0..d {
                        layer_grads[0].bff2[i] += dff2_grad[i];
                        for j in 0..ff {
                            layer_grads[0].wff2[idx2d(i, j, ff)] += dff2_grad[i] * ff_act[j];
                            dff_act[j] += layer.wff2[idx2d(i, j, ff)] * dff2_grad[i];
                        }
                    }

                    let mut dff_pre_grad = vec![0.0f32; ff];
                    for i in 0..ff { dff_pre_grad[i] = if ff_pre[i] > 0.0 { dff_act[i] } else { 0.0 }; }
                    for i in 0..ff {
                        layer_grads[0].bff1[i] += dff_pre_grad[i];
                        for j in 0..d {
                            layer_grads[0].wff1[idx2d(i, j, d)] += dff_pre_grad[i] * h1[j];
                            dh1[j] += layer.wff1[idx2d(i, j, d)] * dff_pre_grad[i];
                        }
                    }

                    let mut dln1g = vec![0.0f32; d];
                    let mut dln1b = vec![0.0f32; d];
                    let dpre1 = layer_norm_backward(&pre1, &layer.ln1_gamma, &dh1, &mut dln1g, &mut dln1b);
                    for j in 0..d {
                        layer_grads[0].ln1_gamma[j] += dln1g[j];
                        layer_grads[0].ln1_beta[j] += dln1b[j];
                    }

                    let mut dhead = vec![0.0f32; d];
                    for i in 0..d {
                        for j in 0..d {
                            layer_grads[0].wo[idx2d(i, j, d)] += dpre1[i] * head[j];
                            dhead[j] += layer.wo[idx2d(i, j, d)] * dpre1[i];
                        }
                    }

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

                    let mut dx_last = dpre1.clone();
                    for i in 0..d {
                        for j in 0..d {
                            layer_grads[0].wq[idx2d(i, j, d)] += dq_last[i] * x[last][j];
                            dx_last[j] += layer.wq[idx2d(i, j, d)] * dq_last[i];
                        }
                    }

                    let mut dx_all: Vec<Vec<f32>> = vec![vec![0.0; d]; t];
                    for j in 0..t {
                        for i in 0..d {
                            for dd in 0..d {
                                layer_grads[0].wk[idx2d(i, dd, d)] += dk_all[j][i] * x[j][dd];
                                layer_grads[0].wv[idx2d(i, dd, d)] += dv_all[j][i] * x[j][dd];
                                dx_all[j][dd] += layer.wk[idx2d(i, dd, d)] * dk_all[j][i]
                                    + layer.wv[idx2d(i, dd, d)] * dv_all[j][i];
                            }
                        }
                    }
                    for dd in 0..d { dx_all[last][dd] += dx_last[dd]; }

                    for ti in 0..t {
                        let tok = ctx_tokens[ti] as usize;
                        for dd in 0..d {
                            g_token_emb[idx2d(tok, dd, d)] += dx_all[ti][dd];
                        }
                    }
                }

                accum_samples += b_count;
                accum_steps += 1;

                let is_last_batch = batch_end == train_starts.len();
                let should_step = accum_steps >= grad_accum_steps || is_last_batch;

                if should_step {
                    adam_step += 1;
                    let inv = 1.0 / accum_samples as f32;

                    for v in g_token_emb.iter_mut() { *v *= inv; }
                    layer_grads[0].wq.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].wk.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].wv.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].wo.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].wff1.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].bff1.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].wff2.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].bff2.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].ln1_gamma.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].ln1_beta.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].ln2_gamma.iter_mut().for_each(|v| *v *= inv);
                    layer_grads[0].ln2_beta.iter_mut().for_each(|v| *v *= inv);
                    for v in g_wout.iter_mut() { *v *= inv; }
                    for v in g_bout.iter_mut() { *v *= inv; }

                    am_token_emb.update(&mut self.token_emb, &g_token_emb, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wq.update(&mut self.layers[0].wq, &layer_grads[0].wq, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wk.update(&mut self.layers[0].wk, &layer_grads[0].wk, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wv.update(&mut self.layers[0].wv, &layer_grads[0].wv, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wo.update(&mut self.layers[0].wo, &layer_grads[0].wo, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wff1.update(&mut self.layers[0].wff1, &layer_grads[0].wff1, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].bff1.update(&mut self.layers[0].bff1, &layer_grads[0].bff1, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].wff2.update(&mut self.layers[0].wff2, &layer_grads[0].wff2, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].bff2.update(&mut self.layers[0].bff2, &layer_grads[0].bff2, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].ln1_gamma.update(&mut self.layers[0].ln1_gamma, &layer_grads[0].ln1_gamma, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].ln1_beta.update(&mut self.layers[0].ln1_beta, &layer_grads[0].ln1_beta, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].ln2_gamma.update(&mut self.layers[0].ln2_gamma, &layer_grads[0].ln2_gamma, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    layer_adam[0].ln2_beta.update(&mut self.layers[0].ln2_beta, &layer_grads[0].ln2_beta, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    am_wout.update(&mut self.wout, &g_wout, lr, adam_step, adam_beta1, adam_beta2, adam_eps);
                    am_bout.update(&mut self.bout, &g_bout, lr, adam_step, adam_beta1, adam_beta2, adam_eps);

                    g_token_emb.iter_mut().for_each(|v| *v = 0.0);
                    layer_grads[0].zero();
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
                for kk in 0..vocab { logits[kk] += self.bout[kk]; }
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

fn read_mat<R: Read>(r: &mut R, mat: &mut [f32]) -> bool {
    let mut buf = vec![0u8; mat.len() * 4];
    if r.read_exact(&mut buf).is_err() {
        return false;
    }
    for (i, chunk) in buf.chunks_exact(4).enumerate() {
        mat[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    true
}

fn write_mat<W: Write>(w: &mut W, mat: &[f32]) -> bool {
    let mut buf = vec![0u8; mat.len() * 4];
    for (i, &v) in mat.iter().enumerate() {
        let b = v.to_le_bytes();
        buf[i * 4..i * 4 + 4].copy_from_slice(&b);
    }
    w.write_all(&buf).is_ok()
}
