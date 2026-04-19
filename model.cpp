#include "model.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(USE_CBLAS)
#include <cblas.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace std;

static inline float dot_simd(const float *a, const float *b, int n) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

static inline void axpy_simd(float *y, const float *x, float alpha, int n) {
#if defined(__AVX2__)
    __m256 aval = _mm256_set1_ps(alpha);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 yv = _mm256_loadu_ps(y + i);
        __m256 xv = _mm256_loadu_ps(x + i);
        yv = _mm256_add_ps(yv, _mm256_mul_ps(aval, xv));
        _mm256_storeu_ps(y + i, yv);
    }
    for (; i < n; ++i) y[i] += alpha * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
#endif
}

static inline void matvec_rm(const float *A, const float *x, float *y, int rows, int cols) {
#if defined(USE_CBLAS)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, A, cols, x, 1, 0.0f, y, 1);
#else
    for (int r = 0; r < rows; ++r) y[r] = dot_simd(A + (size_t)r * cols, x, cols);
#endif
}

static inline int snap_batch_size(int batch_size) {
    if (batch_size <= 4) return 4;
    if (batch_size <= 8) return 8;
    if (batch_size <= 16) return 16;
    return 32;
}

ChatModel::ChatModel(int vocab_size, int model_dim, int seq_len)
    : vocab(vocab_size), D(model_dim), T(seq_len), FF(model_dim * 4),
      token_emb(vocab * D),
      pos_emb(T * D),
      Wq(D * D), Wk(D * D), Wv(D * D), Wo(D * D),
      Wff1(FF * D), bff1(FF, 0.0),
      Wff2(D * FF), bff2(D, 0.0),
      ln1_gamma(D, 1.0), ln1_beta(D, 0.0),
      ln2_gamma(D, 1.0), ln2_beta(D, 0.0),
      Wout(vocab * D), bout(vocab, 0.0) {
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < D; ++j) token_emb[idx2d(i, j, D)] = rand_weight();
    }
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < D; ++j) pos_emb[idx2d(t, j, D)] = rand_weight();
    }

    auto init_mat = [&](vector<float> &M) {
        for (float &v : M) v = rand_weight();
    };

    init_mat(Wq); init_mat(Wk); init_mat(Wv); init_mat(Wo);
    init_mat(Wff1); init_mat(Wff2); init_mat(Wout);
}

size_t ChatModel::idx2d(int r, int c, int cols) {
    return (size_t)r * (size_t)cols + (size_t)c;
}

float ChatModel::rand_weight() {
    return ((float)rand() / RAND_MAX - 0.5f) / 5.0f;
}

vector<float> ChatModel::softmax(const vector<float> &z) {
    vector<float> y(z.size());
    float maxv = z[0];
    for (float v : z) if (v > maxv) maxv = v;
    float sum = 0.0f;
    for (size_t i = 0; i < z.size(); ++i) {
        y[i] = exp(z[i] - maxv);
        sum += y[i];
    }
    for (size_t i = 0; i < z.size(); ++i) y[i] /= sum;
    return y;
}

int ChatModel::sample_next_token(const vector<float> &logits, double temperature, bool deterministic) {
    if (deterministic) {
        int best = 0;
        for (int i = 1; i < (int)logits.size(); ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    float temp = (float)temperature;
    if (temp < 1e-6f) temp = 1e-6f;

    vector<float> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) scaled[i] = logits[i] / temp;
    vector<float> y = softmax(scaled);

    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int k = 0; k < (int)y.size(); ++k) {
        cum += y[k];
        if (r <= cum) return k;
    }
    return (int)y.size() - 1;
}

vector<float> ChatModel::layer_norm_forward(const vector<float> &x, const vector<float> &gamma, const vector<float> &beta) {
    const float eps = 1e-5f;
    float mean = 0.0f;
    for (float v : x) mean += v;
    mean /= (float)x.size();

    float var = 0.0f;
    for (float v : x) {
        float d = v - mean;
        var += d * d;
    }
    var /= (float)x.size();
    float inv_std = 1.0f / sqrtf(var + eps);

    vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float xhat = (x[i] - mean) * inv_std;
        y[i] = gamma[i] * xhat + beta[i];
    }
    return y;
}

vector<float> ChatModel::layer_norm_backward(
    const vector<float> &x,
    const vector<float> &gamma,
    const vector<float> &dout,
    vector<float> &dgamma,
    vector<float> &dbeta) {
    const float eps = 1e-5f;
    int n = (int)x.size();

    float mean = 0.0f;
    for (float v : x) mean += v;
    mean /= (float)n;

    float var = 0.0f;
    for (float v : x) {
        float d = v - mean;
        var += d * d;
    }
    var /= (float)n;
    float inv_std = 1.0f / sqrtf(var + eps);

    vector<float> xhat(n), dxhat(n);
    for (int i = 0; i < n; ++i) {
        xhat[i] = (x[i] - mean) * inv_std;
        dgamma[i] += dout[i] * xhat[i];
        dbeta[i] += dout[i];
        dxhat[i] = dout[i] * gamma[i];
    }

    float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum_dxhat += dxhat[i];
        sum_dxhat_xhat += dxhat[i] * xhat[i];
    }

    vector<float> dx(n);
    for (int i = 0; i < n; ++i) {
        dx[i] = (1.0f / n) * inv_std * (n * dxhat[i] - sum_dxhat - xhat[i] * sum_dxhat_xhat);
    }
    return dx;
}

vector<int> ChatModel::normalize_context(const vector<int> &tokens) const {
    vector<int> ctx;
    if (tokens.empty()) {
        ctx.assign(T, 0);
        return ctx;
    }

    if ((int)tokens.size() >= T) {
        ctx.assign(tokens.end() - T, tokens.end());
    } else {
        ctx = tokens;
        while ((int)ctx.size() < T) ctx.insert(ctx.begin(), ctx.front());
    }
    return ctx;
}

vector<float> ChatModel::forward_last_hidden(const vector<int> &tokens) const {
    vector<int> ctx = normalize_context(tokens);

    // x[t] = token_emb + pos_emb
    vector<vector<float>> x(T, vector<float>(D));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        int tok = ctx[t];
        for (int j = 0; j < D; ++j) x[t][j] = token_emb[idx2d(tok, j, D)] + pos_emb[idx2d(t, j, D)];
    }

    // q, k, v
    vector<vector<float>> q(T, vector<float>(D)), k(T, vector<float>(D)), v(T, vector<float>(D));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        matvec_rm(Wq.data(), x[t].data(), q[t].data(), D, D);
        matvec_rm(Wk.data(), x[t].data(), k[t].data(), D, D);
        matvec_rm(Wv.data(), x[t].data(), v[t].data(), D, D);
    }

    // Causal self-attention.
    vector<vector<float>> attn_out(T, vector<float>(D, 0.0));
    const float scale = 1.0f / sqrtf((float)D);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        vector<float> scores(T, -1e9);
        for (int j = 0; j <= t; ++j) {
            scores[j] = dot_simd(q[t].data(), k[j].data(), D) * scale;
        }
        vector<float> a = softmax(scores);

        vector<float> head(D, 0.0);
        for (int j = 0; j <= t; ++j) axpy_simd(head.data(), v[j].data(), a[j], D);

        matvec_rm(Wo.data(), head.data(), attn_out[t].data(), D, D);
    }

    // Residual + LayerNorm after attention.
    vector<vector<float>> h1(T, vector<float>(D));
    for (int t = 0; t < T; ++t) {
        vector<float> pre(D);
        for (int d = 0; d < D; ++d) pre[d] = x[t][d] + attn_out[t][d];
        h1[t] = layer_norm_forward(pre, ln1_gamma, ln1_beta);
    }

    // FFN + residual + LayerNorm.
    vector<vector<float>> h2(T, vector<float>(D));
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        vector<float> ff(FF, 0.0), ff2(D, 0.0);
        matvec_rm(Wff1.data(), h1[t].data(), ff.data(), FF, D);
        for (int i = 0; i < FF; ++i) {
            float z = bff1[i] + ff[i];
            ff[i] = z > 0.0 ? z : 0.0; // ReLU
        }

        matvec_rm(Wff2.data(), ff.data(), ff2.data(), D, FF);
        for (int i = 0; i < D; ++i) h2[t][i] = h1[t][i] + bff2[i] + ff2[i];
        h2[t] = layer_norm_forward(h2[t], ln2_gamma, ln2_beta);
    }

    return h2[T - 1];
}

bool ChatModel::load_weights(const string &filename) {
    ifstream in(filename, ios::binary);
    if (!in) return false;

    auto read_mat = [&](vector<float> &M) {
        in.read(reinterpret_cast<char*>(M.data()), (long long)M.size() * sizeof(float));
        return (bool)in;
    };
    auto read_vec = [&](vector<float> &v) {
        in.read(reinterpret_cast<char*>(v.data()), (long long)v.size() * sizeof(float));
        return (bool)in;
    };

    if (!read_mat(token_emb)) return false;
    if (!read_mat(pos_emb)) return false;
    if (!read_mat(Wq)) return false;
    if (!read_mat(Wk)) return false;
    if (!read_mat(Wv)) return false;
    if (!read_mat(Wo)) return false;
    if (!read_mat(Wff1)) return false;
    if (!read_vec(bff1)) return false;
    if (!read_mat(Wff2)) return false;
    if (!read_vec(bff2)) return false;
    if (!read_vec(ln1_gamma)) return false;
    if (!read_vec(ln1_beta)) return false;
    if (!read_vec(ln2_gamma)) return false;
    if (!read_vec(ln2_beta)) return false;
    if (!read_mat(Wout)) return false;
    if (!read_vec(bout)) return false;

    return true;
}

bool ChatModel::save_weights(const string &filename) const {
    ofstream out(filename, ios::binary);
    if (!out) return false;

    auto write_mat = [&](const vector<float> &M) {
        out.write(reinterpret_cast<const char*>(M.data()), (long long)M.size() * sizeof(float));
    };
    auto write_vec = [&](const vector<float> &v) {
        out.write(reinterpret_cast<const char*>(v.data()), (long long)v.size() * sizeof(float));
    };

    write_mat(token_emb);
    write_mat(pos_emb);
    write_mat(Wq);
    write_mat(Wk);
    write_mat(Wv);
    write_mat(Wo);
    write_mat(Wff1);
    write_vec(bff1);
    write_mat(Wff2);
    write_vec(bff2);
    write_vec(ln1_gamma);
    write_vec(ln1_beta);
    write_vec(ln2_gamma);
    write_vec(ln2_beta);
    write_mat(Wout);
    write_vec(bout);

    return (bool)out;
}

void ChatModel::train(const vector<int> &data, int epochs, float lr, int batch_size, int sample_stride) {
    if ((int)data.size() <= T) return;
    if (batch_size < 1) batch_size = 1;
    if (sample_stride < 1) sample_stride = 1;
    batch_size = snap_batch_size(batch_size);

    vector<int> sample_starts;
    sample_starts.reserve(data.size());
    for (size_t i = 0; i + T < data.size(); i += (size_t)sample_stride) sample_starts.push_back((int)i);
    const int sample_count = (int)sample_starts.size();
    if (sample_count == 0) return;

    int grad_accum_steps = 1;
    if (batch_size == 4) grad_accum_steps = 4;      // effective batch 16
    else if (batch_size == 8) grad_accum_steps = 2; // effective batch 16

    cout << "Samples: " << sample_count
         << ", stride: " << sample_stride
         << ", micro-batch size: " << batch_size
         << ", grad_accum_steps: " << grad_accum_steps
         << ", effective batch: " << (batch_size * grad_accum_steps) << endl;

    const float scale = 1.0f / sqrtf((float)D);
    int last = T - 1;

    for (int ep = 0; ep < epochs; ++ep) {
        float total_loss = 0.0f;

        vector<vector<float>> gWout_accum(vocab, vector<float>(D, 0.0f));
        vector<float> gbout_accum(vocab, 0.0f);
        vector<vector<float>> gWo_accum(D, vector<float>(D, 0.0f));
        vector<vector<float>> gWq_accum(D, vector<float>(D, 0.0f));
        vector<vector<float>> gWk_accum(D, vector<float>(D, 0.0f));
        vector<vector<float>> gWv_accum(D, vector<float>(D, 0.0f));
        vector<vector<float>> gWff1_accum(FF, vector<float>(D, 0.0f));
        vector<float> gbff1_accum(FF, 0.0f);
        vector<vector<float>> gWff2_accum(D, vector<float>(FF, 0.0f));
        vector<float> gbff2_accum(D, 0.0f);
        vector<float> gln1_gamma_accum(D, 0.0f), gln1_beta_accum(D, 0.0f);
        vector<float> gln2_gamma_accum(D, 0.0f), gln2_beta_accum(D, 0.0f);
        vector<vector<float>> dtoken_emb_accum(vocab, vector<float>(D, 0.0f));
        vector<vector<float>> dpos_emb_accum(T, vector<float>(D, 0.0f));
        int accum_samples = 0;
        int accum_steps = 0;

        for (size_t batch_start = 0; batch_start < sample_starts.size(); batch_start += (size_t)batch_size) {
            size_t batch_end = min(sample_starts.size(), batch_start + (size_t)batch_size);
            int B = (int)(batch_end - batch_start);

            vector<vector<int>> batch_ctx(B, vector<int>(T));
            vector<int> batch_target(B);
            for (int b = 0; b < B; ++b) {
                int start = sample_starts[batch_start + b];
                for (int t = 0; t < T; ++t) batch_ctx[b][t] = data[start + t];
                batch_target[b] = data[start + T];
            }

            // Batch forward pass.
            vector<vector<vector<float>>> x(B, vector<vector<float>>(T, vector<float>(D, 0.0)));
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    int tok = batch_ctx[b][t];
                    for (int d = 0; d < D; ++d) x[b][t][d] = token_emb[idx2d(tok, d, D)] + pos_emb[idx2d(t, d, D)];
                }
            }

            vector<vector<float>> q_last(B, vector<float>(D, 0.0));
            vector<vector<vector<float>>> k_all(B, vector<vector<float>>(T, vector<float>(D, 0.0)));
            vector<vector<vector<float>>> v_all(B, vector<vector<float>>(T, vector<float>(D, 0.0)));
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                matvec_rm(Wq.data(), x[b][last].data(), q_last[b].data(), D, D);
                for (int t = 0; t < T; ++t) {
                    matvec_rm(Wk.data(), x[b][t].data(), k_all[b][t].data(), D, D);
                    matvec_rm(Wv.data(), x[b][t].data(), v_all[b][t].data(), D, D);
                }
            }

            vector<vector<float>> a(B, vector<float>(T, 0.0));
            vector<vector<float>> head(B, vector<float>(D, 0.0));
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                vector<float> scores(T);
                for (int t = 0; t < T; ++t) {
                    scores[t] = dot_simd(q_last[b].data(), k_all[b][t].data(), D) * scale;
                }
                a[b] = softmax(scores);
                for (int t = 0; t < T; ++t) axpy_simd(head[b].data(), v_all[b][t].data(), a[b][t], D);
            }

            vector<vector<float>> attn_out(B, vector<float>(D, 0.0));
            vector<vector<float>> pre1(B, vector<float>(D, 0.0));
            vector<vector<float>> h1_norm(B, vector<float>(D, 0.0));
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                matvec_rm(Wo.data(), head[b].data(), attn_out[b].data(), D, D);
                for (int i = 0; i < D; ++i) pre1[b][i] = x[b][last][i] + attn_out[b][i];
                h1_norm[b] = layer_norm_forward(pre1[b], ln1_gamma, ln1_beta);
            }

            vector<vector<float>> ff_pre(B, vector<float>(FF, 0.0));
            vector<vector<float>> ff(B, vector<float>(FF, 0.0));
            vector<vector<float>> ff2(B, vector<float>(D, 0.0));
            vector<vector<float>> pre2(B, vector<float>(D, 0.0));
            vector<vector<float>> h2_norm(B, vector<float>(D, 0.0));
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; ++b) {
                matvec_rm(Wff1.data(), h1_norm[b].data(), ff_pre[b].data(), FF, D);
                for (int i = 0; i < FF; ++i) {
                    float z = bff1[i] + ff_pre[b][i];
                    ff_pre[b][i] = z;
                    ff[b][i] = z > 0.0 ? z : 0.0;
                }
                matvec_rm(Wff2.data(), ff[b].data(), ff2[b].data(), D, FF);
                for (int i = 0; i < D; ++i) pre2[b][i] = h1_norm[b][i] + bff2[i] + ff2[b][i];
                h2_norm[b] = layer_norm_forward(pre2[b], ln2_gamma, ln2_beta);
            }

            vector<vector<float>> y(B, vector<float>(vocab, 0.0));
            #pragma omp parallel for reduction(+:total_loss) schedule(static)
            for (int b = 0; b < B; ++b) {
                vector<float> logits(vocab, 0.0);
                matvec_rm(Wout.data(), h2_norm[b].data(), logits.data(), vocab, D);
                for (int k = 0; k < vocab; ++k) logits[k] += bout[k];
                y[b] = softmax(logits);
                total_loss += -log(y[b][batch_target[b]] + 1e-12);
            }

            // Accumulated gradients over the whole batch.
            vector<vector<float>> gWout_sum(vocab, vector<float>(D, 0.0));
            vector<float> gbout_sum(vocab, 0.0);
            vector<vector<float>> gWo_sum(D, vector<float>(D, 0.0));
            vector<vector<float>> gWq_sum(D, vector<float>(D, 0.0));
            vector<vector<float>> gWk_sum(D, vector<float>(D, 0.0));
            vector<vector<float>> gWv_sum(D, vector<float>(D, 0.0));
            vector<vector<float>> gWff1_sum(FF, vector<float>(D, 0.0));
            vector<float> gbff1_sum(FF, 0.0);
            vector<vector<float>> gWff2_sum(D, vector<float>(FF, 0.0));
            vector<float> gbff2_sum(D, 0.0);
            vector<float> gln1_gamma_sum(D, 0.0), gln1_beta_sum(D, 0.0);
            vector<float> gln2_gamma_sum(D, 0.0), gln2_beta_sum(D, 0.0);
            vector<vector<float>> dtoken_emb_sum(vocab, vector<float>(D, 0.0));
            vector<vector<float>> dpos_emb_sum(T, vector<float>(D, 0.0));

            for (int b = 0; b < B; ++b) {
                vector<float> dz = y[b];
                dz[batch_target[b]] -= 1.0;

                vector<float> dh2_norm(D, 0.0);
                for (int k = 0; k < vocab; ++k) {
                    gbout_sum[k] += dz[k];
                    for (int j = 0; j < D; ++j) {
                        gWout_sum[k][j] += dz[k] * h2_norm[b][j];
                        dh2_norm[j] += Wout[idx2d(k, j, D)] * dz[k];
                    }
                }

                vector<float> gln2_gamma(D, 0.0), gln2_beta(D, 0.0);
                vector<float> dpre2 = layer_norm_backward(pre2[b], ln2_gamma, dh2_norm, gln2_gamma, gln2_beta);

                vector<float> dh1_norm = dpre2;
                vector<float> dff2 = dpre2;

                vector<float> dff(FF, 0.0);
                for (int i = 0; i < D; ++i) {
                    gbff2_sum[i] += dff2[i];
                    for (int j = 0; j < FF; ++j) {
                        gWff2_sum[i][j] += dff2[i] * ff[b][j];
                        dff[j] += Wff2[idx2d(i, j, FF)] * dff2[i];
                    }
                }

                vector<float> dff_pre(FF, 0.0);
                for (int i = 0; i < FF; ++i) dff_pre[i] = ff_pre[b][i] > 0.0 ? dff[i] : 0.0;
                for (int i = 0; i < FF; ++i) {
                    gbff1_sum[i] += dff_pre[i];
                    for (int j = 0; j < D; ++j) {
                        gWff1_sum[i][j] += dff_pre[i] * h1_norm[b][j];
                        dh1_norm[j] += Wff1[idx2d(i, j, D)] * dff_pre[i];
                    }
                }

                vector<float> gln1_gamma(D, 0.0), gln1_beta(D, 0.0);
                vector<float> dpre1 = layer_norm_backward(pre1[b], ln1_gamma, dh1_norm, gln1_gamma, gln1_beta);

                for (int d = 0; d < D; ++d) {
                    gln1_gamma_sum[d] += gln1_gamma[d];
                    gln1_beta_sum[d] += gln1_beta[d];
                    gln2_gamma_sum[d] += gln2_gamma[d];
                    gln2_beta_sum[d] += gln2_beta[d];
                }

                vector<float> dx_last = dpre1;

                vector<float> dhead(D, 0.0);
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        gWo_sum[i][j] += dpre1[i] * head[b][j];
                        dhead[j] += Wo[idx2d(i, j, D)] * dpre1[i];
                    }
                }

                vector<float> da(T, 0.0);
                vector<vector<float>> dv_all(T, vector<float>(D, 0.0));
                for (int t = 0; t < T; ++t) {
                    for (int d = 0; d < D; ++d) {
                        dv_all[t][d] += a[b][t] * dhead[d];
                        da[t] += dhead[d] * v_all[b][t][d];
                    }
                }

                float sum_da_a = 0.0f;
                for (int t = 0; t < T; ++t) sum_da_a += da[t] * a[b][t];
                vector<float> dscores(T, 0.0);
                for (int t = 0; t < T; ++t) dscores[t] = a[b][t] * (da[t] - sum_da_a);

                vector<float> dq_last(D, 0.0);
                vector<vector<float>> dk_all(T, vector<float>(D, 0.0));
                for (int t = 0; t < T; ++t) {
                    for (int d = 0; d < D; ++d) {
                        dq_last[d] += scale * dscores[t] * k_all[b][t][d];
                        dk_all[t][d] += scale * dscores[t] * q_last[b][d];
                    }
                }

                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        gWq_sum[i][j] += dq_last[i] * x[b][last][j];
                        dx_last[j] += Wq[idx2d(i, j, D)] * dq_last[i];
                    }
                }

                vector<vector<float>> dx_all(T, vector<float>(D, 0.0));
                for (int t = 0; t < T; ++t) {
                    for (int i = 0; i < D; ++i) {
                        for (int j = 0; j < D; ++j) {
                            gWk_sum[i][j] += dk_all[t][i] * x[b][t][j];
                            gWv_sum[i][j] += dv_all[t][i] * x[b][t][j];
                            dx_all[t][j] += Wk[idx2d(i, j, D)] * dk_all[t][i] + Wv[idx2d(i, j, D)] * dv_all[t][i];
                        }
                    }
                }
                for (int d = 0; d < D; ++d) dx_all[last][d] += dx_last[d];

                for (int t = 0; t < T; ++t) {
                    int tok = batch_ctx[b][t];
                    for (int d = 0; d < D; ++d) {
                        dtoken_emb_sum[tok][d] += dx_all[t][d];
                        dpos_emb_sum[t][d] += dx_all[t][d];
                    }
                }
            }

            for (int k = 0; k < vocab; ++k) {
                gbout_accum[k] += gbout_sum[k];
                for (int j = 0; j < D; ++j) gWout_accum[k][j] += gWout_sum[k][j];
            }
            for (int i = 0; i < D; ++i) {
                gbff2_accum[i] += gbff2_sum[i];
                gln1_gamma_accum[i] += gln1_gamma_sum[i];
                gln1_beta_accum[i] += gln1_beta_sum[i];
                gln2_gamma_accum[i] += gln2_gamma_sum[i];
                gln2_beta_accum[i] += gln2_beta_sum[i];
                for (int j = 0; j < D; ++j) {
                    gWo_accum[i][j] += gWo_sum[i][j];
                    gWq_accum[i][j] += gWq_sum[i][j];
                    gWk_accum[i][j] += gWk_sum[i][j];
                    gWv_accum[i][j] += gWv_sum[i][j];
                }
                for (int j = 0; j < FF; ++j) gWff2_accum[i][j] += gWff2_sum[i][j];
            }
            for (int i = 0; i < FF; ++i) {
                gbff1_accum[i] += gbff1_sum[i];
                for (int j = 0; j < D; ++j) gWff1_accum[i][j] += gWff1_sum[i][j];
            }
            for (int tok = 0; tok < vocab; ++tok) {
                for (int d = 0; d < D; ++d) dtoken_emb_accum[tok][d] += dtoken_emb_sum[tok][d];
            }
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) dpos_emb_accum[t][d] += dpos_emb_sum[t][d];
            }

            accum_samples += B;
            accum_steps += 1;
            bool should_step = (accum_steps >= grad_accum_steps) || (batch_end == sample_starts.size());
            if (!should_step) continue;

            float step_lr = lr / (float)accum_samples;
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < vocab; ++k) {
                for (int j = 0; j < D; ++j) Wout[idx2d(k, j, D)] -= step_lr * gWout_accum[k][j];
                bout[k] -= step_lr * gbout_accum[k];
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    Wo[idx2d(i, j, D)] -= step_lr * gWo_accum[i][j];
                    Wq[idx2d(i, j, D)] -= step_lr * gWq_accum[i][j];
                    Wk[idx2d(i, j, D)] -= step_lr * gWk_accum[i][j];
                    Wv[idx2d(i, j, D)] -= step_lr * gWv_accum[i][j];
                }
                for (int j = 0; j < FF; ++j) Wff2[idx2d(i, j, FF)] -= step_lr * gWff2_accum[i][j];
                bff2[i] -= step_lr * gbff2_accum[i];

                ln1_gamma[i] -= step_lr * gln1_gamma_accum[i];
                ln1_beta[i] -= step_lr * gln1_beta_accum[i];
                ln2_gamma[i] -= step_lr * gln2_gamma_accum[i];
                ln2_beta[i] -= step_lr * gln2_beta_accum[i];
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < FF; ++i) {
                for (int j = 0; j < D; ++j) Wff1[idx2d(i, j, D)] -= step_lr * gWff1_accum[i][j];
                bff1[i] -= step_lr * gbff1_accum[i];
            }
            #pragma omp parallel for schedule(static)
            for (int tok = 0; tok < vocab; ++tok) {
                for (int d = 0; d < D; ++d) token_emb[idx2d(tok, d, D)] -= step_lr * dtoken_emb_accum[tok][d];
            }
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) pos_emb[idx2d(t, d, D)] -= step_lr * dpos_emb_accum[t][d];
            }

            for (int k = 0; k < vocab; ++k) {
                gbout_accum[k] = 0.0f;
                fill(gWout_accum[k].begin(), gWout_accum[k].end(), 0.0f);
            }
            for (int i = 0; i < D; ++i) {
                gbff2_accum[i] = 0.0f;
                gln1_gamma_accum[i] = gln1_beta_accum[i] = 0.0f;
                gln2_gamma_accum[i] = gln2_beta_accum[i] = 0.0f;
                fill(gWo_accum[i].begin(), gWo_accum[i].end(), 0.0f);
                fill(gWq_accum[i].begin(), gWq_accum[i].end(), 0.0f);
                fill(gWk_accum[i].begin(), gWk_accum[i].end(), 0.0f);
                fill(gWv_accum[i].begin(), gWv_accum[i].end(), 0.0f);
                fill(gWff2_accum[i].begin(), gWff2_accum[i].end(), 0.0f);
            }
            for (int i = 0; i < FF; ++i) {
                gbff1_accum[i] = 0.0f;
                fill(gWff1_accum[i].begin(), gWff1_accum[i].end(), 0.0f);
            }
            for (int tok = 0; tok < vocab; ++tok) {
                fill(dtoken_emb_accum[tok].begin(), dtoken_emb_accum[tok].end(), 0.0f);
            }
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < T; ++t) {
                fill(dpos_emb_accum[t].begin(), dpos_emb_accum[t].end(), 0.0f);
            }
            accum_samples = 0;
            accum_steps = 0;
        }

        float avg_loss = total_loss / (float)sample_count;
        cout << "Epoch " << (ep + 1) << "/" << epochs << " avg loss: " << avg_loss << endl;
    }
}

string ChatModel::generate(const vector<int> &context, int length, double temperature, bool deterministic) const {
    vector<int> ctx = context;
    if (ctx.empty()) ctx.push_back(0);

    vector<int> out_tokens = ctx;
    for (int step = 0; step < length; ++step) {
        vector<int> norm = normalize_context(ctx);
        vector<float> h_last = forward_last_hidden(norm);

        vector<float> logits(vocab);
        matvec_rm(Wout.data(), h_last.data(), logits.data(), vocab, D);
        for (int k = 0; k < vocab; ++k) logits[k] += bout[k];

        int next_idx = sample_next_token(logits, temperature, deterministic);
        ctx.push_back(next_idx);
        out_tokens.push_back(next_idx);
    }

    return detokenize_bytes(out_tokens);
}
