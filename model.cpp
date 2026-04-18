#include "model.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace std;

static inline double dot_simd(const double *a, const double *b, int n) {
#if defined(__AVX2__)
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d av = _mm256_loadu_pd(a + i);
        __m256d bv = _mm256_loadu_pd(b + i);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(av, bv));
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

static inline void axpy_simd(double *y, const double *x, double alpha, int n) {
#if defined(__AVX2__)
    __m256d aval = _mm256_set1_pd(alpha);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m256d yv = _mm256_loadu_pd(y + i);
        __m256d xv = _mm256_loadu_pd(x + i);
        yv = _mm256_add_pd(yv, _mm256_mul_pd(aval, xv));
        _mm256_storeu_pd(y + i, yv);
    }
    for (; i < n; ++i) y[i] += alpha * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
#endif
}

ChatModel::ChatModel(int vocab_size, int model_dim, int seq_len)
    : vocab(vocab_size), D(model_dim), T(seq_len), FF(model_dim * 2),
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

    auto init_mat = [&](vector<double> &M) {
        for (double &v : M) v = rand_weight();
    };

    init_mat(Wq); init_mat(Wk); init_mat(Wv); init_mat(Wo);
    init_mat(Wff1); init_mat(Wff2); init_mat(Wout);
}

size_t ChatModel::idx2d(int r, int c, int cols) {
    return (size_t)r * (size_t)cols + (size_t)c;
}

double ChatModel::rand_weight() {
    return ((double)rand() / RAND_MAX - 0.5) / 5.0;
}

vector<double> ChatModel::softmax(const vector<double> &z) {
    vector<double> y(z.size());
    double maxv = z[0];
    for (double v : z) if (v > maxv) maxv = v;
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        y[i] = exp(z[i] - maxv);
        sum += y[i];
    }
    for (size_t i = 0; i < z.size(); ++i) y[i] /= sum;
    return y;
}

int ChatModel::sample_next_token(const vector<double> &logits, double temperature, bool deterministic) {
    if (deterministic) {
        int best = 0;
        for (int i = 1; i < (int)logits.size(); ++i) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    double temp = temperature;
    if (temp < 1e-6) temp = 1e-6;

    vector<double> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) scaled[i] = logits[i] / temp;
    vector<double> y = softmax(scaled);

    double r = (double)rand() / RAND_MAX;
    double cum = 0.0;
    for (int k = 0; k < (int)y.size(); ++k) {
        cum += y[k];
        if (r <= cum) return k;
    }
    return (int)y.size() - 1;
}

vector<double> ChatModel::layer_norm_forward(const vector<double> &x, const vector<double> &gamma, const vector<double> &beta) {
    const double eps = 1e-5;
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= (double)x.size();

    double var = 0.0;
    for (double v : x) {
        double d = v - mean;
        var += d * d;
    }
    var /= (double)x.size();
    double inv_std = 1.0 / sqrt(var + eps);

    vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        double xhat = (x[i] - mean) * inv_std;
        y[i] = gamma[i] * xhat + beta[i];
    }
    return y;
}

vector<double> ChatModel::layer_norm_backward(
    const vector<double> &x,
    const vector<double> &gamma,
    const vector<double> &dout,
    vector<double> &dgamma,
    vector<double> &dbeta) {
    const double eps = 1e-5;
    int n = (int)x.size();

    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= (double)n;

    double var = 0.0;
    for (double v : x) {
        double d = v - mean;
        var += d * d;
    }
    var /= (double)n;
    double inv_std = 1.0 / sqrt(var + eps);

    vector<double> xhat(n), dxhat(n);
    for (int i = 0; i < n; ++i) {
        xhat[i] = (x[i] - mean) * inv_std;
        dgamma[i] += dout[i] * xhat[i];
        dbeta[i] += dout[i];
        dxhat[i] = dout[i] * gamma[i];
    }

    double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_dxhat += dxhat[i];
        sum_dxhat_xhat += dxhat[i] * xhat[i];
    }

    vector<double> dx(n);
    for (int i = 0; i < n; ++i) {
        dx[i] = (1.0 / n) * inv_std * (n * dxhat[i] - sum_dxhat - xhat[i] * sum_dxhat_xhat);
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

vector<double> ChatModel::forward_last_hidden(const vector<int> &tokens) const {
    vector<int> ctx = normalize_context(tokens);

    // x[t] = token_emb + pos_emb
    vector<vector<double>> x(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        int tok = ctx[t];
        for (int j = 0; j < D; ++j) x[t][j] = token_emb[idx2d(tok, j, D)] + pos_emb[idx2d(t, j, D)];
    }

    // q, k, v
    vector<vector<double>> q(T, vector<double>(D)), k(T, vector<double>(D)), v(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < D; ++i) {
            q[t][i] = dot_simd(&Wq[idx2d(i, 0, D)], x[t].data(), D);
            k[t][i] = dot_simd(&Wk[idx2d(i, 0, D)], x[t].data(), D);
            v[t][i] = dot_simd(&Wv[idx2d(i, 0, D)], x[t].data(), D);
        }
    }

    // Causal self-attention.
    vector<vector<double>> attn_out(T, vector<double>(D, 0.0));
    const double scale = 1.0 / sqrt((double)D);
    for (int t = 0; t < T; ++t) {
        vector<double> scores(T, -1e9);
        for (int j = 0; j <= t; ++j) {
            scores[j] = dot_simd(q[t].data(), k[j].data(), D) * scale;
        }
        vector<double> a = softmax(scores);

        vector<double> head(D, 0.0);
        for (int j = 0; j <= t; ++j) axpy_simd(head.data(), v[j].data(), a[j], D);

        for (int i = 0; i < D; ++i) {
            attn_out[t][i] = dot_simd(&Wo[idx2d(i, 0, D)], head.data(), D);
        }
    }

    // Residual + LayerNorm after attention.
    vector<vector<double>> h1(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        vector<double> pre(D);
        for (int d = 0; d < D; ++d) pre[d] = x[t][d] + attn_out[t][d];
        h1[t] = layer_norm_forward(pre, ln1_gamma, ln1_beta);
    }

    // FFN + residual + LayerNorm.
    vector<vector<double>> h2(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        vector<double> ff(FF, 0.0);
        for (int i = 0; i < FF; ++i) {
            double z = bff1[i] + dot_simd(&Wff1[idx2d(i, 0, D)], h1[t].data(), D);
            ff[i] = z > 0.0 ? z : 0.0; // ReLU
        }

        for (int i = 0; i < D; ++i) {
            double z = bff2[i] + dot_simd(&Wff2[idx2d(i, 0, FF)], ff.data(), FF);
            h2[t][i] = h1[t][i] + z;
        }
        h2[t] = layer_norm_forward(h2[t], ln2_gamma, ln2_beta);
    }

    return h2[T - 1];
}

bool ChatModel::load_weights(const string &filename) {
    ifstream in(filename, ios::binary);
    if (!in) return false;

    auto read_mat = [&](vector<double> &M) {
        in.read(reinterpret_cast<char*>(M.data()), (long long)M.size() * sizeof(double));
        return (bool)in;
    };
    auto read_vec = [&](vector<double> &v) {
        in.read(reinterpret_cast<char*>(v.data()), (long long)v.size() * sizeof(double));
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

    auto write_mat = [&](const vector<double> &M) {
        out.write(reinterpret_cast<const char*>(M.data()), (long long)M.size() * sizeof(double));
    };
    auto write_vec = [&](const vector<double> &v) {
        out.write(reinterpret_cast<const char*>(v.data()), (long long)v.size() * sizeof(double));
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

void ChatModel::train(const vector<int> &data, int epochs, double lr, int batch_size) {
    if ((int)data.size() <= T) return;
    if (batch_size < 1) batch_size = 1;

    if (batch_size <= 8) batch_size = 8;
    else if (batch_size <= 16) batch_size = 16;
    else batch_size = 32;

    if (batch_size <= 8) batch_size = 8;
    else if (batch_size <= 16) batch_size = 16;
    else batch_size = 32;

    if (batch_size <= 8) batch_size = 8;
    else if (batch_size <= 16) batch_size = 16;
    else batch_size = 32;

    struct Sample { vector<int> ctx; int target; };
    vector<Sample> samples;
    samples.reserve(data.size());
    for (size_t i = 0; i + T < data.size(); ++i) {
        vector<int> ctx(data.begin() + i, data.begin() + i + T);
        samples.push_back({ctx, data[i + T]});
    }

    cout << "Samples: " << samples.size() << ", batch size: " << batch_size << endl;

    const double scale = 1.0 / sqrt((double)D);
    int last = T - 1;

    for (int ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;

        for (size_t batch_start = 0; batch_start < samples.size(); batch_start += (size_t)batch_size) {
            size_t batch_end = min(samples.size(), batch_start + (size_t)batch_size);
            int B = (int)(batch_end - batch_start);
            double step_lr = lr / (double)B;

            vector<vector<int>> batch_ctx(B, vector<int>(T));
            vector<int> batch_target(B);
            for (int b = 0; b < B; ++b) {
                batch_ctx[b] = samples[batch_start + b].ctx;
                batch_target[b] = samples[batch_start + b].target;
            }

            // Batch forward pass.
            vector<vector<vector<double>>> x(B, vector<vector<double>>(T, vector<double>(D, 0.0)));
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    int tok = batch_ctx[b][t];
                    for (int d = 0; d < D; ++d) x[b][t][d] = token_emb[idx2d(tok, d, D)] + pos_emb[idx2d(t, d, D)];
                }
            }

            vector<vector<double>> q_last(B, vector<double>(D, 0.0));
            vector<vector<vector<double>>> k_all(B, vector<vector<double>>(T, vector<double>(D, 0.0)));
            vector<vector<vector<double>>> v_all(B, vector<vector<double>>(T, vector<double>(D, 0.0)));
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < D; ++i) {
                    q_last[b][i] = dot_simd(&Wq[idx2d(i, 0, D)], x[b][last].data(), D);
                }
                for (int t = 0; t < T; ++t) {
                    for (int i = 0; i < D; ++i) {
                        k_all[b][t][i] = dot_simd(&Wk[idx2d(i, 0, D)], x[b][t].data(), D);
                        v_all[b][t][i] = dot_simd(&Wv[idx2d(i, 0, D)], x[b][t].data(), D);
                    }
                }
            }

            vector<vector<double>> a(B, vector<double>(T, 0.0));
            vector<vector<double>> head(B, vector<double>(D, 0.0));
            for (int b = 0; b < B; ++b) {
                vector<double> scores(T);
                for (int t = 0; t < T; ++t) {
                    scores[t] = dot_simd(q_last[b].data(), k_all[b][t].data(), D) * scale;
                }
                a[b] = softmax(scores);
                for (int t = 0; t < T; ++t) axpy_simd(head[b].data(), v_all[b][t].data(), a[b][t], D);
            }

            vector<vector<double>> attn_out(B, vector<double>(D, 0.0));
            vector<vector<double>> pre1(B, vector<double>(D, 0.0));
            vector<vector<double>> h1_norm(B, vector<double>(D, 0.0));
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < D; ++i) {
                    attn_out[b][i] = dot_simd(&Wo[idx2d(i, 0, D)], head[b].data(), D);
                    pre1[b][i] = x[b][last][i] + attn_out[b][i];
                }
                h1_norm[b] = layer_norm_forward(pre1[b], ln1_gamma, ln1_beta);
            }

            vector<vector<double>> ff_pre(B, vector<double>(FF, 0.0));
            vector<vector<double>> ff(B, vector<double>(FF, 0.0));
            vector<vector<double>> ff2(B, vector<double>(D, 0.0));
            vector<vector<double>> pre2(B, vector<double>(D, 0.0));
            vector<vector<double>> h2_norm(B, vector<double>(D, 0.0));
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < FF; ++i) {
                    double z = bff1[i] + dot_simd(&Wff1[idx2d(i, 0, D)], h1_norm[b].data(), D);
                    ff_pre[b][i] = z;
                    ff[b][i] = z > 0.0 ? z : 0.0;
                }
                for (int i = 0; i < D; ++i) {
                    double z = bff2[i] + dot_simd(&Wff2[idx2d(i, 0, FF)], ff[b].data(), FF);
                    ff2[b][i] = z;
                    pre2[b][i] = h1_norm[b][i] + z;
                }
                h2_norm[b] = layer_norm_forward(pre2[b], ln2_gamma, ln2_beta);
            }

            vector<vector<double>> y(B, vector<double>(vocab, 0.0));
            for (int b = 0; b < B; ++b) {
                vector<double> logits(vocab, 0.0);
                for (int k = 0; k < vocab; ++k) {
                    logits[k] = bout[k] + dot_simd(&Wout[idx2d(k, 0, D)], h2_norm[b].data(), D);
                }
                y[b] = softmax(logits);
                total_loss += -log(y[b][batch_target[b]] + 1e-12);
            }

            // Accumulated gradients over the whole batch.
            vector<vector<double>> gWout_sum(vocab, vector<double>(D, 0.0));
            vector<double> gbout_sum(vocab, 0.0);
            vector<vector<double>> gWo_sum(D, vector<double>(D, 0.0));
            vector<vector<double>> gWq_sum(D, vector<double>(D, 0.0));
            vector<vector<double>> gWk_sum(D, vector<double>(D, 0.0));
            vector<vector<double>> gWv_sum(D, vector<double>(D, 0.0));
            vector<vector<double>> gWff1_sum(FF, vector<double>(D, 0.0));
            vector<double> gbff1_sum(FF, 0.0);
            vector<vector<double>> gWff2_sum(D, vector<double>(FF, 0.0));
            vector<double> gbff2_sum(D, 0.0);
            vector<double> gln1_gamma_sum(D, 0.0), gln1_beta_sum(D, 0.0);
            vector<double> gln2_gamma_sum(D, 0.0), gln2_beta_sum(D, 0.0);
            vector<vector<double>> dtoken_emb_sum(vocab, vector<double>(D, 0.0));
            vector<vector<double>> dpos_emb_sum(T, vector<double>(D, 0.0));

            for (int b = 0; b < B; ++b) {
                vector<double> dz = y[b];
                dz[batch_target[b]] -= 1.0;

                vector<double> dh2_norm(D, 0.0);
                for (int k = 0; k < vocab; ++k) {
                    gbout_sum[k] += dz[k];
                    for (int j = 0; j < D; ++j) {
                        gWout_sum[k][j] += dz[k] * h2_norm[b][j];
                        dh2_norm[j] += Wout[idx2d(k, j, D)] * dz[k];
                    }
                }

                vector<double> gln2_gamma(D, 0.0), gln2_beta(D, 0.0);
                vector<double> dpre2 = layer_norm_backward(pre2[b], ln2_gamma, dh2_norm, gln2_gamma, gln2_beta);

                vector<double> dh1_norm = dpre2;
                vector<double> dff2 = dpre2;

                vector<double> dff(FF, 0.0);
                for (int i = 0; i < D; ++i) {
                    gbff2_sum[i] += dff2[i];
                    for (int j = 0; j < FF; ++j) {
                        gWff2_sum[i][j] += dff2[i] * ff[b][j];
                        dff[j] += Wff2[idx2d(i, j, FF)] * dff2[i];
                    }
                }

                vector<double> dff_pre(FF, 0.0);
                for (int i = 0; i < FF; ++i) dff_pre[i] = ff_pre[b][i] > 0.0 ? dff[i] : 0.0;
                for (int i = 0; i < FF; ++i) {
                    gbff1_sum[i] += dff_pre[i];
                    for (int j = 0; j < D; ++j) {
                        gWff1_sum[i][j] += dff_pre[i] * h1_norm[b][j];
                        dh1_norm[j] += Wff1[idx2d(i, j, D)] * dff_pre[i];
                    }
                }

                vector<double> gln1_gamma(D, 0.0), gln1_beta(D, 0.0);
                vector<double> dpre1 = layer_norm_backward(pre1[b], ln1_gamma, dh1_norm, gln1_gamma, gln1_beta);

                for (int d = 0; d < D; ++d) {
                    gln1_gamma_sum[d] += gln1_gamma[d];
                    gln1_beta_sum[d] += gln1_beta[d];
                    gln2_gamma_sum[d] += gln2_gamma[d];
                    gln2_beta_sum[d] += gln2_beta[d];
                }

                vector<double> dx_last = dpre1;

                vector<double> dhead(D, 0.0);
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        gWo_sum[i][j] += dpre1[i] * head[b][j];
                        dhead[j] += Wo[idx2d(i, j, D)] * dpre1[i];
                    }
                }

                vector<double> da(T, 0.0);
                vector<vector<double>> dv_all(T, vector<double>(D, 0.0));
                for (int t = 0; t < T; ++t) {
                    for (int d = 0; d < D; ++d) {
                        dv_all[t][d] += a[b][t] * dhead[d];
                        da[t] += dhead[d] * v_all[b][t][d];
                    }
                }

                double sum_da_a = 0.0;
                for (int t = 0; t < T; ++t) sum_da_a += da[t] * a[b][t];
                vector<double> dscores(T, 0.0);
                for (int t = 0; t < T; ++t) dscores[t] = a[b][t] * (da[t] - sum_da_a);

                vector<double> dq_last(D, 0.0);
                vector<vector<double>> dk_all(T, vector<double>(D, 0.0));
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

                vector<vector<double>> dx_all(T, vector<double>(D, 0.0));
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
                for (int j = 0; j < D; ++j) Wout[idx2d(k, j, D)] -= step_lr * gWout_sum[k][j];
                bout[k] -= step_lr * gbout_sum[k];
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    Wo[idx2d(i, j, D)] -= step_lr * gWo_sum[i][j];
                    Wq[idx2d(i, j, D)] -= step_lr * gWq_sum[i][j];
                    Wk[idx2d(i, j, D)] -= step_lr * gWk_sum[i][j];
                    Wv[idx2d(i, j, D)] -= step_lr * gWv_sum[i][j];
                }
                for (int j = 0; j < FF; ++j) Wff2[idx2d(i, j, FF)] -= step_lr * gWff2_sum[i][j];
                bff2[i] -= step_lr * gbff2_sum[i];

                ln1_gamma[i] -= step_lr * gln1_gamma_sum[i];
                ln1_beta[i] -= step_lr * gln1_beta_sum[i];
                ln2_gamma[i] -= step_lr * gln2_gamma_sum[i];
                ln2_beta[i] -= step_lr * gln2_beta_sum[i];
            }
            for (int i = 0; i < FF; ++i) {
                for (int j = 0; j < D; ++j) Wff1[idx2d(i, j, D)] -= step_lr * gWff1_sum[i][j];
                bff1[i] -= step_lr * gbff1_sum[i];
            }
            for (int tok = 0; tok < vocab; ++tok) {
                for (int d = 0; d < D; ++d) token_emb[idx2d(tok, d, D)] -= step_lr * dtoken_emb_sum[tok][d];
            }
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) pos_emb[idx2d(t, d, D)] -= step_lr * dpos_emb_sum[t][d];
            }
        }

        double avg_loss = total_loss / (double)samples.size();
        cout << "Epoch " << (ep + 1) << "/" << epochs << " avg loss: " << avg_loss << endl;
    }
}

string ChatModel::generate(const vector<int> &context, int length, double temperature, bool deterministic) const {
    vector<int> ctx = context;
    if (ctx.empty()) ctx.push_back(0);

    vector<int> out_tokens = ctx;
    for (int step = 0; step < length; ++step) {
        vector<int> norm = normalize_context(ctx);
        vector<double> h_last = forward_last_hidden(norm);

        vector<double> logits(vocab);
        for (int k = 0; k < vocab; ++k) {
            logits[k] = bout[k] + dot_simd(&Wout[idx2d(k, 0, D)], h_last.data(), D);
        }

        int next_idx = sample_next_token(logits, temperature, deterministic);
        ctx.push_back(next_idx);
        out_tokens.push_back(next_idx);
    }

    return detokenize_bytes(out_tokens);
}
