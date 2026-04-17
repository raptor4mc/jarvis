#include "model.h"
#include "tokenizer.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

ChatModel::ChatModel(int vocab_size, int model_dim, int seq_len)
    : vocab(vocab_size), D(model_dim), T(seq_len), FF(model_dim * 2),
      token_emb(vocab, vector<double>(D)),
      pos_emb(T, vector<double>(D)),
      Wq(D, vector<double>(D)), Wk(D, vector<double>(D)), Wv(D, vector<double>(D)), Wo(D, vector<double>(D)),
      Wff1(FF, vector<double>(D)), bff1(FF, 0.0),
      Wff2(D, vector<double>(FF)), bff2(D, 0.0),
      Wout(vocab, vector<double>(D)), bout(vocab, 0.0) {
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < D; ++j) token_emb[i][j] = rand_weight();
    }
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < D; ++j) pos_emb[t][j] = rand_weight();
    }

    auto init_mat = [&](vector<vector<double>> &M) {
        for (auto &row : M) for (double &v : row) v = rand_weight();
    };

    init_mat(Wq); init_mat(Wk); init_mat(Wv); init_mat(Wo);
    init_mat(Wff1); init_mat(Wff2); init_mat(Wout);
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
        for (int j = 0; j < D; ++j) x[t][j] = token_emb[tok][j] + pos_emb[t][j];
    }

    // q, k, v
    vector<vector<double>> q(T, vector<double>(D)), k(T, vector<double>(D)), v(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < D; ++i) {
            double qv = 0.0, kv = 0.0, vv = 0.0;
            for (int j = 0; j < D; ++j) {
                qv += Wq[i][j] * x[t][j];
                kv += Wk[i][j] * x[t][j];
                vv += Wv[i][j] * x[t][j];
            }
            q[t][i] = qv;
            k[t][i] = kv;
            v[t][i] = vv;
        }
    }

    // Causal self-attention.
    vector<vector<double>> attn_out(T, vector<double>(D, 0.0));
    const double scale = 1.0 / sqrt((double)D);
    for (int t = 0; t < T; ++t) {
        vector<double> scores(T, -1e9);
        for (int j = 0; j <= t; ++j) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) dot += q[t][d] * k[j][d];
            scores[j] = dot * scale;
        }
        vector<double> a = softmax(scores);

        vector<double> head(D, 0.0);
        for (int j = 0; j <= t; ++j) {
            for (int d = 0; d < D; ++d) head[d] += a[j] * v[j][d];
        }

        for (int i = 0; i < D; ++i) {
            double out = 0.0;
            for (int j = 0; j < D; ++j) out += Wo[i][j] * head[j];
            attn_out[t][i] = out;
        }
    }

    // Residual after attention.
    vector<vector<double>> h1(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) h1[t][d] = x[t][d] + attn_out[t][d];
    }

    // FFN + residual.
    vector<vector<double>> h2(T, vector<double>(D));
    for (int t = 0; t < T; ++t) {
        vector<double> ff(FF, 0.0);
        for (int i = 0; i < FF; ++i) {
            double z = bff1[i];
            for (int j = 0; j < D; ++j) z += Wff1[i][j] * h1[t][j];
            ff[i] = z > 0.0 ? z : 0.0; // ReLU
        }

        for (int i = 0; i < D; ++i) {
            double z = bff2[i];
            for (int j = 0; j < FF; ++j) z += Wff2[i][j] * ff[j];
            h2[t][i] = h1[t][i] + z;
        }
    }

    return h2[T - 1];
}

bool ChatModel::load_weights(const string &filename) {
    ifstream in(filename, ios::binary);
    if (!in) return false;

    auto read_mat = [&](vector<vector<double>> &M) {
        for (auto &row : M) {
            in.read(reinterpret_cast<char*>(row.data()), (long long)row.size() * sizeof(double));
            if (!in) return false;
        }
        return true;
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
    if (!read_mat(Wout)) return false;
    if (!read_vec(bout)) return false;

    return true;
}

bool ChatModel::save_weights(const string &filename) const {
    ofstream out(filename, ios::binary);
    if (!out) return false;

    auto write_mat = [&](const vector<vector<double>> &M) {
        for (const auto &row : M) out.write(reinterpret_cast<const char*>(row.data()), (long long)row.size() * sizeof(double));
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
    write_mat(Wout);
    write_vec(bout);

    return (bool)out;
}

void ChatModel::train(const vector<int> &data, int epochs, double lr) {
    if ((int)data.size() <= T) return;

    struct Sample { vector<int> ctx; int target; };
    vector<Sample> samples;
    samples.reserve(data.size());
    for (size_t i = 0; i + T < data.size(); ++i) {
        vector<int> ctx(data.begin() + i, data.begin() + i + T);
        int target = data[i + T];
        samples.push_back({ctx, target});
    }

    const size_t max_samples = 8;
    if (samples.size() > max_samples) samples.resize(max_samples);

    cout << "Samples: " << samples.size() << endl;

    for (int ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;

        for (const auto &s : samples) {
            vector<int> ctx = normalize_context(s.ctx);

            vector<vector<double>> x(T, vector<double>(D));
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) x[t][d] = token_emb[ctx[t]][d] + pos_emb[t][d];
            }

            int last = T - 1;
            vector<double> q_last(D);
            vector<vector<double>> k_all(T, vector<double>(D));
            vector<vector<double>> v_all(T, vector<double>(D));

            for (int i = 0; i < D; ++i) {
                double qv = 0.0;
                for (int j = 0; j < D; ++j) qv += Wq[i][j] * x[last][j];
                q_last[i] = qv;
            }
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < D; ++i) {
                    double kv = 0.0, vv = 0.0;
                    for (int j = 0; j < D; ++j) {
                        kv += Wk[i][j] * x[t][j];
                        vv += Wv[i][j] * x[t][j];
                    }
                    k_all[t][i] = kv;
                    v_all[t][i] = vv;
                }
            }

            const double scale = 1.0 / sqrt((double)D);
            vector<double> scores(T);
            for (int t = 0; t < T; ++t) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) dot += q_last[d] * k_all[t][d];
                scores[t] = dot * scale;
            }
            vector<double> a = softmax(scores);

            vector<double> head(D, 0.0);
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) head[d] += a[t] * v_all[t][d];
            }

            vector<double> attn_out(D, 0.0);
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) attn_out[i] += Wo[i][j] * head[j];
            }

            vector<double> h1_last(D);
            for (int d = 0; d < D; ++d) h1_last[d] = x[last][d] + attn_out[d];

            vector<double> ff_pre(FF), ff(FF);
            for (int i = 0; i < FF; ++i) {
                double z = bff1[i];
                for (int j = 0; j < D; ++j) z += Wff1[i][j] * h1_last[j];
                ff_pre[i] = z;
                ff[i] = z > 0.0 ? z : 0.0;
            }

            vector<double> ff2(D);
            for (int i = 0; i < D; ++i) {
                double z = bff2[i];
                for (int j = 0; j < FF; ++j) z += Wff2[i][j] * ff[j];
                ff2[i] = z;
            }

            vector<double> h2_last(D);
            for (int d = 0; d < D; ++d) h2_last[d] = h1_last[d] + ff2[d];

            vector<double> logits(vocab);
            for (int k = 0; k < vocab; ++k) {
                double z = bout[k];
                for (int j = 0; j < D; ++j) z += Wout[k][j] * h2_last[j];
                logits[k] = z;
            }

            vector<double> y = softmax(logits);
            total_loss += -log(y[s.target] + 1e-12);

            vector<double> dz(vocab);
            for (int k = 0; k < vocab; ++k) dz[k] = y[k];
            dz[s.target] -= 1.0;

            vector<vector<double>> gWout(vocab, vector<double>(D, 0.0));
            vector<double> gbout(vocab, 0.0);
            vector<double> dh2_last(D, 0.0);
            for (int k = 0; k < vocab; ++k) {
                gbout[k] = dz[k];
                for (int j = 0; j < D; ++j) {
                    gWout[k][j] = dz[k] * h2_last[j];
                    dh2_last[j] += Wout[k][j] * dz[k];
                }
            }

            // Backprop FFN and residual.
            vector<double> dh1_last = dh2_last;
            vector<double> dff2 = dh2_last;

            vector<vector<double>> gWff2(D, vector<double>(FF, 0.0));
            vector<double> gbff2(D, 0.0);
            vector<double> dff(FF, 0.0);
            for (int i = 0; i < D; ++i) {
                gbff2[i] = dff2[i];
                for (int j = 0; j < FF; ++j) {
                    gWff2[i][j] = dff2[i] * ff[j];
                    dff[j] += Wff2[i][j] * dff2[i];
                }
            }

            vector<double> dff_pre(FF, 0.0);
            for (int i = 0; i < FF; ++i) dff_pre[i] = ff_pre[i] > 0.0 ? dff[i] : 0.0;

            vector<vector<double>> gWff1(FF, vector<double>(D, 0.0));
            vector<double> gbff1(FF, 0.0);
            for (int i = 0; i < FF; ++i) {
                gbff1[i] = dff_pre[i];
                for (int j = 0; j < D; ++j) {
                    gWff1[i][j] = dff_pre[i] * h1_last[j];
                    dh1_last[j] += Wff1[i][j] * dff_pre[i];
                }
            }

            // Backprop attention path.
            vector<double> dattn_out = dh1_last;
            vector<double> dx_last(D, 0.0);
            for (int d = 0; d < D; ++d) dx_last[d] += dh1_last[d]; // residual x[last] -> h1[last]

            vector<vector<double>> gWo(D, vector<double>(D, 0.0));
            vector<double> dhead(D, 0.0);
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    gWo[i][j] = dattn_out[i] * head[j];
                    dhead[j] += Wo[i][j] * dattn_out[i];
                }
            }

            vector<double> da(T, 0.0);
            vector<vector<double>> dv_all(T, vector<double>(D, 0.0));
            for (int t = 0; t < T; ++t) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d) {
                    dv_all[t][d] += a[t] * dhead[d];
                    dot += dhead[d] * v_all[t][d];
                }
                da[t] = dot;
            }

            double sum_da_a = 0.0;
            for (int t = 0; t < T; ++t) sum_da_a += da[t] * a[t];
            vector<double> dscores(T, 0.0);
            for (int t = 0; t < T; ++t) dscores[t] = a[t] * (da[t] - sum_da_a);

            vector<double> dq_last(D, 0.0);
            vector<vector<double>> dk_all(T, vector<double>(D, 0.0));
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) {
                    dq_last[d] += scale * dscores[t] * k_all[t][d];
                    dk_all[t][d] += scale * dscores[t] * q_last[d];
                }
            }

            vector<vector<double>> gWq(D, vector<double>(D, 0.0));
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    gWq[i][j] = dq_last[i] * x[last][j];
                    dx_last[j] += Wq[i][j] * dq_last[i];
                }
            }

            vector<vector<double>> gWk(D, vector<double>(D, 0.0));
            vector<vector<double>> gWv(D, vector<double>(D, 0.0));
            vector<vector<double>> dx_all(T, vector<double>(D, 0.0));
            for (int t = 0; t < T; ++t) {
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        gWk[i][j] += dk_all[t][i] * x[t][j];
                        gWv[i][j] += dv_all[t][i] * x[t][j];
                        dx_all[t][j] += Wk[i][j] * dk_all[t][i] + Wv[i][j] * dv_all[t][i];
                    }
                }
            }
            for (int d = 0; d < D; ++d) dx_all[last][d] += dx_last[d];

            // SGD update.
            for (int k = 0; k < vocab; ++k) {
                for (int j = 0; j < D; ++j) Wout[k][j] -= lr * gWout[k][j];
                bout[k] -= lr * gbout[k];
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    Wo[i][j] -= lr * gWo[i][j];
                    Wq[i][j] -= lr * gWq[i][j];
                    Wk[i][j] -= lr * gWk[i][j];
                    Wv[i][j] -= lr * gWv[i][j];
                }
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < FF; ++j) Wff2[i][j] -= lr * gWff2[i][j];
                bff2[i] -= lr * gbff2[i];
            }
            for (int i = 0; i < FF; ++i) {
                for (int j = 0; j < D; ++j) Wff1[i][j] -= lr * gWff1[i][j];
                bff1[i] -= lr * gbff1[i];
            }
            for (int t = 0; t < T; ++t) {
                int tok = ctx[t];
                for (int d = 0; d < D; ++d) {
                    token_emb[tok][d] -= lr * dx_all[t][d];
                    pos_emb[t][d] -= lr * dx_all[t][d];
                }
            }
        }

        if (ep % 100 == 0) {
            cout << "Epoch " << ep << " loss: " << total_loss << endl;
        }
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
            double z = bout[k];
            for (int j = 0; j < D; ++j) z += Wout[k][j] * h_last[j];
            logits[k] = z;
        }

        int next_idx = sample_next_token(logits, temperature, deterministic);
        ctx.push_back(next_idx);
        out_tokens.push_back(next_idx);
    }

    return detokenize_bytes(out_tokens);
}
