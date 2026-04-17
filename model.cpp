#include "model.h"
#include "tokenizer.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;

ChatModel::ChatModel(int vocab_size, int hidden_size, int embedding_size)
    : vocab(vocab_size), H(hidden_size), D(embedding_size),
      E(vocab, vector<double>(D)),
      W1(H, vector<double>(3 * D)), b1(H, 0.0),
      W2(H, vector<double>(H)), b2(H, 0.0),
      W3(vocab, vector<double>(H)), b3(vocab, 0.0) {
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < D; ++j) E[i][j] = rand_weight();
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < 3 * D; ++j) W1[i][j] = rand_weight();
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) W2[i][j] = rand_weight();
    }
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < H; ++j) W3[i][j] = rand_weight();
    }
}

double ChatModel::rand_weight() {
    return ((double)rand() / RAND_MAX - 0.5) / 5.0;
}

double ChatModel::fast_tanh(double x) {
    return tanh(x);
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

bool ChatModel::load_weights(const string &filename) {
    ifstream in(filename, ios::binary);
    if (!in) return false;

    for (int i = 0; i < vocab; ++i) {
        in.read(reinterpret_cast<char*>(E[i].data()), D * sizeof(double));
        if (!in) return false;
    }

    for (int i = 0; i < H; ++i) {
        in.read(reinterpret_cast<char*>(W1[i].data()), (3 * D) * sizeof(double));
        if (!in) return false;
    }
    in.read(reinterpret_cast<char*>(b1.data()), H * sizeof(double));
    if (!in) return false;

    for (int i = 0; i < H; ++i) {
        in.read(reinterpret_cast<char*>(W2[i].data()), H * sizeof(double));
        if (!in) return false;
    }
    in.read(reinterpret_cast<char*>(b2.data()), H * sizeof(double));
    if (!in) return false;

    for (int i = 0; i < vocab; ++i) {
        in.read(reinterpret_cast<char*>(W3[i].data()), H * sizeof(double));
        if (!in) return false;
    }
    in.read(reinterpret_cast<char*>(b3.data()), vocab * sizeof(double));
    if (!in) return false;

    return true;
}

bool ChatModel::save_weights(const string &filename) const {
    ofstream out(filename, ios::binary);
    if (!out) return false;

    for (int i = 0; i < vocab; ++i) out.write(reinterpret_cast<const char*>(E[i].data()), D * sizeof(double));
    for (int i = 0; i < H; ++i) out.write(reinterpret_cast<const char*>(W1[i].data()), (3 * D) * sizeof(double));
    out.write(reinterpret_cast<const char*>(b1.data()), H * sizeof(double));

    for (int i = 0; i < H; ++i) out.write(reinterpret_cast<const char*>(W2[i].data()), H * sizeof(double));
    out.write(reinterpret_cast<const char*>(b2.data()), H * sizeof(double));

    for (int i = 0; i < vocab; ++i) out.write(reinterpret_cast<const char*>(W3[i].data()), H * sizeof(double));
    out.write(reinterpret_cast<const char*>(b3.data()), vocab * sizeof(double));

    return (bool)out;
}

void ChatModel::train(const vector<int> &data, int epochs, double lr) {
    if (data.size() < 4) return;

    struct Sample { int w0, w1, w2, target; };
    vector<Sample> samples;
    samples.reserve(data.size());
    for (size_t i = 0; i + 3 < data.size(); ++i) {
        samples.push_back({data[i], data[i + 1], data[i + 2], data[i + 3]});
    }

    cout << "Samples: " << samples.size() << endl;

    for (int ep = 0; ep < epochs; ++ep) {
        double total_loss = 0.0;

        for (const auto &s : samples) {
            int w0 = s.w0, w1 = s.w1, w2 = s.w2, t = s.target;

            vector<double> x(3 * D);
            for (int d = 0; d < D; ++d) {
                x[d] = E[w0][d];
                x[D + d] = E[w1][d];
                x[2 * D + d] = E[w2][d];
            }

            vector<double> z1(H), h1(H);
            for (int i = 0; i < H; ++i) {
                double z = b1[i];
                for (int j = 0; j < 3 * D; ++j) z += W1[i][j] * x[j];
                z1[i] = z;
                h1[i] = fast_tanh(z);
            }

            vector<double> z2(H), h2(H);
            for (int i = 0; i < H; ++i) {
                double z = b2[i];
                for (int j = 0; j < H; ++j) z += W2[i][j] * h1[j];
                z2[i] = z;
                h2[i] = fast_tanh(z);
            }

            vector<double> z3(vocab);
            for (int k = 0; k < vocab; ++k) {
                double z = b3[k];
                for (int j = 0; j < H; ++j) z += W3[k][j] * h2[j];
                z3[k] = z;
            }

            vector<double> y = softmax(z3);
            total_loss += -log(y[t] + 1e-12);

            vector<double> dz3(vocab);
            for (int k = 0; k < vocab; ++k) dz3[k] = y[k];
            dz3[t] -= 1.0;

            vector<double> dh2(H, 0.0);
            for (int k = 0; k < vocab; ++k) {
                for (int j = 0; j < H; ++j) {
                    dh2[j] += dz3[k] * W3[k][j];
                    W3[k][j] -= lr * dz3[k] * h2[j];
                }
                b3[k] -= lr * dz3[k];
            }

            vector<double> dz2(H);
            for (int j = 0; j < H; ++j) dz2[j] = dh2[j] * (1.0 - h2[j] * h2[j]);

            vector<double> dh1(H, 0.0);
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < H; ++j) {
                    dh1[j] += dz2[i] * W2[i][j];
                    W2[i][j] -= lr * dz2[i] * h1[j];
                }
                b2[i] -= lr * dz2[i];
            }

            vector<double> dz1(H);
            for (int j = 0; j < H; ++j) dz1[j] = dh1[j] * (1.0 - h1[j] * h1[j]);

            vector<double> dx(3 * D, 0.0);
            for (int j = 0; j < H; ++j) {
                for (int k = 0; k < 3 * D; ++k) {
                    dx[k] += dz1[j] * W1[j][k];
                    W1[j][k] -= lr * dz1[j] * x[k];
                }
                b1[j] -= lr * dz1[j];
            }

            for (int d = 0; d < D; ++d) {
                E[w0][d] -= lr * dx[d];
                E[w1][d] -= lr * dx[D + d];
                E[w2][d] -= lr * dx[2 * D + d];
            }
        }

        if (ep % 100 == 0) {
            cout << "Epoch " << ep << " loss: " << total_loss << endl;
        }
    }
}

string ChatModel::generate(const vector<int> &context, int length, double temperature, bool deterministic) const {
    vector<int> ctx = context;
    while (ctx.size() < 3) ctx.insert(ctx.begin(), ctx.front());
    vector<int> out_tokens = ctx;

    for (int step = 0; step < length; ++step) {
        int w0 = ctx[ctx.size() - 3];
        int w1 = ctx[ctx.size() - 2];
        int w2 = ctx[ctx.size() - 1];

        vector<double> x(3 * D);
        for (int d = 0; d < D; ++d) {
            x[d] = E[w0][d];
            x[D + d] = E[w1][d];
            x[2 * D + d] = E[w2][d];
        }

        vector<double> h1(H);
        for (int i = 0; i < H; ++i) {
            double z = b1[i];
            for (int j = 0; j < 3 * D; ++j) z += W1[i][j] * x[j];
            h1[i] = fast_tanh(z);
        }

        vector<double> h2(H);
        for (int i = 0; i < H; ++i) {
            double z = b2[i];
            for (int j = 0; j < H; ++j) z += W2[i][j] * h1[j];
            h2[i] = fast_tanh(z);
        }

        vector<double> z3(vocab);
        for (int k = 0; k < vocab; ++k) {
            double z = b3[k];
            for (int j = 0; j < H; ++j) z += W3[k][j] * h2[j];
            z3[k] = z;
        }

        int next_idx = sample_next_token(z3, temperature, deterministic);
        ctx.push_back(next_idx);
        out_tokens.push_back(next_idx);
    }

    return detokenize_bytes(out_tokens);
}
