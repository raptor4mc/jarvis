#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>

using namespace std;

double rand_weight() {
    return ((double)rand() / RAND_MAX - 0.5) / 5.0;
}

double fast_tanh(double x) {
    return tanh(x);
}

vector<string> split_words(const string &s) {
    vector<string> words;
    string cur;
    for (char c : s) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

vector<double> softmax(const vector<double> &z) {
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

int sample_next_token(const vector<double> &logits, double temperature, bool deterministic) {
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

int main() {
    srand((unsigned)time(nullptr));

    // Training text (original, self-contained)
    string text =
        "hello there i am an offline chatbot created to demonstrate how a small neural network can learn patterns from text "
        "i do not use the internet and i do not rely on external data everything i know is written inside this program "
        "you can ask simple questions and i will try to respond based on the patterns i learned during training "
        "my goal is not to be perfect but to show how a tiny model can generate sentences and imitate conversation "
        "the more text i am trained on the better i become at forming coherent replies and continuing your messages "
        "this dataset is small but it teaches basic structure like greetings statements and simple explanations "
        "feel free to experiment with different prompts and see how the model reacts to your input "
        "remember that this chatbot is only a demonstration and not a full language model but it can still be fun to interact with ";

    vector<string> words = split_words(text);
    if (words.size() < 10) {
        cout << "Not enough training words.\n";
        return 0;
    }

    // Build vocabulary
    map<string,int> stoi;
    map<int,string> itos;
    for (const string &w : words) {
        if (stoi.find(w) == stoi.end()) {
            int idx = (int)stoi.size();
            stoi[w] = idx;
            itos[idx] = w;
        }
    }
    int vocab = (int)stoi.size();
    cout << "Vocab size: " << vocab << ", training words: " << words.size() << endl;

    // Encode words as indices
    vector<int> data;
    for (const string &w : words) data.push_back(stoi[w]);

    // Build training samples: 3-word context -> next word
    struct Sample { int w0, w1, w2, target; };
    vector<Sample> samples;
    for (size_t i = 0; i + 3 < data.size(); ++i) {
        Sample s;
        s.w0 = data[i];
        s.w1 = data[i+1];
        s.w2 = data[i+2];
        s.target = data[i+3];
        samples.push_back(s);
    }
    cout << "Samples: " << samples.size() << endl;

    // Neural network: 3-word context -> 2 hidden layers -> vocab
    const int H = 64; // hidden size
    const int D = 12; // embedding size

    // Embedding table: vocab x D
    vector<vector<double>> E(vocab, vector<double>(D));

    // First hidden layer: H x (3*D) for concatenated 3-word embeddings
    vector<vector<double>> W1(H, vector<double>(3 * D));
    vector<double> b1(H, 0.0);

    // Second hidden layer: H x H
    vector<vector<double>> W2(H, vector<double>(H));
    vector<double> b2(H, 0.0);

    // Output layer: vocab x H
    vector<vector<double>> W3(vocab, vector<double>(H));
    vector<double> b3(vocab, 0.0);

    // Init weights
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < D; ++j) E[i][j] = rand_weight();
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < 3 * D; ++j) W1[i][j] = rand_weight();
        b1[i] = 0.0;
    }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) W2[i][j] = rand_weight();
        b2[i] = 0.0;
    }
    for (int i = 0; i < vocab; ++i) {
        for (int j = 0; j < H; ++j) W3[i][j] = rand_weight();
        b3[i] = 0.0;
    }

    double lr = 0.05;
    const int epochs_if_loaded = 100;
    const int epochs_if_fresh = 600;
    const string weights_file = "weights.bin";

    auto load_weights = [&](const string &filename) {
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
    };

    auto save_weights = [&](const string &filename) {
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
    };

    bool loaded = load_weights(weights_file);
    int epochs_to_train = loaded ? epochs_if_loaded : epochs_if_fresh;
    if (loaded) {
        cout << "Loaded weights from " << weights_file << ". Continuing training for "
             << epochs_to_train << " epochs.\n";
    } else {
        cout << "No valid weights found. Training model for "
             << epochs_to_train << " epochs from scratch...\n";
    }

    auto train_model = [&](int epochs) {
        for (int ep = 0; ep < epochs; ++ep) {
            double total_loss = 0.0;

            for (const auto &s : samples) {
                int w0 = s.w0, w1 = s.w1, w2 = s.w2, t = s.target;

                // Forward
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
                double loss = -log(y[t] + 1e-12);
                total_loss += loss;

                // Backprop
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
                for (int j = 0; j < H; ++j) {
                    dz2[j] = dh2[j] * (1.0 - h2[j] * h2[j]);
                }

                vector<double> dh1(H, 0.0);
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < H; ++j) {
                        dh1[j] += dz2[i] * W2[i][j];
                        W2[i][j] -= lr * dz2[i] * h1[j];
                    }
                    b2[i] -= lr * dz2[i];
                }

                vector<double> dz1(H);
                for (int j = 0; j < H; ++j) {
                    dz1[j] = dh1[j] * (1.0 - h1[j] * h1[j]);
                }

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
    };

    train_model(epochs_to_train);

    if (save_weights(weights_file)) {
        cout << "Saved weights to " << weights_file << ".\n";
    } else {
        cout << "Warning: failed to save weights to " << weights_file << ".\n";
    }

    // Generation: given last 3 known words, predict continuation
    auto generate = [&](const vector<int> &context, int length, double temperature, bool deterministic) {
        vector<int> ctx = context;
        while (ctx.size() < 3) ctx.insert(ctx.begin(), ctx.front());
        string out;
        for (int idx : ctx) {
            out += itos[idx] + " ";
        }

        for (int step = 0; step < length; ++step) {
            int w0 = ctx[ctx.size()-3];
            int w1 = ctx[ctx.size()-2];
            int w2 = ctx[ctx.size()-1];

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

            int next_idx = sample_next_token(z3, temperature, deterministic);

            ctx.push_back(next_idx);
            out += itos[next_idx] + " ";
        }
        return out;
    };

    cout << "\nChatbot ready.\n";
    cout << "Type a message (using simple words like in the training text).\n";
    cout << "Use /temp <value> to control randomness (example: /temp 0.7).\n";
    cout << "Use /det on or /det off for deterministic generation.\n";
    cout << "Type 'quit' to exit.\n\n";

    double temperature = 1.0;
    bool deterministic = false;

    while (true) {
        cout << "You: ";
        string line;
        if (!getline(cin, line)) break;
        if (line == "quit") break;
        if (line.rfind("/temp", 0) == 0) {
            istringstream iss(line);
            string cmd;
            double t;
            iss >> cmd >> t;
            if (iss && t > 0.0) {
                temperature = t;
                cout << "Bot: temperature set to " << temperature << "\n";
            } else {
                cout << "Bot: invalid temperature. use /temp <positive number>\n";
            }
            continue;
        }
        if (line == "/det on") {
            deterministic = true;
            cout << "Bot: deterministic generation enabled\n";
            continue;
        }
        if (line == "/det off") {
            deterministic = false;
            cout << "Bot: deterministic generation disabled\n";
            continue;
        }

        vector<string> in_words = split_words(line);
        vector<int> ctx;
        for (const string &w : in_words) {
            auto it = stoi.find(w);
            if (it != stoi.end()) ctx.push_back(it->second);
        }
        if (ctx.empty()) {
            cout << "Bot: i do not know these words try simpler ones like hello or chatbot\n";
            continue;
        }
        if (ctx.size() < 3) {
            while (ctx.size() < 3) ctx.insert(ctx.begin(), ctx.front());
        }

        string reply = generate(ctx, 15, temperature, deterministic);
        cout << "Bot: " << reply << "\n";
    }

    return 0;
}
