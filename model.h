#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>

class ChatModel {
public:
    // model_dim is Transformer width, seq_len is context length.
    ChatModel(int vocab, int model_dim, int seq_len);

    bool load_weights(const std::string &filename);
    bool save_weights(const std::string &filename) const;

    void train(const std::vector<int> &data, int epochs, double lr);
    std::string generate(const std::vector<int> &context, int length, double temperature, bool deterministic) const;

private:
    int vocab;
    int D;
    int T;
    int FF;

    std::vector<std::vector<double>> token_emb; // vocab x D
    std::vector<std::vector<double>> pos_emb;   // T x D

    // Single-head self-attention block parameters.
    std::vector<std::vector<double>> Wq, Wk, Wv, Wo; // D x D

    // Feed-forward network parameters.
    std::vector<std::vector<double>> Wff1; // FF x D
    std::vector<double> bff1;              // FF
    std::vector<std::vector<double>> Wff2; // D x FF
    std::vector<double> bff2;              // D

    // LM head.
    std::vector<std::vector<double>> Wout; // vocab x D
    std::vector<double> bout;              // vocab

    static double rand_weight();
    static std::vector<double> softmax(const std::vector<double> &z);
    static int sample_next_token(const std::vector<double> &logits, double temperature, bool deterministic);

    std::vector<double> forward_last_hidden(const std::vector<int> &tokens) const;
    std::vector<int> normalize_context(const std::vector<int> &tokens) const;
};

#endif
