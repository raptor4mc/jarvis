#ifndef MODEL_H
#define MODEL_H

#include <cstddef>
#include <string>
#include <vector>

class ChatModel {
public:
    // model_dim is Transformer width, seq_len is context length.
    ChatModel(int vocab, int model_dim, int seq_len);

    bool load_weights(const std::string &filename);
    bool save_weights(const std::string &filename) const;

    void train(const std::vector<int> &data, int epochs, float lr, int batch_size = 1, int sample_stride = 1);
    std::string generate(const std::vector<int> &context, int length, double temperature, bool deterministic) const;

private:
    static size_t idx2d(int r, int c, int cols);

    int vocab;
    int D;
    int T;
    int FF;

    std::vector<float> token_emb; // vocab x D
    std::vector<float> pos_emb;   // T x D

    // Single-head self-attention block parameters.
    std::vector<float> Wq, Wk, Wv, Wo; // D x D

    // Feed-forward network parameters.
    std::vector<float> Wff1;              // FF x D
    std::vector<float> bff1;              // FF
    std::vector<float> Wff2;              // D x FF
    std::vector<float> bff2;              // D

    // LayerNorm parameters (post-attention and post-FFN).
    std::vector<float> ln1_gamma, ln1_beta; // D
    std::vector<float> ln2_gamma, ln2_beta; // D

    // LM head.
    std::vector<float> Wout;              // vocab x D
    std::vector<float> bout;              // vocab

    static float rand_weight();
    static std::vector<float> softmax(const std::vector<float> &z);
    static int sample_next_token(const std::vector<float> &logits, double temperature, bool deterministic);
    static std::vector<float> layer_norm_forward(const std::vector<float> &x, const std::vector<float> &gamma, const std::vector<float> &beta);
    static std::vector<float> layer_norm_backward(
        const std::vector<float> &x,
        const std::vector<float> &gamma,
        const std::vector<float> &dout,
        std::vector<float> &dgamma,
        std::vector<float> &dbeta);

    std::vector<float> forward_last_hidden(const std::vector<int> &tokens) const;
    std::vector<int> normalize_context(const std::vector<int> &tokens) const;
};

#endif
