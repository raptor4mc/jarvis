#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>

class ChatModel {
public:
    ChatModel(int vocab, int hidden_size, int embedding_size);

    bool load_weights(const std::string &filename);
    bool save_weights(const std::string &filename) const;

    void train(const std::vector<int> &data, int epochs, double lr);
    std::string generate(const std::vector<int> &context, int length, double temperature, bool deterministic) const;

private:
    int vocab;
    int H;
    int D;

    std::vector<std::vector<double>> E;
    std::vector<std::vector<double>> W1;
    std::vector<double> b1;

    std::vector<std::vector<double>> W2;
    std::vector<double> b2;

    std::vector<std::vector<double>> W3;
    std::vector<double> b3;

    static double rand_weight();
    static double fast_tanh(double x);
    static std::vector<double> softmax(const std::vector<double> &z);
    static int sample_next_token(const std::vector<double> &logits, double temperature, bool deterministic);
};

#endif
