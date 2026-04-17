#include "tokenizer.h"

std::vector<int> tokenize_bytes(const std::string &s) {
    std::vector<int> tokens;
    tokens.reserve(s.size());
    for (unsigned char c : s) tokens.push_back((int)c);
    return tokens;
}

std::string detokenize_bytes(const std::vector<int> &tokens) {
    std::string out;
    out.reserve(tokens.size());
    for (int t : tokens) {
        if (t >= 0 && t <= 255) out.push_back((char)t);
    }
    return out;
}
