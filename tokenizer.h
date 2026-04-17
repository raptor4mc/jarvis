#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

std::vector<int> tokenize_bytes(const std::string &s);
std::string detokenize_bytes(const std::vector<int> &tokens);

#endif
