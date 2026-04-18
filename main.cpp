#include "model.h"
#include "tokenizer.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main() {
    srand((unsigned)time(nullptr));

    string text =
        "hello there i am an offline chatbot created to demonstrate how a small neural network can learn patterns from text "
        "i do not use the internet and i do not rely on external data everything i know is written inside this program "
        "you can ask simple questions and i will try to respond based on the patterns i learned during training "
        "my goal is not to be perfect but to show how a tiny model can generate sentences and imitate conversation "
        "the more text i am trained on the better i become at forming coherent replies and continuing your messages "
        "this dataset is small but it teaches basic structure like greetings statements and simple explanations "
        "feel free to experiment with different prompts and see how the model reacts to your input "
        "remember that this chatbot is only a demonstration and not a full language model but it can still be fun to interact with ";

    ifstream wiki_in("wikipedia.txt");
    if (wiki_in) {
        string wiki((istreambuf_iterator<char>(wiki_in)), istreambuf_iterator<char>());
        text += "\n";
        text += wiki;
    }

    vector<int> data = tokenize_bytes(text);
    if (data.size() < 16) {
        cout << "Not enough training tokens.\n";
        return 0;
    }

    const int vocab = 256;
    const int model_dim = 64;
    const int seq_len = 32;
    const float learning_rate = 0.05f;
    const int epochs_if_loaded = 10;
    const int epochs_if_fresh = 50;
    const int batch_size = 16;
    const string weights_file = "weights.bin";

    cout << "Vocab size: " << vocab << ", training tokens: " << data.size() << endl;

    ChatModel model(vocab, model_dim, seq_len);

    bool loaded = model.load_weights(weights_file);
    int epochs_to_train = loaded ? epochs_if_loaded : epochs_if_fresh;
    if (loaded) {
        cout << "Loaded weights from " << weights_file << ". Continuing training for "
             << epochs_to_train << " epochs.\n";
    } else {
        cout << "No valid weights found. Training model for "
             << epochs_to_train << " epochs from scratch...\n";
    }

    model.train(data, epochs_to_train, learning_rate, batch_size);

    if (model.save_weights(weights_file)) {
        cout << "Saved weights to " << weights_file << ".\n";
    } else {
        cout << "Warning: failed to save weights to " << weights_file << ".\n";
    }

    cout << "\nChatbot ready.\n";
    cout << "Type a message.\n";
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

        vector<int> ctx = tokenize_bytes(line);
        if (ctx.empty()) ctx.push_back(0);

        string reply = model.generate(ctx, 15, temperature, deterministic);
        cout << "Bot: " << reply << "\n";
    }

    return 0;
}
