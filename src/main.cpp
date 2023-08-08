
#include "essential.hpp"
#include "neural_network.hpp"
#include "xor.hpp"

void showValues(std::vector<float> predicted)
{
    for (float v : predicted) {
        std::cout << " " << v;
    }
    std::cout << std::endl;

}

std::tuple<std::vector<float>, std::vector<float>> getExemple()
{
    int v1 = std::rand() % 2;
    int v2 = std::rand() % 2;
    int expect = XOR_GATE(v1, v2);

    std::vector<float> entry = {static_cast<float>(v1), static_cast<float>(v2)};
    std::vector<float> output = {static_cast<float>(expect)};
    return {entry, output};
}

int main(int ac, char **argc)
{
    NeuralNetwork nn(2);
    nn.stackLayer(1);
    // learning
    int epoch = 10;
    for (int i = 0; i < epoch; i++) {
        auto sample = getExemple();
        auto data = std::get<0>(sample);
        auto expectation = std::get<1>(sample);
        nn.learnSample(data, expectation);
    }
    // entry values to network
    std::vector<float> inputs;
    inputs.push_back(4);
    // prediction
    std::vector<float> predicted = nn.predict(inputs);
    std::cout << "inputs :" << std::endl;
    showValues(inputs);
    std::cout << "prediction :" << std::endl;
    showValues(predicted);
}