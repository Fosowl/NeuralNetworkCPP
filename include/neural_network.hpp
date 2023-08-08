
#ifndef NETWORK_H
#define NETWORK_H

#include "layer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

class NeuralNetwork {
    public:
        NeuralNetwork(int numInputs);
        void stackLayer(int numNeurons);
        void forwardPropagation(std::vector<float> inputs);
        void backPropagation(std::vector<float> outputs, std::vector<float> expectation);
        std::vector<float> predict(std::vector<float> inputs);
        void learnSample(std::vector<float> data, std::vector<float> expectation);
        float cumulativeCost(std::vector<float> prediction, std::vector<float> actual);
        int numberInputs;
    private:
        std::vector<float> chainRule(std::vector<float> grad1, std::vector<float> grad2);
        std::vector<float> costGradient(std::vector<float> outputs, std::vector<float> expectation);
        std::vector<float> gradientDescentStep(std::vector<float> weights, std::vector<float> gradients);
        float costDerivative(float predicted, float expected);
        float calculateCost(float predicted, float expected);

        std::vector<Layer> layers;
        std::vector<float> modelInputs;
        std::vector<float> modelOutputs;
};

#endif