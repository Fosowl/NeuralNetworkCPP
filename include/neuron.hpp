
#ifndef NEURON_H
#define NEURON_H

#include "essential.hpp"

class Neuron {
    public:
        Neuron(int weightCount);
        float compute(std::vector<float> inputs);
        void updateBias(float b) { bias = b; };
        float getOutput() { return output; };
        float sigmoidDerivative(float input);
        void updateWeights(float single_gradient, float learning_rate);
    private:
        float bias;
        float output;
        std::vector<float> weights;
        float sigmoid(float input);
        void initializeWeight(int weightCount);
};

#endif