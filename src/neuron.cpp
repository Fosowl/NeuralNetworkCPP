
#include "neuron.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>

Neuron::Neuron(int weightCount)
{
    bias = 0.0f;
    output = 0.0f;
    initializeWeight(weightCount);
}

float Neuron::sigmoid(float z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

void Neuron::initializeWeight(int weightCount)
{
    for (size_t i = 0; i < weightCount; i++) {
        float weight = static_cast<float>(std::rand()) / RAND_MAX;
        weights.push_back(sigmoid(weight));
    }
}

float Neuron::sigmoidDerivative(float z)
{
    float sigmoid_value = sigmoid(z);
    return sigmoid_value * (1 - sigmoid_value);
}

float Neuron::compute(std::vector<float> inputs)
{
    float y = 0.0f;
    for (int i = 0; i < inputs.size(); i++) {
        y += inputs[i] * weights[i];
    }
    float z = sigmoid(y + bias);
    this->output = z;
    return z;
}

void Neuron::updateWeights(float single_gradient, float learning_rate)
{
    for (int i = 0; i < weights.size(); i++) {
        float weight_gradient = single_gradient * weights[i];
        // negative step because downhill direction is the negative of the gradient
        weights[i] -= learning_rate * weight_gradient;
    }
}

