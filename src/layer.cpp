
#include "layer.hpp"
#include "neuron.hpp"

Layer::Layer(int numNeurons, int numInputs)
{
    initializeNeurons(numNeurons, numInputs);
}

void Layer::initializeNeurons(int numNeurons, int numInputs)
{
    for (int i = 0; i < numNeurons; i++) {
        neurons.push_back(Neuron(numInputs));
    }
}

std::vector<float> Layer::forwardPropagate(std::vector<float> inputs)
{
    for (Neuron n : neurons) {
        float z = n.compute(inputs);
        neurons_output.push_back(z);
    }
    return neurons_output;
}

std::vector<float> Layer::computeGradients()
{
    std::vector<float> gradients;
    for (int i = 0; i < neurons_output.size(); i++) {
        gradients.push_back(neurons[i].sigmoidDerivative(neurons_output[i]));
    }
    return gradients;
}

void Layer::localGradientDescent(std::vector<float> gradients, float learning_rate)
{
    for (int i = 0; i < gradients.size(); i++) {
        neurons[i].updateWeights(gradients[i], learning_rate);
    }
}