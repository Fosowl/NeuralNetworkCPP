
#ifndef LAYER_H
#define LAYER_H

#include "neuron.hpp"
#include "essential.hpp"

class Layer {
    public:
        Layer(int numNeurons, int numInputs);
        std::vector<float> forwardPropagate(std::vector<float> inputs);
        int neuronCount() { return neurons.size(); };
        std::vector<float> computeGradients();
        void localGradientDescent(std::vector<float> gradients, float learning_rate);
        std::vector<float> getNeuronsOutputs() { return neurons_output; };
    private:
        void initializeNeurons(int numNeurons, int numInputs);
        std::vector<Neuron> neurons;
        std::vector<float> neurons_output;
};

#endif
