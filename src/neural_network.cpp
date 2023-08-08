
#include "neural_network.hpp"
#include "essential.hpp"

NeuralNetwork::NeuralNetwork(int numInputs)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    numberInputs = numInputs;
    modelInputs.resize(numInputs);
    std::fill(modelInputs.begin(), modelInputs.end(), 0);
}

void NeuralNetwork::stackLayer(int numNeurons)
{
    int lastLayerSize = layers.size() > 0 ? layers.back().neuronCount() : numberInputs;
    layers.push_back(Layer(numNeurons, lastLayerSize));
}

void NeuralNetwork::forwardPropagation(std::vector<float> inputs)
{
    if (inputs.size() != numberInputs) {
        ERROR("Bad input size for forward propagation")
        return;
    }
    for (auto layer : layers) {
        inputs = layer.forwardPropagate(inputs);
    }
    modelOutputs = inputs;
}


std::vector<float> NeuralNetwork::predict(std::vector<float> inputs)
{
    forwardPropagation(inputs);
    return modelOutputs;
}

float NeuralNetwork::calculateCost(float predicted, float expected)
{
    return pow(predicted - expected, 2);
}

float NeuralNetwork::cumulativeCost(std::vector<float> prediction, std::vector<float> actual)
{
    float cumul_cost = 0;
    for (int i = 0; i < prediction.size(); i++) {
        float cost = calculateCost(prediction[i], actual[i]);
        cumul_cost += cost;
    }
    float average_cost = cumul_cost / prediction.size();
    return average_cost;
}

float NeuralNetwork::costDerivative(float predicted, float expected)
{
    return 2 * (predicted - expected);
}

std::vector<float> NeuralNetwork::costGradient(std::vector<float> predicted, std::vector<float> expectation)
{
    std::vector<float> gradients;

    for (int i = 0; i < predicted.size(); i++) {
        float v = costDerivative(predicted[i], expectation[i]);
        gradients.push_back(v);
    }
    return gradients;
}

std::vector<float>NeuralNetwork::chainRule(std::vector<float> grad1, std::vector<float> grad2)
{
    std::vector<float> resultant;

    for (int i = 0; i < grad1.size(); i++) {
        resultant.push_back(grad1[i] * grad2[i]);
    }
    return resultant;
}

void NeuralNetwork::backPropagation(std::vector<float> predicted, std::vector<float> expectation)
{
    std::vector<Layer> backward_layers = layers;
    std::reverse(backward_layers.begin(), backward_layers.end());
    std::vector<float> gradients;
    float learning_rate = 0.005f;

    // compute gradient of costs (error)
    std::vector<float> prev_gradients = costGradient(predicted, expectation);
    for (Layer layer : backward_layers) {
        // get local gradients for layer
        gradients = layer.computeGradients();
        // chain rule 
        prev_gradients = chainRule(prev_gradients, gradients);
        // backpropgate + gradient descent
        layer.localGradientDescent(gradients, learning_rate);
    }
}

void NeuralNetwork::learnSample(std::vector<float> data, std::vector<float> expectation)
{
    auto predicted = predict(data);
    float loss = cumulativeCost(predicted, expectation);
    backPropagation(predicted, expectation);
    std::cout << "Loss : " << loss << std::endl;
}
