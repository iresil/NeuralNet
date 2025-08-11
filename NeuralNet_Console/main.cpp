// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Training/model_engine.h"

const std::vector<NeuralNetwork::LayerSpec> net_config =
{
    { "Flatten", {} },
    { "Linear", { 28 * 28, 512 } },
    { "Relu", {} },
    { "Linear", { 512, 512 } },
    { "Relu", {} },
    { "Linear", { 512, 10} }
};
const auto layer_registry = LayerFactory::make_registry();

int main()
{
    ModelEngine::train_new_mnist_model(layer_registry, net_config);
    //ModelEngine::inference_on_saved_model(layer_registry, net_config);

    return 0;
}
