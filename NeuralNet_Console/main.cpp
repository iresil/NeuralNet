// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Training/model_engine.h"

const std::string train_data_path = "data/train-images-idx3-ubyte";
const std::string train_labels_path = "data/train-labels-idx1-ubyte";
const std::string test_data_path = "data/t10k-images-idx3-ubyte";
const std::string test_labels_path = "data/t10k-labels-idx1-ubyte";

const auto layer_registry = LayerFactory::make_registry();
const std::vector<NeuralNetwork::LayerSpec> net_config =
{
    { "Flatten", {} },
    { "Linear", { 28 * 28, 512 } },
    { "Relu", {} },
    { "Linear", { 512, 512 } },
    { "Relu", {} },
    { "Linear", { 512, 10} }
};

int main()
{
    ModelEngine::train_new_model(train_data_path, train_labels_path, test_data_path, test_labels_path, layer_registry, net_config, 10, 3, 0.001f);
    //ModelEngine::inference_on_saved_model(test_data_path, test_labels_path, layer_registry, net_config, 10);

    return 0;
}
