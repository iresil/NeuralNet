// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <map>
#include <string>
#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Data/input_data.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Training/model_engine.h"


enum Mode
{
    TRAINING,
    INFERENCE
};

const Mode selected_mode = Mode::TRAINING;
const InputData selected_dataset = InputData::MNIST;

const std::map<InputData, std::map<std::string, std::string>> datasets =
{
    {
        InputData::MNIST,
        {
            { "train_data_path", "data/mnist/train-images-idx3-ubyte" },
            { "train_labels_path", "data/mnist/train-labels-idx1-ubyte" },
            { "test_data_path", "data/mnist/t10k-images-idx3-ubyte" },
            { "test_labels_path", "data/mnist/t10k-labels-idx1-ubyte" },
            { "model_path", "models/mnist.nn" }
        }
    },
    {
        InputData::FASHION_MNIST,
        {
            { "train_data_path", "data/fashion-mnist/train-images-idx3-ubyte" },
            { "train_labels_path", "data/fashion-mnist/train-labels-idx1-ubyte" },
            { "test_data_path", "data/fashion-mnist/t10k-images-idx3-ubyte" },
            { "test_labels_path", "data/fashion-mnist/t10k-labels-idx1-ubyte" },
            { "model_path", "models/fashion-mnist.nn" }
        }
    }
};

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
const int train_batch_size = 10;
const int train_epochs = 3;
const float learning_rate = 0.001f;
const int infer_samples = 10;

int main()
{
    std::string train_data_path = datasets.at(selected_dataset).at("train_data_path");
    std::string train_labels_path = datasets.at(selected_dataset).at("train_labels_path");
    std::string test_data_path = datasets.at(selected_dataset).at("test_data_path");
    std::string test_labels_path = datasets.at(selected_dataset).at("test_labels_path");
    std::string model_path = datasets.at(selected_dataset).at("model_path");

    if (selected_mode == Mode::TRAINING)
    {
        ModelEngine::train_new_model(selected_dataset, train_data_path, train_labels_path, test_data_path, test_labels_path, layer_registry, net_config, train_batch_size, train_epochs, learning_rate, model_path);
    }
    else
    {
        ModelEngine::inference_on_saved_model(selected_dataset, test_data_path, test_labels_path, layer_registry, net_config, infer_samples, model_path);
    }

    return 0;
}
