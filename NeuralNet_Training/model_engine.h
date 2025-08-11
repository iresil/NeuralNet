#pragma once
class Module;
class DataLoader;
class CrossEntropy;
class SGD;
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <any>
#include "../NeuralNet_Core/neural_network.h"

class ModelEngine
{
    private:
        static void train(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn, SGD &optimizer);
        static void test(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn);

    public:
        static void train_new_mnist_model(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                          const std::vector<NeuralNetwork::LayerSpec> &layer_specs);
        static void inference_on_saved_model(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                             const std::vector<NeuralNetwork::LayerSpec> &layer_specs);
};
