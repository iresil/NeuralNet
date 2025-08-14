#pragma once
class Module;
class DataLoader;
class CrossEntropy;
class SGD;
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <any>
#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Data/input_data.h"

class ModelEngine
{
    private:
        static void train(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn, SGD &optimizer);
        static void test(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn);

    public:
        static void train_new_model(InputData selected_dataset, std::string train_data_path, std::string train_labels_path, std::string test_data_path, std::string test_labels_path,
                                    const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                    const std::vector<NeuralNetwork::LayerSpec> &layer_specs, int batch_size, int n_epochs, float learning_rate, std::string model_path);
        static void inference_on_saved_model(InputData selected_dataset, std::string test_data_path, std::string test_labels_path,
                                             const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                             const std::vector<NeuralNetwork::LayerSpec> &layer_specs, int n_samples, std::string model_path);
};
