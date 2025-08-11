#include "pch.h"
#include "model_engine.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <chrono>
#include "../NeuralNet_Data/dataset_mnist.h"
#include "../NeuralNet_Data/dataloader.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Training/loss_crossentropy.h"
#include "../NeuralNet_Training/optimizer_sgd.h"
#include "../NeuralNet_Data/serializer.h"
#include "../NeuralNet_Data/path_provider.h"

void ModelEngine::train(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn, SGD &optimizer)
{
    std::size_t log_interval = 10;
    std::size_t batch_n = 0;
    std::size_t seen_samples = 0;

    for (const auto &batch : dataloader)
    {
        batch_n++;
        std::shared_ptr<Tensor> total_loss = nullptr;
        std::size_t batch_size = batch.size();

        for (const auto &[label, tensor] : batch)
        {
            auto output = model(tensor);
            auto loss = loss_fn(output, label);
            if (total_loss == nullptr)
            {
                total_loss = loss;
            }
            else
            {
                total_loss = (*total_loss) + loss;
            }
            seen_samples++;
        }
        total_loss->item() /= batch_size;

        if (batch_n % log_interval == 0)
        {
            std::cout << "Loss: " << std::fixed << std::setprecision(6) << total_loss->item() << "  [" << seen_samples
                << "/" << dataloader.n_samples() << "]" << std::endl;
        }

        total_loss->backward();
        optimizer.step();
        optimizer.zero_grad();
    }
}

void ModelEngine::test(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn)
{
    float running_loss = 0.0f;
    std::size_t correct = 0;
    std::size_t n_samples = 0;

    for (const auto &batch : dataloader)
    {
        for (const auto &[label, tensor] : batch)
        {
            auto output = model(tensor);
            if (output->argmax() == label)
            {
                correct++;
            }
            running_loss += loss_fn(output, label)->item();
            n_samples++;
        }
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(n_samples);
    float avg_loss = running_loss / n_samples;

    std::cout << std::fixed << "Test error:\n  accuracy: " << std::setprecision(1) << accuracy * 100.0f
        << "%\n  avg loss: " << std::setprecision(6) << avg_loss << std::endl;
}

void ModelEngine::train_new_mnist_model(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                        const std::vector<NeuralNetwork::LayerSpec> &layer_specs)
{
    using namespace std::chrono;
    std::chrono::zoned_time time_now = zoned_time{ current_zone(), system_clock::now() };
    std::cout << "[" << time_now << "]" << std::endl;

    std::cout << "Loading dataset ..." << std::endl;
    std::string train_data_path = PathProvider::get_full_path("data/train-images-idx3-ubyte");
    std::string train_labels_path = PathProvider::get_full_path("data/train-labels-idx1-ubyte");
    MNIST mnist_train = MNIST(train_data_path, train_labels_path);
    std::string test_data_path = PathProvider::get_full_path("data/t10k-images-idx3-ubyte");
    std::string test_labels_path = PathProvider::get_full_path("data/t10k-labels-idx1-ubyte");
    MNIST mnist_test = MNIST(test_data_path, test_labels_path);
    std::cout << "Dataset loaded." << std::endl;

    int batch_size = 10;
    DataLoader train_dataloader(&mnist_train, batch_size);
    DataLoader test_dataloader(&mnist_test, batch_size);

    NeuralNetwork model(layer_registry, layer_specs);
    CrossEntropy loss_fn;
    float learning_rate = 0.001f;
    SGD optimizer(model.parameters(), learning_rate);

    int n_epochs = 3;
    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        time_now = zoned_time{ current_zone(), system_clock::now() };
        std::cout << std::endl << "[" << time_now << "]" << std::endl;
        std::cout << "[Epoch " << (epoch + 1) << "/" << n_epochs << "] Training ..." << std::endl;
        train(train_dataloader, model, loss_fn, optimizer);

        time_now = zoned_time{ current_zone(), system_clock::now() };
        std::cout << std::endl << "[" << time_now << "]" << std::endl;
        std::cout << "[Epoch " << (epoch + 1) << "/" << n_epochs << "] Testing ..." << std::endl;
        test(train_dataloader, model, loss_fn);
    }

    auto state_dict = model.state_dict();
    std::string path = PathProvider::get_full_path();
    Serializer::save(state_dict, path);
}

void ModelEngine::inference_on_saved_model(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &layer_registry,
                                           const std::vector<NeuralNetwork::LayerSpec> &layer_specs)
{
    NeuralNetwork model(layer_registry, layer_specs);
    std::cout << "Loading model ..." << std::endl;
    std::string path = PathProvider::get_full_path();
    auto loaded_state_dict = Serializer::load(path);
    model.load_state_dict(loaded_state_dict);

    std::cout << "Loading test set ..." << std::endl;
    std::string test_data_path = PathProvider::get_full_path("data/t10k-images-idx3-ubyte");
    std::string test_labels_path = PathProvider::get_full_path("data/t10k-labels-idx1-ubyte");
    MNIST mnist_test = MNIST(test_data_path, test_labels_path);

    int n_samples = 10;
    std::vector<int> all_indices(mnist_test.get_length());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_indices.begin(), all_indices.end(), g);
    std::vector<int> indices(all_indices.begin(), all_indices.begin() + n_samples);

    for (int i = 0; i < n_samples; i++)
    {
        std::cout << "Sample " << i << " of " << n_samples << std::endl;
        std::pair<int, std::shared_ptr<Tensor>> sample_image = mnist_test.get_item(indices[i]);
        Dataset::visualize_image(sample_image.second);
        auto output = model(sample_image.second);
        int predicted_class = output->argmax();
        std::cout << "Predicted Class: " << mnist_test.label_to_class(predicted_class) << std::endl;
        std::cout << "Actual Class: " << mnist_test.label_to_class(sample_image.first) << std::endl;
        std::cout << "=======================================" << std::endl;
    }
}