// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <numeric>
#include <random>
#include "NeuralNetwork.cpp"
#include "../NeuralNet_Data/dataset_mnist.h"
#include "../NeuralNet_Data/dataloader.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Training/loss_crossentropy.h"
#include "../NeuralNet_Training/optimizer_sgd.h"
#include "../NeuralNet_Core/serializer.h"

void train(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn, SGD& optimizer)
{
    std::size_t log_interval = 100;
    std::size_t batch_n = 0;
    std::size_t seen_samples = 0;

    for (const auto &batch : dataloader)
    {
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
        batch_n++;
    }
}

void test(DataLoader &dataloader, NeuralNetwork &model, CrossEntropy &loss_fn)
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
              << "%\n  avg loss" << std::setprecision(6) << avg_loss << std::endl;
}

std::string safe_getenv(const char *name)
{
    char *buffer = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buffer, &len, name) == 0 && buffer != nullptr)
    {
        std::string value(buffer);
        free(buffer); // Free memory allocated by _dupenv_s
        return value;
    }
    return std::string();
}

bool is_visual_studio()
{
    // Environment Variables set at runtime
    std::string vs_env = safe_getenv("VSINSTALLDIR");
    bool is_env_var_present = vs_env.c_str() != nullptr;

    // Might be true for non-VS debuggers, like WinDbg
    bool is_debugger_present = IsDebuggerPresent();

    return is_env_var_present && is_debugger_present;
}

std::string get_model_path(std::string filename = "mnist.nn")
{
    std::filesystem::path folderPath;

    if (is_visual_studio())
    {
        folderPath = std::filesystem::path(__FILE__).parent_path().parent_path().append("models");
    }
    else
    {
        char path[MAX_PATH];
        DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
        if (length == 0)
        {
            return std::string();
        }

        std::filesystem::path exePath(path);
        folderPath = exePath.parent_path();
    }
    folderPath.append(filename);
    return folderPath.string();
}

void train_new_mnist_model()
{
    std::cout << "Loading dataset ..." << std::endl;
    std::string train_data_path = get_model_path("train-images-idx3-ubyte");
    std::string train_labels_path = get_model_path("train-labels-idx1-ubyte");
    MNIST mnist_train = MNIST(train_data_path, train_labels_path);
    std::string test_data_path = get_model_path("t10k-images-idx3-ubyte");
    std::string test_labels_path = get_model_path("t10k-labels-idx1-ubyte");
    MNIST mnist_test = MNIST(test_data_path, test_labels_path);
    std::cout << "Dataset loaded." << std::endl;

    int batch_size = 10;
    DataLoader train_dataloader(&mnist_train, batch_size);
    DataLoader test_dataloader(&mnist_test, batch_size);

    NeuralNetwork model;
    CrossEntropy loss_fn;
    float learning_rate = 0.001f;
    SGD optimizer(model.parameters(), learning_rate);

    int n_epochs = 3;
    for (int epoch = 0; epoch < n_epochs; epoch++)
    {
        std::cout << "[Epoch " << (epoch + 1) << "/" << n_epochs << "] Training ..." << std::endl;
        train(train_dataloader, model, loss_fn, optimizer);
        std::cout << "[Epoch " << (epoch + 1) << "/" << n_epochs << "] Testing ..." << std::endl;
        test(train_dataloader, model, loss_fn);
    }

    auto state_dict = model.state_dict();
    std::string path = get_model_path();
    Serializer::save(state_dict, path);
}

void inference_on_saved_model()
{
    NeuralNetwork model;
    std::cout << "Loading model ..." << std::endl;
    std::string path = get_model_path();
    auto loaded_state_dict = Serializer::load(path);
    model.load_state_dict(loaded_state_dict);

    std::cout << "Loading test set ..." << std::endl;
    std::string test_data_path = get_model_path("t10k-images-idx3-ubyte");
    std::string test_labels_path = get_model_path("t10k-labels-idx1-ubyte");
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

int main()
{
    train_new_mnist_model();
    //inference_on_saved_model();

    return 0;
}
