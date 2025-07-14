// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <filesystem>
#include <iostream>
#include <string>
#include "../NeuralNet_Data/dataset_mnist.h"
#include "../NeuralNet_Data/dataloader.h"

std::string get_model_path(std::string filename = "mnist.nn") {
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (length == 0) {
        return std::string();
    }

    std::filesystem::path exePath(path);
    std::filesystem::path folderPath = exePath.parent_path();
    folderPath.append(filename);
    return folderPath.string();
}

int main()
{
    std::string data_path = get_model_path("train-images-idx3-ubyte");
    std::string labels_path = get_model_path("train-labels-idx1-ubyte");
    MNIST mnist_train = MNIST(data_path, labels_path);
    std::cout << "Datasets successfully loaded" << std::endl;

    int batch_size = 10;
    DataLoader mnist_train_loader(&mnist_train, batch_size);

    std::cout << "Visualizing first batch of training data" << std::endl;
    for (auto batch : mnist_train_loader)
    {
        for (auto item : batch)
        {
            Dataset::visualize_image(item.second);
            std::cout << mnist_train.label_to_class(item.first) << std::endl;
        }
        break;
    }

    return 0;
}
