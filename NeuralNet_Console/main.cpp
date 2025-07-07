// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <random>
#include "../NeuralNet_Core/tensor.h"
#include "NeuralNetwork.cpp"

int main()
{
    NeuralNetwork model;

    // Randomized input
    std::vector<std::vector<float>> input_data(28, std::vector<float>(28));
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &row : input_data)
    {
        for (auto &val : row)
        {
            val = dist(rng);
        }
    }

    // Create input tensor
    std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(input_data);

    // Forward pass
    std::shared_ptr<Tensor> output_tensor = model(input_tensor);

    std::cout << (*output_tensor) << std::endl;

    return 0;
}
