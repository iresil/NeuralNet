#include "pch.h"
#include "neural_network.h"
#include <algorithm>
#include <cctype>

NeuralNetwork::NeuralNetwork(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &registry,
                             const std::vector<LayerSpec> &specs)
{
    // Step 1: Count occurrences of each type
    std::unordered_map<std::string, int> type_counts;
    for (const auto &spec : specs)
    {
        type_counts[spec.type]++;
    }

    // Step 2: Track index per type
    std::unordered_map<std::string, int> type_indices;

    // Step 3: Build layers and register with or without index
    for (const auto &spec : specs)
    {
        auto it = registry.find(spec.type);
        if (it != registry.end())
        {
            auto layer = it->second(spec.params);
            _layers.push_back(layer);

            std::string name = spec.type;
            std::transform(name.begin(), name.end(), name.begin(),
                [](unsigned char c) { return std::tolower(c); }
            );
            if (type_counts[spec.type] > 1)
            {
                int idx = ++type_indices[spec.type];
                name += "_" + std::to_string(idx);
            }

            register_module(name, layer);
        }
        else
        {
            throw std::runtime_error("Unknown layer type: " + spec.type);
        }
    }
}

std::shared_ptr<Tensor> NeuralNetwork::forward(std::shared_ptr<Tensor> input)
{
    std::shared_ptr<Tensor> output = input;
    for (const auto &layer : _layers)
    {
        output = (*layer)(output);
    }
    return output;
}
