#pragma once
class Tensor;
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <any>
#include "module.h"

class NeuralNetwork : public Module
{
    private:
        std::vector<std::shared_ptr<Module>> _layers;

    public:
        struct LayerSpec
        {
            std::string type;
            std::vector<std::any> params;
        };

        NeuralNetwork(const std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> &registry,
                      const std::vector<LayerSpec> &specs);
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
