#pragma once
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Core/module.h"

class Relu : public Module
{
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
