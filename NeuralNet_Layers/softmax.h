#pragma once
class Tensor;
#include <memory>
#include "../NeuralNet_Core/module.h"

class SoftMax : public Module
{
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
