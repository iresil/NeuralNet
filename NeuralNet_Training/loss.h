#pragma once
class Tensor;
#include <memory>
#include <cstddef>
#include "../NeuralNet_Core/module.h"

class Loss : public Module
{
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
        virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target);
        std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input, std::size_t target);
};
