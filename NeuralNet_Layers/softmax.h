#pragma once
class Tensor;
#include <memory>
#include <vector>
#include "../NeuralNet_Core/module.h"

class SoftMax : public Module
{
    private:
        static float forward_scalar(std::shared_ptr<Tensor> input);
        static void backward_scalar(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output);
        static std::vector<float> forward_1D(std::shared_ptr<Tensor> input);
        static void backward_1D(std::shared_ptr<Tensor> input, std::vector<float> result, const std::vector<float> &grad_output);

    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
