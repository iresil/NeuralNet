#pragma once
class Tensor;
#include <memory>
#include <cstddef>
#include <vector>
#include "loss.h"

class NLL : public Loss
{
    private:
        static float _forward(std::shared_ptr<Tensor> input, std::size_t target);
        static void _backward(std::shared_ptr<Tensor> input, std::size_t target, const std::vector<float> &grad_output);

    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;
};
