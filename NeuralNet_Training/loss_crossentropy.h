#pragma once
class Tensor;
#include <memory>
#include <cstddef>
#include "loss.h"

class CrossEntropy : public Loss
{
    public:
        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input, std::size_t target) override;
};
