#include "pch.h"
#include "loss.h"

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input)
{
    throw std::runtime_error("Loss expects both an input and a target");
}

std::shared_ptr<Tensor> Loss::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    throw std::runtime_error("Forward not implemented");
}

std::shared_ptr<Tensor> Loss::operator()(std::shared_ptr<Tensor> input, std::size_t target)
{
    return forward(input, target);
}
