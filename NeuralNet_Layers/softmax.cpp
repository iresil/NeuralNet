#include "pch.h"
#include "softmax.h"
#include <cmath>
#include <stdexcept>
#include "../NeuralNet_Core/tensor.h"

float SoftMax::forward_scalar(std::shared_ptr<Tensor> input)
{
    return 1.0f;
}

void SoftMax::backward_scalar(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output)
{
    // Softmax is designed for multi-class classification and operates on vectors (e.g. 3 or more logits).
    // If you apply softmax to a single scalar value, the result is always 1.
    // Its derivative (gradient) then becomes zero, which means no meaningful gradient flows backward.
    // This is why Sigmoid is used for binary classificatiom, since it is designed for scalar inputs,
    // has a non-zero gradient and is thus useful for optimization.
    std::vector<float> grad_input = { 0.0f };
    input->add_to_grad(grad_input);
}

std::vector<float> SoftMax::forward_1D(std::shared_ptr<Tensor> input)
{
    // Get max value to avoid overflow
    float max_val = (*input)(0);
    for (std::size_t i = 0; i < input->count(); i++)
    {
        if ((*input)(i) > max_val)
        {
            max_val = (*input)(i);
        }
    }
    std::vector<float> result;
    float sum_exp = 0.0f;
    for (std::size_t i = 0; i < input->count(); i++)
    {
        sum_exp += std::exp((*input)(i) - max_val);
    }
    for (std::size_t i = 0; i < input->count(); i++)
    {
        result.push_back(std::exp((*input)(i) - max_val) / sum_exp);
    }
    return result;
}

void SoftMax::backward_1D(std::shared_ptr<Tensor> input, std::vector<float> result, const std::vector<float> &grad_output)
{
    std::vector<float> grad_input;
    for (std::size_t j = 0; j < input->count(); j++)
    {
        float grad_j = 0.0f;
        for (std::size_t i = 0; i < grad_output.size(); i++)
        {
            if (i == j)
            {
                grad_j += grad_output[i] * (result[i] * (1 - result[i]));
            }
            else
            {
                grad_j += grad_output[i] * (-result[i] * result[j]);
            }
        }
        grad_input.push_back(grad_j);
    }
    input->add_to_grad(grad_input);
}

std::shared_ptr<Tensor> SoftMax::forward(std::shared_ptr<Tensor> input)
{
    if (input->shape().size() == 0)  // Scalar
    {
        float result = forward_scalar(input);
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{ input };
            std::function<void (const std::vector<float>&)> gradfn = [input](const std::vector<float> &grad_output)
            {
                backward_scalar(input, grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (input->shape().size() == 1)  // 1D
    {
        std::vector<float> result = forward_1D(input);
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{ input };
            std::function<void (const std::vector<float>&)> gradfn = [input, result](const std::vector<float> &grad_output)
            {
                backward_1D(input, result, grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else  // 2D
    {
        throw std::runtime_error("Softmax is only allowed for 1D vectors");
    }
}
