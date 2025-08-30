#include "pch.h"
#include "relu.h"
#include <map>
#include "../NeuralNet_Core/tensor.h"

std::map<std::size_t, Relu::Operation> relu_operations =
{
    { 0, { Relu::forward_scalar, Relu::backward_scalar } },
    { 1, { Relu::forward_1D, Relu::backward_1D } },
    { 2, { Relu::forward_2D, Relu::backward_2D } }
};

std::shared_ptr<Tensor> Relu::forward_scalar(std::shared_ptr<Tensor> input)
{
    float result = 0.0f;
    if (input->item() > 0)
    {
        result = input->item();
    }
    return std::make_shared<Tensor>(result);
}

void Relu::backward_scalar(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output)
{
    std::vector<float> grad_input;
    if (input->item() > 0)
    {
        grad_input.push_back(grad_output[0]);
    }
    else
    {
        grad_input.push_back(0.0f);
    }
    input->add_to_grad(grad_input);
}

std::shared_ptr<Tensor> Relu::forward_1D(std::shared_ptr<Tensor> input)
{
    std::vector<float> result;
    for (std::size_t i = 0; i < input->shape()[0]; i++)
    {
        if ((*input)(i) > 0)
        {
            result.push_back((*input)(i));
        }
        else
        {
            result.push_back(0.0f);
        }
    }
    return std::make_shared<Tensor>(result);
}

void Relu::backward_1D(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output)
{
    std::vector<float> grad_input;
    for (std::size_t i = 0; i < input->count(); i++)
    {
        if ((*input)(i) > 0)
        {
            grad_input.push_back(grad_output[i]);
        }
        else
        {
            grad_input.push_back(0.0f);
        }
    }
    input->add_to_grad(grad_input);
}

std::shared_ptr<Tensor> Relu::forward_2D(std::shared_ptr<Tensor> input)
{
    std::vector<std::vector<float>> result;
    for (std::size_t i = 0; i < input->shape()[0]; i++)
    {
        std::vector<float> result_i;
        for (std::size_t j = 0; j < input->shape()[1]; j++)
        {
            if ((*input)(i, j) > 0)
            {
                result_i.push_back((*input)(i, j));
            }
            else
            {
                result_i.push_back(0.0f);
            }
        }
        result.push_back(result_i);
    }
    return std::make_shared<Tensor>(result);
}

void Relu::backward_2D(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output)
{
    // All gradients are stored in row-major order
    std::vector<float> grad_input;
    for (std::size_t i = 0; i < input->count(); i++)
    {
        if ((*input)(i) > 0)
        {
            grad_input.push_back(grad_output[i]);
        }
        else
        {
            grad_input.push_back(0.0f);
        }
    }
    input->add_to_grad(grad_input);
}

std::shared_ptr<Tensor> Relu::create_tensor_with_grad(std::shared_ptr<Tensor> result,
                                                      std::shared_ptr<Tensor> input, Relu::GradFunc backward)
{
    std::vector<std::shared_ptr<Tensor>> parents{ input };
    std::function<void(const std::vector<float>&)> gradfn = [input, backward](const std::vector<float> &grad_output)
    {
        backward(input, grad_output);
    };

    result->make_with_grad(true, parents, gradfn);
    return result;
}

std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input)
{
    std::size_t rank = input->shape().size();
    auto it = relu_operations.find(rank);
    if (it == relu_operations.end())
        throw std::invalid_argument("Unsupported tensor rank for ReLU");

    std::shared_ptr<Tensor> result = it->second.forward(input);

    if (input->requires_grad())
        return create_tensor_with_grad(result, input, it->second.backward);

    return result;
}
