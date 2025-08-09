#include "pch.h"
#include "loss_nll.h"
#include <functional>
#include <algorithm>
#include <cmath>
#include "../NeuralNet_Core/tensor.h"

std::shared_ptr<Tensor> NLL::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if (input->shape().size() != 1)
    {
        throw std::runtime_error("NLL Loss expects a 1D input tensor");
    }
    if (target >= input->count())
    {
        throw std::runtime_error("NLL Loss target out of bounds");
    }
    // Clamp probability to prevent log(0)
    float prob = std::max((*input)(target), 1e-12f);
    float loss = -std::log(prob);

    if (input->requires_grad())
    {
        std::vector<std::shared_ptr<Tensor>> parents{ input };
        std::function<void (const std::vector<float> &)> gradfn = [input, target](const std::vector<float> &grad_output)
        {
            std::vector<float> grad_input;
            for (std::size_t i = 0; i < input->count(); i++)
            {
                if (i == target)
                {
                    grad_input.push_back(grad_output[0] * (-1.0f / (*input)(i)));
                }
                else
                {
                    grad_input.push_back(0.0f);
                }
            }
            input->add_to_grad(grad_input);
        };
        return std::make_shared<Tensor>(loss, true, gradfn, parents);
    }
    return std::make_shared<Tensor>(loss);
}
