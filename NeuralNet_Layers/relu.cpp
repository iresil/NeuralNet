#include "pch.h"
#include "relu.h"

std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> input)
{
    if (input->shape().size() == 0)
    {
        float result = 0.0f;
        if (input->item() > 0)
        {
            result = input->item();
        }
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{ input };
            
            std::function<void(const std::vector<float>&)> gradfn = [input](const std::vector<float> &grad_output)
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
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (input->shape().size() == 1)
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
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{ input };

            std::function<void(const std::vector<float>&)> gradfn = [input](const std::vector<float> &grad_output)
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
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (input->shape().size() == 2)
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
        if (input->requires_grad())
        {
            std::vector<std::shared_ptr<Tensor>> parents{ input };

            std::function<void(const std::vector<float>&)> gradfn = [input](const std::vector<float> &grad_output)
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
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
}
