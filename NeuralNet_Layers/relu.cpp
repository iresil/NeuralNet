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
        return std::make_shared<Tensor>(result);
    }
}
