#include "pch.h"
#include "optimizer_sgd.h"
#include <cstddef>
#include <execution>
#include "../NeuralNet_Core/tensor.h"

SGD::SGD(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params, float lr) :
    _params(params), _learning_rate(lr) { }

void SGD::step()
{
    for (auto &param : _params)
    {
        std::transform(std::execution::par, param.second->data().begin(), param.second->data().end(),
            param.second->grad().begin(), param.second->data().begin(),
            [lr = _learning_rate](auto data_val, auto grad_val)
            {
                return data_val - lr * grad_val;
            }
        );
    }
}

void SGD::zero_grad()
{
    for (auto &param : _params)
    {
        param.second->zero_grad();
    }
}
