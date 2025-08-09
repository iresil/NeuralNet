#include "pch.h"
#include "optimizer_sgd.h"
#include <cstddef>
#include "../NeuralNet_Core/tensor.h"

SGD::SGD(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params, float lr) :
    _params(params), _learning_rate(lr) { }

void SGD::step()
{
    for (auto &param : _params)
    {
        for (std::size_t i = 0; i < param.second->count(); i++)
        {
            param.second->data()[i] -= _learning_rate * param.second->grad()[i];
        }
    }
}

void SGD::zero_grad()
{
    for (auto &param : _params)
    {
        param.second->zero_grad();
    }
}
