#include "pch.h"
#include "loss_crossentropy.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Layers/softmax.h"
#include "loss_nll.h"

std::shared_ptr<Tensor> CrossEntropy::forward(std::shared_ptr<Tensor> input, std::size_t target)
{
    if (input->shape().size() != 1)
    {
        throw std::runtime_error("Cross-Entropy Loss expects a 1D input tensor");
    }
    if (target >= input->count())
    {
        throw std::runtime_error("Cross-Entropy Loss target out of bounds");
    }
    SoftMax softmax;
    NLL nll_loss;
    std::shared_ptr<Tensor> softmax_output = softmax(input);
    return nll_loss(softmax_output, target);
}
