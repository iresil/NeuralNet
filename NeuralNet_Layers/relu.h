#pragma once
class Tensor;
#include <functional>
#include <memory>
#include <vector>
#include "../NeuralNet_Core/module.h"

class Relu : public Module
{
    public:
        using ReluFunc = std::function<std::shared_ptr<Tensor>(std::shared_ptr<Tensor>)>;
        using GradFunc = std::function<void(std::shared_ptr<Tensor>, const std::vector<float>&)>;

        struct Operation {
            ReluFunc forward;
            GradFunc backward;
        };

        static std::shared_ptr<Tensor> forward_scalar(std::shared_ptr<Tensor> input);
        static void backward_scalar(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output);
        static std::shared_ptr<Tensor> forward_1D(std::shared_ptr<Tensor> input);
        static void backward_1D(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output);
        static std::shared_ptr<Tensor> forward_2D(std::shared_ptr<Tensor> input);
        static void backward_2D(std::shared_ptr<Tensor> input, const std::vector<float> &grad_output);

        std::shared_ptr<Tensor> create_tensor_with_grad(std::shared_ptr<Tensor> result,
                                                        std::shared_ptr<Tensor> input, Relu::GradFunc backward);

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;
};
