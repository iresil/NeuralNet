#include "pch.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Core/module.h"
#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Layers/linear.h"
#include "../NeuralNet_Layers/relu.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Training/loss_crossentropy.h"
#include "../NeuralNet_Training/optimizer_sgd.h"

namespace OptimizerTests
{
    TEST(SGD, Step)
    {
        const std::vector<NeuralNetwork::LayerSpec> config =
        {
            { "Linear", { 5, 5 } },
            { "Relu", {} }
        };
        const auto reg = LayerFactory::make_registry(7);
        NeuralNetwork network(reg, config);
        auto params = network.parameters();

        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>(5.0f, 1.0f));

        // Copy linear weight
        std::vector<std::vector<float>> copy_linear_weight(params[0].second->shape()[0], std::vector<float>(params[0].second->shape()[1]));
        for (size_t i = 0; i < params[0].second->shape()[0]; i++)
        {
            for (size_t j = 0; j < params[0].second->shape()[1]; j++)
            {
                copy_linear_weight[i][j] = (*params[0].second)(i, j);
            }
        }

        // Copy linear bias
        std::vector<float> copy_linear_bias(params[1].second->shape()[0]);
        for (size_t i = 0; i < params[1].second->shape()[0]; i++)
        {
            copy_linear_bias[i] = (*params[1].second)(i);
        }

        auto copy_linear_weight_tensor = std::make_shared<Tensor>(copy_linear_weight);
        auto copy_linear_bias_tensor = std::make_shared<Tensor>(copy_linear_bias);

        auto xW = (*input) * copy_linear_weight_tensor;
        auto expected_linear_output = (*xW) + copy_linear_bias_tensor;

        SGD sgd(params);
        CrossEntropy ce_loss;
        std::size_t target = 1;

        auto output = network(input);
        auto loss = ce_loss(output, target);
        loss->backward();

        if ((*expected_linear_output)(1) <= 0)
        {
            for (int i = 0; i < params[0].second->shape()[0]; i++)
            {
                float grad_val = (*params[0].second).grad()[i * params[0].second->stride()[0] + 1];
                EXPECT_EQ(grad_val, 0.0f) << "Expected gradient to be 0 at col 1, row " << i;
            }
        }

        sgd.step();

        // If gradient is 0, weight must be unchanged
        if ((*params[0].second).grad()[1] == 0)
        {
            float new_weight = (*params[0].second)(0, 1);
            EXPECT_FLOAT_EQ(new_weight, copy_linear_weight[0][1]) << "Weight changed despite zero gradient";
        }

        // If gradient is > 0, weight should have decreased
        if ((*params[0].second).grad()[0] > 0)
        {
            float updated_weight = (*params[0].second)(0, 0);
            EXPECT_LT(updated_weight, copy_linear_weight[0][0]) << "Weight not updated correctly for non-zero grad";
        }

        sgd.zero_grad();

        for (size_t i = 0; i < params[0].second->count(); i++)
        {
            EXPECT_FLOAT_EQ((*params[0].second).grad()[i], 0.0f) << "Gradient not reset to 0 at index " << i;
        }
    }
}
