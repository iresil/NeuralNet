#include "pch.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Training/loss_nll.h"
#include "../NeuralNet_Training/loss_crossentropy.h"

namespace LossTests
{
    TEST(NLL, Forward_No_Grad)
    {
        NLL nll_loss;

        // Basic forward pass without grad
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 0.1f, 0.2f, 0.7f });
        std::size_t target = 2;
        std::shared_ptr<Tensor> loss = nll_loss(input, target);

        EXPECT_EQ(loss->shape().size(), 0);
        EXPECT_FALSE(loss->requires_grad());
        EXPECT_NEAR(loss->item(), -std::log(0.7f), 1e-6f);
    }

    TEST(NLL, Near_Zero_Probability_Clamped)
    {
        NLL nll_loss;

        // Near-zero probability clamped
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 1e-15f, 0.5f, 0.5f });
        std::size_t target = 0;
        std::shared_ptr<Tensor> loss = nll_loss(input, target);

        EXPECT_NEAR(loss->item(), -std::log(1e-12f), 1e-6f);
    }

    TEST(NLL, Backward)
    {
        NLL nll_loss;

        // Basic backward pass
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 0.1f, 0.2f, 0.7f }, true);
        std::size_t target = 2;
        std::shared_ptr<Tensor> loss = nll_loss(input, target);

        EXPECT_TRUE(loss->requires_grad());

        loss->backward();

        ASSERT_EQ(input->grad().size(), 3);
        EXPECT_FLOAT_EQ(input->grad()[0], 0.0f);
        EXPECT_FLOAT_EQ(input->grad()[1], 0.0f);
        EXPECT_NEAR(input->grad()[2], -1.0f / 0.7f, 1e-6f);
    }

    TEST(NLL, Error)
    {
        NLL nll_loss;

        // Error handling
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<std::vector<float>>{ { 0.1f, 0.9f } });
        std::size_t target = 0;

        EXPECT_THROW(nll_loss(input, target), std::runtime_error);
    }

    TEST(NLL, Target_Out_Of_Bounds)
    {
        NLL nll_loss;

        // Test target out of bounds
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 0.1f, 0.9f });
        std::size_t target = 2;

        EXPECT_THROW(nll_loss(input, target), std::runtime_error);
    }

    TEST(NLL, Target_Out_Of_Bounds_Negative)
    {
        NLL nll_loss;

        // Test target out of bounds (negative equivalent, size_t wrap around)
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 0.1f, 0.9f });
        std::size_t target = -1;

        EXPECT_THROW(nll_loss(input, target), std::runtime_error);
    }

    TEST(CrossEntropy, Forward_No_Grad)
    {
        CrossEntropy ce_loss;

        // Basic forward pass without grad
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 1.0f, 2.0f, 3.0f });
        std::size_t target = 2;
        std::shared_ptr<Tensor> loss = ce_loss(input, target);

        EXPECT_EQ(loss->shape().size(), 0);
        EXPECT_FALSE(loss->requires_grad());
        EXPECT_NEAR(loss->item(), 0.40761f, 1e-5f);
    }

    TEST(CrossEntropy, Backward)
    {
        CrossEntropy ce_loss;

        // Basic backward pass
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 1.0f, 2.0f, 3.0f }, true);
        std::size_t target = 2;
        std::shared_ptr<Tensor> loss = ce_loss(input, target);

        EXPECT_TRUE(loss->requires_grad());

        loss->backward();

        ASSERT_EQ(input->grad().size(), 3);
        EXPECT_NEAR(input->grad()[0], 0.09003f, 1e-5f);
        EXPECT_NEAR(input->grad()[1], 0.24473f, 1e-5f);
        EXPECT_NEAR(input->grad()[2], -0.33476f, 1e-5f);
    }

    TEST(CrossEntropy, Error)
    {
        CrossEntropy ce_loss;

        // Error handling: incorrect input shape
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<std::vector<float>>{ { 1.0f, 2.0f } });
        std::size_t target = 0;

        EXPECT_THROW(ce_loss(input, target), std::runtime_error);
    }

    TEST(CrossEntropy, Target_Out_Of_Bounds)
    {
        CrossEntropy ce_loss;

        // Test target out of bounds
        std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<float>{ 1.0f, 2.0f });
        std::size_t target = 2;

        EXPECT_THROW(ce_loss(input, target), std::runtime_error);
    }
}
