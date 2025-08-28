#include "pch.h"
#include "../NeuralNet_Core/tensor.h"

namespace TensorBackwardTests
{
    TEST(BackwardAddition, Scalar_Scalar)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2.0f, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(3.0f, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        c->backward();
        EXPECT_EQ(a->grad()[0], 1.0f);
        EXPECT_EQ(b->grad()[0], 1.0f);
    }

    TEST(BackwardAddition, Scalar_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(1.0f, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 2.0f, 3.0f, 4.0f }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 1D_Scalar_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0f, 2.0f, 3.0f }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(4.0f, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 1D_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0f, 2.0f, 3.0f }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 4.0f, 5.0f, 6.0f }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, Scalar_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(1.0f, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 2D_Scalar_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(1.0f, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 2D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 7.0f, 8.0f, 9.0f }, { 10.0f, 11.0f, 12.0f } }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 1D_1D)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0f, 2.0f, 3.0f }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 4.0f, 5.0f, 6.0f }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        c->backward();
        EXPECT_FLOAT_EQ(a->grad()[0], 4.0f);
        EXPECT_FLOAT_EQ(a->grad()[1], 5.0f);
        EXPECT_FLOAT_EQ(a->grad()[2], 6.0f);
        EXPECT_FLOAT_EQ(b->grad()[0], 1.0f);
        EXPECT_FLOAT_EQ(b->grad()[1], 2.0f);
        EXPECT_FLOAT_EQ(b->grad()[2], 3.0f);
    }

    TEST(BackwardMultiplication, 1D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0f, 2.0f, 3.0f }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 4.0f, 5.0f }, { 6.0f, 7.0f }, { 8.0f, 9.0f } }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 2D_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 7.0f, 8.0f, 9.0f }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 2D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<std::vector<float>>({ { 7.0f, 8.0f }, { 9.0f, 10.0f }, { 11.0f, 12.0f } }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }
}