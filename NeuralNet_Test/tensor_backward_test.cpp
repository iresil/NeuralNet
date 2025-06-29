#include "pch.h"
#include "../NeuralNet_Core/tensor.h"

namespace TensorBackwardTests
{
    TEST(BackwardAddition, Scalar_Scalar)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2.0, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(3.0, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        c->backward();
        EXPECT_EQ(a->grad()[0], 1.0);
        EXPECT_EQ(b->grad()[0], 1.0);
    }

    TEST(BackwardAddition, Scalar_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(1.0, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 2.0, 3.0, 4.0 }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 1D_Scalar_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(4.0, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 1D_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 4.0, 5.0, 6.0 }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, Scalar_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(1.0, true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 2D_Scalar_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(1.0, true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardAddition, 2D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0} }), true);
        std::shared_ptr<Tensor> c = (*a) + b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 1D_1D)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 4.0, 5.0, 6.0 }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        c->backward();
        EXPECT_FLOAT_EQ(a->grad()[0], 4.0);
        EXPECT_FLOAT_EQ(a->grad()[1], 5.0);
        EXPECT_FLOAT_EQ(a->grad()[2], 6.0);
        EXPECT_FLOAT_EQ(b->grad()[0], 1.0);
        EXPECT_FLOAT_EQ(b->grad()[1], 2.0);
        EXPECT_FLOAT_EQ(b->grad()[2], 3.0);
    }

    TEST(BackwardMultiplication, 1D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {4.0, 5.0}, {6.0, 7.0}, {8.0, 9.0} }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 2D_1D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(std::vector<float>({ 7.0, 8.0, 9.0 }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }

    TEST(BackwardMultiplication, 2D_2D_Invalid)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }), true);
        std::shared_ptr<Tensor> b = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0} }), true);
        std::shared_ptr<Tensor> c = (*a) * b;
        EXPECT_THROW(c->backward(), std::runtime_error);
    }
}