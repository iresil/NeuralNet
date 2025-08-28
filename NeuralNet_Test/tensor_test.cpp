#include "pch.h"
#include "../NeuralNet_Core/tensor.h"

namespace TensorTests
{
    TEST(Creation, Scalar)
    {
        Tensor tensor = Tensor(5.0f);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({}));
        EXPECT_THROW(tensor(0), std::invalid_argument);
        EXPECT_EQ(tensor.item(), 5.0f);
    }

    TEST(Creation, 1D)
    {
        std::vector<float> v = { 1.0f, 2.0f, 3.0f };
        Tensor tensor = Tensor(v);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ(tensor(0), 1.0f);
        EXPECT_EQ(tensor(1), 2.0f);
        EXPECT_EQ(tensor(2), 3.0f);
        EXPECT_THROW(tensor(3), std::invalid_argument);
        EXPECT_THROW(tensor.item(), std::runtime_error);
    }

    TEST(Creation, 2D)
    {
        std::vector<std::vector<float>> v_2 = { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };
        Tensor tensor = Tensor(v_2);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({ 2, 3 }));
        EXPECT_EQ(tensor.stride(), std::vector<std::size_t>({ 3, 1 }));
        EXPECT_EQ(tensor(0, 0), 1.0f);
        EXPECT_EQ(tensor(0, 1), 2.0f);
        EXPECT_EQ(tensor(0, 2), 3.0f);
        EXPECT_EQ(tensor(1, 0), 4.0f);
        EXPECT_EQ(tensor(1, 1), 5.0f);
        EXPECT_EQ(tensor(1, 2), 6.0f);
        EXPECT_THROW(tensor(2, 0), std::invalid_argument);
        EXPECT_THROW(tensor(0, 3), std::invalid_argument);
        EXPECT_THROW(tensor.item(), std::runtime_error);
    }
}
