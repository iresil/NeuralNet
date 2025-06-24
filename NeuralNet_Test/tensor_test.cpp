#include "pch.h"
#include "../NeuralNet_Core/tensor.h"

namespace TensorTests {
    TEST(Creation, Scalar)
    {
        Tensor tensor = Tensor(5.0);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({}));
        EXPECT_THROW(tensor(0), std::invalid_argument);
        EXPECT_EQ(tensor.item(), 5.0);
    }

    TEST(Creation, 1D)
    {
        std::vector<float> v = { 1.0, 2.0, 3.0 };
        Tensor tensor = Tensor(v);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ(tensor(0), 1.0);
        EXPECT_EQ(tensor(1), 2.0);
        EXPECT_EQ(tensor(2), 3.0);
        EXPECT_THROW(tensor(3), std::invalid_argument);
        EXPECT_THROW(tensor.item(), std::runtime_error);
    }

    TEST(Creation, 2D)
    {
        std::vector<std::vector<float>> v_2 = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} };
        Tensor tensor = Tensor(v_2);
        EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({ 2, 3 }));
        EXPECT_EQ(tensor.stride(), std::vector<std::size_t>({ 3, 1 }));
        EXPECT_EQ(tensor(0, 0), 1.0);
        EXPECT_EQ(tensor(0, 1), 2.0);
        EXPECT_EQ(tensor(0, 2), 3.0);
        EXPECT_EQ(tensor(1, 0), 4.0);
        EXPECT_EQ(tensor(1, 1), 5.0);
        EXPECT_EQ(tensor(1, 2), 6.0);
        EXPECT_THROW(tensor(2, 0), std::invalid_argument);
        EXPECT_THROW(tensor(0, 3), std::invalid_argument);
        EXPECT_THROW(tensor.item(), std::runtime_error);
    }

    TEST(Addition, Scalar_Scalar)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
        std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
        EXPECT_EQ(tensor3->item(), 3.0);
    }

    TEST(Addition, Scalar_1D)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(std::vector<float>({ 2.0, 3.0, 4.0 }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*tensor3)(0), 3.0);
        EXPECT_EQ((*tensor3)(1), 4.0);
        EXPECT_EQ((*tensor3)(2), 5.0);
    }

    TEST(Addition, 1D_1D)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(std::vector<float>({ 4.0, 5.0, 6.0 }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*tensor3)(0), 5.0);
        EXPECT_EQ((*tensor3)(1), 7.0);
        EXPECT_EQ((*tensor3)(2), 9.0);
    }

    TEST(Addition, 1D_Scalar)
    {
        std::shared_ptr<Tensor> tensor1 =
            std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(4.0);
        std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*tensor3)(0), 5.0);
        EXPECT_EQ((*tensor3)(1), 6.0);
        EXPECT_EQ((*tensor3)(2), 7.0);
    }

    TEST(Addition, 2D_2D)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {7.0, 8.0, 9.0}, {10.0, 11.0, 12.0} }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) + tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 2, 3 }));
        EXPECT_EQ((*tensor3)(0, 0), 8.0);
        EXPECT_EQ((*tensor3)(0, 1), 10.0);
        EXPECT_EQ((*tensor3)(0, 2), 12.0);
        EXPECT_EQ((*tensor3)(1, 0), 14.0);
        EXPECT_EQ((*tensor3)(1, 1), 16.0);
        EXPECT_EQ((*tensor3)(1, 2), 18.0);
    }

    TEST(Multiplication, Scalar_Scalar)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(2.0);
        EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);
    }

    TEST(Multiplication, Scalar_1D)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(1.0);
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(std::vector<float>({ 2.0, 3.0, 4.0 }));
        EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);
    }

    TEST(Multiplication, 1D_1D)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(std::vector<float>({ 4.0, 5.0, 6.0 }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) * tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({}));
        EXPECT_EQ(tensor3->item(), 32.0);
    }

    TEST(Multiplication, 1D_2D_Dimensions_Mismatched)
    {
        std::shared_ptr<Tensor> tensor1 =
            std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 5.0 }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} }));
        EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);
    }

    TEST(Multiplication, 1D_2D_Dimensions_Matching)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0 }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0} }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) * tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*tensor3)(0), 18.0);
        EXPECT_EQ((*tensor3)(1), 21.0);
        EXPECT_EQ((*tensor3)(2), 24.0);
    }

    TEST(Multiplication, 2D_1D_Dimensions_Matching)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }));
        std::shared_ptr<Tensor> tensor2 =
            std::make_shared<Tensor>(std::vector<float>({ 1.0, 2.0, 3.0 }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) * tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 2 }));
        EXPECT_EQ((*tensor3)(0), 14.0);
        EXPECT_EQ((*tensor3)(1), 32.0);
    }

    TEST(Multiplication, 2D_2D_Dimensions_Mismatched)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }));
        EXPECT_THROW((*tensor1) * tensor2, std::invalid_argument);
    }

    TEST(Multiplication, 2D_2D_Dimensions_Matching)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} }));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>({ {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0} }));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) * tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 2, 2 }));
        EXPECT_EQ((*tensor3)(0, 0), 58.0);
        EXPECT_EQ((*tensor3)(0, 1), 64.0);
        EXPECT_EQ((*tensor3)(1, 0), 139.0);
        EXPECT_EQ((*tensor3)(1, 1), 154.0);
    }

    TEST(Multiplication, 2D_2D_Large_Size)
    {
        std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>(200, std::vector<float>(300, 1.0)));
        std::shared_ptr<Tensor> tensor2 = std::make_shared<Tensor>(
            std::vector<std::vector<float>>(300, std::vector<float>(400, 1.0)));
        std::shared_ptr<Tensor> tensor3 = (*tensor1) * tensor2;
        EXPECT_EQ(tensor3->shape(), std::vector<std::size_t>({ 200, 400 }));
        for (std::size_t i = 0; i < 200; i++)
        {
            for (std::size_t j = 0; j < 400; j++)
            {
                EXPECT_EQ((*tensor3)(i, j), 300.0);
            }
        }
    }
}
