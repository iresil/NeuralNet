#include "pch.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Layers/flatten.h"
#include "../NeuralNet_Layers/linear.h"
#include "../NeuralNet_Layers/relu.h"
#include "../NeuralNet_Layers/softmax.h"

namespace LayerTests
{
    TEST(Flatten, Scalar)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(2.0f);
        std::shared_ptr<Flatten> flatten = std::make_shared<Flatten>();
        std::shared_ptr<Tensor> b = (*flatten)(a);
        EXPECT_EQ(b->shape(), std::vector<std::size_t>({ 1 }));
        EXPECT_EQ(b->item(), 2.0f);
    }

    TEST(Flatten, 1D)
    {
        std::shared_ptr<Tensor> c = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
        std::shared_ptr<Flatten> flatten = std::make_shared<Flatten>();
        std::shared_ptr<Tensor> d = (*flatten)(c);
        EXPECT_EQ(d->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*d)(0), 1.0f);
        EXPECT_EQ((*d)(1), 2.0f);
        EXPECT_EQ((*d)(2), 3.0f);
    }

    TEST(Flatten, 2D)
    {
        std::shared_ptr<Tensor> e = std::make_shared<Tensor>(std::vector<std::vector<float>>{{1.0, 2.0, 3.0}, { 4.0, 5.0, 6.0 }});
        std::shared_ptr<Flatten> flatten = std::make_shared<Flatten>();
        std::shared_ptr<Tensor> f = (*flatten)(e);
        EXPECT_EQ(f->shape(), std::vector<std::size_t>({ 6 }));
        EXPECT_EQ((*f)(0), 1.0f);
        EXPECT_EQ((*f)(1), 2.0f);
        EXPECT_EQ((*f)(2), 3.0f);
        EXPECT_EQ((*f)(3), 4.0f);
        EXPECT_EQ((*f)(4), 5.0f);
        EXPECT_EQ((*f)(5), 6.0f);
    }

    TEST(Linear, Transform)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
        Linear linear(3, 2, 7);
        std::shared_ptr<Tensor> b = linear(a);
        EXPECT_EQ(b->shape(), std::vector<std::size_t>({ 2 }));
        EXPECT_NEAR((*b)(0), -0.13753, 1e-5);
        EXPECT_NEAR((*b)(1), 2.26260, 1e-5);
    }

    TEST(ReLU, Positive_Input)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
        Relu relu;
        std::shared_ptr<Tensor> b = relu(a);
        EXPECT_EQ(b->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*b)(0), 1.0f);
        EXPECT_EQ((*b)(1), 2.0f);
        EXPECT_EQ((*b)(2), 3.0f);
    }

    TEST(ReLU, Negative_Input)
    {
        std::shared_ptr<Tensor> c = std::make_shared<Tensor>(std::vector<float>{-1.0, -2.0, -3.0});
        Relu relu;
        std::shared_ptr<Tensor> d = relu(c);
        EXPECT_EQ(d->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_EQ((*d)(0), 0.0f);
        EXPECT_EQ((*d)(1), 0.0f);
        EXPECT_EQ((*d)(2), 0.0f);
    }

    TEST(Softmax, Transform)
    {
        std::shared_ptr<Tensor> a = std::make_shared<Tensor>(std::vector<float>{1.0, 2.0, 3.0});
        SoftMax softmax;
        std::shared_ptr<Tensor> b = softmax(a);
        EXPECT_EQ(b->shape(), std::vector<std::size_t>({ 3 }));
        EXPECT_NEAR((*b)(0), 0.09003, 1e-5);
        EXPECT_NEAR((*b)(1), 0.24473, 1e-5);
        EXPECT_NEAR((*b)(2), 0.66524, 1e-5);
    }
}
