#include "pch.h"
#include "../NeuralNet_Core/module.h"
#include "../NeuralNet_Layers/linear.h"

TEST(ModuleTest, Linear_Simple)
{
    Linear linear(3, 2, 42);
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params[0].first, "weight");
    EXPECT_EQ(params[1].first, "bias");
}
