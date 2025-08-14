#include "pch.h"
#include "../NeuralNet_Core/tensor.h"
#include "../NeuralNet_Core/module.h"
#include "../NeuralNet_Core/neural_network.h"
#include "../NeuralNet_Layers/flatten.h"
#include "../NeuralNet_Layers/linear.h"
#include "../NeuralNet_Layers/relu.h"
#include "../NeuralNet_Layers/layer_factory.h"
#include "../NeuralNet_Data/serializer.h"
#include "../NeuralNet_Data/path_provider.h"

namespace ModuleTests
{
    TEST(Creation, Linear_Simple_Parameters)
    {
        Linear linear(3, 2, 42);
        auto params = linear.parameters();
        EXPECT_EQ(params.size(), 2);
        EXPECT_EQ(params[0].first, "weight");
        EXPECT_EQ(params[1].first, "bias");
    }

    TEST(Serialization, NeuralNetwork_Seeded_Parameters)
    {
        const std::vector<NeuralNetwork::LayerSpec> config =
        {
            { "Flatten", {} },
            { "Linear", { 5 * 5, 5 } },
            { "Relu", {} },
            { "Linear", { 5, 10 } },
            { "Relu", {} },
            { "Linear", { 10, 10 } }
        };

        const auto reg1 = LayerFactory::make_registry(42);
        NeuralNetwork network_1(reg1, config);
        auto params1 = network_1.parameters();

        // Save state_dict
        auto state_dict = network_1.state_dict();
        std::string path = PathProvider::get_full_path("test/state_dict.nn");
        Serializer::save(state_dict, path);

        const auto reg2 = LayerFactory::make_registry(77);
        NeuralNetwork network_2(reg2, config);
        auto params2 = network_2.parameters();

        // Confirm that params1 is different from params2 for layers 1, 3, 5
        EXPECT_NE(params1[0].second->data()[0], params2[0].second->data()[0]);
        EXPECT_NE(params1[2].second->data()[0], params2[2].second->data()[0]);
        EXPECT_NE(params1[4].second->data()[0], params2[4].second->data()[0]);

        // Load state_dict
        auto loaded_state_dict = Serializer::load(path);
        network_2.load_state_dict(loaded_state_dict);

        // Confirm that params1 is the same as params2 for layers 1, 3, 5
        EXPECT_EQ(params1[0].second->data()[0], params2[0].second->data()[0]);
        EXPECT_EQ(params1[2].second->data()[0], params2[2].second->data()[0]);
        EXPECT_EQ(params1[4].second->data()[0], params2[4].second->data()[0]);

        // Delete state_dict file
        std::remove(path.c_str());
    }

    TEST(Creation, NeuralNetwork_Randomized)
    {
        const std::vector<NeuralNetwork::LayerSpec> config =
        {
            { "Flatten", {} },
            { "Linear", { 28 * 28, 512 } },
            { "Relu", {} },
            { "Linear", { 512, 512 } },
            { "Relu", {} },
            { "Linear", { 512, 10 } }
        };
        const auto registry = LayerFactory::make_registry();
        NeuralNetwork network(registry, config);

        auto params = network.parameters();

        EXPECT_EQ(params.size(), 6);
        EXPECT_EQ(params[0].first, "linear_1.weight");
        EXPECT_EQ(params[1].first, "linear_1.bias");
        EXPECT_EQ(params[2].first, "linear_2.weight");
        EXPECT_EQ(params[3].first, "linear_2.bias");
        EXPECT_EQ(params[4].first, "linear_3.weight");
        EXPECT_EQ(params[5].first, "linear_3.bias");
    }
}
