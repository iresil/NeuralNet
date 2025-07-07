#include "../NeuralNet_Core/module.h"
#include "../NeuralNet_Layers/flatten.h"
#include "../NeuralNet_Layers/linear.h"
#include "../NeuralNet_Layers/relu.h"
#include <memory>

class NeuralNetwork : public Module
{
    private:
        // Layers
        std::shared_ptr<Flatten> _flatten = std::make_shared<Flatten>();
        std::shared_ptr<Linear> _linear_1 = std::make_shared<Linear>(28 * 28, 512);
        std::shared_ptr<Linear> _linear_2 = std::make_shared<Linear>(512, 512);
        std::shared_ptr<Linear> _linear_3 = std::make_shared<Linear>(512, 10);
        // Activation
        std::shared_ptr<Relu> _relu = std::make_shared<Relu>();

    public:
        NeuralNetwork()
        {
            register_module("linear_1", _linear_1);
            register_module("linear_2", _linear_2);
            register_module("linear_3", _linear_3);
        }

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input)
        {
            std::shared_ptr<Tensor> flat = (*_flatten)(input);
            std::shared_ptr<Tensor> linear1 = (*_linear_1)(flat);
            std::shared_ptr<Tensor> relu1 = (*_relu)(linear1);
            std::shared_ptr<Tensor> linear2 = (*_linear_2)(relu1);
            std::shared_ptr<Tensor> relu2 = (*_relu)(linear2);
            std::shared_ptr<Tensor> linear3 = (*_linear_3)(relu2);
            return linear3;
        }
};