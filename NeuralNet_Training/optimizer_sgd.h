#pragma once
class Tensor;
#include <memory>
#include <string>
#include <vector>

class SGD
{
    private:
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> _params;
        float _learning_rate;

    public:
        SGD(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> params, float lr = 0.001f);
        void step();
        void zero_grad();
};
