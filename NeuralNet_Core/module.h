#pragma once
class Tensor;
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

class Module
{
    private:
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> _parameters;
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> _modules;

    public:
        virtual ~Module() = default;
        virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
        std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input);
        void register_parameter(std::string name, std::shared_ptr<Tensor> param);
        void register_module(std::string name, std::shared_ptr<Module> param);
        std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> parameters() const;
        std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict() const;
        void load_state_dict(std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);
};
