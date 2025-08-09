#pragma once
class Module;
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <any>

class LayerFactory
{
    public:
        static std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> make_registry(int seed = -1);
};
