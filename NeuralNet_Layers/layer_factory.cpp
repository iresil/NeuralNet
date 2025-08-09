#include "pch.h"
#include "layer_factory.h"
#include "flatten.h"
#include "linear.h"
#include "relu.h"

std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> LayerFactory::make_registry(int seed)
{
    std::unordered_map<std::string, std::function<std::shared_ptr<Module>(std::vector<std::any>)>> layer_registry =
    {
        {
            "Flatten",
            [](std::vector<std::any>)
            {
                return std::make_shared<Flatten>();
            }
        },
        {
            "Linear",
            [seed](std::vector<std::any> args)
            {
                if (seed == -1)
                {
                    return std::make_shared<Linear>(std::any_cast<int>(args[0]), std::any_cast<int>(args[1]));
                }
                else
                {
                    return std::make_shared<Linear>(std::any_cast<int>(args[0]), std::any_cast<int>(args[1]), seed);
                }
            }
        },
        {
            "Relu",
            [](std::vector<std::any>)
            {
                return std::make_shared<Relu>();
            }
        }
    };
    return layer_registry;
}
