#pragma once
#include "tensor.h"

class Serializer
{
    private:
        static constexpr int MAGIC_NUMBER_ = 7777;

    public:
        void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict, const std::string &filename);
        std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename);
};
