#pragma once
class Tensor;
#include <memory>
#include <string>
#include <unordered_map>

class Serializer
{
    private:
        static constexpr int MAGIC_NUMBER_ = 7777;

    public:
        static void save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict, const std::string &filename);
        static std::unordered_map<std::string, std::shared_ptr<Tensor>> load(const std::string &filename);
};
