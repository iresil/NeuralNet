#pragma once
class Tensor;
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "dataset.h"

class MNIST : public Dataset
{
    private:
        std::vector<std::vector<std::vector<float>>> _images;
        std::vector<int> _labels;
        std::vector<std::string> classes = { "zero", "one", "two", "three", "four",
                                             "five", "six", "seven", "eight", "nine" };

    public:
        MNIST(std::string data_path, std::string labels_path);
        std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
        int get_length() override;
        std::string label_to_class(int label);
};
