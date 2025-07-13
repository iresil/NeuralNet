#pragma once
class Tensor;
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "dataset.h"

class FashionMNIST : public Dataset
{
    private:
        std::vector<std::vector<std::vector<float>>> _images;
        std::vector<int> _labels;
        std::vector<std::string> _classes = { "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"};

    public:
        FashionMNIST(std::string data_path, std::string labels_path);
        std::pair<int, std::shared_ptr<Tensor>> get_item(int index) override;
        int get_length() override;
        std::string label_to_class(int label);
};
