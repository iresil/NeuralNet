#pragma once
class Tensor;
#include <memory>
#include <string>
#include <vector>
#include <utility>

class Dataset
{
    private:
        int reverse_int(int i);
        float convert_to_float(unsigned char px);

    public:
        virtual std::pair<int, std::shared_ptr<Tensor>> get_item(int index) = 0;
        virtual int get_length() = 0;
        std::vector<int> read_mnist_labels(std::string path);
        std::vector<std::vector<std::vector<float>>> read_mnist(std::string path);
        void visualize_image(std::shared_ptr<Tensor> image);
};
