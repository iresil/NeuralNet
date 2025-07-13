#include "pch.h"
#include "dataset_fashionmnist.h"
#include "../NeuralNet_Core/tensor.h"

FashionMNIST::FashionMNIST(std::string data_path, std::string labels_path)
{
    _labels = read_mnist_labels(labels_path);
    _images = read_mnist(data_path);
}

std::pair<int, std::shared_ptr<Tensor>> FashionMNIST::get_item(int index)
{
    return std::make_pair(_labels[index], std::make_shared<Tensor>(_images[index]));
}

int FashionMNIST::get_length()
{
    return _images.size();
}

std::string FashionMNIST::label_to_class(int label)
{
    return _classes[label];
}
