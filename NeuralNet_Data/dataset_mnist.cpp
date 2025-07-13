#include "pch.h"
#include "dataset_mnist.h"
#include "../NeuralNet_Core/tensor.h"

MNIST::MNIST(std::string data_path, std::string labels_path)
{
    _labels = read_mnist_labels(labels_path);
    _images = read_mnist(data_path);
}

std::pair<int, std::shared_ptr<Tensor>> MNIST::get_item(int index)
{
    return std::make_pair(_labels[index], std::make_shared<Tensor>(_images[index]));
}

int MNIST::get_length()
{
    return _images.size();
}

std::string MNIST::label_to_class(int label)
{
    return _classes[label];
}
