#include "pch.h"
#include "dataset.h"
#include <fstream>
#include <iostream>
#include "../NeuralNet_Core/tensor.h"

std::vector<int> Dataset::read_mnist_labels(std::string path)
{
    std::ifstream file(path);
    std::vector<int> labels;
    if (file.is_open())
    {
        int magic_number = 0;
        int item_count = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        // We're reading the bytes of a big-endian file while running the code on a little-endian system,
        // so byte order will be reversed during reading (which will also reverse the resulting number).
        // We reverse the number back, to ensure that the most significant digit will be first.
        magic_number = reverse_int(magic_number);
        if (magic_number != 2049)
        {
            throw std::runtime_error("Invalid MNIST label file!");
        }
        file.read((char*)&item_count, sizeof(item_count));
        item_count = reverse_int(item_count);
        for (int i = 0; i < item_count; ++i)
        {
            unsigned char label = 0;
            file.read((char*)&label, sizeof(label));
            labels.push_back(label);
        }
    }
    return labels;
}

std::vector<std::vector<std::vector<float>>> Dataset::read_mnist(std::string path)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<std::vector<float>>> dataset;
    if (file.is_open())
    {
        int magic_number = 0;
        int image_count = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        // We're reading the bytes of a big-endian file while running the code on a little-endian system,
        // so byte order will be reversed during reading (which will also reverse the resulting number).
        // We reverse the number back, to ensure that the most significant digit will be first.
        magic_number = reverse_int(magic_number);
        if (magic_number != 2051)
        {
            throw std::runtime_error("Invalid MNIST image file!");
        }
        file.read((char*)&image_count, sizeof(image_count));
        image_count = reverse_int(image_count);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        for (int i = 0; i < image_count; ++i)
        {
            std::vector<std::vector<float>> image;
            for (int r = 0; r < n_rows; ++r)
            {
                std::vector<float> row;
                for (int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    row.push_back(convert_to_float(temp));
                }
                image.push_back(row);
            }
            dataset.push_back(image);
        }
    }
    return dataset;
}

void Dataset::visualize_image(std::shared_ptr<Tensor> image)
{
    for (int i = 0; i < image->shape()[0]; ++i)
    {
        for (int j = 0; j < image->shape()[1]; ++j)
        {
            float px = (*image)(i, j);
            std::cout << (px > 0.75f ? '@' : (px > 0.5f ? '#' : (px > 0.25f ? '+' : (px > 0.1f ? '.' : ' '))));
        }
        std::cout << std::endl;
    }
}

int Dataset::reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float Dataset::convert_to_float(unsigned char px)
{
    return (float)px / 255.0f;
}
