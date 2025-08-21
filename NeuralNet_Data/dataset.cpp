#include "pch.h"
#include "dataset.h"
#include <fstream>
#include <iostream>
#include <execution>
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

        const std::size_t images_per_chunk = 10;
        std::vector<unsigned char> buffer(images_per_chunk * n_rows * n_cols);

        dataset.resize(image_count);

        for (std::size_t start = 0; start < image_count; start += images_per_chunk)
        {
            std::size_t count = std::min(images_per_chunk, image_count - start);

            // Sequential disk read into buffer
            file.read(reinterpret_cast<char *>(buffer.data()), count * n_rows * n_cols);

            // Parallel transform into dataset
            std::vector<std::size_t> indices(count);
            std::iota(indices.begin(), indices.end(), 0);

            std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t idx)
            {
                std::size_t i = start + idx;
                std::vector<std::vector<float>> &image = dataset[i];
                image.resize(n_rows);

                for (int r = 0; r < n_rows; r++)
                {
                    std::vector<float> &row = image[r];
                    row.resize(n_cols);

                    for (int c = 0; c < n_cols; c++)
                    {
                        std::size_t buf_idx = idx * n_rows * n_cols + r * n_cols + c;
                        row[c] = convert_to_float(buffer[buf_idx]);
                    }
                }
            });
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
