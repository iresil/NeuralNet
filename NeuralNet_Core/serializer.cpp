#include "pch.h"
#include "serializer.h"

void Serializer::save(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
                      const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&MAGIC_NUMBER_), sizeof(int));
    for (const auto &[weight_name, weight] : state_dict)
    {
        std::size_t name_len = weight_name.size();
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(std::size_t));
        file.write(weight_name.data(), name_len);

        std::size_t shape_len = weight->shape().size();
        file.write(reinterpret_cast<const char *>(&shape_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(weight->shape().data()), shape_len * sizeof(std::size_t));

        std::size_t data_len = weight->count();
        file.write(reinterpret_cast<const char *>(&data_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(weight->data().data()), data_len * sizeof(float));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Serializer::load(const std::string &filename)
{
    std::unordered_map<std::string, std::shared_ptr<Tensor>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open " + filename);
    }

    int magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(int));
    if (magic != MAGIC_NUMBER_)
    {
        throw std::runtime_error("Bad file format: Wrong magic number");
    }

    while (file.peek() != EOF)
    {
        std::size_t name_len = 0;
        if (!file.read(reinterpret_cast<char*>(&name_len), sizeof(std::size_t)))
        {
            break;
        }

        std::string weight_name(name_len, '\0');
        file.read(weight_name.data(), name_len);

        std::size_t shape_len = 0;
        file.read(reinterpret_cast<char *>(&shape_len), sizeof(std::size_t));

        std::vector<std::size_t> shape(shape_len);
        file.read(reinterpret_cast<char *>(shape.data()), shape_len * sizeof(std::size_t));

        std::size_t data_len = 0;
        file.read(reinterpret_cast<char *>(&data_len), sizeof(std::size_t));

        std::vector<float> raw(data_len);
        file.read(reinterpret_cast<char *>(raw.data()), data_len * sizeof(float));

        std::shared_ptr<Tensor> tensor;
        if (shape_len == 0)
        {
            tensor = std::make_shared<Tensor>(raw[0]);
        }
        else if (shape_len == 1)
        {
            tensor = std::make_shared<Tensor>(raw);
        }
        else if (shape_len == 2)
        {
            std::vector<std::vector<float>> data_2d(shape[0], std::vector<float>(shape[1]));
            for (std::size_t i = 0; i < shape[0]; i++)
            {
                for (std::size_t j = 0; j < shape[1]; j++)
                {
                    data_2d[i][j] = raw[i * shape[1] + j];
                }
            }
            tensor = std::make_shared<Tensor>(data_2d);
        }
        else
        {
            throw std::runtime_error("Unsupported tensor dimensionality: " + std::to_string(shape_len));
        }

        state_dict[weight_name] = tensor;
    }

    return state_dict;
}
