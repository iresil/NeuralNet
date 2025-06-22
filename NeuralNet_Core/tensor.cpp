#include "pch.h"
#include "tensor.h"

Tensor::Tensor(float data) : _data{ data }, _shape{}, _stride{} {}

Tensor::Tensor(std::vector<float> data) : _data{ data }, _shape{ data.size() }, _stride{ 1 } {}

Tensor::Tensor(std::vector<std::vector<float>> data) :
    _shape{ data.size(), data[0].size() }, _stride{ data[0].size(), 1 }
{
    // Validate dimensions
    std::size_t expected_cols = data[0].size();
    for (std::size_t i = 0; i < data.size(); i++)
    {
        if (data[i].size() != expected_cols)
        {
            throw std::invalid_argument("Dimensions are inconsistent");
        }
    }

    // Store in row-major format
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < data[i].size(); j++)
        {
            _data.push_back(data[i][j]);
        }
    }
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }
const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

const float &Tensor::item() const
{
    // Works only with scalars and 1d tensors
    if (_data.size() == 1)
    {
        return _data[0];
    }
    else
    {
        throw::std::runtime_error("item() can only be called on tensors with a single element");
    }
}

float &Tensor::item()
{
    // Works only with scalars and 1d tensors
    if (_data.size() == 1)
    {
        return _data[0];
    }
    else
    {
        throw::std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const float &Tensor::operator()(std::size_t i) const
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

float &Tensor::operator()(std::size_t i)
{
    if (_shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (_shape.size() == 1)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size "
                + std::to_string(_shape[0]));
        }
        return _data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row index " + std::to_string(i) + " is out of bounds for tensor with "
                + std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column index " + std::to_string(j) + " is out of bounds for tensor with "
                + std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    if (_shape.size() == 2)
    {
        if (i >= _shape[0])
        {
            throw std::invalid_argument("Row index " + std::to_string(i) + " is out of bounds for tensor with "
                + std::to_string(_shape[0]) + " rows");
        }
        if (j >= _shape[1])
        {
            throw std::invalid_argument("Column index " + std::to_string(j) + " is out of bounds for tensor with "
                + std::to_string(_shape[1]) + " columns");
        }
        return _data[i * _stride[0] + j * _stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj)
{
    std::string string_repr = "[";
    if (obj.shape().size() == 0)
    {
        os << obj.item();
        return os;
    }
    else if (obj.shape().size() == 1)
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    else
    {
        for (std::size_t i = 0; i < obj.shape()[0]; i++)
        {
            string_repr += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; j++)
            {
                string_repr += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1)
                {
                    string_repr += ", ";
                }
            }
            string_repr += "]";
            if (i != obj.shape()[0] - 1)
            {
                string_repr += ", ";
            }
        }
        string_repr += "]";
    }
    os << string_repr;
    return os;
}
