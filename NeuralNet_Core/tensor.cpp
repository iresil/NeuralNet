#include "pch.h"
#include "tensor.h"
#include <map>
#include "tensor_operations.h"

std::map<std::pair<std::size_t, std::size_t>, Tensor::AddOperation> addition_operations =
{
    { { 0, 0 }, { TensorOperations::add_scalar_scalar, TensorOperations::add_scalar_scalar_reverse } },
    { { 0, 1 }, { TensorOperations::add_scalar_1D, TensorOperations::add_scalar_1D_reverse } },
    { { 0, 2 }, { TensorOperations::add_scalar_2D, TensorOperations::add_scalar_2D_reverse } },
    { { 1, 0 }, { TensorOperations::add_1D_scalar, TensorOperations::add_1D_scalar_reverse } },
    { { 2, 0 }, { TensorOperations::add_2D_scalar, TensorOperations::add_2D_scalar_reverse } },
    { { 1, 1 }, { TensorOperations::add_1D_1D, TensorOperations::add_1D_1D_reverse } },
    { { 2, 2 }, { TensorOperations::add_2D_2D, TensorOperations::add_2D_2D_reverse } }
};

std::map<std::pair<std::size_t, std::size_t>, Tensor::MultOperation> multiplication_operations =
{
    { { 1, 1 }, { TensorOperations::mult_1D_1D, TensorOperations::mult_1D_1D_reverse } },
    { { 2, 1 }, { TensorOperations::mult_2D_1D, TensorOperations::mult_2D_1D_reverse } },
    { { 1, 2 }, { TensorOperations::mult_1D_2D, TensorOperations::mult_1D_2D_reverse } },
    { { 2, 2 }, { TensorOperations::mult_2D_2D, TensorOperations::mult_2D_2D_reverse } }
};

Tensor::Tensor(float data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents) :
    _data{ data }, _shape{}, _stride{}, _dimension_x{ 1 }, _dimension_y{ 1 },
    _requires_grad{ requires_grad }, _gradfn{ gradfn }, _parents{ parents }
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<float> data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents) :
    _data{ data }, _shape{ data.size() }, _stride{ 1 }, _dimension_x{ data.size() }, _dimension_y{ 1 },
    _requires_grad{ requires_grad }, _gradfn{ gradfn }, _parents{ parents }
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<std::vector<float>> data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents) :
    _shape{ data.size(), data[0].size() }, _stride{ data[0].size(), 1 }, _dimension_x{ data.size() }, _dimension_y{ data[0].size() },
    _requires_grad{ requires_grad }, _gradfn{ gradfn }, _parents{ parents }
{
    // Validate dimensions
    _dimension_x = data.size();
    _dimension_y = data[0].size();

    std::vector<std::size_t> indices(_dimension_x);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
    {
        if (data[i].size() != _dimension_y)
        {
            throw std::invalid_argument("Dimensions are inconsistent");
        }
    });

    _data.resize(_dimension_x * _dimension_y);

    // Store in row-major format
    std::vector<std::size_t> indices_i(_dimension_x);
    std::iota(indices_i.begin(), indices_i.end(), 0);
    std::vector<std::size_t> indices_j(_dimension_y);
    std::iota(indices_j.begin(), indices_j.end(), 0);
    std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
    {
        std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
        {
            _data[i * _dimension_y + j] = data[i][j];
        });
    });

    if (_requires_grad)
    {
        zero_grad();
    }
}

void Tensor::make_with_grad(bool requires_grad, std::vector<std::shared_ptr<Tensor>> parents,
                            std::function<void(const std::vector<float>&)> gradfn)
{
    _requires_grad = requires_grad;
    _parents = parents;
    _gradfn = gradfn;
    zero_grad();
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }
const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

bool Tensor::requires_grad() const { return _requires_grad; }
const std::vector<float> &Tensor::grad() const { return _grad; }
void Tensor::zero_grad() { _grad = std::vector<float>(_data.size(), 0.0f); }
void Tensor::add_to_grad(const std::vector<float> &grad_update)
{
    if (!_requires_grad)
    {
        return;
    }
    if (_grad.size() != grad_update.size())
    {
        throw std::runtime_error("Gradient shape mismatch during accumulation");
    }
    std::transform(std::execution::par, _grad.begin(), _grad.end(), grad_update.begin(), _grad.begin(), std::plus());
}
std::size_t Tensor::count() const { return _data.size(); }

std::vector<float> &Tensor::data() { return _data; }

std::size_t Tensor::argmax() const
{
    return std::distance(_data.begin(), std::max_element(_data.begin(), _data.end()));
}

void Tensor::_reset_graph_visit()
{
    if (!_visited)
    {
        return;
    }
    _visited = false;
    for (std::size_t i = 0; i < _parents.size(); i++)
    {
        _parents[i]->_reset_graph_visit();
    }
}

void Tensor::_backward()
{
    if (!_requires_grad)
    {
        return;
    }
    if (_visited)
    {
        return;
    }
    _visited = true;
    if (_gradfn)
    {
        _gradfn(_grad);
    }
    for (std::size_t i = 0; i < _parents.size(); i++)
    {
        _parents[i]->_backward();
    }
}

void Tensor::backward()
{
    if (!_requires_grad)
    {
        throw std::runtime_error("Element does not require grad");
    }
    if (_shape.size() != 0)
    {
        throw std::runtime_error("Gradient can only be calculated for scalar outputs");
    }
    _reset_graph_visit();
    _grad = { 1 };
    _backward();
}

template <typename T>
typename std::conditional<
    std::is_const<T>::value,
    const float &,
    float &
>::type Tensor::_get_item(T &tensor)
{
    // Works only with scalars and 1d tensors
    if (tensor._dimension_x == 1)
    {
        return tensor._data[0];
    }
    else
    {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}
template float &Tensor::_get_item<Tensor>(Tensor &tensor);
template const float &Tensor::_get_item<const Tensor>(const Tensor &tensor);

template <typename T>
typename std::conditional<
    std::is_const<T>::value,
    const float &,
    float &
>::type Tensor::_get_item(T &tensor, std::size_t i)
{
    if (tensor._shape.size() == 0)
    {
        throw std::invalid_argument("Can't index into a scalar. Use item() instead");
    }
    if (tensor._shape.size() == 1)
    {
        if (i >= tensor._shape[0])
        {
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(tensor._shape[0]));
        }
        return tensor._data[i];
    }
    throw std::invalid_argument("This is a 1D tensor. Use two indices for 2D tensors.");
}
template float &Tensor::_get_item<Tensor>(Tensor &tensor, std::size_t i);
template const float &Tensor::_get_item<const Tensor>(const Tensor &tensor, std::size_t i);

template <typename T>
typename std::conditional<
    std::is_const<T>::value,
    const float &,
    float &
>::type Tensor::_get_item(T &tensor, std::size_t i, std::size_t j)
{
    if (tensor._shape.size() == 2)
    {
        if (i >= tensor._shape[0])
        {
            throw std::invalid_argument("Row index " + std::to_string(i) + " is out of bounds for tensor with "
                + std::to_string(tensor._shape[0]) + " rows");
        }
        if (j >= tensor._shape[1])
        {
            throw std::invalid_argument("Column index " + std::to_string(j) + " is out of bounds for tensor with "
                + std::to_string(tensor._shape[1]) + " columns");
        }
        return tensor._data[i * tensor._stride[0] + j * tensor._stride[1]];
    }
    throw std::invalid_argument("Can only double index into 2D tensors");
}
template float &Tensor::_get_item<Tensor>(Tensor &tensor, std::size_t i, std::size_t j);
template const float &Tensor::_get_item<const Tensor>(const Tensor &tensor, std::size_t i, std::size_t j);

const float &Tensor::item() const
{
    return _get_item<const Tensor>(*this);
}

float &Tensor::item()
{
    return _get_item<Tensor>(*this);
}

const float &Tensor::operator()(std::size_t i) const
{
    return _get_item<const Tensor>(*this, i);
}

float &Tensor::operator()(std::size_t i)
{
    return _get_item<Tensor>(*this, i);
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const
{
    return _get_item<const Tensor>(*this, i, j);
}

float &Tensor::operator()(std::size_t i, std::size_t j)
{
    return _get_item<Tensor>(*this, i, j);
}

std::shared_ptr<Tensor> Tensor::create_tensor_with_grad(std::shared_ptr<Tensor> result, std::shared_ptr<Tensor> self,
                                                        std::shared_ptr<Tensor> other, GradFunc backward)
{
    std::vector<std::shared_ptr<Tensor>> parents{ self, other };
    std::function<void(const std::vector<float>&)> gradfn = [self, other, backward](const std::vector<float> &grad_output)
    {
        backward(self, other, grad_output);
    };

    result->make_with_grad(true, parents, gradfn);

    return result;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other)
{
    std::pair<std::size_t, std::size_t> key = std::make_pair(_shape.size(), other->shape().size());
    auto it = addition_operations.find(key);
    if (it == addition_operations.end())
    {
        throw std::invalid_argument("Unsupported shape combination");
    }

    std::shared_ptr<Tensor> self = shared_from_this();
    std::shared_ptr<Tensor> result = it->second.forward(self, other);

    if (_requires_grad || other->requires_grad())
    {
        return create_tensor_with_grad(result, self, other, it->second.backward);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other)
{
    if (_shape.size() == 0 || other->shape().size() == 0)
    {
        throw std::invalid_argument("Both arguments need to be at least 1D for matrix multiplication");
    }

    if (_shape.back() != other->shape().front())
    {
        throw std::invalid_argument("Shape mismatch: incompatible dimensions for multiplication");
    }

    std::pair<std::size_t, std::size_t> key = std::make_pair(_shape.size(), other->shape().size());
    auto it = multiplication_operations.find(key);
    if (it == multiplication_operations.end())
    {
        throw std::invalid_argument("Unsupported shape combination for multiplication");
    }

    std::shared_ptr<Tensor> self = shared_from_this();
    std::shared_ptr<Tensor> result = it->second.forward(self, other);

    if (_requires_grad || other->requires_grad())
    {
        return create_tensor_with_grad(result, self, other, it->second.backward);
    }

    return result;
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
