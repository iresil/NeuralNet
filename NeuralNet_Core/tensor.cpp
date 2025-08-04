#include "pch.h"
#include <atomic>
#include <numeric>
#include <execution>
#include "tensor.h"

Tensor::Tensor(float data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents) :
    _data{ data }, _shape{}, _stride{}, _requires_grad{ requires_grad }, _gradfn{ gradfn }, _parents{ parents }
{
    if (_requires_grad)
    {
        zero_grad();
    }
}

Tensor::Tensor(std::vector<float> data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Tensor>> parents) :
    _data{ data }, _shape{ data.size() }, _stride{ 1 },
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
    _shape{ data.size(), data[0].size() }, _stride{ data[0].size(), 1 },
    _requires_grad{ requires_grad }, _gradfn{ gradfn }, _parents{ parents }
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

    if (_requires_grad)
    {
        zero_grad();
    }
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
    for (std::size_t i = 0; i < _grad.size(); i++)
    {
        _grad[i] += grad_update[i];
    }
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

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other)
{
    if (_shape.size() == 0 && other->shape().size() == 0)  // Scalar + Scalar
    {
        float result = item() + other->item();
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Two scalars are added and the result is a scalar (no broadcast during forward pass).
                // The gradient during backpropagation is also a scalar, which affects both inputs equally.
                // So we pass the gradient unchanged to both self and other.
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 0 && other->shape().size() == 1)  // Scalar + 1D
    {
        std::size_t count_i = other->shape()[0];
        std::vector<float> result(count_i);

        std::vector<std::size_t> indices(count_i);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
        {
            result[i] = item() + other->operator()(i);
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
                // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
                // The vector receives grad_output directly because it wasn't broadcast.
                std::atomic<float> grad_self = 0;

                std::size_t count_i = grad_output.size();
                std::vector<std::size_t> indices(count_i);
                std::iota(indices.begin(), indices.end(), 0);
                std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
                {
                    std::atomic_fetch_add(&grad_self, grad_output[i]);
                });

                self->add_to_grad({ grad_self });
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 0 && other->shape().size() == 2)  // Scalar + 2D
    {
        std::size_t count_i = other->shape()[0];
        std::size_t count_j = other->shape()[1];
        std::vector<std::vector<float>> result(count_i);
        std::vector<float> result_i(count_j);

        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                result_i[j] = item() + other->operator()(i, j);
            });
            result[i] = result_i;
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
                // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
                // The vector receives grad_output directly because it wasn't broadcast.
                std::atomic<float> grad_self = 0;
                std::size_t count_i = grad_output.size();

                std::vector<std::size_t> indices(count_i);
                std::iota(indices.begin(), indices.end(), 0);
                std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
                {
                    std::atomic_fetch_add(&grad_self, grad_output[i]);
                });

                self->add_to_grad({ grad_self });
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 1 && other->shape().size() == 0)  // 1D + Scalar
    {
        std::size_t count_i = _shape[0];
        std::vector<float> result(count_i);

        std::vector<std::size_t> indices(count_i);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
        {
            result[i] = operator()(i) + other->item();
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
                // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
                // The vector receives grad_output directly because it wasn't broadcast.
                std::atomic<float> grad_other = 0;
                std::size_t count_i = grad_output.size();

                std::vector<std::size_t> indices(count_i);
                std::iota(indices.begin(), indices.end(), 0);
                std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
                {
                    std::atomic_fetch_add(&grad_other, grad_output[i]);
                });

                self->add_to_grad(grad_output);
                other->add_to_grad({ grad_other });
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 2 && other->shape().size() == 0)  // 2D + Scalar
    {
        std::size_t count_i = _shape[0];
        std::size_t count_j = _shape[1];
        std::vector<std::vector<float>> result(count_i);
        std::vector<float> result_i(count_j);

        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                result_i[j] = operator()(i, j) + other->item();
            });
            result[i] = result_i;
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
                // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
                // The vector receives grad_output directly because it wasn't broadcast.
                std::atomic<float> grad_other = 0;
                std::size_t count_i = grad_output.size();

                std::vector<std::size_t> indices(count_i);
                std::iota(indices.begin(), indices.end(), 0);
                std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
                {
                    std::atomic_fetch_add(&grad_other, grad_output[i]);
                });

                self->add_to_grad(grad_output);
                other->add_to_grad({ grad_other });
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 1 && other->shape().size() == 1 && _shape[0] == other->shape()[0])  // 1D + 1D
    {
        std::size_t count_i = _shape[0];
        std::vector<float> result(count_i);

        std::vector<std::size_t> indices(count_i);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
        {
            result[i] = operator()(i) + other->operator()(i);
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents { self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Two vectors are added and the result is a vector (no broadcast during forward pass).
                // So during backpropagation we pass the gradient unchanged to both self and other.
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 2 && other->shape().size() == 2 && _shape[0] == other->shape()[0])  // 2D + 2D
    {
        if (_shape[1] != other->shape()[1])
        {
            throw std::invalid_argument("Second dimensions are not equal");
        }

        std::size_t count_i = _shape[0];
        std::size_t count_j = _shape[1];
        std::vector<std::vector<float>> result(count_i);
        std::vector<float> result_i(count_j);

        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                result_i[j] = operator()(i, j) + other->operator()(i, j);
            });
            result[i] = result_i;
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                // Two vectors are added and the result is a vector (no broadcast during forward pass).
                // So during backpropagation we pass the gradient unchanged to both self and other.
                self->add_to_grad(grad_output);
                other->add_to_grad(grad_output);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else
    {
        if (_shape[0] != other->shape()[0])
        {
            throw std::invalid_argument("First dimensions are not equal");
        }
    }
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other)
{
    if (_shape.size() == 0 || other->shape().size() == 0)
    {
        throw std::invalid_argument("Both arguments need to be at least 1D for matrix multiplication");
    }
    if (_shape[_shape.size() - 1] != other->shape()[0])
    {
        throw std::invalid_argument("Last dimension of the first tensor doesn't have the same size as the first dimension of the second");
    }

    if (_shape.size() == 1 && other->shape().size() == 1)  // Dot Product: 1D x 1D -> float
    {
        std::atomic<float> result = 0;

        std::size_t count_i = _shape[0];
        std::vector<std::size_t> indices(count_i);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
        {
            std::atomic_fetch_add(&result, operator()(i) * other->operator()(i));
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                std::size_t count_i = self->count();
                std::vector<float> grad_self(count_i);
                std::vector<float> grad_other(count_i);

                std::vector<std::size_t> indices(count_i);
                std::iota(indices.begin(), indices.end(), 0);
                std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
                {
                    grad_self[i] = other->operator()(i) * grad_output[0];
                    grad_other[i] = self->operator()(i) * grad_output[0];
                });

                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 2 && other->shape().size() == 1)  // Matrix-Vector Product: 2D x 1D -> 1D
    {
        std::vector<float> result(_shape[0]);

        std::size_t count_i = _shape[0];
        std::size_t count_j = _shape[1];
        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::atomic<float> result_i = 0;
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                std::atomic_fetch_add(&result_i, operator()(i, j) * other->operator()(j));
            });
            result[i] = result_i;
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                std::size_t count_i = self->shape()[0];
                std::size_t count_j = self->shape()[1];
                std::vector<float> grad_self(count_i * count_j);

                std::vector<std::size_t> indices_i(count_i);
                std::iota(indices_i.begin(), indices_i.end(), 0);
                std::vector<std::size_t> indices_j(count_j);
                std::iota(indices_j.begin(), indices_j.end(), 0);
                std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
                {
                    std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
                    {
                        grad_self[i * count_j + j] = other->operator()(j) * grad_output[i];
                    });
                });

                std::size_t count_i_other = other->shape()[0];
                std::size_t count_j_other = self->shape()[0];
                std::vector<float> grad_other(count_i_other);

                std::vector<std::size_t> indices_i_other(count_i_other);
                std::iota(indices_i_other.begin(), indices_i_other.end(), 0);
                std::vector<std::size_t> indices_j_other(count_j_other);
                std::iota(indices_j_other.begin(), indices_j_other.end(), 0);
                std::for_each(std::execution::par, indices_i_other.begin(), indices_i_other.end(), [&](std::size_t i)
                {
                    std::atomic<float> grad_other_i = 0;
                    std::for_each(std::execution::par, indices_j_other.begin(), indices_j_other.end(), [&](std::size_t j)
                    {
                        std::atomic_fetch_add(&grad_other_i, self->operator()(j, i) * grad_output[j]);
                    });
                    grad_other[i] = grad_other_i;
                });

                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 1 && other->shape().size() == 2)  // Vector-Matrix Product: 1D x 2D -> 1D
    {
        std::size_t count_i = other->shape()[1];
        std::size_t count_j = other->shape()[0];
        std::vector<float> result(count_i);

        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::atomic<float> result_i = 0;
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                std::atomic_fetch_add(&result_i, operator()(j) * other->operator()(j, i));
            });
            result[i] = result_i;
        });
        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                std::size_t count_i = self->shape()[0];
                std::size_t count_j = other->shape()[1];
                std::vector<float> grad_self(count_i);

                std::vector<std::size_t> indices_i(count_i);
                std::iota(indices_i.begin(), indices_i.end(), 0);
                std::vector<std::size_t> indices_j(count_j);
                std::iota(indices_j.begin(), indices_j.end(), 0);
                std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
                {
                    std::atomic<float> grad_self_i = 0;
                    std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
                    {
                        std::atomic_fetch_add(&grad_self_i, other->operator()(i, j) * grad_output[j]);
                    });
                    grad_self[i] = grad_self_i;
                });

                std::size_t count_i_other = other->shape()[0];
                std::size_t count_j_other = other->shape()[1];
                std::vector<float> grad_other(count_i_other * count_j_other);

                std::vector<std::size_t> indices_i_other(count_i_other);
                std::iota(indices_i_other.begin(), indices_i_other.end(), 0);
                std::vector<std::size_t> indices_j_other(count_j_other);
                std::iota(indices_j_other.begin(), indices_j_other.end(), 0);
                std::for_each(std::execution::par, indices_i_other.begin(), indices_i_other.end(), [&](std::size_t i)
                {
                    std::for_each(std::execution::par, indices_j_other.begin(), indices_j_other.end(), [&](std::size_t j)
                    {
                        grad_other[i * count_j_other + j] = self->operator()(i) * grad_output[j];
                    });
                });

                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else if (_shape.size() == 2 && other->shape().size() == 2) // Matrix-Matrix Product: 2D x 2D -> 2D
    {
        std::size_t count_i = _shape[0];
        std::size_t count_j = other->shape()[1];
        std::size_t count_k = _shape[1];
        std::vector<std::vector<float>> result(count_i);
        std::vector<float> result_i(count_j);

        std::vector<std::size_t> indices_i(count_i);
        std::iota(indices_i.begin(), indices_i.end(), 0);
        std::vector<std::size_t> indices_j(count_j);
        std::iota(indices_j.begin(), indices_j.end(), 0);
        std::vector<std::size_t> indices_k(count_k);
        std::iota(indices_k.begin(), indices_k.end(), 0);
        std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
        {
            std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
            {
                std::atomic<float> result_i_j = 0;
                std::for_each(std::execution::par, indices_k.begin(), indices_k.end(), [&](std::size_t k)
                {
                    std::atomic_fetch_add(&result_i_j, operator()(i, k) * other->operator()(k, j));
                });
                result_i[j] = result_i_j;
            });
            result[i] = result_i;
        });

        if (_requires_grad || other->requires_grad())
        {
            std::shared_ptr<Tensor> self = shared_from_this();
            std::vector<std::shared_ptr<Tensor>> parents{ self, other };
            std::function<void(const std::vector<float>&)> gradfn = [self, other](const std::vector<float> &grad_output)
            {
                std::size_t count_i = self->shape()[0];
                std::size_t count_j = self->shape()[1];
                std::size_t count_k = other->shape()[1];
                std::vector<float> grad_self(count_i *count_j);

                std::vector<std::size_t> indices_i(count_i);
                std::iota(indices_i.begin(), indices_i.end(), 0);
                std::vector<std::size_t> indices_j(count_j);
                std::iota(indices_j.begin(), indices_j.end(), 0);
                std::vector<std::size_t> indices_k(count_k);
                std::iota(indices_k.begin(), indices_k.end(), 0);
                std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
                {
                    std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
                    {
                        std::atomic<float> grad_self_i_j = 0;
                        std::for_each(std::execution::par, indices_k.begin(), indices_k.end(), [&](std::size_t k)
                        {
                            std::atomic_fetch_add(&grad_self_i_j, other->operator()(j, k) * grad_output[i * count_k + k]);
                        });
                        grad_self[i * count_j + j] = grad_self_i_j;
                    });
                });

                std::size_t count_i_other = other->shape()[0];
                std::size_t count_j_other = other->shape()[1];
                std::size_t count_k_other = self->shape()[0];
                std::vector<float> grad_other(count_i_other * count_j_other);

                std::vector<std::size_t> indices_i_other(count_i_other);
                std::iota(indices_i_other.begin(), indices_i_other.end(), 0);
                std::vector<std::size_t> indices_j_other(count_j_other);
                std::iota(indices_j_other.begin(), indices_j_other.end(), 0);
                std::vector<std::size_t> indices_k_other(count_k_other);
                std::iota(indices_k_other.begin(), indices_k_other.end(), 0);
                std::for_each(std::execution::par, indices_i_other.begin(), indices_i_other.end(), [&](std::size_t i)
                {
                    std::for_each(std::execution::par, indices_j_other.begin(), indices_j_other.end(), [&](std::size_t j)
                    {
                        std::atomic<float> grad_other_i_j = 0;
                        std::for_each(std::execution::par, indices_k_other.begin(), indices_k_other.end(), [&](std::size_t k)
                        {
                            std::atomic_fetch_add(&grad_other_i_j, self->operator()(k, i) * grad_output[k * count_j_other + j]);
                        });
                        grad_other[i * count_j_other + j] = grad_other_i_j;
                    });
                });

                self->add_to_grad(grad_self);
                other->add_to_grad(grad_other);
            };
            return std::make_shared<Tensor>(result, true, gradfn, parents);
        }
        return std::make_shared<Tensor>(result);
    }
    else
    {
        throw std::invalid_argument("One or more of the tensors is a scalar");
    }
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
