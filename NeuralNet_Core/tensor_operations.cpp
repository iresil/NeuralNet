#include "pch.h"
#include "tensor_operations.h"
#include "tensor.h"

std::shared_ptr<Tensor> TensorOperations::add_scalar_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    float result = self->item() + other->item();
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_scalar_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                                 const std::vector<float> &grad_output)
{
    // Two scalars are added and the result is a scalar (no broadcast during forward pass).
    // The gradient during backpropagation is also a scalar, which affects both inputs equally.
    // So we pass the gradient unchanged to both self and other.
    self->add_to_grad(grad_output);
    other->add_to_grad(grad_output);
}

std::shared_ptr<Tensor> TensorOperations::add_scalar_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = other->shape()[0];
    std::vector<float> result(count_i);

    std::vector<std::size_t> indices(count_i);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
    {
        result[i] = self->item() + (*other)(i);
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_scalar_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                             const std::vector<float> &grad_output)
{
    // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
    // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
    // The vector receives grad_output directly because it wasn't broadcast.
    float grad_self = 0.0f;
    std::size_t count_i = grad_output.size();
    for (std::size_t i = 0; i < count_i; i++)
    {
        grad_self += grad_output[i];
    }

    self->add_to_grad({ grad_self });
    other->add_to_grad(grad_output);
}

std::shared_ptr<Tensor> TensorOperations::add_scalar_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
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
            result_i[j] = self->item() + (*other)(i, j);
        });
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_scalar_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                             const std::vector<float> &grad_output)
{
    // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
    // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
    // The vector receives grad_output directly because it wasn't broadcast.
    float grad_self = 0.0f;
    std::size_t count_i = grad_output.size();
    for (std::size_t i = 0; i < count_i; i++)
    {
        grad_self += grad_output[i];
    }

    self->add_to_grad({ grad_self });
    other->add_to_grad(grad_output);
}

std::shared_ptr<Tensor> TensorOperations::add_1D_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::vector<float> result(count_i);

    std::vector<std::size_t> indices(count_i);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
    {
        result[i] = (*self)(i) + other->item();
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_1D_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                             const std::vector<float> &grad_output)
{
    // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
    // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
    // The vector receives grad_output directly because it wasn't broadcast.
    float grad_other = 0.0f;
    std::size_t count_i = grad_output.size();
    for (std::size_t i = 0; i < count_i; i++)
    {
        grad_other += grad_output[i];
    }

    self->add_to_grad(grad_output);
    other->add_to_grad({ grad_other });
}

std::shared_ptr<Tensor> TensorOperations::add_2D_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = self->shape()[1];
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
            result_i[j] = (*self)(i, j) + other->item();
        });
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_2D_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                             const std::vector<float> &grad_output)
{
    // Because a scalar is added to a vector, the scalar was broadcast during forward pass.
    // When a scalar is broadcast forward, its gradient must be the sum of all gradients from the broadcasted elements.
    // The vector receives grad_output directly because it wasn't broadcast.
    float grad_other = 0.0f;
    std::size_t count_i = grad_output.size();
    for (std::size_t i = 0; i < count_i; i++)
    {
        grad_other += grad_output[i];
    }

    self->add_to_grad(grad_output);
    other->add_to_grad({ grad_other });
}

std::shared_ptr<Tensor> TensorOperations::add_1D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::vector<float> result(count_i);

    std::vector<std::size_t> indices(count_i);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
    {
        result[i] = (*self)(i) + (*other)(i);
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_1D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                         const std::vector<float> &grad_output)
{
    // Two vectors are added and the result is a vector (no broadcast during forward pass).
    // So during backpropagation we pass the gradient unchanged to both self and other.
    self->add_to_grad(grad_output);
    other->add_to_grad(grad_output);
}

std::shared_ptr<Tensor> TensorOperations::add_2D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = self->shape()[1];
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
            result_i[j] = (*self)(i, j) + (*other)(i, j);
        });
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::add_2D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                         const std::vector<float> &grad_output)
{
    // Two vectors are added and the result is a vector (no broadcast during forward pass).
    // So during backpropagation we pass the gradient unchanged to both self and other.
    self->add_to_grad(grad_output);
    other->add_to_grad(grad_output);
}

std::shared_ptr<Tensor> TensorOperations::mult_1D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    float result = 0.0f;
    std::size_t count_i = self->shape()[0];
    for (std::size_t i = 0; i < count_i; i++)
    {
        result += (*self)(i) * (*other)(i);
    }
    return std::make_shared<Tensor>(result);
}

void TensorOperations::mult_1D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other, const std::vector<float> &grad_output)
{
    std::size_t count_i = self->count();
    std::vector<float> grad_self(count_i);
    std::vector<float> grad_other(count_i);

    std::vector<std::size_t> indices(count_i);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](std::size_t i)
    {
        grad_self[i] = (*other)(i) * grad_output[0];
        grad_other[i] = (*self)(i) * grad_output[0];
    });

    self->add_to_grad(grad_self);
    other->add_to_grad(grad_other);
}

std::shared_ptr<Tensor> TensorOperations::mult_2D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = self->shape()[1];
    std::vector<float> result(count_i);

    std::vector<std::size_t> indices_i(count_i);
    std::iota(indices_i.begin(), indices_i.end(), 0);
    std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
    {
        float result_i = 0.0f;
        for (std::size_t j = 0; j < count_j; j++)
        {
            result_i += (*self)(i, j) * (*other)(j);
        }
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::mult_2D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other, const std::vector<float> &grad_output)
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
            grad_self[i * count_j + j] = (*other)(j) * grad_output[i];
        });
    });

    std::size_t count_i_other = other->shape()[0];
    std::size_t count_j_other = self->shape()[0];
    std::vector<float> grad_other(count_i_other);

    std::vector<std::size_t> indices_i_other(count_i_other);
    std::iota(indices_i_other.begin(), indices_i_other.end(), 0);
    std::for_each(std::execution::par, indices_i_other.begin(), indices_i_other.end(), [&](std::size_t i)
    {
        float grad_other_i = 0.0f;
        for (std::size_t j = 0; j < count_j; j++)
        {
            grad_other_i += (*self)(j, i) * grad_output[j];
        }
        grad_other[i] = grad_other_i;
    });

    self->add_to_grad(grad_self);
    other->add_to_grad(grad_other);
}

std::shared_ptr<Tensor> TensorOperations::mult_1D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = other->shape()[1];
    std::size_t count_j = other->shape()[0];
    std::vector<float> result(count_i);

    std::vector<std::size_t> indices_i(count_i);
    std::iota(indices_i.begin(), indices_i.end(), 0);
    std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
    {
        float result_i = 0.0f;
        for (std::size_t j = 0; j < count_j; j++)
        {
            result_i += (*self)(j) * (*other)(j, i);
        }
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::mult_1D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other, const std::vector<float> &grad_output)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = other->shape()[1];
    std::vector<float> grad_self(count_i);

    std::vector<std::size_t> indices_i(count_i);
    std::iota(indices_i.begin(), indices_i.end(), 0);
    std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
    {
        float grad_self_i = 0.0f;
        for (std::size_t j = 0; j < count_j; j++)
        {
            grad_self_i += (*other)(i, j) * grad_output[j];
        }
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
            grad_other[i * count_j_other + j] = (*self)(i) * grad_output[j];
        });
    });

    self->add_to_grad(grad_self);
    other->add_to_grad(grad_other);
}

std::shared_ptr<Tensor> TensorOperations::mult_2D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = other->shape()[1];
    std::size_t count_k = self->shape()[1];
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
            float result_i_j = 0.0f;
            for (std::size_t k = 0; k < count_k; k++)
            {
                result_i_j += (*self)(i, k) * (*other)(k, j);
            }
            result_i[j] = result_i_j;
        });
        result[i] = result_i;
    });
    return std::make_shared<Tensor>(result);
}

void TensorOperations::mult_2D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other, const std::vector<float> &grad_output)
{
    std::size_t count_i = self->shape()[0];
    std::size_t count_j = self->shape()[1];
    std::size_t count_k = other->shape()[1];
    std::vector<float> grad_self(count_i * count_j);

    std::vector<std::size_t> indices_i(count_i);
    std::iota(indices_i.begin(), indices_i.end(), 0);
    std::vector<std::size_t> indices_j(count_j);
    std::iota(indices_j.begin(), indices_j.end(), 0);
    std::for_each(std::execution::par, indices_i.begin(), indices_i.end(), [&](std::size_t i)
    {
        std::for_each(std::execution::par, indices_j.begin(), indices_j.end(), [&](std::size_t j)
        {
            float grad_self_i_j = 0.0f;
            for (std::size_t k = 0; k < count_k; k++)
            {
                grad_self_i_j += (*other)(j, k) * grad_output[i * count_k + k];
            }
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
    std::for_each(std::execution::par, indices_i_other.begin(), indices_i_other.end(), [&](std::size_t i)
    {
        std::for_each(std::execution::par, indices_j_other.begin(), indices_j_other.end(), [&](std::size_t j)
        {
            float grad_other_i_j = 0.0f;
            for (std::size_t k = 0; k < count_k_other; k++)
            {
                grad_other_i_j += (*self)(k, i) * grad_output[k * count_j_other + j];
            }
            grad_other[i * count_j_other + j] = grad_other_i_j;
        });
    });

    self->add_to_grad(grad_self);
    other->add_to_grad(grad_other);
}
