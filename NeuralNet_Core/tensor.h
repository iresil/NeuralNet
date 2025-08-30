#pragma once
#include <functional>
#include <iostream>
#include <memory>
#include <cstddef>
#include <type_traits>
#include <vector>

class Tensor : public std::enable_shared_from_this<Tensor>
{
    private:
        std::vector<float> _data;
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;
        std::size_t _dimension_x;
        std::size_t _dimension_y;

        std::vector<float> _grad;
        std::function<void(const std::vector<float>&)> _gradfn;
        std::vector<std::shared_ptr<Tensor>> _parents;
        bool _requires_grad;

        void _backward();
        bool _visited = false;
        void _reset_graph_visit();

        template <typename T>
        static typename std::conditional<
            std::is_const<T>::value,
            const float &,
            float &
        >::type _get_item(T &tensor);

        template <typename T>
        static typename std::conditional<
            std::is_const<T>::value,
            const float &,
            float &
        >::type _get_item(T &tensor, std::size_t i);

        template <typename T>
        static typename std::conditional<
            std::is_const<T>::value,
            const float &,
            float &
        >::type _get_item(T &tensor, std::size_t i, std::size_t j);

    public:
        Tensor(float data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {});
        Tensor(std::vector<float> data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {});
        Tensor(std::vector<std::vector<float>> data, bool requires_grad = false,
               std::function<void(const std::vector<float>&)> gradfn = nullptr,
               std::vector<std::shared_ptr<Tensor>> parents = {});

        const std::vector<std::size_t> &shape() const;
        const std::vector<std::size_t> &stride() const;

        bool requires_grad() const;
        const std::vector<float> &grad() const;
        void zero_grad();
        void add_to_grad(const std::vector<float> &grad_update);
        std::size_t count() const;
        std::vector<float> &data();
        std::size_t argmax() const;

        void backward();

        const float &item() const;
        float &item();
        const float &operator()(std::size_t i) const;
        float &operator()(std::size_t i);
        const float &operator()(std::size_t i, std::size_t j) const;
        float &operator()(std::size_t i, std::size_t j);

        std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
        std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);
        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};
