#pragma once
class Tensor;
#include <memory>
#include <vector>

class TensorOperations
{
    public:
        static float add_scalar_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_scalar_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                              const std::vector<float> &grad_output);
        static std::vector<float> add_scalar_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_scalar_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                          const std::vector<float> &grad_output);
        static std::vector<std::vector<float>> add_scalar_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> & other);
        static void add_scalar_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                          const std::vector<float> &grad_output);
        static std::vector<float> add_1D_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_1D_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                          const std::vector<float> &grad_output);
        static std::vector<std::vector<float>> add_2D_scalar(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_2D_scalar_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                          const std::vector<float> &grad_output);
        static std::vector<float> add_1D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_1D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                      const std::vector<float> &grad_output);
        static std::vector<std::vector<float>> add_2D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void add_2D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                      const std::vector<float> &grad_output);

        static float mult_1D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void mult_1D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                       const std::vector<float> &grad_output);
        static std::vector<float> mult_2D_1D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void mult_2D_1D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                       const std::vector<float> &grad_output);
        static std::vector<float> mult_1D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void mult_1D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                       const std::vector<float> &grad_output);
        static std::vector<std::vector<float>> mult_2D_2D(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other);
        static void mult_2D_2D_reverse(const std::shared_ptr<Tensor> &self, const std::shared_ptr<Tensor> &other,
                                       const std::vector<float> &grad_output);
};
