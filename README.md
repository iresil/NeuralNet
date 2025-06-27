# NeuralNet

### Automatic Differentiation (Autograd) rules for Tensor Addition
| Forward Type | Who Broadcast? | Who Gets Gradient Summed? |
| --- | :---: | --- |
| Scalar + Scalar | Neither | No summing needed |
| Scalar + 1D | Scalar | Scalar (Sum Gradients) |
| Scalar + 2D | Scalar | Scalar (Sum Gradients) |
| 1D + Scalar | Scalar | Scalar (Sum Gradients) |
| 2D + Scalar | Scalar | Scalar (Sum Gradients) |
| 1D + 1D | Neither | No summing needed |
| 2D + 2D | Neither | No summing needed |
