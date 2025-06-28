# NeuralNet

## Automatic Differentiation (Autograd)
### Rules for Tensor Addition
| Forward Type | Who Broadcast? | Who Gets Gradient Summed? | Notes |
| --- | :---: | --- | --- |
| Scalar + Scalar | Neither | No summing needed | |
| Scalar + 1D | Scalar | Scalar (Sum Gradients) | |
| Scalar + 2D | Scalar | Scalar (Sum Gradients) | |
| 1D + Scalar | Scalar | Scalar (Sum Gradients) | |
| 2D + Scalar | Scalar | Scalar (Sum Gradients) | |
| 1D + 1D | Neither | No summing needed | |
| 1D + 2D | 1D | 1D (Sum Over Rows) | Not Implemented, Covered by 1D + (2D&#8594;1D) |
| 2D + 1D | 1D | 1D (Sum Over Rows) | Not Implemented, Covered by (2D&#8594;1D) + 1D |
| 2D + 2D | Neither | No summing needed | |
