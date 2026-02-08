from settings import *
import math
import torch
import cupy as cp

torch._dynamo.allow_in_graph(cp.RawKernel)

KERNEL_THREADS_COUNT = HIDDEN_SIZE

ft_forward_kernel = "ft_forward_kernel"
ft_backward_kernel = "ft_backward_kernel"

module = cp.RawModule(
    code=open('cuda/ft_kernels.cu').read(),
    options=("-std=c++14", "-O3"),
    name_expressions=[ft_forward_kernel, ft_backward_kernel]
)

ft_forward_kernel = module.get_function(ft_forward_kernel)
ft_backward_kernel = module.get_function(ft_backward_kernel)

class FeatureTransformerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, active_features: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
        assert len(active_features.shape) == 2
        assert active_features.dtype == torch.int32
        assert active_features.device == DEVICE
        assert active_features.is_cuda
        assert active_features.is_contiguous()

        assert len(weights.shape) == 2
        assert weights.dtype == torch.float32
        assert weights.device == DEVICE
        assert weights.is_cuda
        assert weights.is_contiguous()

        assert len(biases.shape) == 1
        assert biases.dtype == torch.float32
        assert biases.device == DEVICE
        assert biases.is_cuda
        assert biases.is_contiguous()

        ctx.save_for_backward(active_features, weights, biases)

        batch_size = active_features.shape[0]

        output = torch.empty(
            batch_size,
            HIDDEN_SIZE,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True
        )

        kernel_blocks_count = batch_size

        ft_forward_kernel(
            (kernel_blocks_count,),
            (KERNEL_THREADS_COUNT,),
            (active_features.data_ptr(), weights.data_ptr(), biases.data_ptr(), output.data_ptr())
        )

        return output

    @staticmethod
    def backward(ctx, out_grad):
        assert not ctx.needs_input_grad[0]
        assert ctx.needs_input_grad[1]
        assert ctx.needs_input_grad[2]

        active_features, weights, biases = ctx.saved_tensors

        batch_size = active_features.shape[0]

        out_grad = out_grad.contiguous()

        w_grad = torch.zeros(weights.shape, dtype=torch.float32, device=DEVICE)
        b_grad = torch.zeros(biases.shape, dtype=torch.float32, device=DEVICE)

        kernel_blocks_count = batch_size

        ft_backward_kernel(
            (kernel_blocks_count,),
            (KERNEL_THREADS_COUNT,),
            (active_features.data_ptr(), w_grad.data_ptr(), b_grad.data_ptr(), out_grad.data_ptr())
        )

        return None, w_grad, b_grad

class FeatureTransformer(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(FeatureTransformer, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.rand(num_inputs, num_outputs, dtype=torch.float32) * 0.2 - 0.1
        )

        self.bias = torch.nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * 0.2 - 0.1)

    def forward(self, active_features: torch.Tensor):
        return FeatureTransformerFunction.apply(active_features, self.weight, self.bias)
