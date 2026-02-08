from settings import *
import math
import re
import torch
import cupy as cp

torch._dynamo.allow_in_graph(cp.RawKernel)

KERNEL_THREADS = (HIDDEN_SIZE,)

hidden_to_logits_forward_kernel = "hidden_to_logits_forward_kernel"
hidden_to_logits_backward_kernel = "hidden_to_logits_backward_kernel"

module = cp.RawModule(
    code=open("cuda/hidden_to_logits_kernels.cu").read(),
    options=("-std=c++14", "-O3"),
    name_expressions=[hidden_to_logits_forward_kernel, hidden_to_logits_backward_kernel]
)

hidden_to_logits_forward_kernel = module.get_function(hidden_to_logits_forward_kernel)
hidden_to_logits_backward_kernel = module.get_function(hidden_to_logits_backward_kernel)

class HiddenToLogitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_layer: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor,
        legal_moves_idxs: torch.Tensor
    ):
        assert len(hidden_layer.shape) == 2
        assert hidden_layer.dtype == torch.float32
        assert hidden_layer.device == DEVICE
        assert hidden_layer.is_cuda
        assert hidden_layer.is_contiguous()

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

        assert len(legal_moves_idxs.shape) == 2
        assert legal_moves_idxs.dtype == torch.int32
        assert legal_moves_idxs.device == DEVICE
        assert legal_moves_idxs.is_cuda
        assert legal_moves_idxs.is_contiguous()

        ctx.save_for_backward(hidden_layer, weights, biases, legal_moves_idxs)

        batch_size = legal_moves_idxs.shape[0]

        output = torch.zeros(
            batch_size,
            MAX_MOVES_PER_POS,
            dtype=torch.float32,
            device=DEVICE,
            requires_grad=True
        )

        kernel_blocks = (MAX_MOVES_PER_POS, batch_size)

        hidden_to_logits_forward_kernel(
            kernel_blocks,
            KERNEL_THREADS,
            (
                hidden_layer.data_ptr(),
                weights.data_ptr(),
                biases.data_ptr(),
                legal_moves_idxs.data_ptr(),
                output.data_ptr()
            )
        )

        return output

    @staticmethod
    def backward(ctx, out_grad):
        assert ctx.needs_input_grad[0]
        assert ctx.needs_input_grad[1]
        assert ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]

        hidden_layer, weights, biases, legal_moves_idxs = ctx.saved_tensors

        batch_size = legal_moves_idxs.shape[0]

        out_grad = out_grad.contiguous()

        hidden_grad = torch.zeros(hidden_layer.shape, dtype=torch.float32, device=DEVICE)
        w_grad = torch.zeros(weights.shape, dtype=torch.float32, device=DEVICE)
        b_grad = torch.zeros(biases.shape, dtype=torch.float32, device=DEVICE)

        kernel_blocks = (MAX_MOVES_PER_POS, batch_size)

        hidden_to_logits_backward_kernel(
            kernel_blocks,
            KERNEL_THREADS,
            (
                hidden_layer.data_ptr(),
                weights.data_ptr(),
                hidden_grad.data_ptr(),
                w_grad.data_ptr(),
                b_grad.data_ptr(),
                legal_moves_idxs.data_ptr(),
                out_grad.data_ptr()
            )
        )

        return hidden_grad, w_grad, b_grad, None

class HiddenToLogits(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(HiddenToLogits, self).__init__()

        self.weight = torch.nn.Parameter(
            torch.rand(num_outputs, num_inputs, dtype=torch.float32) * 0.2 - 0.1
        )

        self.bias = torch.nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * 0.2 - 0.1)

    def forward(self, hidden_layer: torch.Tensor, legal_moves_idxs: torch.Tensor):
        return HiddenToLogitsFunction.apply(hidden_layer, self.weight, self.bias, legal_moves_idxs)
