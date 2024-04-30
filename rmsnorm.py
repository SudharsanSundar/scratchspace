import triton
import triton.language as tl
import torch
from typing import Any

def RMSNorm_grad_wrt_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
    H = g.shape[0]

    sum_eps = x.pow(2).mean(-1, keepdim=True) + 1e-5

    add_term = g * torch.rsqrt(sum_eps)

    mult_term = x * ((1 / H) * torch.pow(sum_eps, -1.5))
    final_mult_term = g * x * torch.sum(mult_term, dim=-1, keepdim=True)

    output_grad = add_term - final_mult_term

    final_grad = grad_output * output_grad

    print(grad_output.shape, x.shape, g.shape, sum_eps.shape)
    print(add_term.shape)
    print(output_grad.shape)
    print(final_grad.shape)

    return final_grad
