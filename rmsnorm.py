import triton
import triton.language as tl
import torch
from typing import Any

def RMSNorm_grad_wrt_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
    H = g.shape[0]

    sum_eps = x.pow(2).mean(-1, keepdim=True) + 1e-5

    add_term = g * torch.rsqrt(sum_eps)

    mult_term = x * (1 / H) * torch.pow(sum_eps, -1.5)
    final_mult_term = g * x * torch.sum(mult_term, dim=-1, keepdim=True)

    output_grad = add_term - final_mult_term

    final_grad = grad_output * output_grad

    print(grad_output.shape, x.shape, g.shape, sum_eps.shape)
    print(add_term.shape)
    print(output_grad.shape)
    print(final_grad.shape)
    
    return final_grad

    # > pytest -k test_rmsnorm_backward_x_pytorch
    #
    # False = <built-in method allclose of type object at 0x14f7c1471840>(
    # tensor([[[-6.4320e-03,  1.4323e-02,  5.9898e-02,  ...,  2.9904e-04,\n           6.8773e-02, -3.2441e-02],\n         [-1....1.3778e-01, -5.8441e-02, -3.1317e-01,  ..., -2.2864e-03,\n          -7.0831e-02, -1.5222e-01]]], grad_fn=<MulBackward0>),
    # tensor([[[-3.8991e-02,  3.7541e-02,  5.9241e-02,  ...,  6.7614e-02,\n           4.9535e-02, -8.1552e-02],\n         [-1....1.8454e-01],\n         [-1.2955e-01, -6.2788e-02, -3.1281e-01,  ..., -5.3356e-03,\n          -7.2525e-02, -1.5264e-01]]])
    # , rtol=0.0001, atol=1e-05)
    #
    # [STDOUT:
    # torch.Size([10, 20, 50]) torch.Size([10, 20, 50]) torch.Size([50]) torch.Size([10, 20, 1])
    # torch.Size([10, 20, 50])
    # torch.Size([10, 20, 50])
    # torch.Size([10, 20, 50])
    # ]
