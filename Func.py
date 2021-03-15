from torch import Tensor, Generator, strided, memory_format, contiguous_format
from typing import List, Tuple, Optional, Union, Any, ContextManager, Callable, overload, Iterator, NamedTuple, \
    Sequence, TypeVar
from torch._six import inf
import torch

from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout

import builtins
import time


# def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[_int, _size] = 1,
#            padding: Union[_int, _size] = 0, dilation: Union[_int, _size] = 1, groups: _int = 1) -> Tensor: ...


def _zero_padding2d(x: Tensor, padding: int) -> Tensor:
    """零填充(F.pad()) - 已重写简化
    :param x: shape = (N, Cin, Hin, Win)
    :param padding: int
    :return: shape = (N, Cin, Hout, Wout)"""

    output = torch.zeros((*x.shape[:2],  # N, Cin
                          x.shape[-2] + 2 * padding,  # Hout
                          x.shape[-1] + 2 * padding), dtype=x.dtype, device=x.device)  # Wout
    h_out, w_out = output.shape[-2:]
    output[:, :, padding:h_out - padding, padding:w_out - padding] = x
    return output


def _conv2d(x: Tensor, weight: Tensor, bias: Tensor = None,
            stride: int = 1, padding: int = 0,
            dilation: int = 1, groups: int = 1) -> Tensor:
    """2d卷积(F.conv2d()) - 已重写简化
    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (Cout, Cin, KH, KW)
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :return: shape = (N, Cout, Hout, Wout)
    """
    if padding:
        x = _zero_padding2d(x, padding)

    kernel_size = weight.shape[-2:]

    # Out(H, W) = (In(H, W) + 2 * padding − kernel_size) // stride + 1
    output_h, output_w = (x.shape[2] - kernel_size[0]) // stride + 1, (x.shape[3] - kernel_size[1]) // stride + 1
    output = torch.empty((x.shape[0], weight.shape[0], output_h, output_w), dtype=x.dtype, device=x.device)

    slice_time = 0
    dot_time = 0
    sum_time = 0

    for i in range(output.shape[2]):  # Hout
        for j in range(output.shape[3]):  # # Wout
            h_start, w_start = i * stride, j * stride
            start_time = time.time()
            h_pos, w_pos = slice(h_start, (h_start + kernel_size[0])), slice(w_start, (w_start + kernel_size[1]))
            end_time = time.time()
            slice_time += (end_time - start_time)

            # N, Cout, Cin, KH, KW
            start_time = time.time()
            out = x[:, None, :, h_pos, w_pos] * weight[None, :, :, :, :]
            end_time = time.time()
            dot_time += (end_time - start_time)

            start_time = time.time()
            output[:, :, i, j] = torch.sum(out, dim=(-3, -2, -1)) + (bias if bias is not None else 0)
            end_time = time.time()
            sum_time += (end_time - start_time)

    return output, slice_time, dot_time, sum_time
