"""
nn1d.grad —— 纯 NumPy 的基础算子、数值梯度工具与迷你反向模式自动微分
=====================================================================
本模块是 nn1d 的"数学底座"，**只依赖 numpy**（环境里没有 torch/tf/scipy，
且禁止安装）。

内容分五块：

1. 卷积展开算子（卷积层的矩阵化实现）
   - ``im2col_1d(x, k, stride)``  : 把 (N, C, L) 展成 (N*L_out, C*k) 的列矩阵，
     于是 conv1d 前向 = 一次大矩阵乘 ``cols @ W.reshape(out, C*k).T``。
     前向公式:  out[n, o, l] = sum_{c,j} x[n, c, l*stride + j] * W[o, c, j] + b[o]
   - ``col2im_1d(dcols, x_shape, k, stride)`` : 反向的散射累加
     反向公式:  dx[n, c, l] = sum_{o,j 且 l = l'*stride + j} dcols[n, l', c*k + j]
   - 二者互为**转置（伴随）算子**：<im2col(x), dcols> == <x, col2im(dcols)>。

2. 数值稳定的 softmax 与交叉熵（内部先减去每行最大值，防止 exp 溢出）
   公式:  L = mean_i( -log softmax(logits_i)[y_i] )
          dL/dlogits = (softmax(logits) - onehot(y)) / N

3. 全局梯度范数与按范数裁剪（GRAD_CLIP）

4. 中心差分数值梯度 + 相对误差
   公式:  df/dx_i ≈ ( f(x + eps*e_i) - f(x - eps*e_i) ) / (2*eps)
   ``tests/test_nn1d.py`` 的手写反向传播校验完全依赖这两个函数。

5. ``Var``：约 70 行的反向模式自动微分引擎（micrograd 风格）。
   它**不参与正式训练路径**，只作为"独立的第三方实现"，在测试里复核
   手写层的解析梯度，避免"自己验证自己"的循环论证。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "im2col_1d",
    "col2im_1d",
    "softmax",
    "softmax_cross_entropy",
    "global_norm",
    "clip_by_global_norm",
    "numerical_gradient",
    "relative_error",
    "Var",
]


# --------------------------------------------------------------------------
# 1. 卷积展开算子
# --------------------------------------------------------------------------
def im2col_1d(x, k, stride=1):
    """
    1D 卷积的列矩阵展开（无拷贝版本用 stride tricks，最后一次性物化）。

    参数
    ----
    x      : (N, C, L) float 数组
    k      : 卷积核长度
    stride : 步长（本项目恒为 1）

    返回
    ----
    cols  : (N * L_out, C * k) —— 第 (n*L_out + l) 行 = x[n, :, l*stride : l*stride+k]
    L_out : 输出长度 = (L - k) // stride + 1
    """
    x = np.asarray(x)
    if x.ndim != 3:
        raise ValueError(f"im2col_1d 期望 (N, C, L) 三维输入，实际 {x.shape}")
    N, C, L = x.shape
    if L < k:
        raise ValueError(f"输入长度 L={L} 小于卷积核 k={k}，无法做 valid 卷积")
    L_out = (L - k) // stride + 1
    # (N, C, L-k+1, k) 的**视图**，再按 stride 抽样并截到 L_out 个窗口
    windows = sliding_window_view(x, k, axis=2)[:, :, ::stride, :][:, :, :L_out, :]
    # (N, C, L_out, k) -> (N, L_out, C, k) -> (N*L_out, C*k)
    cols = np.ascontiguousarray(windows.transpose(0, 2, 1, 3)).reshape(N * L_out, C * k)
    return cols, L_out


def col2im_1d(dcols, x_shape, k, stride=1):
    """
    ``im2col_1d`` 的转置算子：把列矩阵上的梯度散射回输入空间。

    dx[n, c, l*stride + j] += dcols[n*L_out + l, c*k + j]

    对 k 做一次循环（k<=9，代价可忽略），每次对一条步长为 stride 的切片做累加。
    """
    dcols = np.asarray(dcols)
    N, C, L = x_shape
    L_out = (L - k) // stride + 1
    if dcols.shape != (N * L_out, C * k):
        raise ValueError(f"dcols 形状应为 {(N * L_out, C * k)}，实际 {dcols.shape}")
    dc = np.ascontiguousarray(dcols.reshape(N, L_out, C, k).transpose(0, 2, 1, 3))
    dx = np.zeros(x_shape, dtype=dcols.dtype)
    for j in range(k):
        dx[:, :, j:j + L_out * stride:stride] += dc[:, :, :, j]
    return dx


# --------------------------------------------------------------------------
# 2. softmax / 交叉熵
# --------------------------------------------------------------------------
def softmax(logits):
    """数值稳定 softmax：先减每行最大值再取指数（输出按行归一）。"""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """
    logits 输入的多分类交叉熵（内部做数值稳定处理）。

    loss = mean_i( logsumexp(z_i) - z_i[y_i] )，其中 z = logits - rowmax
    dloss/dlogits = (softmax(logits) - onehot(y)) / N

    返回 ``(loss: float, dlogits: ndarray)``。
    """
    logits = np.asarray(logits, dtype=np.float64) if logits.dtype == np.float64 else np.asarray(logits)
    y = np.asarray(y, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"logits 应为 (N, C)，实际 {logits.shape}")
    N = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1))
    loss = float(np.mean(logsumexp - z[np.arange(N), y]))
    p = np.exp(z - logsumexp[:, None])
    dlogits = p
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N
    return loss, dlogits


# --------------------------------------------------------------------------
# 3. 梯度范数与裁剪
# --------------------------------------------------------------------------
def global_norm(tensors):
    """所有张量梯度的全局 L2 范数 sqrt(sum_i ||t_i||^2)（用 float64 累加防溢出）。"""
    total = 0.0
    for t in tensors:
        total += float(np.sum(np.asarray(t, dtype=np.float64) ** 2))
    return float(np.sqrt(total))


def clip_by_global_norm(tensors, max_norm):
    """
    按全局范数裁剪（**原地**修改 tensors）：若 ||g|| > max_norm 则整体乘以
    max_norm / ||g||，保持梯度方向不变。返回 (scale, 裁剪前的全局范数)。
    """
    total = global_norm(tensors)
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for t in tensors:
            t *= scale
        return scale, total
    return 1.0, total


# --------------------------------------------------------------------------
# 4. 数值梯度
# --------------------------------------------------------------------------
def numerical_gradient(f, x, eps=1e-5):
    """
    中心差分数值梯度：g_i = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)。

    ``f`` 接收与 x 同形状的 float64 数组并返回标量；x 本身不被修改。
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel().copy()
    grad = np.zeros_like(flat)
    it = np.nditer(flat, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index[0]
        old = flat[i]
        flat[i] = old + eps
        fp = float(f(flat.reshape(x.shape)))
        flat[i] = old - eps
        fm = float(f(flat.reshape(x.shape)))
        flat[i] = old
        grad[i] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad.reshape(x.shape)


def relative_error(a, b):
    """
    整体相对误差 max|a-b| / max(|a|max, |b|max, 1e-8)。

    归一化取两侧**最大幅值**而非逐元素，避免某个恰好为 0 的元素把相对误差炸到 1。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-8)
    return float(np.max(np.abs(a - b)) / denom)


# --------------------------------------------------------------------------
# 5. 迷你反向模式自动微分（仅用于测试中的独立复核）
# --------------------------------------------------------------------------
def _unbroadcast(g, shape):
    """
    把上游梯度 g 沿广播出来的轴求和，还原成变量本身的形状。

    例如 bias (C,) 与 (N, C) 相加后梯度是 (N, C)，回传给 bias 前必须对
    第 0 轴求和。缺少这一步会导致 dW/db 形状错误（是广播算子里最经典的坑）。
    """
    g = np.asarray(g)
    for _ in range(g.ndim - len(shape)):
        g = g.sum(axis=0)
    axes = tuple(i for i in range(len(shape)) if shape[i] == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g


def _acc(v, g):
    """把梯度 g（先按广播还原形状）累加到变量 v 上。"""
    g = _unbroadcast(g, v.data.shape)
    v.grad = g if v.grad is None else v.grad + g


class Var:
    """
    最小可用的反向模式自动微分标量/张量（micrograd 风格）。

    用法::

        W = Var(np.random.standard_normal((3, 2)))
        x = Var(np.random.standard_normal((4, 3)))
        loss = ((x @ W).relu() * 2.0).sum()
        loss.backward()
        W.grad          # 自动求得的解析梯度

    支持的算子：``+ - * @ .T .relu() .sum() .mean()``，全部按链式法则注册
    局部反向闭包，``backward()`` 先做拓扑排序再逆序回传。
    """

    __slots__ = ("data", "grad", "_backward", "_prev")

    def __init__(self, data, _backward=None, _prev=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self._backward = _backward
        self._prev = tuple(_prev)

    # ---- 便捷属性 ----
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        out = Var(self.data.T, _prev=(self,))

        def _bw():
            _acc(self, out.grad.T)

        out._backward = _bw
        return out

    # ---- 逐元素 / 矩阵算子 ----
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data + other.data, _prev=(self, other))

        def _bw():
            _acc(self, out.grad)
            _acc(other, out.grad)

        out._backward = _bw
        return out

    __radd__ = __add__

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data - other.data, _prev=(self, other))

        def _bw():
            _acc(self, out.grad)
            _acc(other, -out.grad)

        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data * other.data, _prev=(self, other))

        def _bw():
            _acc(self, out.grad * other.data)
            _acc(other, out.grad * self.data)

        out._backward = _bw
        return out

    __rmul__ = __mul__

    def __matmul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.data @ other.data, _prev=(self, other))

        def _bw():
            _acc(self, out.grad @ other.data.T)
            _acc(other, self.data.T @ out.grad)

        out._backward = _bw
        return out

    def relu(self):
        out = Var(np.maximum(self.data, 0.0), _prev=(self,))

        def _bw():
            _acc(self, out.grad * (self.data > 0.0))

        out._backward = _bw
        return out

    def sum(self):
        out = Var(self.data.sum(), _prev=(self,))

        def _bw():
            _acc(self, np.broadcast_to(out.grad, self.data.shape).copy())

        out._backward = _bw
        return out

    def mean(self):
        out = Var(self.data.mean(), _prev=(self,))

        def _bw():
            _acc(self, np.broadcast_to(out.grad, self.data.shape).copy() / self.data.size)

        out._backward = _bw
        return out

    # ---- 反向传播 ----
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    def __repr__(self):  # pragma: no cover - 仅调试用
        return f"Var(shape={self.data.shape})"
