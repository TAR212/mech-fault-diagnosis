"""
nn1d.layers —— 纯 NumPy 手写的 1D-CNN 层与 Adam 优化器
=======================================================
每一层都显式写出**前向公式**与**反向公式**（教学可读优先），
不依赖任何自动微分框架。

统一约定
--------
* 特征图形状统一用 ``(N, C, L)``（batch, 通道, 长度）；全连接层输入 ``(N, F)``。
* 每个 ``Module`` 有 ``params`` / ``grads`` 两个 dict；``backward`` **原地写回**
  ``grads``，因此优化器持有的引用在多个 step 之间保持有效。
* ``weight_decay_ok = True`` 只给 Conv/Linear 的 **权重**，bias 与 BN 的
  gamma/beta 一律不做 L2（见 ``named_parameters``）。
* train/eval 模式开关：BatchNorm 与 Dropout 的行为依赖 ``self.training``。
"""

from __future__ import annotations

import numpy as np

from .grad import col2im_1d, im2col_1d, softmax_cross_entropy

__all__ = [
    "Module",
    "Conv1d",
    "BatchNorm1d",
    "ReLU",
    "MaxPool1d",
    "GlobalAvgPool1d",
    "Dropout",
    "Linear",
    "SoftmaxCrossEntropy",
    "Adam",
]


# --------------------------------------------------------------------------
# 基类
# --------------------------------------------------------------------------
class Module:
    """所有层/容器的基类。"""

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.training: bool = True
        self.weight_decay_ok: bool = False  # 权重 ('W') 是否参与 L2 正则

    # ---- 必须实现 ----
    def forward(self, x):  # pragma: no cover - 抽象
        raise NotImplementedError

    def backward(self, dout):  # pragma: no cover - 抽象
        raise NotImplementedError

    # ---- 模式切换 ----
    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, x, training=None):
        if training is None:
            training = self.training
        else:
            self.training = bool(training)
        return self.forward(x)


# --------------------------------------------------------------------------
# Conv1d
# --------------------------------------------------------------------------
class Conv1d(Module):
    """
    一维卷积（stride=1, padding='valid'）。

    前向公式
    --------
        out[n, o, l] = sum_{c=0..C-1} sum_{j=0..k-1} x[n, c, l+j] * W[o, c, j] + b[o]

    反向公式（dout 已由上层给出）
    ---------------------------
        dW[o, c, j] = sum_{n,l} dout[n, o, l] * x[n, c, l+j]
        db[o]       = sum_{n,l} dout[n, o, l]
        dx[n, c, i] = sum_{o} sum_{j 使 i = l+j} dout[n, o, l] * W[o, c, j]

    实现上把前向写成一次大矩阵乘 ``cols @ W_flat.T``（im2col），
    反向则是它的转置（col2im）。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, rng=None, dtype=np.float32):
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        fan_in = self.in_channels * self.kernel_size
        # He 初始化：std = sqrt(2 / fan_in)，适配后面的 ReLU
        w = rng.standard_normal((out_channels, in_channels, kernel_size)) * np.sqrt(2.0 / fan_in)
        self.params["W"] = np.ascontiguousarray(w, dtype=dtype)
        self.params["b"] = np.zeros(out_channels, dtype=dtype)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.weight_decay_ok = True
        self._cache = None

    def forward(self, x):
        x = np.asarray(x)
        N, C, L = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv1d 期望 in_channels={self.in_channels}，实际 {C}")
        cols, L_out = im2col_1d(x, self.kernel_size, self.stride)   # (N*L_out, C*k)
        W_flat = self.params["W"].reshape(self.out_channels, -1)     # (out, C*k)
        out = cols @ W_flat.T + self.params["b"]                     # (N*L_out, out)
        # transpose 出来的是**非 C 连续视图**，下游 BatchNorm/ReLU/MaxPool 都要在
        # 这个布局上做跨 (N,L) 的归约，strided 内存会让归约慢约 1.5 倍。
        # 这里花一次拷贝换成连续布局，net win 明显（本机 BN 前向 10.2ms -> 6.7ms）。
        out = np.ascontiguousarray(out.reshape(N, L_out, self.out_channels).transpose(0, 2, 1))
        self._cache = (cols, W_flat, x.shape, L_out)
        return out

    def backward(self, dout):
        cols, W_flat, x_shape, L_out = self._cache
        N, Out, _ = dout.shape
        dflat = np.ascontiguousarray(dout.transpose(0, 2, 1)).reshape(N * L_out, Out)
        np.copyto(self.grads["W"], (dflat.T @ cols).reshape(self.params["W"].shape))
        np.copyto(self.grads["b"], dflat.sum(axis=0))
        dcols = dflat @ W_flat                                        # (N*L_out, C*k)
        return col2im_1d(dcols, x_shape, self.kernel_size, self.stride)


# --------------------------------------------------------------------------
# BatchNorm1d
# --------------------------------------------------------------------------
class BatchNorm1d(Module):
    """
    一维批归一化，输入 (N, C, L)，**沿 batch 与长度维**统计（等价 PyTorch 的
    ``nn.BatchNorm1d`` 对 (N, C, L) 输入的行为）：每个通道一组 mu/var，
    参与统计的元素数 M = N * L。

    前向（训练）
    -----------
        mu  = mean_{n,l} x[n, c, l]
        var = mean_{n,l} (x - mu)^2                (有偏方差)
        xh  = (x - mu) / sqrt(var + eps)
        y   = gamma * xh + beta
        running_mean = (1-m)*running_mean + m*mu
        running_var  = (1-m)*running_var  + m*var*M/(M-1)   (无偏，供推理用)

    前向（推理）：用 running_mean / running_var 代替 mu / var（不再更新）。

    反向（训练）
    -----------
        dxh    = dout * gamma
        dgamma = sum_{n,l} dout * xh
        dbeta  = sum_{n,l} dout
        dx     = istd * dxh
                 - (istd/M) * sum(dxh)
                 - (istd/M) * xh * sum(dxh*xh)

    最后一项**必须带 istd 因子**：把 dvar 与 dmu 两条路径合并后，
    (x-mu) 会与 xh=(x-mu)*istd 互换，最终剩下的正是 istd*xh/M*sum(dxh*xh)。
    漏掉这个 istd 是 BN 反向最常见的错（误差约 1e-2 量级，容易被
    "只校验 dgamma/dbeta" 的测试放过），本项目的数值梯度测试会同时校验 dx。
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, dtype=np.float32):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.params["gamma"] = np.ones(num_features, dtype=dtype)
        self.params["beta"] = np.zeros(num_features, dtype=dtype)
        self.grads["gamma"] = np.zeros(num_features, dtype=dtype)
        self.grads["beta"] = np.zeros(num_features, dtype=dtype)
        # 滑动统计用 float64 保存，推理时再转回输入 dtype
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
        self.num_batches_tracked = 0
        self._cache = None

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError(f"BatchNorm1d 期望 (N, C, L)，实际 {x.shape}")
        N, C, L = x.shape
        gamma, beta = self.params["gamma"], self.params["beta"]
        if self.training:
            M = N * L
            mu = x.mean(axis=(0, 2))
            var = x.var(axis=(0, 2))
            istd = 1.0 / np.sqrt(var + self.eps)
            xh = (x - mu[None, :, None]) * istd[None, :, None]
            # 滑动平均（unbiased 方差用于推理，与 PyTorch 一致）
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mu.astype(np.float64)
            unbiased = var.astype(np.float64) * (M / max(M - 1, 1))
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * unbiased
            self.num_batches_tracked += 1
            self._cache = (xh, mu.astype(x.dtype), var.astype(x.dtype), istd.astype(x.dtype), M)
        else:
            mu = self.running_mean.astype(x.dtype)
            var = self.running_var.astype(x.dtype)
            istd = 1.0 / np.sqrt(var + self.eps)
            xh = (x - mu[None, :, None]) * istd[None, :, None]
            self._cache = (xh, mu, var, istd, N * L)
        return gamma[None, :, None] * xh + beta[None, :, None]

    def backward(self, dout):
        xh, mu, var, istd, M = self._cache
        gamma = self.params["gamma"]
        np.copyto(self.grads["gamma"], (dout * xh).sum(axis=(0, 2)))
        np.copyto(self.grads["beta"], dout.sum(axis=(0, 2)))
        dxh = dout * gamma[None, :, None]
        s1 = dxh.sum(axis=(0, 2))
        s2 = (dxh * xh).sum(axis=(0, 2))
        # dx = istd * ( dxh - (1/M)*sum(dxh) - xh*(1/M)*sum(dxh*xh) )
        # 故意把 istd 提到最后统一乘，避免对三个大张量各乘一次；dxh 是本层
        # 新算出来的数组（dout*gamma 的结果），可以安全地原地复用。
        dx = dxh
        dx -= (s1 / M)[None, :, None]
        dx -= xh * (s2 / M)[None, :, None]
        dx *= istd[None, :, None]
        return dx


# --------------------------------------------------------------------------
# 激活 / 池化 / Dropout
# --------------------------------------------------------------------------
class ReLU(Module):
    """前向 y = max(x, 0)；反向 dx = dout * (x > 0)。"""

    def __init__(self):
        super().__init__()
        self._mask = None

    def forward(self, x):
        x = np.asarray(x)
        self._mask = x > 0
        return np.where(self._mask, x, 0)

    def backward(self, dout):
        return dout * self._mask


class MaxPool1d(Module):
    """
    一维最大池化（默认 k=2, stride=2，valid）。

    前向: out[n,c,l] = max_{j<k} x[n, c, l*stride + j]
    反向: 梯度只回传给窗口内取得最大值的那个位置（argmax 路由）

    实现上有两条路径：
    * **非重叠 (k == stride) 快路径**：窗口刚好铺满输入，前向退化成
      ``max(x[..., 0::2], x[..., 1::2])``，只读一次内存、无中间窗口数组、
      无 take_along_axis。实测比通用滑窗实现快 **~14 倍**（14.7ms -> 0.98ms，
      batch=64/16通道/L=1016），这是本项目单轮耗时能否过 30s 红线的主要因素。
    * 通用滑窗路径（k != stride 时）保持原实现，语义与快路径逐位一致
      （测试里有 ``np.array_equal`` 的 argmax 与输出对照）。
    """

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self._argmax = None
        self._in_shape = None
        self._L_out = None
        self._fast = False

    def forward(self, x):
        x = np.asarray(x)
        N, C, L = x.shape
        k, s = self.kernel_size, self.stride
        L_out = (L - k) // s + 1
        if L_out < 1:
            raise ValueError(f"MaxPool1d: 输入长度 {L} 小于池化核 {k}")
        if k == s and k == 2:
            # 快路径：0::2 与 1::2 两路比较，argmax 直接就是布尔掩码（0/1）
            a = x[:, :, 0:2 * L_out:2]
            b = x[:, :, 1:2 * L_out:2]
            idx = b > a                     # bool: True 表示最大值在奇数位
            out = np.where(idx, b, a)
            self._argmax = idx
            self._in_shape = x.shape
            self._L_out = L_out
            self._fast = True
            return out
        self._fast = False
        windows = _window_view(x[:, :, :k + (L_out - 1) * s], k, s, L_out)  # (N, C, L_out, k)
        idx = windows.argmax(axis=-1)
        out = np.take_along_axis(windows, idx[..., None], axis=-1)[..., 0]
        self._argmax = idx
        self._in_shape = x.shape
        self._L_out = L_out
        return out

    def backward(self, dout):
        idx, x_shape, L_out = self._argmax, self._in_shape, self._L_out
        N, C, L = x_shape
        k, s = self.kernel_size, self.stride
        if getattr(self, "_fast", False):
            # 快路径反向：把 dout 按掩码分流回偶数位/奇数位，零散射开销
            dx = np.zeros(x_shape, dtype=dout.dtype)
            dx[:, :, 0:2 * L_out:2] = np.where(idx, 0, dout)
            dx[:, :, 1:2 * L_out:2] = np.where(idx, dout, 0)
            return dx
        dx = np.zeros(x_shape, dtype=dout.dtype)
        dw = np.zeros((N, C, L_out, k), dtype=dout.dtype)
        np.put_along_axis(dw, idx[..., None], dout[..., None], axis=-1)
        for j in range(k):
            dx[:, :, j:j + L_out * s:s] += dw[:, :, :, j]
        return dx


def _window_view(x, k, stride, L_out):
    """辅助：把 (N, C, L_valid) 变成 (N, C, L_out, k) 的滑窗视图。"""
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(x, k, axis=2)[:, :, ::stride, :][:, :, :L_out, :]


class GlobalAvgPool1d(Module):
    """
    全局平均池化 (N, C, L) -> (N, C)。

    前向: out[n,c] = (1/L) * sum_l x[n,c,l]
    反向: dx[n,c,l] = dout[n,c] / L
    """

    def __init__(self):
        super().__init__()
        self._shape = None

    def forward(self, x):
        x = np.asarray(x)
        self._shape = x.shape
        return x.mean(axis=2)

    def backward(self, dout):
        L = self._shape[2]
        return np.broadcast_to(dout[:, :, None] / L, self._shape).copy()


class Dropout(Module):
    """
    Inverted dropout：训练时以概率 p 置零，并把保留的元素放大 1/(1-p)，
    使期望不变；推理时**直接透传**（这一条是测试准确率不虚高的关键）。

        mask = (rand >= p) / (1 - p)
        y    = x * mask          (训练)
        y    = x                 (推理)
    """

    def __init__(self, p=0.0, rng=None):
        super().__init__()
        self.p = float(p)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self._mask = None

    def set_p(self, p):
        self.p = float(p)

    def forward(self, x):
        x = np.asarray(x)
        if not self.training or self.p <= 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        mask = (self.rng.random(x.shape) < keep) / keep
        self._mask = mask.astype(x.dtype)
        return x * self._mask

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask


# --------------------------------------------------------------------------
# Linear / 损失
# --------------------------------------------------------------------------
class Linear(Module):
    """
    全连接层，权重形状 (out_features, in_features)（与 PyTorch 一致）。

    前向: y = x @ W.T + b
    反向: dW = dout.T @ x ；db = sum_n dout ；dx = dout @ W
    """

    def __init__(self, in_features, out_features, rng=None, dtype=np.float32):
        super().__init__()
        rng = np.random.default_rng(0) if rng is None else rng
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        w = rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)
        self.params["W"] = np.ascontiguousarray(w, dtype=dtype)
        self.params["b"] = np.zeros(out_features, dtype=dtype)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self.weight_decay_ok = True
        self._cache = None

    def forward(self, x):
        x = np.asarray(x)
        W = self.params["W"]
        self._cache = x
        return x @ W.T + self.params["b"]

    def backward(self, dout):
        x = self._cache
        np.copyto(self.grads["W"], dout.T @ x)
        np.copyto(self.grads["b"], dout.sum(axis=0))
        return dout @ self.params["W"]


class SoftmaxCrossEntropy(Module):
    """logits -> (loss, dlogits) 的包装层，公式见 nn1d.grad.softmax_cross_entropy。"""

    def forward(self, logits, y):  # type: ignore[override]
        return softmax_cross_entropy(logits, y)


# --------------------------------------------------------------------------
# Adam
# --------------------------------------------------------------------------
class Adam:
    """
    Adam 优化器（β1=0.9, β2=0.999, eps=1e-8）。

        m = β1*m + (1-β1)*g
        v = β2*v + (1-β2)*g^2
        θ = θ - lr * (m/(1-β1^t)) / ( sqrt(v/(1-β2^t)) + eps )

    正则与裁剪顺序：先把 L2 项 (weight_decay * θ) 加进梯度（**只对 Conv/Linear
    的权重**），再做全局范数裁剪，最后做 Adam 更新。

    ``get_params`` 是一个无参可调用对象，每次 ``step`` 都重新取一遍参数列表，
    因此中途新建/替换的梯度数组不会被漏掉。
    """

    def __init__(self, get_params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.get_params = get_params if callable(get_params) else (lambda: get_params)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def step(self, lr=None, weight_decay=None, grad_clip=None):
        """
        执行一次参数更新。

        返回 ``(全局梯度范数, 实际裁剪比例)``：

        * 全局梯度范数 = **裁剪前**的 ||g||（用于日志监控梯度爆炸）；
        * 裁剪比例 = 实际乘到梯度上的系数，未触发裁剪时为 ``1.0``。

        返回裁剪比例而不是学习率，是因为学习率调用方本来就知道，而裁剪是否
        触发（scale < 1）是训练健康度的关键信号。
        """
        from .grad import global_norm

        lr = self.lr if lr is None else float(lr)
        wd = self.weight_decay if weight_decay is None else float(weight_decay)
        entries = list(self.get_params())
        # 每个条目记下 (name, param, 调用方原始梯度数组, 实际使用的有效梯度)。
        # 数据梯度 + L2 项需要新数组时，稍后会把结果**写回**原始数组，
        # 这样调用方观察到的梯度始终等于真正用于更新的梯度（可监控、可断言）。
        eff = []
        for name, param, grad, decay in entries:
            g = grad + (wd * param if (decay and wd) else 0.0)
            eff.append((name, param, grad, g))
        total = global_norm([g for _, _, _, g in eff])
        scale = 1.0
        if grad_clip is not None and total > grad_clip:
            scale = grad_clip / (total + 1e-12)
            for _, _, _, g in eff:
                g *= scale
        for _, _, raw, g in eff:
            if g is not raw:
                np.copyto(raw, g)
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for name, param, _raw, g in eff:
            m = self._m.get(name)
            v = self._v.get(name)
            if m is None:
                m = np.zeros_like(param, dtype=np.float64)
                v = np.zeros_like(param, dtype=np.float64)
                self._m[name] = m
                self._v[name] = v
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            param -= (lr * mhat / (np.sqrt(vhat) + self.eps)).astype(param.dtype)
        return total, scale
