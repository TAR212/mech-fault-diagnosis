"""
tests/test_nn1d.py —— nn1d 手写反向传播的自证测试
==================================================
本文件是 task-3「数值梯度校验」红线的实现，含四类测试：

A. 算子级测试
   - im2col / col2im 互为转置（伴随）算子；
   - softmax 数值稳定性与交叉熵的解析梯度 vs 数值梯度；
   - MaxPool / GlobalAvgPool / Dropout 的梯度正确性；
   - Adam 在二次函数上确实下降；梯度裁剪的正确缩放。

B. **数值梯度校验（核心）**
   对每个含参数模块 Conv1d / BatchNorm1d / Linear，用中心差分
   (eps=1e-5) 逐一校验 dW/db（以及 BN 的 dgamma/dbeta）：
       相对误差 = max|g_ana - g_num| / max(max|g_ana|, max|g_num|) < 1e-4

C. 训练/推理模式测试
   - BatchNorm 训练模式下 running 统计逐渐逼近真实统计量，推理用 running；
   - Dropout 训练时置零并放大、推理时原样透传；
   - 同一批数据在 eval 下两次前向结果完全一致（无随机性）。

D. **小样本过拟合测试**
   32 个样本训练 200 步，训练准确率必须达到 100%（证明反向传播真的在学习）。

运行::

    cd mech-fault-diagnosis
    python tests/test_nn1d.py

兼容 pytest（所有用例都是无参 ``test_*`` 函数）。退出码 0 = 全绿。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np


# ---- 控制台中文：stdout 被重定向时 Windows 按 locale（GBK）编码，遇到非 GBK
#      字符会抛 UnicodeEncodeError 并以退出码 1 结束。尽力切到 UTF-8，失败静默跳过。 ----
def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass


_force_utf8_stdout()

# ---- 让脚本在任意工作目录下都能 import 到项目根目录的 config / nn1d ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as C  # noqa: E402
import nn1d  # noqa: E402
from nn1d.grad import (  # noqa: E402
    Var,
    col2im_1d,
    im2col_1d,
    numerical_gradient,
    relative_error,
    softmax,
    softmax_cross_entropy,
)
from nn1d.layers import (  # noqa: E402
    Adam,
    BatchNorm1d,
    Conv1d,
    Dropout,
    GlobalAvgPool1d,
    Linear,
    MaxPool1d,
    ReLU,
)
from nn1d.model import CNN1D, classification_report, confusion_matrix, evaluate, train_model  # noqa: E402

RTOL = 1e-4        # 相对误差阈值（task-3 规定 rtol≈1e-4）
EPS = 1e-5         # 中心差分步长


# ==========================================================================
# 工具
# ==========================================================================
def _assert_close(actual, expected, tol, msg):
    err = relative_error(np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64))
    assert err < tol, f"{msg}: 相对误差 {err:.3e} >= {tol:.1e}"
    return err


def _numerical_param_check(layer, x, layer_name, tol=RTOL, eps=EPS):
    """
    对 ``layer`` 的所有参数做中心差分数值梯度校验。

    损失定义为 L = sum(forward(x) * coef)，coef 为固定随机张量（不参与求导），
    这样每个输出元素都会对参数产生非零影响，避免"梯度恰好为 0 的假通过"。
    """
    rng = np.random.default_rng(12345)
    x = np.asarray(x, dtype=np.float64)
    layer.train(True)

    out = layer.forward(x)
    coef = rng.standard_normal(out.shape).astype(np.float64)
    layer.backward(coef)
    analytic = {k: v.copy() for k, v in layer.grads.items()}

    errs = {}
    for name in layer.params:
        shape = layer.params[name].shape

        def loss_fn(p, _name=name, _shape=shape):
            layer.params[_name][...] = p.reshape(_shape)
            return float((layer.forward(x) * coef).sum())

        p0 = layer.params[name].copy()          # 先用数值梯度扰动，再恢复原值
        numeric = numerical_gradient(loss_fn, p0, eps=eps)
        layer.params[name][...] = p0
        err = _assert_close(analytic[name], numeric, tol, f"{layer_name}.{name} 梯度校验")
        errs[name] = err
        print(f"    [OK] {layer_name}.{name:6s} shape={str(tuple(shape)):12s} "
              f"最大相对误差={err:.3e}")
    # 校验完成后把 grads 恢复成解析梯度，方便调用方继续使用
    for k, v in analytic.items():
        layer.grads[k][...] = v
    return errs


def _numerical_input_check(layer, x, layer_name, tol=RTOL, eps=EPS):
    """
    对手写层的**输入梯度 dx** 做中心差分数值校验。

    为什么必须单独测 dx：只校验 dW/db/dgamma/dbeta 的测试**无法发现**反向公式
    里"少乘一个因子"这类错误——参数梯度可能完全正确，而回传给上游的 dx 是错的
    （例如 BatchNorm 反向第三项漏掉 istd，参数梯度照样对，dx 却差 1e-2）。
    层内的错误会在深层网络里逐层放大，所以 dx 必须与参数梯度一起校验。

    损失定义为 L = sum(forward(x) * coef)，coef 固定；对 x 的每个元素做中心差分。
    """
    rng = np.random.default_rng(987)
    x = np.asarray(x, dtype=np.float64)
    layer.train(True)

    out = layer.forward(x)
    coef = rng.standard_normal(out.shape).astype(np.float64)
    dx_analytic = np.asarray(layer.backward(coef), dtype=np.float64)

    def loss_fn(xx):
        return float((layer.forward(xx.reshape(x.shape)) * coef).sum())

    dx_numeric = numerical_gradient(loss_fn, x, eps=eps)
    err = _assert_close(dx_analytic, dx_numeric, tol, f"{layer_name}.dx 输入梯度校验")
    print(f"    [OK] {layer_name}.dx     shape={str(tuple(x.shape)):12s} "
          f"最大相对误差={err:.3e}")
    return err


# ==========================================================================
# A. 算子级测试
# ==========================================================================
def test_im2col_col2im_adjoint():
    """<im2col(x), dcols> 必须等于 <x, col2im(dcols)>（矩阵与其转置的内积等价）。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 11))
    for k in (1, 2, 5, 11):
        cols, L_out = im2col_1d(x, k, stride=1)
        assert cols.shape == (2 * L_out, 3 * k), f"im2col 形状错误 {cols.shape}"
        # 逐窗口核对内容
        for n in range(2):
            for l in range(L_out):
                np.testing.assert_allclose(cols[n * L_out + l], x[n, :, l:l + k].ravel())
        dcols = rng.standard_normal(cols.shape)
        dx = col2im_1d(dcols, x.shape, k, stride=1)
        lhs = float((cols * dcols).sum())
        rhs = float((x * dx).sum())
        _assert_close(lhs, rhs, 1e-12, f"im2col/col2im 伴随性 (k={k})")
    print(f"    [OK] im2col/col2im 伴随性通过 (k=1,2,5,11)，x.shape={x.shape}")


def test_softmax_and_ce():
    """softmax 归一化 + 数值稳定 + 交叉熵解析梯度 vs 数值梯度。"""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((5, 4)) * 3.0
    p = softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(5), atol=1e-12)
    assert (p >= 0).all()
    # 极端 logits 不得溢出
    big = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    pb = softmax(big)
    assert np.isfinite(pb).all() and abs(pb[0, 0] - 1.0) < 1e-12

    y = rng.integers(0, 4, size=5)
    loss, dz = softmax_cross_entropy(z, y)
    # 手算交叉熵核对
    ref = -np.log(softmax(z)[np.arange(5), y]).mean()
    _assert_close(loss, ref, 1e-12, "交叉熵数值")

    def f(zz):
        return softmax_cross_entropy(zz, y)[0]

    num = numerical_gradient(f, z)
    _assert_close(dz, num, RTOL, "softmax 交叉熵 dlogits 梯度校验")
    print(f"    [OK] softmax 数值稳定；交叉熵 loss={loss:.6f}；dlogits 相对误差"
          f"={relative_error(dz, num):.3e}")


def test_pool_and_dropout_gradients():
    """MaxPool / GlobalAvgPool / ReLU 的解析梯度 vs 数值梯度；Dropout 的缩放与透传。"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 2, 8))
    coef = rng.standard_normal((3, 2, 4))

    pool = MaxPool1d(2, 2)
    out = pool.forward(x)
    pool.backward(coef)

    def f_pool(xx):
        return float((pool.forward(xx) * coef).sum())

    _assert_close(pool.backward(coef), numerical_gradient(f_pool, x), RTOL, "MaxPool1d 梯度校验")

    gap = GlobalAvgPool1d()
    out2 = gap.forward(x)
    coef2 = rng.standard_normal(out2.shape)

    def f_gap(xx):
        return float((gap.forward(xx) * coef2).sum())

    gap.backward(coef2)
    _assert_close(gap.backward(coef2), numerical_gradient(f_gap, x), RTOL, "GlobalAvgPool 梯度校验")

    relu = ReLU()
    coef3 = rng.standard_normal(x.shape)

    def f_relu(xx):
        return float((relu.forward(xx) * coef3).sum())

    relu.forward(x)
    _assert_close(relu.backward(coef3), numerical_gradient(f_relu, x), RTOL, "ReLU 梯度校验")

    # Dropout：训练时置零比例接近 p，且保留元素的期望不变；推理时原样输出
    dr = Dropout(0.3, rng=np.random.default_rng(3))
    dr.train(True)
    big = np.ones((200, 200))
    y = dr.forward(big)
    zero_ratio = float((y == 0).mean())
    assert abs(zero_ratio - 0.3) < 0.02, f"dropout 置零比例 {zero_ratio:.4f} 偏离 0.3"
    _assert_close(y.mean(), 1.0, 0.02, "inverted dropout 期望")
    assert np.allclose(y[y != 0], 1.0 / 0.7)
    dr.train(False)
    assert dr.forward(big) is big or np.array_equal(dr.forward(big), big), "dropout 推理必须透传"

    # Dropout 反向必须复用前向的同一掩码（重新采样会让梯度与输出不对应）
    with_prob = Dropout(0.5, rng=np.random.default_rng(5))
    with_prob.train(True)
    b = np.ones((50, 50))
    yb = with_prob.forward(b)
    db = with_prob.backward(np.ones_like(b))
    assert np.array_equal(db == 0, yb == 0), "dropout 反向必须复用前向掩码"
    assert np.allclose(db[yb != 0], 1.0 / 0.5)
    print("    [OK] MaxPool/GlobalAvgPool/ReLU 梯度通过；Dropout 置零比例"
          f"={zero_ratio:.4f}，推理透传与反向掩码复用正常")


def test_maxpool_fast_and_generic_paths_equivalent():
    """
    MaxPool1d 的 **(k==stride) 快路径** 与 **通用滑窗路径** 必须逐位等价。

    快路径把窗口恰好铺满的情形退化成 ``max(x[...,0::2], x[...,1::2])``，
    是纯 NumPy 实现能不能过单轮耗时红线的关键优化（本机 14.7ms -> 0.98ms）。
    优化不能悄悄改变语义，所以这里用参考实现（通用滑窗）对照输出、argmax
    与反向梯度，要求 **完全相等** 而不是近似相等。

    另外单独验一次 k != stride（非重叠 → 重叠）的通用分支仍然可用。
    """
    from nn1d.layers import _window_view

    class _GenericPool(MaxPool1d):
        """强制走通用滑窗路径的参考实现（与原实现逐行一致）。"""

        def forward(self, x):
            x = np.asarray(x)
            N, C, L = x.shape
            k, s = self.kernel_size, self.stride
            L_out = (L - k) // s + 1
            w = _window_view(x[:, :, :k + (L_out - 1) * s], k, s, L_out)
            idx = w.argmax(axis=-1)
            out = np.take_along_axis(w, idx[..., None], axis=-1)[..., 0]
            self._argmax = idx
            self._in_shape = x.shape
            self._L_out = L_out
            self._fast = False
            return out

    rng = np.random.default_rng(4)
    # 前三个是真实网络里的形状，后两个是边界（很小的 L、C）
    for shape in ((64, 16, 1016), (64, 32, 502), (64, 64, 247), (3, 2, 10), (2, 1, 6)):
        x = rng.standard_normal(shape)
        fast, generic = MaxPool1d(2, 2), _GenericPool(2, 2)
        o_fast, o_gen = fast.forward(x), generic.forward(x)
        assert np.array_equal(o_fast, o_gen), f"快路径前向与通用路径不一致 shape={shape}"
        d = rng.standard_normal(o_fast.shape)
        # 注意：必须各自先 forward 再用自己的缓存做 backward
        fast.forward(x); dx_fast = fast.backward(d)
        generic.forward(x); dx_gen = generic.backward(d)
        assert np.array_equal(dx_fast, dx_gen), f"快路径反向与通用路径不一致 shape={shape}"
        # 快路径的 argmax 是布尔掩码，语义上等价于 argmax 索引
        assert fast._argmax.dtype == np.bool_, "快路径应保存布尔掩码（省一次 argmax 归约）"
    print("    [OK] MaxPool1d 快路径 vs 通用路径：前向/反向逐位相等 (5 组形状)")

    # 非重叠 -> 重叠：k=3, stride=1 必须走通用分支且梯度正确
    x = rng.standard_normal((2, 2, 9))
    pool = MaxPool1d(3, 1)
    coef = rng.standard_normal(pool.forward(x).shape)

    def f_generic(xx):
        return float((pool.forward(xx) * coef).sum())

    pool.forward(x)
    _assert_close(pool.backward(coef), numerical_gradient(f_generic, x), RTOL,
                  "MaxPool1d(k=3,s=1) 通用路径梯度")
    print("    [OK] MaxPool1d(k=3,s=1) 通用路径梯度校验通过")


def test_adam_and_grad_clip():
    """Adam 在二次函数上必须单调下降；全局范数裁剪必须保持方向、缩放到阈值。"""
    rng = np.random.default_rng(5)
    target = rng.standard_normal((4, 3))
    theta = np.zeros((4, 3), dtype=np.float64)

    def get_params():
        yield ("theta", theta, grad_ref[0], False)

    grad_ref = [np.zeros_like(theta)]
    opt = Adam(get_params, lr=0.1, weight_decay=0.0)
    losses = []
    for _ in range(100):
        grad_ref[0] = 2.0 * (theta - target)
        losses.append(float(((theta - target) ** 2).sum()))
        opt.step(grad_clip=None)
    assert losses[-1] < losses[0] * 1e-3, f"Adam 未有效收敛: {losses[0]:.4f} -> {losses[-1]:.4f}"
    # Adam 带一阶动量，接近最优点时会有微小超调，因此不要求逐步单调；
    # 但任何单步的"上升"都必须远小于初始损失（否则说明动量/偏差校正有问题）。
    worst_rise = float(max(np.max(np.diff(losses)), 0.0))
    assert worst_rise < 1e-3 * losses[0], f"Adam 单步上升幅度异常: {worst_rise:.3e}"
    _assert_close(theta, target, 1e-2, "Adam 收敛到最优解")

    # 梯度裁剪
    theta2 = np.zeros((10,), dtype=np.float64)
    g2 = [np.full((10,), 3.0)]
    opt2 = Adam(lambda: [("t", theta2, g2[0], False)], lr=0.0)

    def get_g():
        return g2[0]

    before = float(np.linalg.norm(get_g()))
    # 契约：Adam.step 返回 (裁剪前全局梯度范数, 实际裁剪比例)。
    total, scale = opt2.step(lr=0.0, weight_decay=0.0, grad_clip=5.0)
    assert abs(total - before) < 1e-9, f"返回的裁剪前范数不对: {total} vs {before}"
    assert abs(float(np.linalg.norm(get_g())) - 5.0) < 1e-9, "裁剪后范数应等于阈值"
    # lr=0 时参数不动，可安全用来观察裁剪比例
    assert abs(scale - 5.0 / before) < 1e-12
    print(f"    [OK] Adam 二次函数 loss {losses[0]:.4f} -> {losses[-1]:.2e}，"
          f"梯度裁剪 {before:.4f} -> 5.0000")


# ==========================================================================
# B. 数值梯度校验（核心红线）
# ==========================================================================
def test_conv1d_numerical_gradient():
    print("  [B1] Conv1d 数值梯度校验 (中心差分 eps=1e-5, rtol=1e-4)")
    rng = np.random.default_rng(10)
    layer = Conv1d(2, 3, 4, stride=1, rng=np.random.default_rng(11), dtype=np.float64)
    layer.params["W"][...] = rng.standard_normal(layer.params["W"].shape)
    layer.params["b"][...] = rng.standard_normal(layer.params["b"].shape)
    x = rng.standard_normal((2, 2, 7))
    _numerical_param_check(layer, x, "Conv1d", tol=RTOL)
    _numerical_input_check(layer, x, "Conv1d", tol=RTOL)

    # 再验一个 stride=2 的配置，确保 col2im 的散射索引正确
    layer2 = Conv1d(1, 2, 3, stride=2, rng=np.random.default_rng(12), dtype=np.float64)
    x2 = rng.standard_normal((2, 1, 9))
    _numerical_param_check(layer2, x2, "Conv1d(stride=2)", tol=RTOL)
    _numerical_input_check(layer2, x2, "Conv1d(stride=2)", tol=RTOL)


def test_batchnorm_numerical_gradient():
    print("  [B2] BatchNorm1d 数值梯度校验 (训练模式批统计, eps=1e-5)")
    rng = np.random.default_rng(20)
    layer = BatchNorm1d(3, eps=1e-5, dtype=np.float64)
    layer.params["gamma"][...] = rng.standard_normal(3) * 0.5 + 1.0
    layer.params["beta"][...] = rng.standard_normal(3) * 0.5
    x = rng.standard_normal((4, 3, 5))
    _numerical_param_check(layer, x, "BatchNorm1d", tol=RTOL)
    _numerical_input_check(layer, x, "BatchNorm1d", tol=RTOL)

    # 再多验两个形状（含 N=1 的边界：此时 batch 与长度维退化成同一组统计）
    for shape in ((6, 3, 4), (1, 2, 8)):
        layer_x = BatchNorm1d(shape[1], eps=1e-5, dtype=np.float64)
        layer_x.params["gamma"][...] = rng.standard_normal(shape[1]) * 0.5 + 1.0
        layer_x.params["beta"][...] = rng.standard_normal(shape[1]) * 0.5
        _numerical_input_check(layer_x, rng.standard_normal(shape), f"BatchNorm1d{shape}", tol=RTOL)

    # 均匀输入 + 有偏方差公式：数值稳定性回归（var=0 时不能 NaN）
    layer2 = BatchNorm1d(2, dtype=np.float64)
    out = layer2.forward(np.ones((3, 2, 4)))
    assert np.isfinite(out).all(), "BN 在常量输入上出现 NaN/Inf"


def test_linear_numerical_gradient():
    print("  [B3] Linear 数值梯度校验")
    rng = np.random.default_rng(30)
    layer = Linear(5, 3, rng=np.random.default_rng(31), dtype=np.float64)
    layer.params["W"][...] = rng.standard_normal(layer.params["W"].shape)
    layer.params["b"][...] = rng.standard_normal(3)
    x = rng.standard_normal((6, 5))
    _numerical_param_check(layer, x, "Linear", tol=RTOL)


def test_var_autodiff_cross_check():
    """
    用独立实现的迷你自动微分引擎复核手写的 Conv1d / Linear 解析梯度。

    这一步是为了避免"自己验证自己"：``Var`` 只认识 + - * @ relu sum，
    完全不知道卷积/BN 的存在，是最简单的交叉验证基线。
    """
    print("  [B4] 迷你自动微分引擎交叉复核 Linear 前向+反向")

    def run(engine):
        rng = np.random.default_rng(40)
        x = rng.standard_normal((4, 3))
        W = rng.standard_normal((5, 3))
        b = rng.standard_normal(5)
        coef = rng.standard_normal((4, 5))
        if engine == "layer":
            layer = Linear(3, 5, dtype=np.float64)
            layer.params["W"][...] = W
            layer.params["b"][...] = b
            layer.train(False)
            out = layer.forward(x)
            layer.backward(coef)
            return layer.grads["W"].copy(), layer.grads["b"].copy()
        vx, vW, vb, vc = Var(x), Var(W), Var(b), Var(coef)
        out = vx @ vW.T + vb
        loss = (out * vc).sum()
        loss.backward()
        return vW.grad, vb.grad

    gw_layer, gb_layer = run("layer")
    gw_var, gb_var = run("var")
    _assert_close(gw_layer, gw_var, 1e-10, "Linear dW 与自动微分交叉复核")
    _assert_close(gb_layer, gb_var, 1e-10, "Linear db 与自动微分交叉复核")

    # 自动微分引擎自身也做一次数值梯度校验
    rng = np.random.default_rng(41)
    x = np.asarray(rng.standard_normal((4, 3)))
    coef = np.asarray(rng.standard_normal((4, 3)))

    def f(xx):
        return float(((Var(xx).relu() * coef).sum()).data)

    # 注意：要校验的"输入"是 x 这个叶子节点，而不是 relu 的输出节点。
    # 输出节点的梯度是 dL/dout = coef（上游梯度，未乘 relu 掩码）；只有输入节点
    # 才携带掩码后的 dL/dx。此前误取输出节点，才会出现 mask 处的假误差。
    vx = Var(x)
    v = vx.relu()
    loss = (v * coef).sum()
    loss.backward()
    _assert_close(vx.grad, numerical_gradient(f, x), 1e-10, "Var.relu 数值梯度")
    print("    [OK] 自动微分交叉复核通过（Linear dW/db 误差 < 1e-10）")


# ==========================================================================
# C. 训练/推理模式
# ==========================================================================
def test_batchnorm_train_eval_modes():
    """BN 训练模式更新滑动统计；推理模式使用滑动统计且**不更新**。"""
    rng = np.random.default_rng(50)
    bn = BatchNorm1d(2, eps=1e-5, momentum=0.1, dtype=np.float64)
    data = rng.standard_normal((8, 2, 16)) * 2.0 + 5.0
    true_mean = data.mean(axis=(0, 2))
    true_var = data.var(axis=(0, 2))

    bn.train(True)
    for _ in range(50):          # 多批累积滑动统计
        bn.forward(data)
    _assert_close(bn.running_mean, true_mean, 1e-2, "BN running_mean 应收敛到批统计")
    _assert_close(np.sqrt(bn.running_var), np.sqrt(true_var), 5e-2, "BN running_var 应收敛")

    rm, rv = bn.running_mean.copy(), bn.running_var.copy()
    bn.train(False)
    for _ in range(5):
        bn.forward(data)
    assert np.array_equal(rm, bn.running_mean) and np.array_equal(rv, bn.running_var), \
        "推理模式绝不能更新 BN 滑动统计"

    # 训练模式两次前向一致（确定性），推理模式也一致
    bn.train(False)
    o1 = bn.forward(data)
    o2 = bn.forward(data)
    assert np.array_equal(o1, o2), "eval 模式前向必须完全确定"
    print(f"    [OK] BN running_mean≈批均值(误差"
          f"{relative_error(bn.running_mean, true_mean):.3e})；推理模式不更新统计")


def test_dropout_train_eval_consistency():
    """同一模型在 eval 下多次评估必须完全一致；train 下应存在随机性。"""
    model = CNN1D(input_len=64, n_filters=(4, 4), kernel_sizes=(5, 3),
                  pool_size=2, hidden_dim=8, num_classes=2, dropout=0.5, seed=7)
    rng = np.random.default_rng(51)
    X = rng.standard_normal((16, 64)).astype(np.float32)
    y = rng.integers(0, 2, size=16)

    l1, a1 = evaluate(model, X, y, batch_size=16)
    l2, a2 = evaluate(model, X, y, batch_size=16)
    assert l1 == l2 and a1 == a2, f"eval 下两次评估必须一致: {(l1,a1)} vs {(l2,a2)}"

    model.set_training(True)
    p1 = model.forward(X[:, None, :])
    p2 = model.forward(X[:, None, :])
    assert not np.array_equal(p1, p2), "train 模式（dropout）下两次前向应不同"
    model.set_training(False)
    q1 = model.forward(X[:, None, :])
    q2 = model.forward(X[:, None, :])
    assert np.array_equal(q1, q2), "eval 模式（dropout 关闭）下两次前向必须相同"
    print("    [OK] Dropout eval 确定 / train 随机；BN+Dropout 模式切换正确")


def test_parameter_count_and_l2_policy():
    """参数量必须等于手算值 18594，且权重衰减只作用于 Conv/Linear 的 W。"""
    model = CNN1D()
    rows = model.param_summary
    print("    [参数量核算] " + " | ".join(
        f"{n} {s}" for n, s, _, _ in rows))
    expected = 18594
    assert model.trainable_params == expected, \
        f"参数量 {model.trainable_params} != 手算值 {expected}"
    decayed = [n for n, _, _, d in rows if d]
    assert len(decayed) == 5, f"应恰好有 5 个权重参与 L2（3 个 Conv.W + 2 个 Linear.W），实际 {decayed}"
    for n, _, _, d in rows:
        if n.endswith(".b") or n.endswith(".gamma") or n.endswith(".beta"):
            assert not d, f"{n} 不应施加权重衰减"
    print(f"    [OK] trainable_params={model.trainable_params}（手算 18594 一致）；"
          f"参与 L2 的参数 {len(decayed)} 个")


def test_metrics_helpers():
    """混淆矩阵与分类报告的手算一致性。"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 1], [1, 2]], f"混淆矩阵错误: {cm.tolist()}"
    assert cm.sum() == y_true.size
    acc = np.trace(cm) / cm.sum()
    assert abs(acc - 4 / 6) < 1e-12
    rep = classification_report(y_true, y_pred)
    _assert_close(rep["normal"]["precision"], 2 / 3, 1e-12, "precision")
    _assert_close(rep["normal"]["recall"], 2 / 3, 1e-12, "recall")
    _assert_close(rep["normal"]["f1"], 2 / 3, 1e-12, "f1")
    print(f"    [OK] 混淆矩阵 {cm.tolist()} 四格和={cm.sum()}，手算 acc={acc:.4f}")


# ==========================================================================
# D. 小样本过拟合测试（证明反向传播真的在工作）
# ==========================================================================
def test_overfit_32_samples():
    """
    32 个样本、200 步全批量 Adam，训练准确率必须达到 100%。

    数据为线性可分的高斯团 + 一点确定性纹理，模型用 config 的真实结构
    （通道/核长/隐藏维/学习率全部读 config），只在输入长度上缩到 256 以
    控制测试时长（网络结构完全一致，梯度来源相同）。
    """
    print("  [D1] 小样本过拟合测试：32 样本 × 200 步，要求训练准确率 = 100%")
    rng = np.random.default_rng(C.SEED)
    n, L = 32, 256
    y = np.repeat([C.NORMAL_LABEL, C.FAULT_LABEL], n // 2).astype(np.int64)
    X = rng.standard_normal((n, L)).astype(np.float32) * 0.1
    t = np.arange(L)
    # 两类的确定性纹理（周期分量幅度不同），保证有可学信号
    X[y == C.FAULT_LABEL] += 0.8 * np.sin(2 * np.pi * t / 16.0)[None, :]
    X[y == C.NORMAL_LABEL] += 0.1 * np.sin(2 * np.pi * t / 16.0)[None, :]

    model = CNN1D(input_len=L, seed=C.SEED)
    opt_lr = 3e-3
    history, info = train_model(
        model, X, y, X, y,
        epochs=200, batch_size=n, lr=opt_lr, lr_decay=1.0, lr_min=opt_lr,
        weight_decay=0.0, grad_clip=C.GRAD_CLIP, early_stop_patience=10_000,
        log_every=40, log_fn=lambda s: None,
    )
    final_acc = history["train_acc"][-1]
    params = model.trainable_params
    assert params == model.trainable_params
    print(f"    200 步后 train_acc={final_acc:.4f}，train_loss={history['train_loss'][-1]:.6f}，"
          f"耗时 {info['total_seconds']:.1f}s")
    assert final_acc == 1.0, (
        f"过拟合测试失败：200 步后训练准确率仅 {final_acc:.4f}（应当为 1.0），"
        "说明反向传播或优化器存在问题"
    )


def test_training_scale_control_semantics():
    """
    ``steps_per_epoch`` / ``time_budget`` 是**训练规模控制**，不能改变语义。

    三件事必须成立：
    1. ``steps_per_epoch=None``（完整轮）与 ``steps_per_epoch=完整步数`` 必须给出
       **逐位相同**的 history —— 证明部分轮开关没有偷偷改动默认训练的数值行为；
    2. ``steps_per_epoch=K`` 时 history 仍有 K 条/轮的真实记录，且同种子两次运行结果
       完全一致（可复现），训练集指标按**实际看过的样本数**归一；
    3. ``time_budget`` 到达后停止，且已写入 history 的每一条都是真实跑过的轮。
    """
    print("  [E1] 训练规模控制：部分轮 / 时间预算 的语义与可复现性")
    L, n = 128, 64
    rng = np.random.default_rng(C.SEED)
    y = np.repeat([C.NORMAL_LABEL, C.FAULT_LABEL], n // 2).astype(np.int64)
    X = rng.standard_normal((n, L)).astype(np.float32) * 0.2
    t = np.arange(L)
    X[y == C.FAULT_LABEL] += 0.6 * np.sin(2 * np.pi * t / 8.0)[None, :]

    def run(steps_per_epoch, epochs=3, time_budget=None, keep=("train_loss", "train_acc")):
        m = CNN1D(input_len=L, seed=C.SEED)
        h, i = train_model(
            m, X, y, X, y, epochs=epochs, batch_size=32, lr=1e-3, lr_decay=1.0, lr_min=1e-3,
            weight_decay=0.0, grad_clip=C.GRAD_CLIP, early_stop_patience=10_000,
            steps_per_epoch=steps_per_epoch, time_budget=time_budget,
            log_every=0, log_fn=lambda s: None,
        )
        return h, i

    # 1) 完整轮：None 与显式给出全部步数（64/32 = 2 步）必须完全一致
    h_none, i_none = run(None)
    h_full, i_full = run(2)
    for k in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc", "grad_norm", "lr"):
        assert h_none[k] == h_full[k], (
            f"steps_per_epoch=完整步数 与 None 的 {k} 不一致：{h_none[k]} vs {h_full[k]}，"
            "说明部分轮开关改变了默认训练的数值行为"
        )
    assert i_none["steps_per_epoch"] is None and i_full["steps_per_epoch"] == 2
    assert i_none["full_steps_per_epoch"] == 2, "完整轮步数应为 ceil(64/32)=2"
    print("    [OK] steps_per_epoch=None 与 =完整步数 的 history 逐位相同（默认行为未变）")

    # 2) 部分轮：每轮只走 1 步 = 32 个窗口，history 仍有 3 条真实记录且可复现
    h_a, i_a = run(1)
    h_b, i_b = run(1)
    assert i_a["steps_run_per_epoch"] == 1 and i_a["full_steps_per_epoch"] == 2
    assert len(h_a["epoch"]) == 3, f"部分轮也应有 3 条真实 history，实际 {len(h_a['epoch'])}"
    assert h_a["train_loss"] == h_b["train_loss"] and h_a["train_acc"] == h_b["train_acc"], \
        "同种子下部分轮训练必须完全可复现"
    # 部分轮每轮只看 32 个窗口，与完整轮看到的样本不同，指标必须不同（否则说明没截断）
    assert h_a["train_loss"] != h_none["train_loss"], "截断后指标不应与完整轮相同"
    print(f"    [OK] 部分轮 1/2 步：3 条真实记录，同种子可复现，"
          f"train_acc={h_a['train_acc'][-1]:.4f}（按实际看过样本数归一）")

    # 3) 时间预算：极小预算必须在第 1 轮后停下，history 只有真实记录
    h_t, i_t = run(None, epochs=8, time_budget=1e-9)
    assert i_t["budget_exhausted"] is True, "到达时间预算后 budget_exhausted 应为 True"
    assert i_t["epochs_run"] == 1, f"预算在第 1 轮后触发，应为 1 轮，实际 {i_t['epochs_run']}"
    assert len(h_t["epoch"]) == 1 and len(h_t["epoch_seconds"]) == 1
    assert all(s > 0 for s in h_t["epoch_seconds"]), "history 的每条耗时都应是真实测量值"
    print(f"    [OK] time_budget 触发：epochs_run={i_t['epochs_run']}，"
          f"history 每条均为真实轮（{h_t['epoch_seconds'][0]:.2f}s）")



ALL_TESTS = [
    test_im2col_col2im_adjoint,
    test_softmax_and_ce,
    test_pool_and_dropout_gradients,
    test_maxpool_fast_and_generic_paths_equivalent,
    test_adam_and_grad_clip,
    test_conv1d_numerical_gradient,
    test_batchnorm_numerical_gradient,
    test_linear_numerical_gradient,
    test_var_autodiff_cross_check,
    test_batchnorm_train_eval_modes,
    test_dropout_train_eval_consistency,
    test_parameter_count_and_l2_policy,
    test_metrics_helpers,
    test_overfit_32_samples,
    test_training_scale_control_semantics,
]


def main() -> int:
    print("=" * 78)
    print("nn1d 自证测试：数值梯度校验 + 训练/推理模式 + 小样本过拟合")
    print(f"numpy {np.__version__} | python {sys.version.split()[0]} | rtol={RTOL} eps={EPS}")
    print("=" * 78)
    failed = []
    for fn in ALL_TESTS:
        name = fn.__name__
        print(f"\n>>> {name}")
        try:
            fn()
            print(f"<<< {name} PASS")
        except Exception as exc:  # noqa: BLE001 - 测试运行器需要收集所有失败
            print(f"<<< {name} FAIL: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 78)
    if failed:
        print(f"结果: {len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} 通过；失败: {failed}")
        return 1
    print(f"结果: {len(ALL_TESTS)}/{len(ALL_TESTS)} 全部通过 √")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
