"""
nn1d —— 纯 NumPy 手写的 1D-CNN（教学用，无 torch/tensorflow/scipy 依赖）
=======================================================================
本包按"数学底座 -> 层 -> 模型"三层组织：

    nn1d.grad    : im2col/col2im、数值稳定 softmax 交叉熵、全局范数裁剪、
                   数值梯度与相对误差、以及一个仅用于测试交叉复核的迷你
                   反向模式自动微分引擎 ``Var``。
    nn1d.layers  : Conv1d / BatchNorm1d / ReLU / MaxPool1d / GlobalAvgPool1d /
                   Dropout / Linear 与 Adam 优化器，每层都写明前向与反向公式。
    nn1d.model   : CNN1D 组装、小批量训练循环、评估、混淆矩阵与分类报告。

设计红线（与 task-3 一致）：
    1) 所有含参数模块的反向传播都必须通过中心差分数值梯度校验；
    2) BatchNorm / Dropout 在 eval 下必须切到推理模式；
    3) 权重衰减只施加于 Conv/Linear 的权重，不施加于 bias 与 BN 参数。
"""

from __future__ import annotations

from . import grad, layers, model
from .grad import im2col_1d, col2im_1d, softmax, softmax_cross_entropy, numerical_gradient, relative_error
from .layers import (
    Adam,
    BatchNorm1d,
    Conv1d,
    Dropout,
    GlobalAvgPool1d,
    Linear,
    MaxPool1d,
    Module,
    ReLU,
)
from .model import (
    CNN1D,
    build_model,
    classification_report,
    confusion_matrix,
    evaluate,
    predict,
    predict_logits,
    train_model,
)

__all__ = [
    "grad", "layers", "model",
    "im2col_1d", "col2im_1d", "softmax", "softmax_cross_entropy",
    "numerical_gradient", "relative_error",
    "Module", "Conv1d", "BatchNorm1d", "ReLU", "MaxPool1d", "GlobalAvgPool1d",
    "Dropout", "Linear", "Adam",
    "CNN1D", "build_model", "train_model", "evaluate", "predict", "predict_logits",
    "confusion_matrix", "classification_report",
]

__version__ = "1.0.0"
