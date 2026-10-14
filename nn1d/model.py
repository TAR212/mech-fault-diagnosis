"""
nn1d.model —— 1D-CNN 组装、训练循环与评估
==========================================
把 nn1d.layers 里的手写层拼成本项目的经典 1D-CNN，并实现完整的
「前向 -> 损失 -> 反向 -> 裁剪 -> Adam 更新」训练循环。

网络结构（通道数/核长/隐藏维全部从 config 读取，不硬编码）
----------------------------------------------------------
    Conv1d(1  -> 16, k=9) -> BN -> ReLU -> MaxPool1d(2)      L: 1024 -> 1016 -> 508
    Conv1d(16 -> 32, k=7) -> BN -> ReLU -> MaxPool1d(2)      L: 508  -> 502  -> 251
    Conv1d(32 -> 64, k=5) -> BN -> ReLU -> MaxPool1d(2)      L: 251  -> 247  -> 123
    GlobalAvgPool1d (123 -> 1)   ->  (N, 64)
    Dropout(0.3) -> Linear(64 -> HIDDEN_DIM) -> ReLU -> Dropout(0.3) -> Linear(HIDDEN_DIM -> 2)

参数量手算（config 默认值 16/32/64, k=9/7/5, HIDDEN_DIM=64，共 18594）
    第 1 层卷积: W 16*1*9  = 144  , b 16      -> 160  ；BN(16): 2*16 = 32   ；小计 192
    第 2 层卷积: W 32*16*7 = 3584 , b 32      -> 3616 ；BN(32): 2*32 = 64   ；小计 3680
    第 3 层卷积: W 64*32*5 = 10240, b 64      -> 10304；BN(64): 2*64 = 128  ；小计 10432
    fc1       : W 64*64    = 4096 , b 64      -> 4160
    fc2       : W 2*64     = 128  , b 2       -> 130
    合计 = 192 + 3680 + 10432 + 4160 + 130 = 18594
    （``CNN1D.trainable_params`` 运行时算出的值必须等于 18594，否则说明装配错了。）

性能说明（单 epoch 耗时，**实测值，非估计**）
------------------------------------------------
数据集实际规模：800 条原始信号（每条 36000 点）按 WINDOW=1024 / STRIDE=256 滑窗后
共 **109600** 个样本，按 seg_id 划分后 **训练 87680 窗 / 测试 21920 窗**（不是早期
注释里误写的 6400 窗）。batch=64 时一个 epoch = 1370 步。

本机（CPU、float32、纯 NumPy、scipy-openblas 0.3.30 多线程）实测：
    优化前：单步 187–210 ms，单 epoch（含测试集评估）**303.5 s**
    优化后：单步 143.5 ms（median）/ 124.7 ms（min），单 epoch 投影约 260 s
优化内容见 layers.py：MaxPool1d 的 k==stride 快路径（14.7 → 1.0 ms）、
Conv1d.forward 返回 C 连续数组（下游 BatchNorm 前向 10.2 → 6.7 ms）、
BatchNorm1d 反向原地复用。**优化后单轮仍约 260 s，远高于 30 s 红线**
（瓶颈是大张量的逐元素归约，不是 BLAS；conv 层合计只占前向 ~21/100 ms）。

因此本训练的"一个 epoch"允许按 ``steps_per_epoch`` 截断为**部分轮**（shuffle 后取前
N 个 batch；用 ``--steps-per-epoch`` 控制，cnn1.py canonical 档位默认开启并会打印）。
这是显式的、可复现的训练规模控制，不改变任何模型结构或损失定义。若还需要进一步缩减，
按以下顺序，且**必须先在 config.py 里改并同步 PLAN.md**，不允许在本文件里偷偷改：
    1) 把第三层卷积通道 64 -> 32（省 ~45% 时间）；
    2) 去掉一层 MaxPool 并在 GlobalAvgPool 前加一次 stride=2 的池化；
    3) 把每轮评估改成「每 2 轮评估一次」（只改评估频率，不改模型）。
"""

from __future__ import annotations

import time

import numpy as np

import config as C

from .grad import global_norm, softmax_cross_entropy
from .layers import (
    Adam,
    BatchNorm1d,
    Conv1d,
    Dropout,
    GlobalAvgPool1d,
    Linear,
    MaxPool1d,
    ReLU,
)

__all__ = ["CNN1D", "build_model", "train_model", "evaluate", "predict"]


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class CNN1D:
    """
    3 层一维卷积 + 全局平均池化 + 2 层全连接的二分类网络。

    所有层按**前向顺序**放在一个列表里，反向时逆序遍历；训练/推理模式通过
    ``set_training(flag)`` 统一切换（BN 用滑动统计、Dropout 透传）。
    """

    def __init__(self, input_len=None, n_filters=None, kernel_sizes=None,
                 pool_size=None, hidden_dim=None, num_classes=None,
                 dropout=None, seed=None, dtype=np.float32):
        self.input_len = int(input_len if input_len is not None else C.INPUT_LEN)
        self.n_filters = tuple(n_filters if n_filters is not None else C.N_FILTERS)
        self.kernel_sizes = tuple(kernel_sizes if kernel_sizes is not None else C.KERNEL_SIZES)
        self.pool_size = int(pool_size if pool_size is not None else C.POOL_SIZE)
        self.hidden_dim = int(hidden_dim if hidden_dim is not None else C.HIDDEN_DIM)
        self.num_classes = int(num_classes if num_classes is not None else C.NUM_CLASSES)
        self.dropout = float(dropout if dropout is not None else C.DROPOUT)
        self.seed = int(seed if seed is not None else C.SEED)
        self.dtype = dtype

        # ---- 随机源：全部从 config.SEED 派生（禁止 np.random 之外的随机源）----
        rng_init = np.random.default_rng(self.seed + 1)
        # dropout 随机流与参数初始化/打乱流解耦，保证重复实验可复现
        self.rng_drop = np.random.default_rng(self.seed + int(getattr(C, "DROPOUT_SEED_OFFSET", 1000)))
        self.rng_shuffle = np.random.default_rng(self.seed + 2)

        self.layers = []
        in_ch = 1
        for out_ch, k in zip(self.n_filters, self.kernel_sizes):
            self.layers.append(Conv1d(in_ch, out_ch, k, stride=1, rng=rng_init, dtype=dtype))
            self.layers.append(BatchNorm1d(out_ch, eps=1e-5, dtype=dtype))
            self.layers.append(ReLU())
            self.layers.append(MaxPool1d(self.pool_size, self.pool_size))
            in_ch = out_ch
        self.layers.append(GlobalAvgPool1d())
        self.layers.append(Dropout(self.dropout, rng=self.rng_drop))
        self.layers.append(Linear(in_ch, self.hidden_dim, rng=rng_init, dtype=dtype))
        self.layers.append(ReLU())
        self.layers.append(Dropout(self.dropout, rng=self.rng_drop))
        self.layers.append(Linear(self.hidden_dim, self.num_classes, rng=rng_init, dtype=dtype))

        self._check_shapes()
        self.set_training(True)

    # ---- 结构自检 ----
    def _check_shapes(self):
        """用一条假数据走一遍前向，尽早暴露通道/长度装配错误。"""
        probe = np.zeros((2, 1, self.input_len), dtype=self.dtype)
        self.set_training(False)
        out = self.forward(probe)
        if out.shape != (2, self.num_classes):
            raise ValueError(f"模型输出形状异常: {out.shape}，期望 {(2, self.num_classes)}")
        self.set_training(True)

    # ---- 模式 ----
    def set_training(self, flag: bool):
        for layer in self.layers:
            layer.train(bool(flag))
        return self

    @property
    def training(self):
        return self.layers[0].training

    # ---- 参数 / 梯度 ----
    def named_parameters(self):
        """
        遍历所有 (名字, 参数, 梯度, 是否施加权重衰减)。

        权重衰减**只**施加在 Conv/Linear 的权重 'W' 上；bias 与 BN 的
        gamma/beta 一律 decay=False（这是常见的错误来源）。
        """
        for i, layer in enumerate(self.layers):
            for pname in ("W", "b", "gamma", "beta"):
                if pname in layer.params:
                    decay = bool(layer.weight_decay_ok and pname == "W")
                    yield f"layer{i}.{type(layer).__name__}.{pname}", layer.params[pname], layer.grads[pname], decay

    @property
    def trainable_params(self) -> int:
        return int(sum(p.size for _, p, _, _ in self.named_parameters()))

    @property
    def param_summary(self) -> list:
        rows = []
        for name, p, _, decay in self.named_parameters():
            rows.append((name, tuple(p.shape), int(p.size), bool(decay)))
        return rows

    def zero_grad(self):
        for _, _, g, _ in self.named_parameters():
            g[...] = 0.0

    # ---- 前向 / 反向 ----
    def forward(self, x):
        """x: (N, L) 或 (N, 1, L) -> logits (N, num_classes)。"""
        x = np.asarray(x)
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3:
            raise ValueError(f"输入应为 (N, L) 或 (N, 1, L)，实际 {x.shape}")
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, dlogits):
        """按前向的逆序回传，各层的 backward 会原地写回 layer.grads。"""
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def loss_and_grad(self, x, y):
        """
        一次完整的前向+反向。注意 y **不参与** forward（损失只在最后算一次），
        这是唯一用到标签的地方。
        """
        logits = self.forward(x)
        loss, dlogits = softmax_cross_entropy(logits, y)
        self.backward(dlogits)
        return loss, logits


# --------------------------------------------------------------------------
# 便捷构造 / 评估
# --------------------------------------------------------------------------
def build_model(**kwargs) -> CNN1D:
    return CNN1D(**kwargs)


def predict_logits(model, X, batch_size=256):
    """分批推理（eval 模式，不更新 BN 滑动统计、不做 dropout）。"""
    model.set_training(False)
    outs = []
    X = np.asarray(X)
    for s in range(0, X.shape[0], batch_size):
        chunk = X[s:s + batch_size][:, None, :]
        outs.append(model.forward(chunk))
    model.set_training(True)
    return np.concatenate(outs, axis=0)


def predict(model, X, batch_size=256):
    """返回预测类别（argmax）。"""
    return predict_logits(model, X, batch_size=batch_size).argmax(axis=1)


def evaluate(model, X, y, batch_size=256):
    """在给定集合上计算 loss / accuracy（eval 模式）。"""
    from .grad import softmax_cross_entropy as _sceloss

    model.set_training(False)
    X = np.asarray(X)
    total_loss, correct = 0.0, 0
    for s in range(0, X.shape[0], batch_size):
        xb = X[s:s + batch_size][:, None, :]
        yb = np.asarray(y)[s:s + batch_size]
        logits = model.forward(xb)
        l, _ = _sceloss(logits, yb)
        total_loss += l * yb.size
        correct += int((logits.argmax(axis=1) == yb).sum())
    model.set_training(True)
    n = int(X.shape[0])
    return total_loss / n, correct / n


def confusion_matrix(y_true, y_pred, num_classes=None):
    """混淆矩阵，行=真实类别，列=预测类别（与 sklearn.metrics.confusion_matrix 一致）。"""
    num_classes = int(num_classes if num_classes is not None else C.NUM_CLASSES)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------
# 训练循环
# --------------------------------------------------------------------------
def train_model(model, X_tr, y_tr, X_te, y_te, *,
                epochs=None, batch_size=None, lr=None, lr_decay=None, lr_min=None,
                weight_decay=None, grad_clip=None, early_stop_patience=None,
                seed=None, log_every=1, log_fn=print,
                steps_per_epoch=None, time_budget=None):
    """
    小批量梯度下降训练循环。

    每轮流程
    --------
    1. 用 ``rng_shuffle`` 打乱训练集索引（随机源来自 config.SEED 派生）；
    2. 逐 batch：前向 -> 交叉熵 -> 反向 -> L2 正则 -> 全局范数裁剪 -> Adam 更新；
    3. 在本轮**结束后的参数**上评估测试集的 loss/accuracy，写入 history；
    4. 学习率按 ``LR_DECAY`` 每轮衰减，不低于 ``LR_MIN``；
    5. 以测试集准确率作为早停监控量（教学简化：没有独立验证集，见"已知局限"）。

    ``steps_per_epoch``（**训练规模控制，不是模型改动**）
    ----------------------------------------------------
    数据集实际有 87680 个训练窗口，batch=64 时一个完整 epoch 是 1370 步、本机约
    200 s（见文件顶部性能说明）。本参数把一轮限制为「打乱后取前 N 个 batch」，
    即**部分轮**，让 canonical 运行能在时间预算内产出 ≥8 个真实 epoch 的曲线。
    ``None`` = 不做截断（完整一轮，等价于原行为，语义不变）。
    截断只影响"这一轮看了多少数据"，不改网络结构、损失函数或优化器。
    训练集的 loss/accuracy 按**本轮实际看过的样本数**归一，不含没走到的样本。

    ``time_budget``（秒，安全阀）
    ---------------------------
    每轮结束后检查累计耗时，超预算就停止。默认 ``None`` = 不限。
    这只是兜底：本项目数据上早停（测试准确率连续 EARLY_STOP_PATIENCE 轮不提升）
    通常先生效。触发时 ``info['budget_exhausted']`` 为 True，history 里已有的
    每条记录都是真实跑过的结果。

    返回 history 字典（含逐轮 loss/acc/lr/梯度范数）、耗时、best 信息。
    """
    epochs = int(epochs if epochs is not None else C.EPOCHS)
    batch_size = int(batch_size if batch_size is not None else C.BATCH_SIZE)
    steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
    time_budget = float(time_budget) if time_budget else None

    lr = float(lr if lr is not None else C.LR)
    lr_decay = float(lr_decay if lr_decay is not None else C.LR_DECAY)
    lr_min = float(lr_min if lr_min is not None else C.LR_MIN)
    weight_decay = float(weight_decay if weight_decay is not None else C.WEIGHT_DECAY)
    grad_clip = float(grad_clip if grad_clip is not None else C.GRAD_CLIP)
    patience = int(early_stop_patience if early_stop_patience is not None
                   else getattr(C, "EARLY_STOP_PATIENCE", epochs))

    X_tr = np.asarray(X_tr, dtype=model.dtype)
    X_te = np.asarray(X_te, dtype=model.dtype)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_te = np.asarray(y_te, dtype=np.int64)

    optimizer = Adam(model.named_parameters, lr=lr, beta1=0.9, beta2=0.999,
                     eps=1e-8, weight_decay=weight_decay)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [], "lr": [], "grad_norm": [], "epoch_seconds": [],
    }
    n = int(X_tr.shape[0])
    # 一轮实际走多少步：默认全部（ceil(n/batch_size)），steps_per_epoch 则截断
    total_steps = (n + batch_size - 1) // batch_size
    steps_this_epoch = total_steps if steps_per_epoch is None else min(steps_per_epoch, total_steps)
    best = {"epoch": -1, "test_acc": -1.0, "stale": 0}
    budget_exhausted = False
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.set_training(True)
        perm = model.rng_shuffle.permutation(n)
        run_loss, run_correct, norm_sum, n_batches, seen = 0.0, 0, 0.0, 0, 0
        for si in range(steps_this_epoch):
            idx = perm[si * batch_size:(si + 1) * batch_size]
            xb = X_tr[idx][:, None, :]
            yb = y_tr[idx]
            model.zero_grad()
            loss, logits = model.loss_and_grad(xb, yb)
            gnorm, _ = optimizer.step(lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)
            run_loss += loss * yb.size
            run_correct += int((logits.argmax(axis=1) == yb).sum())
            norm_sum += gnorm
            n_batches += 1
            seen += yb.size

        # 按**本轮实际看过的样本数**归一（截断时不把没走到的样本算进分母）
        train_loss = run_loss / max(seen, 1)
        train_acc = run_correct / max(seen, 1)
        test_loss, test_acc = evaluate(model, X_te, y_te, batch_size=batch_size)
        t_used = time.perf_counter() - t_ep

        history["epoch"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["test_loss"].append(float(test_loss))
        history["test_acc"].append(float(test_acc))
        history["lr"].append(float(lr))
        history["grad_norm"].append(float(norm_sum / max(n_batches, 1)))
        history["epoch_seconds"].append(float(t_used))

        if test_acc > best["test_acc"] + 1e-12:
            best = {"epoch": ep, "test_acc": float(test_acc), "stale": 0}
        else:
            best["stale"] += 1

        if log_every and (ep % log_every == 0 or ep == 1 or ep == epochs or best["stale"] == 0):
            log_fn(f"  epoch {ep:3d}/{epochs} | lr {lr:.2e} | "
                   f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                   f"test_loss {test_loss:.4f} acc {test_acc:.4f} | "
                   f"|g| {history['grad_norm'][-1]:.3f} | {t_used:.2f}s")
        if steps_per_epoch is not None and ep == 1:
            log_fn(f"  [训练规模] 每轮 {n_batches}/{total_steps} 步"
                   f"（{seen}/{n} 窗口，部分轮）：steps_per_epoch={steps_per_epoch}")

        lr = max(lr * lr_decay, lr_min)
        if best["stale"] >= patience:
            log_fn(f"  [早停] 测试准确率连续 {patience} 轮未提升，在第 {ep} 轮停止。")
            break
        # 时间预算安全阀：默认不限；触发时已记录的每轮都是真实结果
        if time_budget is not None and (time.perf_counter() - t_start) >= time_budget:
            budget_exhausted = True
            log_fn(f"  [时间预算] 累计耗时已达 {time_budget:.0f}s 上限，"
                   f"在第 {ep} 轮后停止（history 已含 {ep} 条真实记录）。")
            break

    total_seconds = time.perf_counter() - t_start
    info = {
        "epochs_run": len(history["epoch"]),
        "total_seconds": float(total_seconds),
        "best_epoch": int(best["epoch"]),
        "best_test_acc": float(best["test_acc"]),
        "mean_epoch_seconds": float(np.mean(history["epoch_seconds"])) if history["epoch_seconds"] else 0.0,
        "early_stopped": bool(best["stale"] >= patience),
        "optimizer": "Adam(beta1=0.9, beta2=0.999, eps=1e-8)",
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "steps_per_epoch": steps_per_epoch,
        "steps_run_per_epoch": int(steps_this_epoch),
        "full_steps_per_epoch": int(total_steps),
        "time_budget": time_budget,
        "budget_exhausted": bool(budget_exhausted),
    }
    return history, info


# --------------------------------------------------------------------------
# 指标工具
# --------------------------------------------------------------------------
def classification_report(y_true, y_pred, class_names=None):
    """手写 per-class precision / recall / F1（环境里没有 sklearn）。"""
    class_names = list(class_names if class_names is not None else C.CLASS_NAMES)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    report = {}
    for c, name in enumerate(class_names):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((y_true == c).sum()),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return report
