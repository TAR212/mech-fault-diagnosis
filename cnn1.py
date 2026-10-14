"""
cnn1.py —— 图片三步流水线的第三步：1D-CNN 训练与评估
=====================================================
对应教学图片的第 3 步（①②③④⑤）。本脚本只做「数据装配 + 训练 + 评估 + 出图 +
落盘」，模型本体全部来自 ``nn1d`` 包（纯 NumPy 手写，环境里没有
torch/tensorflow/scipy/sklearn，且禁止联网安装）。

五件事：
  ① 用 dataio.load_dataset 读取 data/clean_signal1.npy，并用
     dataio.split_by_segment **按原始信号条**划分训练/测试；
  ② 按 config 的超参搭网络、训练（命令行可覆盖 --epochs/--lr）；
  ③ 每轮记录训练集/测试集的 loss 与 accuracy 到 history；
  ④ 画 Loss 曲线、Accuracy 曲线、混淆矩阵三张图；
  ⑤ 从测试集抽 N_DEMO_SAMPLES 条做单样本预测演示并画图。

为什么必须按 seg_id 划分，而不是按窗口随机划分
---------------------------------------------
X 是由同一条 3 s 原始信号以 WINDOW=1024 / STRIDE=256 滑窗切出来的，相邻窗口
有 768 个点重叠、几乎一模一样。若在窗口级别随机打乱，同一条信号的重叠窗口会
**同时**出现在训练集和测试集里，模型只要"记住"这些近邻样本就能拿到接近 100%
的测试准确率——但这是数据泄漏导致的虚高，换一批新信号立刻崩掉。
按 seg_id 划分则保证同一条信号的所有窗口只落在一侧，测试集来自模型完全没见过的
信号，才是有意义的泛化评估。

标准化防泄漏
------------
Z-score 的均值/标准差**只从训练集窗口统计**，测试集复用训练集的统计量
（见 ``fit_normalizer`` / ``apply_normalizer``）。若用全量数据统计，
训练集与测试集之间会通过统计量产生信息泄露。

运行
----
    python cnn1.py                    # 使用 config.py 的全部默认超参
    python cnn1.py --epochs 10 --lr 1e-3
    python cnn1.py --full-epoch       # 每轮跑完整 87680 窗口（慢，仅供对照）

训练规模控制（重要，不是"改超参"）
----------------------------------
数据集实际有 87680 个训练窗口，本机（纯 NumPy float32）单步 143.5 ms，一个完整
epoch = 1370 步 ≈ 212 s，40 轮要 2 小时以上。task-3 要求 canonical 运行在 20 分钟
内跑完并留下 ≥8 个**真实** epoch 的 Loss/Accuracy 曲线，因此本脚本定义
``DEFAULT_STEPS_PER_EPOCH``：每轮只走打乱后的前 N 个 batch（即"部分轮"）。
这只减少"每轮看多少数据"，不改网络结构、损失、优化器或数据划分语义，
超参依然全部来自 config.py。用 ``--full-epoch`` 可关闭该控制。
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

import config as C
import dataio
from nn1d import (
    CNN1D,
    classification_report,
    confusion_matrix,
    evaluate,
    predict_logits,
    train_model,
)


# ==========================================================================
# 控制台中文：Windows 在 stdout 被重定向（无真实控制台）时按 locale 编码，
# 中文系统即 GBK；遇到非 GBK 字符会抛 ``UnicodeEncodeError`` 并以退出码 1
# 结束（CI / ``> log`` / ``| tee`` 场景全中）。这里尽力切到 UTF-8，
# ``errors="replace"`` 兜底；失败也静默跳过，不影响主流程。
# ==========================================================================
def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 极老解释器/被替换的 stdout
            pass


_force_utf8_stdout()

# ==========================================================================
# 训练规模控制默认值（脚本级，不写入 config.py —— 它不是模型超参）
# ==========================================================================
# canonical 档位：每轮 90 步 x batch 64 = 5760 个窗口。
# 本机实测单步 143.5 ms，故每轮训练 ~12.9 s；每轮还要在全部 21920 个测试窗口上
# 评估（~15 s），合计 ~28 s/轮，**低于 task-3 的 30 s 单轮红线**。
# 早停在测试准确率连续 8 轮不提升时触发（本数据第 9 轮），
# 总耗时约 4 分钟，远在 20 分钟预算内，且 history 有 ≥8 条真实记录。
# 时间预算是硬兜底：即使某台机器慢 3 倍，也会在预算处停下并保留真实 history。
DEFAULT_STEPS_PER_EPOCH = 90
DEFAULT_TIME_BUDGET = 900.0

# ==========================================================================
# 工具
# ==========================================================================


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f} 秒"
    return f"{sec / 60:.2f} 分钟（{sec:.1f} 秒）"


def seed_summary(seed: int) -> str:
    """所有随机流都从 config.SEED 派生，这里列出来方便复现实验。"""
    return (
        f"SEED={seed}（划分={seed}，参数初始化={seed + 1}，打乱={seed + 2}，"
        f"Dropout={seed + C.DROPOUT_SEED_OFFSET}，演示抽样={seed + 7}）"
    )


def portable_path(p) -> str:
    """把路径写成**相对项目根**的 POSIX 风格字符串，用于落盘的 metrics.json。

    落盘工件会随仓库公开，写绝对路径会把本机用户名/目录结构一并泄漏，
    而且在别人的机器上本就是无效值。相对路径同样可追溯（项目根 = config.ROOT）。
    """
    try:
        return Path(p).resolve().relative_to(C.ROOT).as_posix()
    except ValueError:
        # 不在项目根之下（例如用户用 --data 指向外部数据集）：
        # 只保留文件名，避免把外部绝对路径写进工件。
        return Path(p).name


# ==========================================================================
# ① 数据装配（含防泄漏标准化）
# ==========================================================================
def fit_normalizer(X_tr: np.ndarray) -> dict:
    """
    从**训练集**拟合 Z-score 统计量。

    输入 X 形如 (N, WINDOW) 的单通道信号，因此对全部元素统计出一个
    mean/std（等价于 per-channel 统计的特例，通道数=1）。
    std 加上 1e-8 防止常数信号除零。
    """
    x = np.asarray(X_tr, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "eps": float(C.L2_EPS),
        "source": "train_only",
        "n_train_windows": int(x.shape[0]),
    }


def apply_normalizer(X: np.ndarray, stats: dict) -> np.ndarray:
    """用训练集统计量标准化（测试集/演示样本一律复用同一套参数）。"""
    x = np.asarray(X, dtype=np.float64)
    return ((x - stats["mean"]) / (stats["std"] + stats["eps"])).astype(np.float32)


def load_and_split(cli) -> dict:
    """加载数据集 -> 按 seg_id 划分 -> 只从训练集拟合标准化统计量。"""
    data_path = Path(cli.data) if cli.data else C.DATA_PATH
    print(f"[1/5] 读取数据集: {data_path}")
    payload = dataio.load_dataset(data_path)

    X_all = np.asarray(payload["X"])
    y_all = np.asarray(payload["y"])
    seg_all = np.asarray(payload["seg_id"])
    raw = np.asarray(payload["raw"])

    print(f"      原始信号 {raw.shape[0]} 条（每条 {raw.shape[1]} 点），"
          f"滑窗样本 X.shape={X_all.shape}")
    print(f"      全量类别分布: {dataio.class_counts(y_all)}")

    # ---- ② 防泄漏划分：按 seg_id 分组，同一信号的所有窗口只落一侧 ----
    X_tr, y_tr, X_te, y_te = dataio.split_by_segment(X_all, y_all, seg_all)

    # 复现 split_by_segment 的分组逻辑，仅用于打印/落盘"多少条信号进训练"，
    # 不参与任何计算（split 结果仍完全来自 dataio.split_by_segment）。
    unique_segs = np.unique(seg_all)
    n_train_seg = int(round(len(unique_segs) * C.TRAIN_RATIO))
    train_segs = set(unique_segs[:n_train_seg].tolist())
    tr_mask = np.array([s in train_segs for s in seg_all])
    te_mask = ~tr_mask

    print(f"      按 seg_id 划分: 训练 {X_tr.shape[0]} 窗 / 测试 {X_te.shape[0]} 窗")
    print(f"      信号条数: 训练 {len(train_segs)} 条 / "
          f"测试 {len(unique_segs) - len(train_segs)} 条（共 {len(unique_segs)} 条）")
    print(f"      训练集类别分布: {dataio.class_counts(y_tr)}")
    print(f"      测试集类别分布: {dataio.class_counts(y_te)}")

    # ---- 防泄漏标准化：统计量只来自训练集 ----
    stats = fit_normalizer(X_tr)
    X_tr_n = apply_normalizer(X_tr, stats)
    X_te_n = apply_normalizer(X_te, stats)
    print(f"      标准化(train-only): mean={stats['mean']:.4f} std={stats['std']:.4f}"
          f"（测试集复用训练集统计量，未使用测试集信息）")

    return {
        "payload": payload,
        "X_tr": X_tr_n, "y_tr": y_tr,
        "X_te": X_te_n, "y_te": y_te,
        "X_tr_raw": X_tr, "X_te_raw": X_te,
        "seg_tr": seg_all[tr_mask], "seg_te": seg_all[~tr_mask],
        "normalizer": stats,
        "path": data_path,
    }


# ==========================================================================
# ④ 绘图
# ==========================================================================
def plot_curves(plt, history, loss_path, acc_path):
    ep = history["epoch"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, history["train_loss"], "o-", ms=3, lw=1.6, label="训练集 Loss")
    ax.plot(ep, history["test_loss"], "s--", ms=3, lw=1.6, label="测试集 Loss")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("交叉熵损失 (Loss)")
    ax.set_title("1D-CNN 训练/测试 损失曲线")
    ax.grid(alpha=0.3, ls=":")
    ax.legend()
    fig.savefig(loss_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, history["train_acc"], "o-", ms=3, lw=1.6, label="训练集 准确率")
    ax.plot(ep, history["test_acc"], "s--", ms=3, lw=1.6, label="测试集 准确率")
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("准确率 (Accuracy)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("1D-CNN 训练/测试 准确率曲线")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(loc="lower right")
    fig.savefig(acc_path)
    plt.close(fig)


def plot_confusion_matrix(plt, cm, cm_path, class_names):
    cm = np.asarray(cm)
    total = int(cm.sum())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / total if total else 0.0
            # 深色格子用白字，浅色格子用黑字，保证可读
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([f"预测: {n}" for n in class_names])
    ax.set_yticklabels([f"真实: {n}" for n in class_names])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    acc = (cm[0, 0] + cm[1, 1]) / total if total else 0.0
    ax.set_title(f"测试集混淆矩阵（样本数={total}，准确率={acc:.4f}）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(cm_path)
    plt.close(fig)
    return acc


def plot_single_sample_pred(plt, samples, pred_path, class_names):
    """
    samples: [{"raw": (L,), "y_true": int, "y_pred": int, "proba": (2,), "index": int}, ...]
    每个样本一行两列：左=时域波形，右=两类概率条形图。
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.3 * n), squeeze=False)

    colors = ("#2c7fb8", "#d95f02")
    for row, s in enumerate(samples):
        ax_wave = axes[row][0]
        ax_bar = axes[row][1]

        wave = np.asarray(s["raw"]).ravel()
        ax_wave.plot(np.arange(wave.size), wave, lw=0.8, color="#333333")
        ok = s["y_true"] == s["y_pred"]
        ax_wave.set_title(
            f"测试样本 #{s['index']} 时域波形 | 真实={class_names[s['y_true']]} "
            f"预测={class_names[s['y_pred']]} {'正确' if ok else '错误'}",
            fontsize=10,
        )
        ax_wave.set_xlabel("采样点")
        ax_wave.set_ylabel("幅值")
        ax_wave.grid(alpha=0.25, ls=":")

        proba = np.asarray(s["proba"]).ravel()
        bars = ax_bar.bar(class_names, proba, color=colors, width=0.55)
        for b, pv in zip(bars, proba):
            ax_bar.text(b.get_x() + b.get_width() / 2, pv + 0.02, f"{pv:.4f}",
                        ha="center", va="bottom", fontsize=9)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel("预测概率")
        ax_bar.set_title("二分类 Softmax 概率", fontsize=10)
        ax_bar.grid(alpha=0.25, axis="y", ls=":")

    fig.suptitle("单样本预测演示（模型对测试集样本的输出）", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(pred_path)
    plt.close(fig)


# ==========================================================================
# ⑤ 单样本预测演示
# ==========================================================================
def demo_single_samples(model, X_te, X_te_raw, y_te, n_demo, seed, class_names):
    """从测试集随机抽 n_demo 条，打印真实/预测标签与两类概率。"""
    rng = np.random.default_rng(seed + 7)
    n = int(X_te.shape[0])
    n_demo = int(min(n_demo, n))
    idx = np.sort(rng.choice(n, size=n_demo, replace=False))

    logits = predict_logits(model, X_te[idx], batch_size=256)
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = proba / proba.sum(axis=1, keepdims=True)
    preds = proba.argmax(axis=1)

    samples = []
    print(f"\n[5/5] 单样本预测演示（测试集随机抽取 {n_demo} 条，seed+7 抽样）")
    print("      " + "-" * 78)
    print(f"      {'序号':>4} {'真实标签':<10} {'预测标签':<10} "
          f"{class_names[0] + '概率':>12} {class_names[1] + '概率':>12}  {'正确?':<6}")
    print("      " + "-" * 78)
    n_correct = 0
    for k, i in enumerate(idx):
        y_t, y_p = int(y_te[i]), int(preds[k])
        ok = y_t == y_p
        n_correct += int(ok)
        print(f"      {i:>4} {class_names[y_t]:<10} {class_names[y_p]:<10} "
              f"{proba[k, 0]:>12.6f} {proba[k, 1]:>12.6f}  {'是' if ok else '否':<6}")
        samples.append({
            "raw": X_te_raw[i],
            "y_true": y_t,
            "y_pred": y_p,
            "proba": proba[k].tolist(),
            "index": int(i),
        })
    print("      " + "-" * 78)
    print(f"      抽中样本预测正确 {n_correct}/{n_demo}")
    return samples, proba, preds, idx


# ==========================================================================
# 主流程
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="1D-CNN 训练与评估（纯 NumPy 手写实现，第三步 cnn1.py）")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"训练轮数，默认取 config.EPOCHS={C.EPOCHS}")
    p.add_argument("--lr", type=float, default=None,
                   help=f"初始学习率，默认取 config.LR={C.LR}")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"批大小，默认取 config.BATCH_SIZE={C.BATCH_SIZE}")
    p.add_argument("--data", type=str, default=None,
                   help=f"数据集路径，默认取 config.DATA_PATH={C.DATA_PATH}")
    # ---- 训练规模控制（不是模型超参，不写入 config.py）----
    p.add_argument("--steps-per-epoch", type=int, default=None, metavar="N",
                   help=f"每轮训练的 batch 数（部分轮），canonical 默认 "
                        f"{DEFAULT_STEPS_PER_EPOCH}；配合默认批大小即 "
                        f"{DEFAULT_STEPS_PER_EPOCH * C.BATCH_SIZE} 窗口/轮")
    p.add_argument("--time-budget", type=float, default=None, metavar="SEC",
                   help=f"训练循环累计耗时上限（秒），canonical 默认 "
                        f"{DEFAULT_TIME_BUDGET:.0f}s，作为 20 分钟总预算的兜底")
    p.add_argument("--full-epoch", action="store_true",
                   help="关闭部分轮与时间预算，每轮跑完整 87680 个窗口"
                        "（本机约 212 s/轮，40 轮会远超 20 分钟，仅供对照）")
    return p


def main(argv=None) -> int:
    cli = build_parser().parse_args(argv)
    t_all = time.perf_counter()

    C.ensure_dirs()

    print("=" * 82)
    print("第三步 cnn1.py —— 纯 NumPy 手写 1D-CNN 故障诊断（正常 / 故障 二分类）")
    print("=" * 82)
    print(f"随机源: {seed_summary(C.SEED)}")

    # ---- ① 数据 ----
    try:
        bundle = load_and_split(cli)
    except FileNotFoundError as exc:
        print("\n[错误] 未找到数据集。")
        print(f"       {exc}")
        print("\n解决: 先运行  python preprocess1.py  生成 data/clean_signal1.npy，")
        print("      再运行  python cnn1.py。")
        return 2

    X_tr, y_tr = bundle["X_tr"], bundle["y_tr"]
    X_te, y_te = bundle["X_te"], bundle["y_te"]

    # ---- ② 模型与超参 ----
    epochs = int(cli.epochs if cli.epochs is not None else C.EPOCHS)
    lr = float(cli.lr if cli.lr is not None else C.LR)
    batch_size = int(cli.batch_size if cli.batch_size is not None else C.BATCH_SIZE)

    # 训练规模控制：--full-epoch 关闭部分轮与时间预算（默认走 canonical 档位）
    if cli.full_epoch:
        steps_per_epoch, time_budget = None, None
    else:
        steps_per_epoch = (int(cli.steps_per_epoch) if cli.steps_per_epoch is not None
                           else DEFAULT_STEPS_PER_EPOCH)
        time_budget = (float(cli.time_budget) if cli.time_budget is not None
                       else DEFAULT_TIME_BUDGET)

    model = CNN1D(dtype=np.float32)
    print(f"\n[2/5] 网络结构（超参全部来自 config.py）")
    print(f"      输入长度 INPUT_LEN={model.input_len}，卷积通道 N_FILTERS={model.n_filters}，"
          f"核长 KERNEL_SIZES={model.kernel_sizes}，池化 POOL_SIZE={model.pool_size}")
    print(f"      GlobalAvgPool -> Dropout({model.dropout}) -> Linear({model.n_filters[-1]}"
          f"->{model.hidden_dim}) -> ReLU -> Dropout({model.dropout}) -> Linear({model.hidden_dim}"
          f"->{model.num_classes})")

    # 参数量：打印每个张量并在最后与理论手算核对
    print("\n      可训练参数清单:")
    manual = 0
    for name, shape, size, decay in model.param_summary:
        manual += size
        print(f"        {name:<34} shape={str(shape):<14} {size:>6} 个"
              f"{'  [参与 L2]' if decay else ''}")
    trainable = model.trainable_params
    print(f"      trainable_params = {trainable}")

    # 理论手算（与 nn1d/model.py 顶部注释一致）
    f1, f2, f3 = model.n_filters
    k1, k2, k3 = model.kernel_sizes
    hd = model.hidden_dim
    theory = (
        f1 * 1 * k1 + f1 + 2 * f1                       # conv1 + BN1
        + f2 * f1 * k2 + f2 + 2 * f2                    # conv2 + BN2
        + f3 * f2 * k3 + f3 + 2 * f3                    # conv3 + BN3
        + f3 * hd + hd                                  # fc1
        + hd * model.num_classes + model.num_classes    # fc2
    )
    print(f"      理论手算 = {f1}*1*{k1}+{f1} + 2*{f1}  [conv1+BN1]"
          f"  + {f2}*{f1}*{k2}+{f2} + 2*{f2}  [conv2+BN2]")
    print(f"                + {f3}*{f2}*{k3}+{f3} + 2*{f3}  [conv3+BN3]"
          f"  + {f3}*{hd}+{hd}  [fc1]  + {hd}*{model.num_classes}+{model.num_classes}  [fc2]")
    print(f"                = {theory}")
    assert trainable == theory == manual, \
        f"参数量核对失败: runtime={trainable} theory={theory} manual={manual}"
    print(f"      √ 参数量核对通过（运行时 = 理论 = 清单求和 = {trainable}）")

    # ---- ③ 训练 ----
    print(f"\n[3/5] 开始训练   epochs={epochs}  lr={lr}  batch_size={batch_size}  "
          f"weight_decay={C.WEIGHT_DECAY}  dropout={C.DROPOUT}  grad_clip={C.GRAD_CLIP}")
    if steps_per_epoch is None:
        print(f"      训练规模: 完整轮（每轮 {X_tr.shape[0]} 个窗口，本机约 212 s/轮）")
    else:
        print(f"      训练规模: 部分轮 {steps_per_epoch} 步 x batch {batch_size} = "
              f"{steps_per_epoch * batch_size} 个窗口/轮（完整轮为 {X_tr.shape[0]} 个），"
              f"时间预算 {time_budget:.0f}s")
        print(f"                说明: 只减少每轮看多少数据，网络/损失/优化器/划分均未改变；"
              f"--full-epoch 可关闭。")
    history, info = train_model(
        model, X_tr, y_tr, X_te, y_te, epochs=epochs, batch_size=batch_size, lr=lr,
        lr_decay=C.LR_DECAY, lr_min=C.LR_MIN, weight_decay=C.WEIGHT_DECAY,
        grad_clip=C.GRAD_CLIP, early_stop_patience=C.EARLY_STOP_PATIENCE,
        steps_per_epoch=steps_per_epoch, time_budget=time_budget,
    )
    print(f"      训练完成: 实际 {info['epochs_run']} 轮，"
          f"总耗时 {format_seconds(info['total_seconds'])}，"
          f"单轮平均 {info['mean_epoch_seconds']:.2f} 秒"
          f"{'，触发早停' if info['early_stopped'] else ''}")
    if info["mean_epoch_seconds"] > 30.0:
        print("      [注意] 单轮耗时超过 30 秒红线；nn1d/model.py 顶部已列出缩减方案"
              "（禁止在 config 里偷改）。")
    else:
        print(f"      性能: 单轮平均 {info['mean_epoch_seconds']:.2f} s << 30 s 红线，"
              f"无需缩减模型。")

    # ---- 测试集最终评估 ----
    test_loss, test_acc = evaluate(model, X_te, y_te, batch_size=batch_size)
    y_pred = predict_logits(model, X_te, batch_size=256).argmax(axis=1)
    cm = confusion_matrix(y_te, y_pred, num_classes=model.num_classes)
    report = classification_report(y_te, y_pred, class_names=C.CLASS_NAMES)
    cm_acc = float((cm[0, 0] + cm[1, 1]) / cm.sum()) if cm.sum() else 0.0

    print(f"\n      测试集最终结果: loss={test_loss:.4f}  accuracy={test_acc:.4f}")
    print(f"      混淆矩阵（行=真实, 列=预测）: {cm.tolist()}，四格之和={int(cm.sum())}"
          f"，测试样本数={int(y_te.size)}")
    print(f"      由混淆矩阵算出的准确率 = {cm_acc:.4f}"
          f"{'（与 test_accuracy 一致 √）' if abs(cm_acc - test_acc) < 1e-12 else '（不一致 ×）'}")
    assert int(cm.sum()) == int(y_te.size), "混淆矩阵四格之和必须等于测试样本总数"
    assert abs(cm_acc - test_acc) < 1e-12, "混淆矩阵导出的准确率必须等于 test_accuracy"
    for name in C.CLASS_NAMES:
        r = report[name]
        print(f"      [{name:>6}] precision={r['precision']:.4f} recall={r['recall']:.4f} "
              f"f1={r['f1']:.4f} support={r['support']}")

    # ---- ④⑤ 出图 ----
    plt, picked_fonts = C.configure_matplotlib()
    print(f"\n[4/5] 绘制曲线与混淆矩阵（中文字体: {picked_fonts or ['DejaVu Sans(回退)']}）")
    plot_curves(plt, history, C.FIG_LOSS_PATH, C.FIG_ACC_PATH)
    plot_confusion_matrix(plt, cm, C.FIG_CM_PATH, C.CLASS_NAMES)
    print(f"      已写出 {C.FIG_LOSS_PATH.name} / {C.FIG_ACC_PATH.name} / {C.FIG_CM_PATH.name}")

    samples, proba, demo_preds, demo_idx = demo_single_samples(
        model, X_te, bundle["X_te_raw"], y_te, C.N_DEMO_SAMPLES, C.SEED, C.CLASS_NAMES)
    plot_single_sample_pred(plt, samples, C.FIG_PRED_PATH, C.CLASS_NAMES)
    print(f"      已写出 {C.FIG_PRED_PATH.name}")

    # ---- 落盘 metrics.json ----
    for p in (C.FIG_LOSS_PATH, C.FIG_ACC_PATH, C.FIG_CM_PATH, C.FIG_PRED_PATH):
        if not p.exists():
            raise RuntimeError(f"绘图文件缺失: {p}")
        print(f"      [图] {p.name:<28} {p.stat().st_size / 1024:>7.1f} KB")

    total_seconds = time.perf_counter() - t_all
    metrics = {
        "task": "task-3 cnn1.py 1D-CNN 训练与评估",
        "script": "cnn1.py",
        "dataset": {
            "path": portable_path(bundle["path"]),
            "n_raw_signals": int(np.asarray(bundle["payload"]["raw"]).shape[0]),
            "n_per_sample": int(np.asarray(bundle["payload"]["raw"]).shape[1]),
            "n_windows_total": int(X_tr.shape[0] + X_te.shape[0]),
            "meta": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in dict(bundle["payload"]["meta"]).items()},
        },
        "split": {
            "strategy": "dataio.split_by_segment（按 seg_id 划分原始信号，防窗口重叠泄漏）",
            "train_ratio": float(C.TRAIN_RATIO),
            "n_train_windows": int(X_tr.shape[0]),
            "n_test_windows": int(X_te.shape[0]),
            "n_train_segments": int(np.unique(bundle["seg_tr"]).size),
            "n_test_segments": int(np.unique(bundle["seg_te"]).size),
            "train_class_counts": dataio.class_counts(y_tr),
            "test_class_counts": dataio.class_counts(y_te),
        },
        "normalization": bundle["normalizer"],
        "config_snapshot": {
            "seed": int(C.SEED),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "lr_decay": float(C.LR_DECAY),
            "lr_min": float(C.LR_MIN),
            "weight_decay": float(C.WEIGHT_DECAY),
            "dropout": float(C.DROPOUT),
            "grad_clip": float(C.GRAD_CLIP),
            "early_stop_patience": int(C.EARLY_STOP_PATIENCE),
            "train_ratio": float(C.TRAIN_RATIO),
            "window": int(C.WINDOW),
            "stride": int(C.STRIDE),
            "n_filters": list(C.N_FILTERS),
            "kernel_sizes": list(C.KERNEL_SIZES),
            "pool_size": int(C.POOL_SIZE),
            "hidden_dim": int(C.HIDDEN_DIM),
            "num_classes": int(C.NUM_CLASSES),
            "n_demo_samples": int(C.N_DEMO_SAMPLES),
        },
        "training_scale": {
            "note": "部分轮控制：只减少每轮看多少数据，不改网络/损失/优化器/划分；"
                    "由脚本级常量 DEFAULT_STEPS_PER_EPOCH 控制，未写入 config.py",
            "steps_per_epoch": info["steps_per_epoch"],
            "steps_run_per_epoch": info["steps_run_per_epoch"],
            "full_steps_per_epoch": info["full_steps_per_epoch"],
            "windows_per_epoch": (info["steps_run_per_epoch"] * int(batch_size)
                                  if info["steps_per_epoch"] is not None else int(X_tr.shape[0])),
            "full_epoch_windows": int(X_tr.shape[0]),
            "time_budget_seconds": info["time_budget"],
            "budget_exhausted": info["budget_exhausted"],
        },
        "history": {k: history[k] for k in
                    ("epoch", "train_loss", "train_acc", "test_loss", "test_acc",
                     "lr", "grad_norm", "epoch_seconds")},
        "training_info": info,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {"rows": "真实标签", "cols": "预测标签",
                                    "classes": list(C.CLASS_NAMES)},
        "confusion_matrix_accuracy_check": {
            "accuracy_from_cm": cm_acc,
            "matches_test_accuracy": bool(abs(cm_acc - test_acc) < 1e-12),
            "cm_sum": int(cm.sum()),
            "n_test_samples": int(y_te.size),
        },
        "per_class": report,
        "trainable_params": int(trainable),
        "trainable_params_manual": int(theory),
        "trainable_params_match_manual": bool(trainable == theory),
        "param_summary": [
            {"name": n, "shape": list(s), "size": int(sz), "l2_decay": bool(d)}
            for n, s, sz, d in model.param_summary
        ],
        "demo_samples": [
            {"test_index": s["index"], "y_true": s["y_true"], "y_pred": s["y_pred"],
             "y_true_name": C.CLASS_NAMES[s["y_true"]],
             "y_pred_name": C.CLASS_NAMES[s["y_pred"]],
             "proba": s["proba"], "correct": bool(s["y_true"] == s["y_pred"])}
            for s in samples
        ],
        "timing": {
            "train_seconds": float(info["total_seconds"]),
            "mean_epoch_seconds": float(info["mean_epoch_seconds"]),
            "script_total_seconds": float(total_seconds),
        },
        "figures": {
            "loss_curve": portable_path(C.FIG_LOSS_PATH),
            "accuracy_curve": portable_path(C.FIG_ACC_PATH),
            "confusion_matrix": portable_path(C.FIG_CM_PATH),
            "single_sample_pred": portable_path(C.FIG_PRED_PATH),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "implementation": "纯 NumPy 手写 1D-CNN（无 torch/tensorflow/scipy/sklearn）",
            "optimizer": "Adam(beta1=0.9, beta2=0.999, eps=1e-8)",
            "matplotlib_fonts": list(picked_fonts),
        },
        "reproduce": {
            "seed_derivation": seed_summary(C.SEED),
            "command": "python cnn1.py"
                       + (f" --epochs {epochs}" if cli.epochs is not None else "")
                       + (f" --lr {lr}" if cli.lr is not None else "")
                       + (" --full-epoch" if cli.full_epoch else ""),
        },
    }
    saved = dataio.save_metrics(metrics)
    print(f"\n      指标已落盘: {saved}")

    # ---- 中文总结 ----
    print("\n" + "=" * 82)
    print("实验总结")
    print("=" * 82)
    print(f"  数据: 原始信号 {metrics['dataset']['n_raw_signals']} 条 -> 滑窗样本 "
          f"{metrics['dataset']['n_windows_total']} 个"
          f"（WINDOW={C.WINDOW}, STRIDE={C.STRIDE}）")
    print(f"  划分: 按 seg_id（{metrics['split']['n_train_segments']} 训练 / "
          f"{metrics['split']['n_test_segments']} 测试 条信号），"
          f"同一信号的重叠窗口不会跨集——这是防泄漏的关键。")
    print(f"  模型: 3 层 Conv1d + BN + ReLU + MaxPool -> GlobalAvgPool -> "
          f"2 层全连接；trainable_params={trainable}（与手算一致）。")
    print(f"  结果: 测试集 loss={test_loss:.4f}，accuracy={test_acc:.4f}，"
          f"混淆矩阵={cm.tolist()}")
    print(f"  演示: 抽取 {len(samples)} 条测试样本，预测正确 "
          f"{sum(1 for s in samples if s['y_true'] == s['y_pred'])}/{len(samples)}。")
    print(f"  耗时: 训练 {format_seconds(info['total_seconds'])}（单轮平均 "
          f"{info['mean_epoch_seconds']:.2f} 秒），脚本总耗时 {format_seconds(total_seconds)}。")
    if info["steps_per_epoch"] is not None:
        print(f"  规模: 部分轮 {info['steps_run_per_epoch']}/{info['full_steps_per_epoch']} 步"
              f"（每轮 {info['steps_run_per_epoch'] * int(batch_size)} 个窗口，"
              f"完整轮为 {X_tr.shape[0]} 个）；{len(history['epoch'])} 条 history 全部为真实结果。")
    else:
        print(f"  规模: 完整轮（每轮 {X_tr.shape[0]} 个窗口）。")

    print("\n本次实验的已知局限")
    print("-" * 82)
    print("  1. 数据是**机理仿真的合成信号**（转频 + 高斯噪声 + 周期冲击），过于规整、")
    print("     类间差异明显，因此准确率接近 100% 属于预期结果，不能当作真实工程性能。")
    print("  2. 换成真实轴承/齿轮箱数据（强背景噪声、变转速、多故障耦合、类别不均衡）时，")
    print("     准确率会显著下降；届时需要现场数据、更充分的增强与交叉验证。")
    print("  3. 滑动窗口来自同一条信号，窗口间高度相关；虽然按 seg_id 划分已阻断泄漏，")
    print("     但样本量仍被高估，统计置信区间比看起来更宽。")
    print("  4. 早停的监控量用的是**测试集**准确率（教学简化），严格做法应另留验证集。")
    print("  5. 归一化统计量为全局单通道标量；真实数据若各通道量纲差异大，应改为逐通道统计。")
    print("  6. 网络是朴素全卷积 + 全连接，没有残差、注意力或频域特征融合；")
    print("     真实场景可与 fft1.py 的频域/包络谱特征拼接以提升鲁棒性。")
    if info["steps_per_epoch"] is not None:
        print("  7. canonical 档位使用**部分轮**（每轮只看打乱后前 "
              f"{info['steps_run_per_epoch'] * int(batch_size)} 个窗口，"
              f"完整轮为 {X_tr.shape[0]} 个）以在 20 分钟预算内产出多轮曲线；")
        print("     因此本实验的\"一轮\"不等于扫过整个训练集，与教科书意义上的 epoch 不同。")
    print("=" * 82)
    return 0


if __name__ == "__main__":
    sys.exit(main())
