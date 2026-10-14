"""
全局配置 —— 全流水线单一事实来源 (Single Source of Truth)
=========================================================
preprocess1.py / fft1.py / cnn1.py 全部从这里读参数，禁止各自硬编码。

对应图片中的三步流水线:
    preprocess1.py -> fft1.py -> cnn1.py

冻结接口说明
------------
本文件与 ``dataio.py`` 是"契约文件"，由架构负责人冻结，子任务不得修改；
如需变更，必须同步更新 PLAN.md 的接口章节。

命名兼容层: 历史/教学材料里出现过 config / config_bp / bp 三种叫法，
本文件在 sys.modules 里注册了别名，因此::

    import config          # 推荐
    import config as bp    # 可用
    from config_bp import *  # 可用

三种写法都指向同一个模块对象、同一份参数值（不存在两份配置漂移的问题）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 0. 路径
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

FUNC_F = 50.0          # 转频 (Hz) —— 电机/轴旋转基频
FS = 12000.0           # 采样频率 (Hz)
T_PER_SAMPLE = 3.0     # 单次采样时长 (s)
N_PER_SAMPLE = int(FS * T_PER_SAMPLE)   # 36000 点 = 3 s
WINDOW = 1024          # 滑动窗口长度 (点) —— 与 1D-CNN 输入长度一致
STRIDE = 256           # 滑动步长 (点) —— 相邻片段重叠，增强样本量
N_SAMPLES_PER_CLASS = 400   # 每类信号条数（正常 400 条 / 故障 400 条）

DATASET_NAME = "clean_signal1.npy"      # 图片指定的数据集文件名

# 目录布局（所有脚本一律用这些常量拼路径，禁止手写相对路径）
DIR_DATA = ROOT / "data"
DIR_FIG = ROOT / "figures"
DIR_RESULT = ROOT / "results"
DIR_DATA_RAW = DIR_DATA / "raw"

DATA_PATH = DIR_DATA / DATASET_NAME     # data/clean_signal1.npy
FIG_TIME_PATH = DIR_FIG / "fig_time_domain.png"
FIG_FREQ_PATH = DIR_FIG / "fig_spectrum.png"
FIG_ENV_PATH = DIR_FIG / "fig_envelope_spectrum.png"
FIG_LOSS_PATH = DIR_FIG / "fig_loss_curve.png"
FIG_ACC_PATH = DIR_FIG / "fig_accuracy_curve.png"
FIG_CM_PATH = DIR_FIG / "fig_confusion_matrix.png"
FIG_PRED_PATH = DIR_FIG / "fig_single_sample_pred.png"
FIG_CACHE_PATH = DIR_FIG / "fig_preprocess_cache.png"
RESULT_JSON = DIR_RESULT / "metrics.json"

# --------------------------------------------------------------------------
# 1. 故障机理参数（第二步 fft1.py 依赖，必须与第一步 preprocess1.py 完全一致）
# --------------------------------------------------------------------------
BEARING = dict(
    n_balls=8,          # 滚动体个数
    d_ball=7.94e-3,     # 滚动体直径 (m)
    D_pitch=38.5e-3,    # 节圆直径 (m)
    contact_deg=0.0,    # 接触角 (度)
)

BPFO = (BEARING["n_balls"] / 2.0) * (1.0 - BEARING["d_ball"] / BEARING["D_pitch"]) * FUNC_F
BPFI = (BEARING["n_balls"] / 2.0) * (1.0 + BEARING["d_ball"] / BEARING["D_pitch"]) * FUNC_F
BSF = (BEARING["D_pitch"] / (2.0 * BEARING["d_ball"])) * (
    1.0 - (BEARING["d_ball"] / BEARING["D_pitch"]) ** 2
) * FUNC_F
FTF = 0.5 * FUNC_F * (1.0 - BEARING["d_ball"] / BEARING["D_pitch"])

# 故障冲击的重复频率
IMPACT_FREQ_FAULT = BPFO        # 外圈缺陷为主故障模式，冲击重复频率 = BPFO
IMPACT_FREQ_NORMAL = FUNC_F     # 正常信号的"旋转"重复频率

FAULT_HARMONICS = (1, 2, 3)     # 早期局部缺陷典型出现的谱峰阶次

# --------------------------------------------------------------------------
# 2. 仿真信号幅度
# --------------------------------------------------------------------------
AMP_ROT = 1.0                   # 转频分量幅度
AMP_NOISE = 0.15                # 高斯白噪声标准差
AMP_IMPULSE = 2.0               # 故障冲击幅度
IMPULSE_DECAY = 800.0           # 冲击衰减时间常数 (1/s)，越小衰减越快

# 标签约定：0 = 正常 (normal)，1 = 故障 (fault)
NORMAL_LABEL = 0
FAULT_LABEL = 1
CLASS_NAMES = ("normal", "fault")

# --------------------------------------------------------------------------
# 3. 模型 / 训练超参数
# --------------------------------------------------------------------------
SEED = 42
TRAIN_RATIO = 0.8               # 按"信号条数"划分，先划分再切窗（防止窗口泄漏）
EPOCHS = 40
BATCH_SIZE = 64
LR = 3e-3
LR_DECAY = 0.97                 # 每轮按此系数衰减
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
L2_EPS = 1e-8
GRAD_CLIP = 5.0                 # 全局梯度范数裁剪阈值
DROPOUT = 0.3
DROPOUT_SEED_OFFSET = 1000      # dropout 随机流与 shuffle 随机流解耦
INPUT_LEN = WINDOW               # 1D-CNN 输入维度
NUM_CLASSES = 2
EARLY_STOP_PATIENCE = 8         # 验证准确率连续 N 轮不提升则早停
N_FILTERS = (16, 32, 64)        # 三层卷积通道数
KERNEL_SIZES = (9, 7, 5)        # 三层卷积核长度
POOL_SIZE = 2
HIDDEN_DIM = 64
N_DEMO_SAMPLES = 5              # 单样本预测演示条数

# 合成数据上模型极易过拟合，故加入显式正则；若在真实数据上出现欠拟合，
# 把 DROPOUT 调到 0.1~0.2、EPOCHS 调大、WEIGHT_DECAY 调到 1e-5。
REGULARIZATION_NOTE = (
    "DROPOUT=0.3 + WEIGHT_DECAY=1e-4 是针对合成数据的过拟合对策；"
    "真实数据效果不佳时优先调小它们，而不是加深网络。"
)

# --------------------------------------------------------------------------
# 4. 绘图统一风格（中文标签，避免乱码）
# --------------------------------------------------------------------------
PLOT_DPI = 150
FONT_CANDIDATES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Source Han Sans SC", "Arial Unicode MS",
)


def configure_matplotlib():
    """在 figures 输出前调用：设置无 GUI 后端 + 中文字体 + 负号正常显示。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [name for name in FONT_CANDIDATES if name in available]
    plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = PLOT_DPI
    plt.rcParams["savefig.bbox"] = "tight"
    return plt, picked


def ensure_dirs() -> None:
    for d in (DIR_DATA, DIR_FIG, DIR_RESULT, DIR_DATA_RAW):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 5. 命名兼容层（见模块 docstring）
# --------------------------------------------------------------------------
sys.modules.setdefault("config_bp", sys.modules[__name__])
sys.modules.setdefault("bp", sys.modules[__name__])
