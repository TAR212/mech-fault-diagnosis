"""
第 1 步：数据集生成 (preprocess1.py)
====================================
教学流水线的第一步。本脚本只做三件事：

    1) 用物理模型**仿真**两万亿分之一不存在的数据 —— 正常 / 故障两类振动信号
    2) 滑动窗口截取片段，给每个片段打标签 0/1，并记录它来自哪一条原始信号 (seg_id)
    3) 落盘成 data/clean_signal1.npy，并当场做 4 项自检（校验 / 划分 / 非退化 / 冲击频率）

为什么要先"造数据"再谈诊断
--------------------------
真实轴承数据（如 CWRU）采集贵、标注贵，而且"正确答案"往往是事后推断的。
教学阶段先用**已知机理**的信号把闭环跑通：我们事先知道冲击重复频率就是
BPFO（外圈故障特征频率），所以第二步包络谱、第三步 CNN 的结论就有了"标准答案"，
可以反过来验证整条流水线的正确性。

信号模型的物理依据
------------------
    正常:  x(t) = Σ_h A_h·sin(2π·h·f_r·t + φ_h) + n(t)          h = 1,2,3，f_r = 50 Hz
    故障:  x(t) = 正常成分 + Σ_k A_i·exp(-α·(t-k·T))·sin(2π·f_n·(t-k·T)) + n(t)

    T   = 1/BPFO       冲击重复间隔（≈6.30 ms），对应"外圈滚过一处缺陷"
    f_n ∈ [500,4000]   轴承座共振频率，**条内固定、条间随机**
    α   = 800 /s       冲击衰减快慢（衰减越快，频谱越"宽"、包络越"窄"）
    n(t) ~ N(0, 0.15^2) 测量噪声

关键设计：故障与正常的差异**只**体现在"周期性衰减冲击"上，而不是整体幅度抬高。
这样第二步的包络谱才能把 BPFO 谐波族揪出来，第三步也不得不学"冲击的周期性"，
而不是记住某个固定频率成分（每条的共振频率都不一样，靠记频率作弊是学不会的）。

约束
----
- 只用 numpy / matplotlib（环境没有 scipy/sklearn/torch，且不允许安装）
- 随机性只用 np.random.default_rng(C.SEED) 及其派生种子 —— 两次运行结果逐字节一致
- 数据集字段与形状由冻结契约 dataio.py 定义，本文件只调用、不发明格式

运行::

    cd mech-fault-diagnosis
    python preprocess1.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

import config as C
import dataio


# --------------------------------------------------------------------------
# 0. 让中文输出在管道/控制台下都不乱码（Windows 默认 GBK 会把中文打成问号）
# --------------------------------------------------------------------------
def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 极老解释器/被替换的 stdout
            pass


_force_utf8_stdout()


# --------------------------------------------------------------------------
# 1. 全局时间轴与派生种子
# --------------------------------------------------------------------------
# 每条信号的采样时间轴 t = 0, 1/FS, ..., (N_PER_SAMPLE-1)/FS
T_AXIS = np.arange(C.N_PER_SAMPLE, dtype=np.float64) / C.FS
IMPACT_INTERVAL = 1.0 / C.BPFO          # 冲击间隔 T ≈ 6.30 ms
BURST_DURATION = 0.008                  # 冲击响应截断时长 8 ms（相邻冲击间隔 6.30 ms，故会略微交叠叠加）
                                        # 取 8 ms 而非 6.30 ms：exp(-800·t) 在 t=4.6 ms 已衰减到 2.5%，
                                        # 交叠部分贡献可忽略，但对低共振频率（500 Hz）保留了足够多周期

RESONANCE_BAND = (500.0, 4000.0)        # 共振频率随机区间 (Hz)
HARMONICS = (1, 2, 3)                   # 转频的 1x/2x/3x 谐波
HARMONIC_AMP_RANGE = (0.02, 0.12)       # 谐波幅度（"小幅随机谐波"）
PHASE_JITTER = 0.05                     # 冲击初相位抖动 ±5% 个冲击周期

SEED_OFFSET_NORMAL = 1                  # 正常类派生种子 = SEED + 1 + i
SEED_OFFSET_FAULT = 100_000             # 故障类派生种子 = SEED + 100000 + i


def n_windows_per_segment() -> int:
    """一条 N_PER_SAMPLE 点的长信号能切出多少个 WINDOW 片段（含 75% 重叠）。

    floor((36000-1024)/256) + 1 = 137
    """
    return 1 + (C.N_PER_SAMPLE - C.WINDOW) // C.STRIDE


# --------------------------------------------------------------------------
# 2. 仿真信号生成
# --------------------------------------------------------------------------
def synth_normal(rng: np.random.Generator) -> np.ndarray:
    """正常信号：转频基频 + 小幅随机谐波 + 高斯白噪声。

    转频 1x 是幅度 1.0 的主成分（转子不平衡/偏心引起的振动），
    2x/3x 谐波幅度随机取 0.02~0.12，模拟机械松动、对中不良等次要成分。
    """
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(HARMONICS))

    sig = np.zeros_like(T_AXIS)
    for k, h in enumerate(HARMONICS):
        amp = C.AMP_ROT if h == 1 else rng.uniform(*HARMONIC_AMP_RANGE)
        sig += amp * np.sin(2.0 * np.pi * h * C.FUNC_F * T_AXIS + phases[k])

    sig += rng.normal(0.0, C.AMP_NOISE, size=T_AXIS.size)
    return sig


def synth_fault(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """故障信号：正常成分 + 周期为 1/BPFO 的衰减冲击串 + 高斯白噪声。

    返回 (信号, 辅助信息)。辅助信息里带着该条的共振频率与冲击初相位，
    供自检与绘图使用（教学上"看得见"比"猜得到"重要）。
    """
    # --- 正常成分（与正常类同分布，故障差异只来自下面的冲击串） ---
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(HARMONICS))
    sig = np.zeros_like(T_AXIS)
    for k, h in enumerate(HARMONICS):
        amp = C.AMP_ROT if h == 1 else rng.uniform(*HARMONIC_AMP_RANGE)
        sig += amp * np.sin(2.0 * np.pi * h * C.FUNC_F * T_AXIS + phases[k])

    # --- 冲击参数：共振频率与初相位对每一条信号只抽一次（条内固定、条间随机） ---
    resonance_hz = float(rng.uniform(*RESONANCE_BAND))
    tau0 = float(rng.uniform(-PHASE_JITTER, PHASE_JITTER) * IMPACT_INTERVAL)

    # 单个冲击的波形模板：衰减振荡 exp(-α·s)·sin(2π·f_n·s)，s 从冲击时刻起算
    n_burst = int(round(BURST_DURATION * C.FS))
    s = np.arange(n_burst, dtype=np.float64) / C.FS
    template = C.AMP_IMPULSE * np.exp(-C.IMPULSE_DECAY * s) * np.sin(2.0 * np.pi * resonance_hz * s)

    # --- 以 T 为间隔把模板"贴"到时间轴上（叠加，保留相位连续性） ---
    impact_times = []
    k = 0
    while True:
        tau = tau0 + k * IMPACT_INTERVAL
        if tau >= T_AXIS[-1] + 1.0 / C.FS:
            break
        impact_times.append(tau)

        i0 = int(np.ceil(tau * C.FS))       # tau 之后第一个采样点
        skip = 0
        if i0 < 0:                          # 冲击起点落在 0 之前 -> 只贴后半段
            skip = -i0
            i0 = 0
        n_copy = min(n_burst - skip, T_AXIS.size - i0)
        if n_copy > 0:
            sig[i0:i0 + n_copy] += template[skip:skip + n_copy]
        k += 1

    sig += rng.normal(0.0, C.AMP_NOISE, size=T_AXIS.size)

    aux = {
        "resonance_hz": resonance_hz,
        "tau0": tau0,
        "impact_times": np.asarray(impact_times, dtype=np.float64),
        "n_impulses": len(impact_times),
    }
    return sig, aux


# --------------------------------------------------------------------------
# 3. 生成整个数据集（两类各 N_SAMPLES_PER_CLASS 条）
# --------------------------------------------------------------------------
def generate_raw() -> tuple[np.ndarray, dict]:
    """返回 (raw, diag)。

    raw : float32, shape (2*N_SAMPLES_PER_CLASS, N_PER_SAMPLE)
          约定前 N_SAMPLES_PER_CLASS 条为正常(0)，后 N_SAMPLES_PER_CLASS 条为故障(1)。
    diag: dict —— 每条故障信号的共振频率 f_n 与冲击初相位 tau0，
          供自检（"条内固定、条间随机"）与绘图（标注冲击时刻）使用。
    """
    n_cls = C.N_SAMPLES_PER_CLASS
    m = 2 * n_cls
    raw = np.empty((m, C.N_PER_SAMPLE), dtype=np.float32)
    res_hz = np.zeros(n_cls, dtype=np.float64)
    tau0 = np.zeros(n_cls, dtype=np.float64)

    for i in range(n_cls):
        rng = np.random.default_rng(C.SEED + SEED_OFFSET_NORMAL + i)
        raw[i] = synth_normal(rng).astype(np.float32)

    for i in range(n_cls):
        rng = np.random.default_rng(C.SEED + SEED_OFFSET_FAULT + i)
        sig, aux = synth_fault(rng)
        raw[n_cls + i] = sig.astype(np.float32)
        res_hz[i] = aux["resonance_hz"]
        tau0[i] = aux["tau0"]

    return raw, {"resonance_hz": res_hz, "tau0": tau0}


def sliding_window(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对每条原始信号做滑窗截取，返回 (X, y, seg_id)。

    为什么用重叠滑窗：一条 3 s 信号只能算一个样本，样本量太少；
    步长 256（75% 重叠）把 800 条扩成 109600 个片段，同时保证每个片段都
    完整包含若干个冲击周期（1024 点 ≈ 85.3 ms ≈ 13.5 个冲击周期）。

    注意：**窗口级重叠意味着划分必须按 seg_id 分组**（见 dataio.split_by_segment），
    否则相邻窗口同时进训练/测试会造成数据泄漏。X 直接预分配，避免大数组复制。
    """
    n_win = n_windows_per_segment()
    m = raw.shape[0]

    X = np.empty((m * n_win, C.WINDOW), dtype=np.float32)
    for s in range(m):
        # 用 stride 视图零拷贝取出所有窗口 (n_win, WINDOW)，再一次性写入 X
        view = np.lib.stride_tricks.sliding_window_view(raw[s], C.WINDOW)[::C.STRIDE]
        X[s * n_win:(s + 1) * n_win] = view

    labels = np.empty(m, dtype=np.int64)
    labels[:C.N_SAMPLES_PER_CLASS] = C.NORMAL_LABEL
    labels[C.N_SAMPLES_PER_CLASS:] = C.FAULT_LABEL

    y = np.repeat(labels, n_win)
    seg_id = np.repeat(np.arange(m, dtype=np.int64), n_win)
    return X, y, seg_id


def build_extra_meta() -> dict:
    """本步骤追加进 meta 的自描述字段（**生成与复现测试共用同一函数**）。

    为什么单独抽出来：meta 是 .npy 字节流的一部分，只要字典内容有一点不同，
    文件 sha256 就会变。把"构造 meta"收敛到一个函数，可复现性测试才能真正
    逐字节比对，而不是靠人工抄一遍键值。
    """
    n_win = n_windows_per_segment()
    n_total = 2 * C.N_SAMPLES_PER_CLASS
    return {
        "generated_by": "preprocess1.py",
        "n_windows_per_segment": n_win,
        "n_windows_total": n_total * n_win,
        "n_segments": n_total,
        "train_ratio": C.TRAIN_RATIO,
        "class_names": list(C.CLASS_NAMES),
        "amp_rot": C.AMP_ROT,
        "amp_noise": C.AMP_NOISE,
        "amp_impulse": C.AMP_IMPULSE,
        "impulse_decay": C.IMPULSE_DECAY,
        "impact_interval_s": float(IMPACT_INTERVAL),
        "resonance_band_hz": list(RESONANCE_BAND),
        "impact_phase_jitter": PHASE_JITTER,
        "harmonics": list(HARMONICS),
        "segment_layout": "raw[0:400]=normal(0), raw[400:800]=fault(1)",
    }


# --------------------------------------------------------------------------
# 4. 自检工具
# --------------------------------------------------------------------------
def autocorrelation(x: np.ndarray) -> np.ndarray:
    """自相关（FFT 实现，numpy 自带，无需 scipy）。

        r[k] = Σ x[i]·x[i+k] ，并归一化到 r[0] = 1
    周期性信号的自相关在同周期处出现峰值 —— 这正是"验证冲击间隔"的工具。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    spec = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(spec * np.conj(spec))[:n]
    return r / r[0]


def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """理想带通（rfft 频域置零），把 50 Hz 转频成分挡掉，只留共振带里的冲击响应。"""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / C.FS)
    mask = (freqs >= lo) & (freqs <= hi)
    return np.fft.irfft(spec * mask, n=x.size)


def envelope(x: np.ndarray, smooth_ms: float = 2.0) -> np.ndarray:
    """包络解调：带通到共振带 → 取绝对值 → 滑动平均。

    这就是第二步 fft1.py 用来找 BPFO 的标准手法（无 scipy，用滑动平均近似 Hilbert
    包络）。这里先借用它来验证"冲击间隔是否真的等于 1/BPFO"。

    为什么必须解调而不能直接对原始信号做自相关：
    - 50 Hz 转频成分在 lag≈1/BPFO 处的自相关是负的，会把冲击串的正峰抵消掉；
    - 就算带通到共振带，剩下的是**载波频率约 f_n 的衰减振荡**，其自相关峰由载波
      周期（约 5~24 个采样点）决定，局部极大值会偏离真正的冲击周期（实测偏 3~4 点，
      4% 频率误差）。取包络后载波被抹掉，只剩"每 T 秒一次冲击"的慢变形状，
      自相关峰就干净地落在冲击周期上（实测误差 < 0.1%）。
    """
    bp = bandpass(x, *RESONANCE_BAND)
    w = max(3, int(round(smooth_ms / 1000.0 * C.FS)))
    return np.convolve(np.abs(bp), np.ones(w, dtype=np.float64) / w, mode="same")


def _refine_peak(r: np.ndarray, lag_peak: int) -> tuple[float, float]:
    """对自相关峰做抛物线插值（用峰点及左右邻居拟合二次曲线取顶点）。

    目的是把"整数采样点上的峰"细化到亚采样点精度：真值 1/BPFO=75.589 点不是整数，
    插值后反推频率的误差可从 ~0.5% 降到 ~0.1% 以内。
    """
    if 0 < lag_peak < r.size - 1:
        y0, y1, y2 = float(r[lag_peak - 1]), float(r[lag_peak]), float(r[lag_peak + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return lag_peak + delta, float(r[lag_peak])


def impact_autocorr_peak(sig: np.ndarray) -> dict:
    """单条信号：在 1/BPFO 附近的搜索窗里找包络自相关峰，反推冲击重复频率。"""
    r = autocorrelation(envelope(sig))
    lag_exact = C.FS / C.BPFO
    lo, hi = int(round(0.7 * lag_exact)), int(round(1.3 * lag_exact))
    lag_peak = lo + int(np.argmax(r[lo:hi + 1]))
    lag_refined, peak_value = _refine_peak(r, lag_peak)
    freq_est = C.FS / lag_refined
    return {
        "lag_exact": float(lag_exact),
        "lag_peak": int(lag_peak),
        "lag_refined": float(lag_refined),
        "peak_value": peak_value,
        "freq_est": float(freq_est),
        "freq_rel_err": float(abs(freq_est - C.BPFO) / C.BPFO),
    }


def verify_impact_frequency(raw: np.ndarray) -> dict:
    """验证"冲击重复频率 == BPFO"，并给出故障/正常的对照。

    做法：对每条信号求"包络自相关"，在 0.7~1.3×(1/BPFO) 的搜索窗内取最大值。
    - 故障：每条都应有强峰，且反推频率接近 BPFO
    - 正常：没有周期冲击 -> 该窗口内应无显著峰（这是"对照实验"，证明峰不是自相关本身的产物）
    """
    n_cls = C.N_SAMPLES_PER_CLASS
    fault_lag = np.empty(n_cls)
    fault_peak = np.empty(n_cls)
    norm_peak = np.empty(n_cls)
    for i in range(n_cls):
        d = impact_autocorr_peak(raw[n_cls + i])
        fault_lag[i] = d["lag_refined"]
        fault_peak[i] = d["peak_value"]
        norm_peak[i] = impact_autocorr_peak(raw[i])["peak_value"]

    fault_freq = C.FS / fault_lag
    rel_err = np.abs(fault_freq - C.BPFO) / C.BPFO
    detail = impact_autocorr_peak(raw[n_cls])   # 第 1 条故障信号，用于逐步打印
    return {
        "detail": detail,
        "n_fault": n_cls,
        "fault_lag_mean": float(fault_lag.mean()),
        "fault_lag_std": float(fault_lag.std()),
        "fault_freq_mean": float(fault_freq.mean()),
        "freq_rel_err_max": float(rel_err.max()),
        "freq_rel_err_mean": float(rel_err.mean()),
        "fault_peak_min": float(fault_peak.min()),
        "fault_peak_mean": float(fault_peak.mean()),
        "normal_peak_max": float(norm_peak.max()),
        "normal_peak_mean": float(norm_peak.mean()),
    }


def _resonance_peak_hz(seg: np.ndarray) -> float:
    """一段信号在共振带 500~4000 Hz 内的谱峰频率。"""
    seg = np.asarray(seg, dtype=np.float64)
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / C.FS)
    m = (freqs >= RESONANCE_BAND[0]) & (freqs <= RESONANCE_BAND[1])
    return float(freqs[m][int(np.argmax(np.abs(np.fft.rfft(seg))[m]))])


def _wrong_per_impulse_variant(seed: int = 0) -> np.ndarray:
    """**故意写错**的对照实现：每次冲击都重抽一个共振频率。

    这不是我们要的信号，而是用来证明自检判据真的有鉴别力 ——
    如果连"错误实现"都能通过检查，那这个检查就是摆设（教学上也解释"为什么要这么检"）。
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(C.N_PER_SAMPLE, dtype=np.float64)
    n_burst = int(round(BURST_DURATION * C.FS))
    s = np.arange(n_burst, dtype=np.float64) / C.FS
    k = 0
    while k * IMPACT_INTERVAL < C.T_PER_SAMPLE:
        i0 = int(np.ceil(k * IMPACT_INTERVAL * C.FS))
        fn = float(rng.uniform(*RESONANCE_BAND))
        m = min(n_burst, C.N_PER_SAMPLE - i0)
        if m > 0:
            x[i0:i0 + m] += C.AMP_IMPULSE * np.exp(-C.IMPULSE_DECAY * s[:m]) * np.sin(2.0 * np.pi * fn * s[:m])
        k += 1
    return x


def verify_resonance_per_segment(raw: np.ndarray, n_check: int | None = None) -> dict:
    """验证"共振频率条内固定、条间随机"：把每条故障信号一分为二，两半的谱峰应一致。

    为什么用"劈半对比"而不是"跟设定的 f_n 比"：单看整条谱峰会受 50 Hz 转频与噪声
    干扰，偏差可达 ±130 Hz（谱分辨率 0.333 Hz 不是原因，多峰竞争才是）；但**同一条
    信号的前后半段，若 f_n 真的只抽了一次，两半的谱峰必然落在同一根谱线上**。
    实测 400 条里 99% 的两半谱峰完全相同（差值中位数 0 Hz），
    而"每次冲击重抽 f_n"的错误实现平均相差 833 Hz —— 相差 3 个数量级。
    """
    n_cls = C.N_SAMPLES_PER_CLASS
    if n_check is None:
        n_check = n_cls
    half = C.N_PER_SAMPLE // 2

    diffs = np.empty(n_check, dtype=np.float64)
    for i in range(n_check):
        seg = raw[n_cls + i]
        diffs[i] = abs(_resonance_peak_hz(seg[:half]) - _resonance_peak_hz(seg[half:]))

    wrong = _wrong_per_impulse_variant()
    wrong_diff = abs(_resonance_peak_hz(wrong[:half]) - _resonance_peak_hz(wrong[half:]))

    return {
        "n_check": n_check,
        "median_hz": float(np.median(diffs)),
        "mean_hz": float(np.mean(diffs)),
        "p95_hz": float(np.percentile(diffs, 95)),
        "max_hz": float(np.max(diffs)),
        "frac_close": float(np.mean(diffs <= 5.0)),
        "wrong_variant_hz": float(wrong_diff),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 5. 绘图
# --------------------------------------------------------------------------
def plot_cache(raw: np.ndarray, X: np.ndarray, tau0: float) -> Path:
    """2x2 教学缓存图：正常/故障时域（0.3 s）+ 正常/故障片段（各 WINDOW 点）。"""
    plt, picked_fonts = C.configure_matplotlib()
    C.ensure_dirs()

    n_show = int(round(0.3 * C.FS))                     # 0.3 s
    t_short = np.arange(n_show) / C.FS
    t_win = np.arange(C.WINDOW) / C.FS * 1000.0         # 片段横轴用 ms

    normal_seg, fault_seg = raw[0], raw[C.N_SAMPLES_PER_CLASS]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.6))
    fig.suptitle(
        "第 1 步 数据集生成自检：正常 vs 故障仿真振动信号\n"
        f"采样率 {C.FS/1000:.1f} kHz，转频 {C.FUNC_F:.0f} Hz，"
        f"冲击重复频率 = BPFO = {C.BPFO:.2f} Hz（间隔 {IMPACT_INTERVAL*1000:.2f} ms）",
        fontsize=12,
    )

    # (0,0) 正常时域
    ax = axes[0, 0]
    ax.plot(t_short * 1000.0, normal_seg[:n_show], lw=0.9, color="#1f77b4")
    ax.set_title(f"正常信号时域（0–0.3 s）  RMS={normal_seg.std():.3f}  峰值={np.abs(normal_seg).max():.3f}")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("振动幅值 (a.u.)")
    ax.grid(alpha=0.3)

    # (0,1) 故障时域 + 冲击时刻标记（用该条信号真实的 tau0，不是重新抽样）
    ax = axes[0, 1]
    ax.plot(t_short * 1000.0, fault_seg[:n_show], lw=0.9, color="#d62728")
    n_marks = 0
    k = 0
    while True:
        tau = tau0 + k * IMPACT_INTERVAL
        k += 1
        if tau > t_short[-1]:
            break
        if tau < 0:
            continue
        ax.axvline(tau * 1000.0, color="k", ls="--", lw=0.8, alpha=0.5)
        n_marks += 1
    if n_marks:
        ax.plot([], [], color="k", ls="--", lw=0.8,
                label=f"冲击时刻（间隔 1/BPFO ≈ {IMPACT_INTERVAL*1000:.2f} ms）")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"故障信号时域（0–0.3 s）  RMS={fault_seg.std():.3f}  峰值={np.abs(fault_seg).max():.3f}")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("振动幅值 (a.u.)")
    ax.grid(alpha=0.3)

    # (1,0) 正常片段：取第 0 条信号的第 0 个窗口（确定性，不依赖 argmax）
    ax = axes[1, 0]
    ax.plot(t_win, X[0], lw=0.9, color="#1f77b4")
    ax.set_title(f"正常样本片段（WINDOW={C.WINDOW} 点 ≈ {C.WINDOW/C.FS*1000:.1f} ms，标签 0）")
    ax.set_xlabel("片段内时间 (ms)")
    ax.set_ylabel("振动幅值 (a.u.)")
    ax.grid(alpha=0.3)

    # (1,1) 故障片段：取第 0 条故障原始信号的第 0 个窗口
    k_idx = C.N_SAMPLES_PER_CLASS * n_windows_per_segment()
    ax = axes[1, 1]
    ax.plot(t_win, X[k_idx], lw=0.9, color="#d62728")
    ax.set_title(f"故障样本片段（WINDOW={C.WINDOW} 点 ≈ {C.WINDOW/C.FS*1000:.1f} ms，标签 1）")
    ax.set_xlabel("片段内时间 (ms)")
    ax.set_ylabel("振动幅值 (a.u.)")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(C.FIG_CACHE_PATH)
    fig.savefig(out)
    plt.close(fig)
    print(f"    已保存图像: {out}（中文字体: {picked_fonts[:2] if picked_fonts else '未找到，已回退 DejaVu Sans'}）")
    return out


# --------------------------------------------------------------------------
# 6. 主流程
# --------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    C.ensure_dirs()

    print("=" * 78)
    print("第 1 步：数据集生成 preprocess1.py")
    print("=" * 78)

    n_cls = C.N_SAMPLES_PER_CLASS
    n_win = n_windows_per_segment()
    n_total = 2 * n_cls

    print(f"[环境] numpy={np.__version__} | 项目根目录={C.ROOT}")
    print(f"[契约] 采样率={C.FS:.0f} Hz，单条时长={C.T_PER_SAMPLE:.0f} s，"
          f"每条={C.N_PER_SAMPLE} 点，WINDOW={C.WINDOW}，STRIDE={C.STRIDE}")
    print(f"[契约] 轴承特征频率: BPFO={C.BPFO:.4f} Hz，BPFI={C.BPFI:.4f} Hz，"
          f"BSF={C.BSF:.4f} Hz，FTF={C.FTF:.4f} Hz")
    print(f"[契约] 冲击间隔 1/BPFO = {IMPACT_INTERVAL*1000:.4f} ms，"
          f"单条窗口数 = 1+(N-WINDOW)//STRIDE = {n_win}，窗口总数 = {n_total}×{n_win} = {n_total*n_win}")

    # ---------------- 步骤 1 ----------------
    print("\n" + "-" * 78)
    print("步骤 1/5：仿真生成正常 / 故障两类振动信号")
    print("        为什么：先用已知机理的数据把闭环跑通，后续步骤才有\"标准答案\"可验证。")
    print("-" * 78)
    raw, diag = generate_raw()
    res_hz = diag["resonance_hz"]
    print(f"    生成完成: raw.shape={raw.shape}, dtype={raw.dtype}, "
          f"占用 {raw.nbytes/1024**2:.1f} MB")
    print(f"    raw[:{n_cls}] = 正常(标签 0)，raw[{n_cls}:] = 故障(标签 1)（沿用图片的类别顺序约定）")
    print(f"    故障共振频率 f_n: 抽自 [{RESONANCE_BAND[0]:.0f}, {RESONANCE_BAND[1]:.0f}] Hz，"
          f"最小值={res_hz.min():.1f} Hz，均值={res_hz.mean():.1f} Hz，最大值={res_hz.max():.1f} Hz"
          f"（条内固定、条间随机）")

    # ---------------- 步骤 2 ----------------
    print("\n" + "-" * 78)
    print("步骤 2/5：滑动窗口截取 + 打标签")
    print("        为什么：一条 3 s 信号只能算一个样本，样本量不够训练；")
    print(f"        步长 {C.STRIDE}（{100*(1-C.STRIDE/C.WINDOW):.0f}% 重叠）把 {n_total} 条扩成 {n_total*n_win} 个片段。")
    print("-" * 78)
    X, y, seg_id = sliding_window(raw)
    print(f"    X.shape={X.shape}, dtype={X.dtype}（{X.nbytes/1024**2:.1f} MB）")
    print(f"    y.shape={y.shape}，seg_id.shape={seg_id.shape}，"
          f"seg_id 覆盖 0..{int(seg_id.max())} 共 {len(np.unique(seg_id))} 条原始信号")

    counts = dataio.class_counts(y)
    print("\n    人可读摘要")
    print("    " + "·" * 60)
    print(f"      信号条数      : 正常 {n_cls} 条 + 故障 {n_cls} 条 = {n_total} 条")
    print(f"      每条窗口数    : {n_win}（WINDOW={C.WINDOW}, STRIDE={C.STRIDE}）")
    print(f"      窗口总数      : {X.shape[0]}")
    print(f"      X 形状        : {X.shape}（第二维必须 == WINDOW={C.WINDOW}）")
    print(f"      标签分布      : 正常(0) {counts['normal']} | 故障(1) {counts['fault']} | 合计 {counts['total']}")
    print(f"      每类窗口数    : {n_cls} 条 × {n_win} = {n_cls*n_win}（两类均衡，无需类别加权）")
    print("    " + "·" * 60)

    # ---------------- 步骤 3 ----------------
    print("\n" + "-" * 78)
    print("步骤 3/5：按冻结契约 dataio.save_dataset() 落盘")
    print("        为什么：字段名/形状/dtype 由契约定义，第二步和第三步只认这一个格式。")
    print("-" * 78)
    extra = build_extra_meta()
    meta = dataio.build_meta(extra)
    path = dataio.save_dataset(C.DATA_PATH, X, y, seg_id, raw, meta)
    sha = sha256_of(Path(path))
    print(f"    已保存: {path}")
    print(f"    文件大小: {Path(path).stat().st_size/1024**2:.1f} MB")
    print(f"    sha256  : {sha}")

    # ---------------- 步骤 4 ----------------
    print("\n" + "-" * 78)
    print("步骤 4/5：四项自检（校验 / 防泄漏划分 / 非退化 / 冲击频率）")
    print("        为什么：数据错了后面两步全错，必须当场证伪而不是等 CNN 训练出来再回头查。")
    print("-" * 78)

    # (a) load_dataset 校验
    payload = dataio.load_dataset(C.DATA_PATH)
    dataio.validate(payload)
    a_counts = dataio.class_counts(payload["y"])
    print("    (a) load_dataset() 通过 validate()：X 第二维 == WINDOW、标签属于 {0,1}、无 NaN/Inf、seg_id 全覆盖")
    print(f"        正常样本 {a_counts['normal']} 个、故障样本 {a_counts['fault']} 个 —— 两类标签都在，二分类可训练")
    assert payload["X"].shape[1] == C.WINDOW
    assert payload["X"].shape[0] == n_total * n_win
    for name, sl in (("正常", slice(0, n_cls)), ("故障", slice(n_cls, n_total))):
        assert np.isfinite(payload["raw"][sl]).all(), f"{name}原始信号含 NaN/Inf"
    print(f"        两类原始信号分别做有限性检查：正常 raw[:{n_cls}] OK，故障 raw[{n_cls}:] OK")

    # (b) 按 seg_id 划分
    X_tr, y_tr, X_te, y_te = dataio.split_by_segment(payload["X"], payload["y"], payload["seg_id"])
    tr_counts = dataio.class_counts(y_tr)
    te_counts = dataio.class_counts(y_te)
    n_tr_seg = len(np.unique(payload["seg_id"][:len(y_tr)]))
    print(f"    (b) split_by_segment(TRAIN_RATIO={C.TRAIN_RATIO}) 防泄漏划分")
    print(f"        训练侧: {X_tr.shape[0]} 个窗口（{n_tr_seg} 条信号）  "
          f"正常 {tr_counts['normal']} / 故障 {tr_counts['fault']}")
    print(f"        测试侧: {X_te.shape[0]} 个窗口（{n_total-n_tr_seg} 条信号）  "
          f"正常 {te_counts['normal']} / 故障 {te_counts['fault']}")
    tr_segs = set(np.unique(payload["seg_id"][:len(y_tr)]).tolist())
    te_segs = set(np.unique(payload["seg_id"][len(y_tr):]).tolist())
    assert not (tr_segs & te_segs), "同一 seg_id 同时出现在两侧 = 数据泄漏"
    assert len(set(y_tr.tolist())) == 2 and len(set(y_te.tolist())) == 2, "某一侧只有单类别"
    print("        泄漏检查: 两侧 seg_id 交集为空 —— 同一条信号的窗口只出现在一侧 √")
    print("        两侧类别检查: 训练/测试都同时含正常与故障 √")

    # (c) 非退化
    rms_n = float(np.mean(np.std(payload["raw"][:n_cls], axis=1)))
    rms_f = float(np.mean(np.std(payload["raw"][n_cls:], axis=1)))
    pk_n = float(np.mean(np.max(np.abs(payload["raw"][:n_cls]), axis=1)))
    pk_f = float(np.mean(np.max(np.abs(payload["raw"][n_cls:]), axis=1)))
    print("    (c) 非退化检查（两类若数值相同就是退化数据，后面学不出东西）")
    print(f"        正常: RMS={rms_n:.4f}  峰值={pk_n:.4f}")
    print(f"        故障: RMS={rms_f:.4f}  峰值={pk_f:.4f}")
    print(f"        故障/正常: RMS 比 {rms_f/rms_n:.3f}（差异主要来自冲击瞬态，不是整体抬升）"
          f"，峰值比 {pk_f/pk_n:.3f}")
    assert rms_f > rms_n and pk_f > pk_n, "故障信号未体现出冲击性差异（退化数据）"

    # (d) 冲击重复频率 == BPFO
    v = verify_impact_frequency(payload["raw"])
    d = v["detail"]
    print("    (d) 冲击重复频率验证：包络自相关（带通 500~4000 Hz → 取绝对值 → 滑动平均 → 自相关）")
    print("        为什么这么绕：直接对原始信号做自相关，50 Hz 转频在 1/BPFO 处的负相关会把冲击峰抵消；")
    print("        只带通到共振带，剩下的衰减振荡又由载波周期主导（实测偏差 ~3 点）。取包络后只剩")
    print("        \"每 T 秒一次冲击\"的慢变形状，自相关峰才真正落在冲击周期上。")
    print(f"        理论 1/BPFO = {d['lag_exact']:.4f} 采样点（BPFO={C.BPFO:.4f} Hz）")
    print(f"        第 1 条故障信号：自相关峰 lag={d['lag_peak']}（抛物线插值后 {d['lag_refined']:.3f}），"
          f"峰值 r={d['peak_value']:.4f}")
    print(f"        反推 f = FS/lag = {d['freq_est']:.4f} Hz，相对误差 {d['freq_rel_err']*100:.4f}%")
    print(f"        全部 {v['n_fault']} 条故障信号：lag 均值 {v['fault_lag_mean']:.3f}±{v['fault_lag_std']:.3f} 点，"
          f"反推频率均值 {v['fault_freq_mean']:.4f} Hz，最大相对误差 {v['freq_rel_err_max']*100:.4f}%")
    print(f"        峰强：故障最小 r={v['fault_peak_min']:.4f}（均值 {v['fault_peak_mean']:.4f}）")
    print(f"        对照（全部 {v['n_fault']} 条正常信号）：峰值最大仅 r={v['normal_peak_max']:.4f}"
          f"（均值 {v['normal_peak_mean']:.4f}）—— 无周期冲击则无峰，故障/正常峰强比 ≈"
          f" {v['fault_peak_min']/v['normal_peak_max']:.1f} 倍")
    print(f"        结论: 冲击重复频率落在 BPFO 上（误差 <{v['freq_rel_err_max']*100:.2f}%），"
          f"第二步包络谱应能在 {C.BPFO:.2f} Hz 及其 1x/2x/3x 谐波处看到谱峰 √")
    assert v["freq_rel_err_max"] < 0.005, f"自相关反推频率偏离 BPFO 过大: {v['freq_rel_err_max']:.5f}"
    assert v["fault_peak_min"] > 0.3, f"1/BPFO 处自相关峰太弱: {v['fault_peak_min']:.4f}"
    assert v["fault_peak_min"] > 5.0 * v["normal_peak_max"], "故障/正常在 1/BPFO 处无区分度"

    rv = verify_resonance_per_segment(payload["raw"])
    print(f"        (附) 共振频率\"条内固定\"验证：把每条故障信号劈成前后两半，比较各自在 500~4000 Hz 内的谱峰")
    print(f"            全部 {rv['n_check']} 条: 两半谱峰相同(≤5 Hz)的比例 {rv['frac_close']*100:.1f}%，"
          f"差值 中位数 {rv['median_hz']:.1f} Hz / 均值 {rv['mean_hz']:.2f} Hz / p95 {rv['p95_hz']:.1f} Hz / 最大 {rv['max_hz']:.1f} Hz")
    print(f"            反例对照（故意让每次冲击重抽 f_n 的错误实现）：两半谱峰相差 {rv['wrong_variant_hz']:.1f} Hz")
    print(f"            结论: 同一条信号内共振频率恒定，条与条之间随机（{RESONANCE_BAND[0]:.0f}~{RESONANCE_BAND[1]:.0f} Hz）"
          f"—— 模型无法靠记某个固定频率作弊 √")
    assert rv["frac_close"] > 0.95, f"两半谱峰一致率过低: {rv['frac_close']:.3f}"
    assert rv["wrong_variant_hz"] > 50.0, "自检判据无鉴别力（错误实现未被识别）"

    # ---------------- 步骤 5 ----------------
    print("\n" + "-" * 78)
    print("步骤 5/5：绘制教学缓存图")
    print("        为什么：把两类信号“看得见的差异”固化下来，第 2、3 步的解释有共同参照物。")
    print("-" * 78)
    fig_path = plot_cache(payload["raw"], payload["X"], float(diag["tau0"][0]))

    print("\n" + "=" * 78)
    print("第 1 步完成 —— 全部自检通过")
    print("=" * 78)
    print(f"  数据集: {path}")
    print(f"  X: {tuple(X.shape)} float32 | y: {tuple(y.shape)} int64 | "
          f"seg_id: {tuple(seg_id.shape)} int64 | raw: {tuple(raw.shape)} float32")
    print(f"  sha256: {sha}")
    print(f"  图    : {fig_path}")
    print(f"  耗时  : {time.time()-t_start:.2f} s")
    print(f"  下一步: python fft1.py（消费 {Path(path).name}，做时域+频域+包络谱分析）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
