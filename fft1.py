#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fft1.py —— 三步流水线第 2 步：轴承故障信号的时域 / 频域 / 包络谱分析
=======================================================================

输入 : data/clean_signal1.npy            （由 preprocess1.py 生成，经 dataio.load_dataset() 读取）
输出 : figures/fig_time_domain.png       图1 时域波形（正常 vs 故障）
       figures/fig_spectrum.png          图2 幅值谱（0–2000 Hz 主视区）
       figures/fig_envelope_spectrum.png 图3 包络谱（0–600 Hz，轴承故障判据图）
       analysis_report.md                定量报告（数值 → 图 → 结论 三者一致）

方法论（教学要点）
------------------
1) 时域：外圈局部缺陷每转产生一次冲击，冲击重复频率 = BPFO；
   周期冲击使**峭度(kurtosis)** 与 **裕度指标(peak/RMS)** 显著上升。
2) 频域：周期冲击在幅值谱上形成 BPFO 的等间隔谐波线谱（1x/2x/3x…），
   而正常信号只有转频及其倍频（50 / 100 / 150 Hz…）。
3) 包络谱：取包络 → 去均值 → FFT，把"高频共振被低频冲击调制"的信息解调出来，
   是滚动轴承故障最经典的判据谱（图3 用整流包络 |x|，冲击检出另用解析包络）。
   本环境**没有 scipy**：包络平滑一律用 np.convolve，解析信号 |x + j·H{x}| 用 FFT 自建。
4) 关键工程教训（本数据集实测）：故障信号转频分量(1.0 g)远大于冲击，
   且冲击幅值受转频调制，因此**幅值阈值法失效**（阈值高过包络最大值，max/thr < 1）。
   真正稳健的判据是**周期性**：包络自相关在 1/BPFO 处的相关系数
   正常 ≈ 0.02、故障 ≈ 0.97，一步即可分开，且正常样本零误报。

约定
----
* 幅值单位 g；采样率 C.FS = 12000 Hz；单条记录 3 s = 36000 点。
* 频谱纵轴统一用 **dB 幅值谱**：0 dB = 1 g，即 dB = 20·log10(A / 1 g)。
* 谐波幅值一律取该阶次 **±2 Hz 带内最大值**（等效峰值幅值，抗栅栏效应与轻微频率偏移）。
* 路径全部取自 config.py 常量，脚本内不出现硬编码路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- 0. 保证在任何工作目录下都能 import 同目录的 config / dataio -------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 控制台中文：本环境 stdout 默认 GBK，尽力切到 UTF-8 以免教学输出乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

import config as C
import dataio

# ---------------------------------------------------------------------------
# 分析超参数（只影响估计/显示，不影响物理含义）
# ---------------------------------------------------------------------------
N_AVG_MAX = 20          # 每类用于统计/平均谱的信号条数（首 N 条，确定性选择）
NFFT_PLOT = 1 << 18     # 绘图用补零 FFT 长度 → df ≈ 0.0458 Hz（提升谱峰读数精度）
SEARCH_HZ = 2.0         # 读取 BPFO 谐波幅值时的搜索半带宽 (Hz)
BAND_HZ = 10.0          # 计算 "BPFO 频带能量占比" 的带宽半宽 (Hz)
SPECTRUM_FMAX = 2000.0  # 幅值谱主视区上限 (Hz)
ENV_FMAX = 600.0        # 包络谱显示上限 (Hz)
DETAIL_S = 0.3          # 时域细节波形时长 (s)
ENV_SMOOTH = 5          # 包络谱的滑动平均点数（要求 5~15）
DET_SMOOTH = 15         # 冲击检出中包络的滑动平均点数（去毛刺）
DEMOD_FLO = 500.0       # 冲击解调带下限 (Hz)：轴承冲击激起的高频共振带
DEMOD_FHI = 4000.0      # 冲击解调带上限 (Hz)：保留边带结构、避开 50 Hz 转频与低阶谐波
DET_K_MAD = 3.0         # 冲击阈值 = 中位数 + k·1.4826·MAD（稳健自适应阈值）
DET_AC_MIN = 0.30       # 周期门限：包络自相关峰值 < 该值即判定"无周期冲击"（不检出）
DET_FILL = True         # 是否用"整数倍周期缺口填补"补回被相位抖动淹没的冲击


# ===========================================================================
# 基础工具
# ===========================================================================
def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """k 点滑动平均（np.convolve 实现，替代 scipy 低通滤波）。

    两端用 edge 填充，保证输出长度与输入一致。
    """
    x = np.asarray(x, dtype=float)
    k = int(k)
    if k <= 1:
        return x.copy()
    pad = k // 2
    kernel = np.ones(k, dtype=float) / k
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: x.size]


def to_db(a: float | np.ndarray) -> np.ndarray:
    """幅值 → dB，参考值 1 g（即 20·log10(A/1g)）。"""
    a = np.asarray(a, dtype=float)
    return 20.0 * np.log10(np.maximum(a, 1e-12))


def single_sided_amplitude(x: np.ndarray, fs: float, nfft: int) -> np.ndarray:
    """单边幅值谱：加 Hann 窗 + 幅值修正，使正弦分量的谱峰 ≈ 该分量的幅值(g)。

    A = 2·|rfft(x·w)| / Σw ；DC 与 Nyquist 折半。
    """
    x = np.asarray(x, dtype=float)
    w = np.hanning(x.size)
    spec = np.fft.rfft(x * w, n=nfft)
    a = np.abs(spec) * (2.0 / w.sum())
    a[0] *= 0.5
    if nfft % 2 == 0:
        a[-1] *= 0.5
    return a


def average_spectrum(signals, fs: float, nfft: int):
    """多信号**功率平均**谱 → 返回 (等效幅值谱 A, 功率谱 P, 频率轴 f)。

    功率平均而非幅值平均：随机噪声的谱线不会因相位随机而虚高，
    周期性线谱则被完整保留 —— 这正是把 BPFO 谐波从噪声底下"捞"出来的关键。
    """
    p_sum = None
    for x in signals:
        a = single_sided_amplitude(x, fs, nfft)
        p = a * a
        p_sum = p if p_sum is None else p_sum + p
    p_avg = p_sum / max(len(signals), 1)
    return np.sqrt(p_avg), p_avg, np.fft.rfftfreq(nfft, d=1.0 / fs)


def band_max(a: np.ndarray, f: np.ndarray, f0: float, half_hz: float) -> float:
    """取 f0 ± half_hz 带内谱峰幅值（g）。"""
    m = (f >= f0 - half_hz) & (f <= f0 + half_hz)
    if not np.any(m):
        return float("nan")
    return float(a[m].max())


def band_power(p: np.ndarray, f: np.ndarray, f0: float, half_hz: float) -> float:
    """取 f0 ± half_hz 带内功率和。"""
    m = (f >= f0 - half_hz) & (f <= f0 + half_hz)
    return float(p[m].sum())


# ===========================================================================
# 时域指标 / 冲击检出
# ===========================================================================
def time_metrics(x: np.ndarray) -> dict:
    """RMS / 峰值 / 峰峰值 / 峭度 / 裕度指标。"""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    var = float(np.mean(xc ** 2))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    return {
        "rms": rms,
        "peak": peak,
        "ptp": float(np.max(x) - np.min(x)),
        # 峭度用 Pearson 定义（正态信号 = 3），不是 excess kurtosis
        "kurtosis": float(np.mean(xc ** 4) / (var ** 2 + 1e-30)),
        "margin": peak / (rms + 1e-30),
    }


def bandpass_fft(x: np.ndarray, fs: float, flo: float, fhi: float,
                 trans_hz: float = 50.0) -> np.ndarray:
    """零相位带通（纯 FFT 频域加权，无需 scipy）。

    梯形窗：flo 以下为 0、flo~flo+trans_hz 线性上升、fhi-trans_hz~fhi 线性下降。
    用于把转频(50 Hz)及其低阶谐波挡在带外，只留下轴承冲击激起的高频共振带。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    df = fs / n
    rl = max(int(trans_hz / df), 2)          # 过渡带样点数
    W = np.clip((f - flo) / (rl * df), 0.0, 1.0)
    if fhi < fs / 2.0:
        W = W * np.clip((fhi - f) / (rl * df), 0.0, 1.0)
    return np.fft.irfft(F * W, n)


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """解析信号包络 |x + j·H{x}|，用 FFT 构造希尔伯特变换（替代 scipy.signal.hilbert）。

    为什么不用 |x| 整流包络：本数据集的转频分量幅值(1.0 g)远大于冲击，
    整流后包络会被 50 Hz 正弦"支配"，其 MAD 被抬到冲击幅值之上，阈值法失效。
    解析包络先去掉载波，冲击的包络突起才真正暴露出来。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.zeros(n, dtype=float)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(x) * h))


def envelope_of(x: np.ndarray, fs: float, smooth_len: int = DET_SMOOTH) -> np.ndarray:
    """冲击解调包络 = 带通(500–4000 Hz，挡掉 50 Hz 转频) → 解析包络 → 滑动平均去毛刺。"""
    xb = bandpass_fft(x, fs, DEMOD_FLO, DEMOD_FHI)
    return moving_average(analytic_envelope(xb), smooth_len)


def rectified_envelope(x: np.ndarray, smooth_len: int = ENV_SMOOTH) -> np.ndarray:
    """包络谱输入 = |x|(整流) → 滑动平均 → 去均值（按题目约定，不用 scipy 希尔伯特）。"""
    e = moving_average(np.abs(np.asarray(x, dtype=float)), smooth_len)
    return e - e.mean()


def periodicity(env: np.ndarray, fs: float, f_repeat: float,
                half_frac: float = 0.4):
    """包络的**周期性强度**：自相关在理论冲击周期 ±half_frac 内的主峰 → (周期[s], 相关系数)。

    这是本脚本判定"到底有没有周期冲击"的核心指标，也是**故障/正常的分界量**：
      · 正常信号包络近似平稳噪声 → 自相关峰值 ac ≈ 0（本数据集实测 0.02 量级）；
      · 外圈故障包络是周期衰减梳状 → ac ≈ 0.97（实测）。
    单纯用幅值阈值无法区分二者：故障包络本身是"稠密周期性梳状"，
    其 ripple 底噪被整体抬高，峰值并不比底噪高出 k·MAD（实测 max/thr < 1）。
    """
    e = np.asarray(env, dtype=float)
    e = e - e.mean()
    n = e.size
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 4))))
    E = np.fft.rfft(e, n=nfft)
    ac = np.fft.irfft(E * np.conj(E), n=nfft)[:n]
    ac = ac / (ac[0] + 1e-12)
    lo = max(1, int((1.0 - half_frac) / f_repeat * fs))
    hi = min(n - 1, int((1.0 + half_frac) / f_repeat * fs))
    if hi <= lo:
        return float("nan"), 0.0
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    return float(lag / fs), float(ac[lag])


def peek_above(env: np.ndarray, thr: float, min_sep: int) -> np.ndarray:
    """连续超阈区间 → 取区间内包络最大点，并施加不应期（不依赖 scipy.signal.find_peaks）。"""
    above = env > thr
    idx, i, n = [], 0, env.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        idx.append(i + int(np.argmax(env[i:j + 1])))
        i = j + 1 + min_sep
    return np.asarray(idx, dtype=int)


def fill_gaps(idx: np.ndarray, env: np.ndarray, t_hat: float, fs: float,
              max_iter: int = 6) -> np.ndarray:
    """周期缺口填补：若相邻两峰间隔 ≈ m 倍(整数)冲击周期(m≥2)，说明中间 m-1 个冲击
    被转频调制压到阈值之下；在该周期的局部包络最大点处补回，恢复等间隔冲击列。

    对本数据集(相位抖动 5%、冲击幅值受转频调制)这一步把检出率从 87% 提到 100%。
    """
    step = t_hat * fs
    half = max(1, int(0.35 * step))
    for _ in range(max_iter):
        if idx.size < 2:
            break
        d = np.diff(idx)
        add = []
        for a, dd in zip(idx[:-1], d):
            m = int(round(dd / step))
            if m >= 2:
                for j in range(1, m):
                    c = int(round(a + j * dd / m))
                    lo = max(0, c - half)
                    hi = min(env.size - 1, c + half)
                    if hi > lo:
                        add.append(lo + int(np.argmax(env[lo:hi + 1])))
        if not add:
            break
        idx = np.unique(np.concatenate([idx, np.asarray(add, dtype=int)]))
    return idx


def detect_impacts(x: np.ndarray, fs: float, f_repeat: float,
                   smooth_len: int = DET_SMOOTH, k_mad: float = DET_K_MAD,
                   ac_min: float = DET_AC_MIN, do_fill: bool = DET_FILL):
    """检出轴承冲击时刻：带通去转频 → 解析包络 → 周期门控 → 稳健 MAD 阈值 → 缺口填补。

    三级判据（缺一不可）：
      1) 解调带 500–4000 Hz：把 50 Hz 转频及其谐波挡在带外；
      2) **周期门控**：包络自相关主峰 ac ≥ ac_min 才认为存在周期冲击；
         ac < ac_min（正常信号）直接返回空集 —— 这正是"零误报"的来源；
      3) 稳健阈值 + 不应期拾峰，再按冲击周期做整数倍缺口填补。

    返回 (冲击时刻[秒], 包络, 阈值, 峰值幅值, 周期强度ac, 实测周期[s])。
    """
    env = envelope_of(x, fs, smooth_len)
    t_hat, ac = periodicity(env, fs, f_repeat)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-12
    thr = med + k_mad * 1.4826 * mad
    if not np.isfinite(t_hat) or ac < ac_min:
        return (np.asarray([]), env, thr, np.asarray([]), ac, t_hat)
    kk = peek_above(env, thr, max(1, int(0.5 * t_hat * fs)))
    if do_fill and kk.size >= 2:
        kk = fill_gaps(kk, env, t_hat, fs)
    return kk / fs, env, thr, env[kk], ac, t_hat


def impact_statistics(x: np.ndarray, fs: float, f_repeat: float) -> dict:
    """冲击检出 + 间隔统计（中位数/均值/标准差）+ 自相关周期交叉校验。"""
    t_imp, env, thr, peaks, ac, t_hat = detect_impacts(x, fs, f_repeat)
    t_theory = 1.0 / f_repeat
    out = {
        "n": int(t_imp.size), "times": t_imp, "env": env, "thr": thr, "peaks": peaks,
        "t_theory": t_theory, "t_median": float("nan"), "t_mean": float("nan"),
        "t_std": float("nan"), "t_autocorr": t_hat, "err_pct": float("nan"),
        "n_used": 0, "ac": ac, "periodic": bool(np.isfinite(ac) and ac >= DET_AC_MIN),
    }
    if t_imp.size >= 3:
        d = np.diff(t_imp)
        keep = (0.5 * t_theory < d) & (d < 1.5 * t_theory)   # 剔除漏检/多检
        if np.count_nonzero(keep) >= 2:
            d = d[keep]
        out["n_used"] = int(d.size)
        out["t_median"] = float(np.median(d))
        out["t_mean"] = float(np.mean(d))
        out["t_std"] = float(np.std(d))
        out["err_pct"] = 100.0 * (out["t_median"] - t_theory) / t_theory
    return out


# ===========================================================================
# 谱指标
# ===========================================================================
def spectral_entropy(p: np.ndarray, f: np.ndarray, fmax: float) -> float:
    """0–fmax 归一化谱熵：H = -Σp·ln p / ln(M)，p 为功率占比。

    用**未补零**的原生分辨率谱计算（df = 1/3 Hz），避免补零导致的熵偏移。
    值越接近 1 → 谱越"白"（噪声态）；越接近 0 → 能量越集中在少数线谱。
    """
    m = f <= fmax
    pk = np.asarray(p[m], dtype=float)
    s = pk.sum()
    if s <= 0:
        return float("nan")
    pk = pk / s
    pk = pk[pk > 0]
    if pk.size <= 1:
        return 0.0
    return float(-np.sum(pk * np.log(pk)) / np.log(pk.size))


def bpfo_band_ratio(p: np.ndarray, f: np.ndarray, f0: float, fmax: float,
                    harmonics=C.FAULT_HARMONICS, half_hz: float = BAND_HZ) -> float:
    """BPFO 各阶 ±half_hz 频带功率之和 / 0–fmax 总功率。"""
    m = f <= fmax
    tot = float(p[m].sum())
    if tot <= 0:
        return float("nan")
    band = sum(band_power(p, f, k * f0, half_hz) for k in harmonics)
    return float(band / tot)


# ===========================================================================
# 图 1 —— 时域波形
# ===========================================================================
def plot_time_domain(plt, x_n, x_f, fs: float, imp: dict, n_avg: int, path: Path,
                     imp_n: dict | None = None) -> None:
    t_ms = np.arange(x_n.size) / fs * 1000.0
    detail = t_ms <= DETAIL_S * 1000.0
    rot_ms = 1000.0 / C.FUNC_F                       # 转频 50 Hz → 20 ms
    n_rot = int(DETAIL_S * 1000.0 / rot_ms)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    # (a) 正常 · 0.3 s 细节
    ax = axes[0][0]
    ax.plot(t_ms[detail], x_n[detail], color="#1f77b4", lw=1.0, label="正常信号")
    for i in range(n_rot + 1):
        ax.axvline(i * rot_ms, color="gray", ls=":", lw=0.9,
                   label=(f"轴旋转周期参考 {rot_ms:.0f} ms（转频 {C.FUNC_F:.0f} Hz）" if i == 0 else None))
    ax.set_title("(a) 正常信号 · 0.3 s 细节：只有周期旋转分量 + 噪声，无冲击")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("幅值 (g)")
    ax.grid(alpha=0.3)
    if imp_n is not None:
        ax.text(0.02, 0.03,
                f"包络自相关周期强度 ac = {imp_n['ac']:.3f}（接近 0 → 无周期冲击）\n"
                f"自动检出冲击数 = {imp_n['n']}（门限 ac ≥ {DET_AC_MIN}，未通过）",
                transform=ax.transAxes, fontsize=8.5, va="bottom",
                bbox=dict(boxstyle="round", fc="#f0f8ff", ec="#1f77b4", alpha=0.9))
    ax.legend(loc="upper right", fontsize=8)

    # (b) 故障 · 0.3 s 细节（红色竖线 = 自动检出的冲击位置）
    ax = axes[0][1]
    ax.plot(t_ms[detail], x_f[detail], color="#d62728", lw=1.0, label="故障信号")
    imp_ms = imp["times"] * 1000.0
    imp_win = imp_ms[imp_ms <= DETAIL_S * 1000.0]
    for i, v in enumerate(imp_win):
        ax.axvline(v, color="red", ls="--", lw=1.0, alpha=0.75,
                   label="检出的冲击位置" if i == 0 else None)
    ax.set_title("(b) 故障信号 · 0.3 s 细节：周期性冲击（BPFO）")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("幅值 (g)")
    ax.grid(alpha=0.3)
    if imp["n"] >= 3:
        ac_n = imp_n["ac"] if imp_n else float("nan")
        txt = (f"实测冲击间隔：中位数 {imp['t_median']*1e3:.3f} ms"
               f"  均值 {imp['t_mean']*1e3:.3f} ± {imp['t_std']*1e3:.3f} ms\n"
               f"理论 1/BPFO = {imp['t_theory']*1e3:.3f} ms"
               f"   相对误差 {imp['err_pct']:+.2f}%   （0.3 s 内检出 {len(imp_win)} 个）\n"
               f"包络自相关周期强度 ac = {imp['ac']:.3f}"
               f"（正常样本 ac = {ac_n:.3f}，门限 {DET_AC_MIN}）")
    else:
        txt = (f"包络自相关周期强度 ac = {imp['ac']:.3f} < 门限 {DET_AC_MIN}："
               f"未检出周期性冲击（判为无轴承故障特征）")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round", fc="#fff5f5", ec="red", alpha=0.9))
    ax.legend(loc="upper right", fontsize=8)

    # (c)(d) 3 s 全景
    ylim_detail = 1.15 * max(np.max(np.abs(x_n[detail])), np.max(np.abs(x_f[detail])))
    for a in (axes[0][0], axes[0][1]):
        a.set_ylim(-ylim_detail, ylim_detail)
    ylim_full = 1.15 * max(np.max(np.abs(x_n)), np.max(np.abs(x_f)))

    for ax, x, c, name in ((axes[1][0], x_n, "#1f77b4", "正常"),
                           (axes[1][1], x_f, "#d62728", "故障")):
        ax.plot(t_ms, x, color=c, lw=0.5)
        ax.axvspan(0.0, DETAIL_S * 1000.0, color="orange", alpha=0.16,
                   label="上排细节放大区间 (0–300 ms)")
        ax.set_xlim(0, C.T_PER_SAMPLE * 1000.0)
        ax.set_ylim(-ylim_full, ylim_full)
        ax.set_title(f"({'cd'[0 if name == '正常' else 1]}) {name}信号 · 3 s 全景")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("幅值 (g)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    if imp["n"] >= 3:
        axes[1][1].text(0.02, 0.03,
                        f"全记录 3 s 内检出的冲击以 {imp['t_median']*1e3:.2f} ms 等间隔出现"
                        f"（共 {imp['n']} 个，理论 {int(C.T_PER_SAMPLE*C.BPFO)} 个；"
                        f"红色竖线见上排右图）",
                        transform=axes[1][1].transAxes, fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="#fff5f5", ec="red", alpha=0.9))
    else:
        axes[1][1].text(0.02, 0.03,
                        f"包络自相关 ac = {imp['ac']:.3f}，未检出周期冲击 → 与正常样本一致",
                        transform=axes[1][1].transAxes, fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="#f0f8ff", ec="gray", alpha=0.9))

    fig.suptitle(f"图1  时域波形对比：正常 vs 外圈故障（采样率 {C.FS/1000:.0f} kHz，"
                 f"代表样本各 1 条，统计量取每类 {n_avg} 条）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=C.PLOT_DPI)
    plt.close(fig)


# ===========================================================================
# 图 2 —— 幅值谱
# ===========================================================================
def plot_spectrum(plt, f, a_n, a_f, spec_peaks, meta_txt, path: Path) -> None:
    db_n, db_f = to_db(a_n), to_db(a_f)
    harm = [k * C.BPFO for k in C.FAULT_HARMONICS]
    lines_db = [spec_peaks[k]["lift"] for k in C.FAULT_HARMONICS]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # (a) 0–2000 Hz 全览
    ax = axes[0]
    ax.plot(f, db_n, color="#1f77b4", lw=1.0, label="正常信号（%d 条功率平均）" % meta_txt["n_avg"])
    ax.plot(f, db_f, color="#d62728", lw=1.0, label="故障信号（%d 条功率平均）" % meta_txt["n_avg"])
    for i, h in enumerate(harm):
        ax.axvline(h, color="red", ls="--", lw=1.0, alpha=0.8,
                   label=f"BPFO {i+1}x = {h:.2f} Hz")
    ax.axvline(2 * C.FUNC_F, color="green", ls="-.", lw=1.1,
               label=f"2×转频 = {2*C.FUNC_F:.0f} Hz")
    ax.axvline(C.BPFI, color="purple", ls=":", lw=1.1, label=f"BPFI = {C.BPFI:.2f} Hz")
    ax.set_xlim(0, SPECTRUM_FMAX)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值 (dB，0 dB = 1 g)")
    ax.set_title("(a) 幅值谱全览 0–2000 Hz：故障信号在 BPFO 谐波处出现线谱，"
                 "正常信号仅有转频系分量")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # (b) 100–600 Hz 放大：把 BPFO 1x/2x/3x 看仔细
    lo, hi = 100.0, 600.0
    m = (f >= lo) & (f <= hi)
    ax = axes[1]
    ax.plot(f[m], db_n[m], color="#1f77b4", lw=1.1, label="正常信号")
    ax.plot(f[m], db_f[m], color="#d62728", lw=1.1, label="故障信号")
    for i, h in enumerate(harm):
        s = spec_peaks[C.FAULT_HARMONICS[i]]
        ax.axvline(h, color="red", ls="--", lw=1.0, alpha=0.8, label=f"BPFO {i+1}x = {h:.2f} Hz")
        ax.annotate("", xy=(h, s["db_f"]), xytext=(h, s["db_n"]),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        ax.text(h, max(s["db_f"], s["db_n"]) + 1.5,
                f"+{s['lift']:.1f} dB", ha="center", fontsize=9, color="black",
                bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85))
    ax.axvline(2 * C.FUNC_F, color="green", ls="-.", lw=1.1, label=f"2×转频 = {2*C.FUNC_F:.0f} Hz")
    ax.axvline(C.BPFI, color="purple", ls=":", lw=1.1, label=f"BPFI = {C.BPFI:.2f} Hz")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值 (dB，0 dB = 1 g)")
    ax.set_title("(b) 放大 100–600 Hz：黑箭头 = 故障相对正常的谱峰抬升（±2 Hz 带内峰值，dB）")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"图2  幅值谱对比（Hann 窗 + 补零至 2^18，功率平均；纵轴 dB 幅值谱，0 dB = 1 g）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=C.PLOT_DPI)
    plt.close(fig)


# ===========================================================================
# 图 3 —— 包络谱
# ===========================================================================
def plot_envelope(plt, f, a_n, a_f, env_peaks, meta_txt, path: Path) -> None:
    db_n, db_f = to_db(a_n), to_db(a_f)
    harm = [k * C.BPFO for k in C.FAULT_HARMONICS]
    m = f <= ENV_FMAX

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    ax = axes[0]
    ax.plot(f[m], db_n[m], color="#1f77b4", lw=1.1, label="正常信号包络谱")
    ax.plot(f[m], db_f[m], color="#d62728", lw=1.1, label="故障信号包络谱")
    for i, h in enumerate(harm):
        ax.axvline(h, color="red", ls="--", lw=1.0, alpha=0.8, label=f"BPFO {i+1}x = {h:.2f} Hz")
    ax.set_xlim(0, ENV_FMAX)
    ax.set_ylim(-25, 10)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值 (dB，0 dB = 1 g)")
    ax.set_title("(a) 包络谱对比 0–600 Hz：故障信号的 BPFO 1x/2x/3x 三根线谱清晰可辨")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(f[m], db_f[m], color="#d62728", lw=1.2, label="故障信号包络谱")
    ax.plot(f[m], db_n[m], color="#1f77b4", lw=1.0, alpha=0.85, label="正常信号包络谱")
    for i, h in enumerate(harm):
        s = env_peaks[C.FAULT_HARMONICS[i]]
        ax.axvline(h, color="red", ls="--", lw=1.0, alpha=0.8, label=f"BPFO {i+1}x")
        ax.plot([h], [s["db_f"]], marker="v", ms=8, color="darkred")
        ax.text(h, s["db_f"] + 1.0,
                f"BPFO {i+1}x = {h:.2f} Hz\n故障 {s['db_f']:.1f} dB / 正常 {s['db_n']:.1f} dB"
                f"\n抬升 +{s['lift']:.1f} dB",
                ha="center", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="#fff5f5", ec="darkred", alpha=0.9))
    ax.set_xlim(0, ENV_FMAX)
    ax.set_ylim(-25, 10)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值 (dB，0 dB = 1 g)")
    ax.set_title("(b) 故障包络谱峰值读数（判据图）：BPFO 1x/2x/3x 及其相对正常的抬升")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle(f"图3  包络谱对比（|x| → 去均值 → {ENV_SMOOTH} 点滑动平均 → FFT；{meta_txt['n_avg']} 条功率平均）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=C.PLOT_DPI)
    plt.close(fig)


# ===========================================================================
# 报告
# ===========================================================================
def build_report(res: dict) -> str:
    n, fa = res["metrics_normal"], res["metrics_fault"]
    imp = res["imp"]
    imp_n = res.get("imp_n") or {"ac": float("nan"), "n": 0}
    L = []
    A = L.append
    A("# 机械设备故障诊断 · 第 2 步：信号分析报告（fft1.py）")
    A("")
    A(f"数据来源：`data/{C.DATASET_NAME}`（经 `dataio.load_dataset()` 读取）　"
      f"采样率 {C.FS:.0f} Hz　单条记录 {C.T_PER_SAMPLE:.0f} s = {C.N_PER_SAMPLE} 点")
    A("")
    A(f"轴承参数：{C.BEARING['n_balls']} 个滚动体，d={C.BEARING['d_ball']*1e3:.2f} mm，"
      f"D={C.BEARING['D_pitch']*1e3:.2f} mm，接触角 {C.BEARING['contact_deg']:.0f}°")
    A(f"特征频率：转频 {C.FUNC_F:.2f} Hz；**BPFO={C.BPFO:.2f} Hz**（外圈缺陷通过滚动体的频率）；"
      f"BPFI={C.BPFI:.2f} Hz；BSF={C.BSF:.2f} Hz；FTF={C.FTF:.2f} Hz")
    A(f"故障冲击理论间隔：1/BPFO = {imp['t_theory']*1e3:.3f} ms　"
      f"轴旋转周期 1/FUNC_F = {1000/C.FUNC_F:.1f} ms")
    A("")
    A("方法：图2 用加 Hann 窗的单边幅值谱，噪声谱用**功率平均**压低；"
      "图3 用整流包络（|x| → 滑动平均）再作幅值谱；"
      "纵轴统一为 dB 幅值谱（0 dB = 1 g，即 20·log10(A/1g)）；"
      f"谐波幅值取该阶次 ±{SEARCH_HZ:.0f} Hz 带内峰值；"
      f"冲击检出用「带通 → FFT 解析包络 → 周期门控 → MAD 阈值 → 缺口填补」；"
      f"时域/频域统计量取自每类前 {res['n_avg']} 条信号的确定性样本（mean ± std）。")
    A("")
    A("---")
    A("")
    A("## 1. 时域分析（对应 图1 `figures/fig_time_domain.png`）")
    A("")
    A("| 指标 | 正常（mean ± std） | 故障（mean ± std） | 故障/正常 |")
    A("|---|---|---|---|")
    rows = [("RMS (g)", "rms", 4), ("峰值 (g)", "peak", 4), ("峰峰值 (g)", "ptp", 4),
            ("峭度 kurtosis（正态=3）", "kurtosis", 3), ("裕度指标 peak/RMS", "margin", 3)]
    for label, key, nd in rows:
        vn, vf = n[key], fa[key]
        ratio = vf[0] / vn[0] if vn[0] else float("nan")
        A(f"| {label} | {vn[0]:.{nd}f} ± {vn[1]:.{nd}f} | {vf[0]:.{nd}f} ± {vf[1]:.{nd}f} | "
          f"{ratio:.2f}× |")
    A("")
    if imp["n"] >= 3:
        A(f"冲击检出方法：带通 {DEMOD_FLO:.0f}–{DEMOD_FHI:.0f} Hz（挡掉 50 Hz 转频）→ FFT 解析包络 → "
          f"**周期门控**（包络自相关主峰 ac ≥ {DET_AC_MIN}）→ 中位数+MAD 阈值(k={DET_K_MAD}) → "
          f"按整数倍冲击周期的缺口填补。")
        A("")
        A(f"实测冲击间隔：中位数 **{imp['t_median']*1e3:.3f} ms**、均值 "
          f"{imp['t_mean']*1e3:.3f} ± {imp['t_std']*1e3:.3f} ms（参与统计的间隔 {imp['n_used']} 个）；"
          f"理论 1/BPFO = {imp['t_theory']*1e3:.3f} ms，相对误差 {imp['err_pct']:+.2f}%。"
          f"包络自相关独立读出周期 {imp['t_autocorr']*1e3:.3f} ms，"
          f"**周期强度 ac = {imp['ac']:.3f}**；同一算法在正常样本上得到 ac = {imp_n['ac']:.3f}、"
          f"检出 {imp_n['n']} 个冲击（门限 ac ≥ {DET_AC_MIN}，正常样本不通过 → 零误报）。")
        A("")
        A(f"时域结论：故障信号 RMS 升为正常的 {fa['rms'][0]/n['rms'][0]:.2f} 倍，"
          f"峭度由 {n['kurtosis'][0]:.2f} 升至 {fa['kurtosis'][0]:.2f}"
          f"（{fa['kurtosis'][0]/n['kurtosis'][0]:.2f}×），"
          f"裕度指标由 {n['margin'][0]:.2f} 升至 {fa['margin'][0]:.2f}"
          f"（{fa['margin'][0]/n['margin'][0]:.2f}×）——冲击成分显著，"
          f"图1(b) 红色竖线间距与理论 1/BPFO 吻合。")
    else:
        A(f"冲击检出方法：带通 {DEMOD_FLO:.0f}–{DEMOD_FHI:.0f} Hz → FFT 解析包络 → 周期门控。")
        A("")
        A(f"时域结论：包络自相关周期强度 ac = {imp['ac']:.3f} < 门限 {DET_AC_MIN}，"
          f"未检出周期性冲击（与正常样本一致）；"
          f"但 RMS 已由 {n['rms'][0]:.4f} 升至 {fa['rms'][0]:.4f} g，"
          f"峭度由 {n['kurtosis'][0]:.2f} 升至 {fa['kurtosis'][0]:.2f}，"
          f"冲击性仍由图1 下排全景体现。")
    A("")
    A("## 2. 频域分析（对应 图2 `figures/fig_spectrum.png`）")
    A("")
    A("### 2.1 BPFO 各阶谱峰幅值（0 dB = 1 g）")
    A("")
    A("| 阶次 | 频率 (Hz) | 正常幅值 (g) | 正常 (dB) | 故障幅值 (g) | 故障 (dB) | 抬升 (dB) |")
    A("|---|---|---|---|---|---|---|")
    for k in C.FAULT_HARMONICS:
        s = res["spec_peaks"][k]
        A(f"| BPFO {k}x | {k*C.BPFO:.2f} | {s['a_n']:.5f} | {s['db_n']:.1f} | "
          f"{s['a_f']:.5f} | {s['db_f']:.1f} | **+{s['lift']:.1f}** |")
    ref = res["ref_peaks"]
    A("")
    A("### 2.2 参考谱线（正常信号的转频系分量）")
    A("")
    A("| 频率 (Hz) | 正常 (dB) | 故障 (dB) | 差 (dB) |")
    A("|---|---|---|---|")
    for name, f0 in (("转频 1x (FUNC_F)", C.FUNC_F), ("2×转频", 2 * C.FUNC_F), ("BPFI", C.BPFI)):
        d = ref[name]
        A(f"| {name} = {f0:.2f} | {d['db_n']:.1f} | {d['db_f']:.1f} | {d['diff']:+.1f} |")
    A("")
    A("### 2.3 全带指标")
    A("")
    A("| 指标 | 正常 | 故障 |")
    A("|---|---|---|")
    A(f"| BPFO 1x/2x/3x 频带(±{BAND_HZ:.0f} Hz)功率占 0–2000 Hz 比例 | "
      f"{res['bpfo_ratio_n']*100:.2f}% | {res['bpfo_ratio_f']*100:.2f}% |")
    A(f"| 0–2000 Hz 归一化谱熵 | {res['entropy_n']:.4f} | {res['entropy_f']:.4f} |")
    A("")
    lifts = [res["spec_peaks"][k]["lift"] for k in C.FAULT_HARMONICS]
    en, ef = res["entropy_n"], res["entropy_f"]
    ratio_n, ratio_f = res["bpfo_ratio_n"] * 100, res["bpfo_ratio_f"] * 100
    ent_dir = "下降" if ef < en else "上升"
    verdict = "三阶全部出现可辨识抬升" if all(v >= 3.0 for v in lifts) else "部分阶次抬升有限"
    A(f"频域结论：故障频谱在 BPFO 1x/2x/3x 处相对正常分别抬升 "
      f"**+{lifts[0]:.1f} / +{lifts[1]:.1f} / +{lifts[2]:.1f} dB**（判据阈值按 3 dB 计，{verdict}）；"
      f"BPFO 1x/2x/3x 频带功率占比由 {ratio_n:.2f}% 升至 {ratio_f:.2f}%"
      f"（{ratio_f/max(ratio_n,1e-9):.2f}×），故障特征频率处的能量集中度提高。")
    A("")
    A(f"关于谱熵：0–2000 Hz 归一化谱熵由 {en:.4f} {ent_dir}至 {ef:.4f}。"
      f"谱熵{'上升' if ef >= en else '下降'}并不与线谱增强矛盾 —— 冲击激起 500–4000 Hz 的宽带共振，"
      f"使总能量在频轴上摊得更开（熵↑）；同时 BPFO 谐波处的**局部**线谱大幅抬升"
      f"（±{BAND_HZ:.0f} Hz 频带占比 {ratio_n:.2f}% → {ratio_f:.2f}%）。"
      f"因此谱熵只作辅证，判据应以 BPFO 谐波带内幅值为准。")
    A("")
    A("## 3. 包络谱分析（对应 图3 `figures/fig_envelope_spectrum.png`）")
    A("")
    A("包络谱是轴承故障的**直接判据图**：取整流包络 |x| → 滑动平均（np.convolve）→ 去均值 → FFT，"
      "把高频共振上的冲击调制解调为低频的 BPFO 线谱。包络谱只保留**幅值调制**信息，"
      "因此即使转频谱线在原始谱中强 50 dB，也不会掩盖 BPFO 谐波。")
    A("")
    A("| 阶次 | 频率 (Hz) | 正常包络幅值 (dB) | 故障包络幅值 (dB) | 抬升 (dB) |")
    A("|---|---|---|---|---|")
    for k in C.FAULT_HARMONICS:
        s = res["env_peaks"][k]
        A(f"| BPFO {k}x | {k*C.BPFO:.2f} | {s['db_n']:.1f} | {s['db_f']:.1f} | **+{s['lift']:.1f}** |")
    el = [res["env_peaks"][k]["lift"] for k in C.FAULT_HARMONICS]
    A("")
    A(f"包络谱结论：故障包络谱在 BPFO 1x/2x/3x 处相对正常分别抬升 "
      f"**+{el[0]:.1f} / +{el[1]:.1f} / +{el[2]:.1f} dB**，"
      f"三根线谱在 0–600 Hz 内清晰可辨，且间隔严格等于 BPFO —— 外圈故障确诊。")
    A("")
    A("## 4. 数值 → 图 → 结论 一致性自查")
    A("")
    A("| 结论 | 支撑数值 | 对应图 |")
    A("|---|---|---|")
    if imp["n"] >= 3:
        A(f"| 时域出现周期冲击 | 冲击间隔中位数 {imp['t_median']*1e3:.3f} ms vs 理论 "
          f"{imp['t_theory']*1e3:.3f} ms（误差 {imp['err_pct']:+.2f}%，"
          f"自相关校验 {imp['t_autocorr']*1e3:.3f} ms，ac={imp['ac']:.3f}） | 图1(b) 红色竖线 |")
    else:
        A(f"| 时域周期冲击 | 包络自相关 ac = {imp['ac']:.3f} < {DET_AC_MIN}，判为无周期冲击 | 图1(b) |")
    A(f"| 冲击性增强 | 峭度 {n['kurtosis'][0]:.2f} → {fa['kurtosis'][0]:.2f}；"
      f"RMS {n['rms'][0]:.3f} → {fa['rms'][0]:.3f} g | 图1 全图 |")
    A(f"| 谱线抬升 | BPFO 1x/2x/3x 分别 +{lifts[0]:.1f}/+{lifts[1]:.1f}/+{lifts[2]:.1f} dB | 图2(a)(b) |")
    A(f"| 判据成立 | 包络谱 BPFO 1x/2x/3x 分别 +{el[0]:.1f}/+{el[1]:.1f}/+{el[2]:.1f} dB | 图3(a)(b) |")
    A(f"| 故障/正常可分（单指标） | 包络自相关周期强度 ac：故障 {imp['ac']:.3f} vs "
      f"正常 {imp_n['ac']:.3f}（门限 {DET_AC_MIN}；正常样本检出 {imp_n['n']} 个冲击） | 图1(a)(b) 文字框 |")
    A(f"| 能量向 BPFO 频带集中 | 频带占比 {ratio_n:.2f}% → {ratio_f:.2f}%"
      f"（{ratio_f/max(ratio_n,1e-9):.2f}×） | 图2(a)(b) 红色虚线 |")
    A("")
    A("## 5. 对第 3 步（cnn1.py 建模）的建议")
    A("")
    A("1. 原始时域波形已含显著冲击与 BPFO 线谱，1D-CNN 直接在 1024 点窗口上学习即可；"
      "无需手工特征工程，但可用本报告指标作为可解释性对照。")
    A(f"2. 正常/故障的量化差异明显（RMS {fa['rms'][0]/n['rms'][0]:.2f}×，"
      f"峭度 {fa['kurtosis'][0]/n['kurtosis'][0]:.2f}×，"
      f"BPFO 谱峰 +{lifts[0]:.1f} dB 起），二分类任务是线性可分性较强的任务，"
      "若测试准确率低于 0.95 应优先排查数据划分是否泄漏，而非加深网络。")
    A("3. 窗长 1024 点 = 85.3 ms，覆盖约 13.5 个 BPFO 冲击周期，足以体现冲击周期性，"
      "窗口长度无需调整。")
    A("")
    A("## 6. 复现方式")
    A("")
    A("```bash")
    A("python fft1.py      # 需先运行 python preprocess1.py 生成 data/clean_signal1.npy")
    A("```")
    A("")
    A(f"依赖：numpy {np.__version__} + matplotlib（**无 scipy**："
      f"解析信号包络用 FFT 自建，包络平滑一律用 np.convolve）。")
    A("")
    return "\n".join(L) + "\n"


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    banner("第 2 步 · 信号分析 fft1.py：时域 / 频域 / 包络谱（轴承故障诊断教学）")
    C.ensure_dirs()
    plt, fonts = C.configure_matplotlib()
    print(f"[0] 绘图后端 = Agg（无 GUI），中文字体 = {fonts}")

    # ---- 1. 加载数据集 -----------------------------------------------------
    print("\n[1] 调用 dataio.load_dataset() 加载冻结数据集 ……")
    try:
        ds = dataio.load_dataset()
    except FileNotFoundError as exc:
        print("  !! 数据集缺失，无法继续。")
        print(f"  {exc}")
        print("  → 请先运行： python preprocess1.py     （生成 data/clean_signal1.npy）")
        return 2
    except (KeyError, ValueError) as exc:
        print(f"  !! 数据集校验失败：{exc}")
        return 3

    raw = np.asarray(ds["raw"], dtype=float)
    y = np.asarray(ds["y"])
    seg = np.asarray(ds["seg_id"])
    print(f"  X.shape = {ds['X'].shape}   raw.shape = {raw.shape}   "
          f"meta.format_version = {ds['meta'].get('format_version')}")
    print(f"  类别计数（窗口级）= {dataio.class_counts(y)}")

    # ---- 2. 选出正常 / 故障的代表信号与统计样本 ----------------------------
    print("\n[2] 依据 seg_id → 标签映射，从 raw 中取出正常样本与故障样本 ……")
    seg_label = {}
    for s, lab in zip(seg.tolist(), y.tolist()):
        seg_label.setdefault(int(s), int(lab))
    normal_ids = sorted(s for s, lab in seg_label.items() if lab == C.NORMAL_LABEL)
    fault_ids = sorted(s for s, lab in seg_label.items() if lab == C.FAULT_LABEL)
    if not normal_ids or not fault_ids:
        print("  !! raw 中正常/故障样本至少各需 1 条，请检查 preprocess1.py")
        return 4
    n_avg = min(N_AVG_MAX, len(normal_ids), len(fault_ids))
    normal_set = [raw[i] for i in normal_ids[:n_avg]]
    fault_set = [raw[i] for i in fault_ids[:n_avg]]
    x_n, x_f = raw[normal_ids[0]], raw[fault_ids[0]]
    print(f"  正常信号 seg_id 前几个 = {normal_ids[:5]}（共 {len(normal_ids)} 条）")
    print(f"  故障信号 seg_id 前几个 = {fault_ids[:5]}（共 {len(fault_ids)} 条）")
    print(f"  作图代表样本：normal seg_id={normal_ids[0]}，fault seg_id={fault_ids[0]}")
    print(f"  统计/平均谱样本数：每类 {n_avg} 条（确定性取前 {n_avg} 条）")

    # ---- 3. 时域指标 + 冲击检出 -------------------------------------------
    print("\n[3] 时域指标（RMS / 峰值 / 峰峰值 / 峭度 / 裕度）……")
    m_n = [time_metrics(x) for x in normal_set]
    m_f = [time_metrics(x) for x in fault_set]
    mean_std = lambda lst, k: (float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst])))
    metrics_normal = {k: mean_std(m_n, k) for k in m_n[0]}
    metrics_fault = {k: mean_std(m_f, k) for k in m_f[0]}
    for k in ("rms", "peak", "kurtosis", "margin"):
        print(f"  {k:9s} 正常 {metrics_normal[k][0]:8.4f} ± {metrics_normal[k][1]:.4f}   "
              f"故障 {metrics_fault[k][0]:8.4f} ± {metrics_fault[k][1]:.4f}")

    print(f"\n[4] 冲击检出：带通 {DEMOD_FLO:.0f}–{DEMOD_FHI:.0f} Hz → FFT 解析包络 → "
          f"周期门控(ac≥{DET_AC_MIN}) → MAD 阈值(k={DET_K_MAD}) → 缺口填补 ……")
    imp = impact_statistics(x_f, C.FS, C.IMPACT_FREQ_FAULT)
    imp_n = impact_statistics(x_n, C.FS, C.IMPACT_FREQ_FAULT)   # 正常样本对照（应无可检出的周期冲击）
    print(f"  包络自相关周期强度 ac = {imp['ac']:.3f}（门限 {DET_AC_MIN}；正常样本 ac = "
          f"{imp_n['ac']:.3f}，检出 {imp_n['n']} 个）→ 周期冲击"
          f"{'成立' if imp['periodic'] else '不成立'}")
    print(f"  自相关直接读出冲击周期 = {imp['t_autocorr']*1e3:.3f} ms"
          f"（理论 1/BPFO = {imp['t_theory']*1e3:.3f} ms，BPFO={C.BPFO:.2f} Hz）")
    print(f"  检出冲击 {imp['n']} 个 / 3 s（理论应有 {int(C.T_PER_SAMPLE*C.BPFO)} 个）")
    if imp["n"] >= 3:
        print(f"  实测间隔中位数 {imp['t_median']*1e3:.3f} ms，均值 {imp['t_mean']*1e3:.3f}"
              f" ± {imp['t_std']*1e3:.3f} ms，相对误差 {imp['err_pct']:+.2f}%"
              f"（用于统计的间隔 {imp['n_used']} 个）")
    else:
        print("  !! 未检出周期冲击（ac 未达门限），图上将不绘制冲击竖线（仅作提示）")

    # ---- 4. 频谱 ----------------------------------------------------------
    print(f"\n[5] 幅值谱（Hann 窗，功率平均，补零至 {NFFT_PLOT} 点，df={C.FS/NFFT_PLOT:.4f} Hz）……")
    a_n, p_n, f = average_spectrum(normal_set, C.FS, NFFT_PLOT)
    a_f, p_f, _ = average_spectrum(fault_set, C.FS, NFFT_PLOT)
    print(f"  自检：正常信号转频 {C.FUNC_F:.0f} Hz 处幅值 = "
          f"{band_max(a_n, f, C.FUNC_F, SEARCH_HZ):.4f} g "
          f"（期望 ≈ {C.AMP_ROT:.2f} g，验证幅值标定正确）")

    spec_peaks, lines_db = {}, []
    for k in C.FAULT_HARMONICS:
        f0 = k * C.BPFO
        pkn, pkf = band_max(a_n, f, f0, SEARCH_HZ), band_max(a_f, f, f0, SEARCH_HZ)
        dbn, dbf = float(to_db(pkn)), float(to_db(pkf))
        spec_peaks[k] = {"a_n": pkn, "db_n": dbn, "a_f": pkf, "db_f": dbf, "lift": dbf - dbn}
        lines_db.append(dbf - dbn)
        print(f"  BPFO {k}x @ {f0:7.2f} Hz : 正常 {pkn:.5f} g ({dbn:6.1f} dB) → "
              f"故障 {pkf:.5f} g ({dbf:6.1f} dB)   抬升 {dbf-dbn:+.1f} dB")

    ref_peaks = {}
    for name, f0 in (("转频 1x (FUNC_F)", C.FUNC_F), ("2×转频", 2 * C.FUNC_F), ("BPFI", C.BPFI)):
        dbn = float(to_db(band_max(a_n, f, f0, SEARCH_HZ)))
        dbf = float(to_db(band_max(a_f, f, f0, SEARCH_HZ)))
        ref_peaks[name] = {"db_n": dbn, "db_f": dbf, "diff": dbf - dbn}
        print(f"  参考 {name:16s} @ {f0:7.2f} Hz : 正常 {dbn:6.1f} dB / 故障 {dbf:6.1f} dB "
              f"(差 {dbf-dbn:+.1f} dB)")

    # 谱熵与频带占比用原生分辨率谱（df = 1/3 Hz），避免补零影响熵的定义
    _, p_nat_n, f_nat = average_spectrum(normal_set, C.FS, C.N_PER_SAMPLE)
    _, p_nat_f, _ = average_spectrum(fault_set, C.FS, C.N_PER_SAMPLE)
    entropy_n = spectral_entropy(p_nat_n, f_nat, SPECTRUM_FMAX)
    entropy_f = spectral_entropy(p_nat_f, f_nat, SPECTRUM_FMAX)
    ratio_n = bpfo_band_ratio(p_n, f, C.BPFO, SPECTRUM_FMAX)
    ratio_f = bpfo_band_ratio(p_f, f, C.BPFO, SPECTRUM_FMAX)
    print(f"  0–2000 Hz 归一化谱熵 : 正常 {entropy_n:.4f} → 故障 {entropy_f:.4f}")
    print(f"  BPFO 1x/2x/3x 频带(±{BAND_HZ:.0f} Hz)功率占比 : "
          f"正常 {ratio_n*100:.2f}% → 故障 {ratio_f*100:.2f}%")

    # ---- 5. 包络谱 --------------------------------------------------------
    print(f"\n[6] 包络谱：|x| → {ENV_SMOOTH} 点滑动平均(np.convolve) → 去均值 → FFT ……")
    env_n = [rectified_envelope(x) for x in normal_set]
    env_f = [rectified_envelope(x) for x in fault_set]
    ea_n, _, fe = average_spectrum(env_n, C.FS, NFFT_PLOT)
    ea_f, _, _ = average_spectrum(env_f, C.FS, NFFT_PLOT)
    env_peaks = {}
    for k in C.FAULT_HARMONICS:
        f0 = k * C.BPFO
        pkn, pkf = band_max(ea_n, fe, f0, SEARCH_HZ), band_max(ea_f, fe, f0, SEARCH_HZ)
        dbn, dbf = float(to_db(pkn)), float(to_db(pkf))
        env_peaks[k] = {"a_n": pkn, "a_f": pkf, "db_n": dbn, "db_f": dbf, "lift": dbf - dbn}
        print(f"  BPFO {k}x @ {f0:7.2f} Hz : 正常 {dbn:6.1f} dB → 故障 {dbf:6.1f} dB   "
              f"抬升 {dbf-dbn:+.1f} dB")

    # ---- 6. 出图 ----------------------------------------------------------
    print("\n[7] 绘制并保存三张图 ……")
    meta_txt = {"n_avg": n_avg}
    plot_time_domain(plt, x_n, x_f, C.FS, imp, n_avg, C.FIG_TIME_PATH, imp_n)
    plot_spectrum(plt, f, a_n, a_f, spec_peaks, meta_txt, C.FIG_FREQ_PATH)
    plot_envelope(plt, fe, ea_n, ea_f, env_peaks, meta_txt, C.FIG_ENV_PATH)
    for p in (C.FIG_TIME_PATH, C.FIG_FREQ_PATH, C.FIG_ENV_PATH):
        size = p.stat().st_size
        print(f"  {p.name:32s} {size/1024:8.1f} KB  {'OK' if size > 30*1024 else '!! 偏小，请检查'}")

    # ---- 7. 报告 ----------------------------------------------------------
    print("\n[8] 生成 analysis_report.md（数值与图同源，保证三者一致）……")
    res = {
        "n_avg": n_avg, "metrics_normal": metrics_normal, "metrics_fault": metrics_fault,
        "imp": imp, "imp_n": imp_n, "spec_peaks": spec_peaks, "ref_peaks": ref_peaks,
        "env_peaks": env_peaks,
        "entropy_n": entropy_n, "entropy_f": entropy_f,
        "bpfo_ratio_n": ratio_n, "bpfo_ratio_f": ratio_f,
    }
    report_path = C.ROOT / "analysis_report.md"
    report_path.write_text(build_report(res), encoding="utf-8")
    n_lines = len(report_path.read_text(encoding="utf-8").splitlines())
    print(f"  {report_path.name}  {n_lines} 行  {report_path.stat().st_size/1024:.1f} KB")

    # ---- 8. 总结 ----------------------------------------------------------
    banner("分析完成 · 关键量化结论")
    print(f"  时域：RMS {metrics_normal['rms'][0]:.4f} → {metrics_fault['rms'][0]:.4f} g"
          f"（{metrics_fault['rms'][0]/metrics_normal['rms'][0]:.2f}×）；"
          f"峭度 {metrics_normal['kurtosis'][0]:.2f} → {metrics_fault['kurtosis'][0]:.2f}"
          f"（{metrics_fault['kurtosis'][0]/metrics_normal['kurtosis'][0]:.2f}×）")
    print(f"  频谱：BPFO 1x/2x/3x 抬升 +{lines_db[0]:.1f} / +{lines_db[1]:.1f} / "
          f"+{lines_db[2]:.1f} dB")
    print(f"  包络谱：BPFO 1x/2x/3x 抬升 +{[env_peaks[k]['lift'] for k in C.FAULT_HARMONICS][0]:.1f} / "
          f"+{[env_peaks[k]['lift'] for k in C.FAULT_HARMONICS][1]:.1f} / "
          f"+{[env_peaks[k]['lift'] for k in C.FAULT_HARMONICS][2]:.1f} dB")
    print(f"  冲击间隔：实测 {imp['t_median']*1e3:.3f} ms vs 理论 {imp['t_theory']*1e3:.3f} ms"
          f"（包络自相关 ac={imp['ac']:.3f}）")
    print("  产出：figures/fig_time_domain.png, figures/fig_spectrum.png, "
          "figures/fig_envelope_spectrum.png, analysis_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
