"""
数据 IO 契约 —— 所有脚本读写数据集/指标的唯一入口
=================================================
冻结文件：任何子任务不得修改本文件语义；需要新增字段时在 PLAN.md 里登记。

数据集格式 (data/clean_signal1.npy)
-----------------------------------
np.save 出来的**字典**(0-d object array), 键为:

    X      : float32 ndarray, shape (N, WINDOW)      # 滑动窗口截取后的片段
    y      : int64   ndarray, shape (N,)             # 0=正常 1=故障
    seg_id : int64   ndarray, shape (N,)             # 该片段来自第几条原始信号
    raw    : float32 ndarray, shape (2*N_SAMPLES_PER_CLASS, N_PER_SAMPLE)
                                                      # 原始长信号（供时域/频域分析）
    meta   : dict    # 版本、参数快照、计数，见 build_meta()

读取统一用 ``load_dataset()``，它同时兼容 npz / npy(字典) 两种容器，
并做形状、标签、NaN 的自检（验证拒收，而不是 raise 到调用方看不懂的地方）。

防泄漏约定
----------
X 的划分**不得**在窗口级别随机进行。必须按 seg_id 先切分原始信号，
再收集对应窗口，见 ``split_by_segment()``。dataio 不做训练集标准化；
标准化统计量必须只从训练集拟合（见 cnn1.py）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import config as C

DATASET_FORMAT_VERSION = "1.0"


# --------------------------------------------------------------------------
# 构造 / 保存
# --------------------------------------------------------------------------
def build_meta(extra: Dict | None = None) -> Dict:
    """生成写入数据集的自描述元信息（可追溯：谁、用什么参数、生成了什么）。"""
    meta = {
        "format_version": DATASET_FORMAT_VERSION,
        "fs": C.FS,
        "t_per_sample": C.T_PER_SAMPLE,
        "n_per_sample": C.N_PER_SAMPLE,
        "window": C.WINDOW,
        "stride": C.STRIDE,
        "n_samples_per_class": C.N_SAMPLES_PER_CLASS,
        "func_f": C.FUNC_F,
        "bpfo": C.BPFO,
        "bpfi": C.BPFI,
        "bsf": C.BSF,
        "ftf": C.FTF,
        "impact_freq_fault": C.IMPACT_FREQ_FAULT,
        "label_map": {"0": "normal", "1": "fault"},
        "seed": C.SEED,
    }
    if extra:
        meta.update(extra)
    return meta


def save_dataset(path, X, y, seg_id, raw, meta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "seg_id": np.asarray(seg_id, dtype=np.int64),
        "raw": np.asarray(raw, dtype=np.float32),
        "meta": meta,
    }
    validate(payload)
    with path.open("wb") as fh:
        np.save(fh, payload, allow_pickle=True)
    return path


# --------------------------------------------------------------------------
# 读取 / 校验
# --------------------------------------------------------------------------
def load_dataset(path=None) -> Dict:
    path = Path(path or C.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {path}\n请先运行: python preprocess1.py"
        )
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            payload = {k: npz[k] for k in npz.files}
    else:
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object:
            payload = payload.item()
    for key in ("X", "y", "seg_id", "raw", "meta"):
        if key not in payload:
            raise KeyError(
                f"数据集缺少字段 '{key}'；实际字段: {sorted(payload.keys())}"
                "（请确认由 preprocess1.py 生成，勿手工拼接）"
            )
    validate(payload)
    return payload


def validate(payload: Dict) -> None:
    X = np.asarray(payload["X"])
    y = np.asarray(payload["y"])
    seg = np.asarray(payload["seg_id"])
    raw = np.asarray(payload["raw"])

    if X.ndim != 2:
        raise ValueError(f"X 必须是二维 (N, WINDOW)，实际 {X.shape}")
    if X.shape[1] != C.WINDOW:
        raise ValueError(f"X 第二维应为 WINDOW={C.WINDOW}，实际 {X.shape[1]}")
    if not (X.shape[0] == y.shape[0] == seg.shape[0]):
        raise ValueError(f"X/y/seg_id 样本数不一致: {X.shape[0]}, {y.shape[0]}, {seg.shape[0]}")
    if raw.ndim != 2 or raw.shape[1] != C.N_PER_SAMPLE:
        raise ValueError(f"raw 形状应为 (M, {C.N_PER_SAMPLE})，实际 {raw.shape}")
    if not np.isfinite(X).all() or not np.isfinite(raw).all():
        raise ValueError("数据集中存在 NaN/Inf，请检查 preprocess1.py 的信号生成")
    labels = set(np.unique(y).tolist())
    if not labels <= {C.NORMAL_LABEL, C.FAULT_LABEL}:
        raise ValueError(f"标签只能是 {{0,1}}，实际出现 {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError(f"数据集里只有一类标签 {labels}，无法训练二分类器")
    if set(np.unique(seg).tolist()) != set(range(raw.shape[0])):
        raise ValueError("seg_id 必须覆盖 raw 的每一条信号（0..M-1）")


# --------------------------------------------------------------------------
# 划分（防窗口泄漏）
# --------------------------------------------------------------------------
def split_by_segment(X, y, seg_id, train_ratio=None, seed=None) -> Tuple[np.ndarray, ...]:
    """
    先按 seg_id 分组划分原始信号，再取窗口。**同一条信号的窗口只出现在一侧**，
    避免相邻重叠窗口同时进训练集和测试集造成的虚高准确率。

    返回: X_tr, y_tr, X_te, y_te
    """
    train_ratio = C.TRAIN_RATIO if train_ratio is None else train_ratio
    seed = C.SEED if seed is None else seed

    X = np.asarray(X)
    y = np.asarray(y)
    seg = np.asarray(seg_id)

    rng = np.random.default_rng(seed)
    unique_segs = np.unique(seg)
    rng.shuffle(unique_segs)
    n_train = int(round(len(unique_segs) * train_ratio))
    train_segs = set(unique_segs[:n_train].tolist())
    test_segs = set(unique_segs[n_train:].tolist())

    m_tr = np.array([s in train_segs for s in seg])
    m_te = ~m_tr
    if m_tr.sum() == 0 or m_te.sum() == 0:
        raise ValueError("划分后某一侧为空，请检查 TRAIN_RATIO 或 seg_id 数量")

    # 两侧都必须同时含两类，否则准确率无意义
    for name, mask in (("训练集", m_tr), ("测试集", m_te)):
        if len(set(y[mask].tolist())) < 2:
            raise ValueError(f"{name}只包含单一类别，请增大 N_SAMPLES_PER_CLASS 或调整划分")

    return X[m_tr], y[m_tr], X[m_te], y[m_te]


def class_counts(y) -> Dict[str, int]:
    y = np.asarray(y)
    return {
        "total": int(y.size),
        C.CLASS_NAMES[0]: int((y == C.NORMAL_LABEL).sum()),
        C.CLASS_NAMES[1]: int((y == C.FAULT_LABEL).sum()),
    }


# --------------------------------------------------------------------------
# 指标落盘
# --------------------------------------------------------------------------
def save_metrics(metrics: Dict, path=None) -> Path:
    path = Path(path or C.RESULT_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_metrics(path=None) -> Dict:
    path = Path(path or C.RESULT_JSON)
    if not path.exists():
        raise FileNotFoundError(f"指标文件不存在: {path}，请先运行 cnn1.py")
    return json.loads(path.read_text(encoding="utf-8"))
