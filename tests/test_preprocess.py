"""
第 1 步自检：tests/test_preprocess.py
======================================
针对 preprocess1.py 的回归测试。两种跑法都必须全绿：

    python tests/test_preprocess.py     # 直接跑（无 pytest 依赖）
    python -m pytest tests/test_preprocess.py

覆盖点（对应任务验收判据）：
  1. 冻结契约参数自洽（BPFO/窗口数/路径常量）
  2. 数据集文件存在且 load_dataset() 通过 validate()
  3. X 形状 (N, WINDOW)、样本数 == 800×137、dtype、标签/seg_id 覆盖
  4. 滑动窗口的边界与内容正确（用可手算的小例子对照）
  5. 防泄漏划分：两侧 seg_id 无交集、两侧都含两类、比例约 0.8
  6. 非退化：故障 RMS/峰值 > 正常
  7. 冲击重复频率 == BPFO（包络自相关峰）
  8. 共振频率条内固定（劈半谱峰一致）+ 反例对照确实被判否
  9. 可复现：种子派生逐位一致；整条流水线重跑一次的文件 sha256 与磁盘产物一致

注意：本文件只读取 data/clean_signal1.npy，不修改任何产物。
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# 让测试可从任意工作目录运行：把项目根目录加入 sys.path
PROJ = Path(__file__).resolve().parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import config as C          # noqa: E402
import dataio               # noqa: E402
import preprocess1 as P     # noqa: E402


# ==========================================================================
# 小工具
# ==========================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_segment(kind: str, index: int) -> np.ndarray:
    """按 preprocess1 约定的派生种子重新生成第 index 条信号（与落盘逻辑同源）。"""
    offset = P.SEED_OFFSET_NORMAL if kind == "normal" else P.SEED_OFFSET_FAULT
    rng = np.random.default_rng(C.SEED + offset + index)
    if kind == "normal":
        return P.synth_normal(rng).astype(np.float32)
    return P.synth_fault(rng)[0].astype(np.float32)


def require_dataset() -> dict:
    """加载并缓存数据集；缺失时给出可执行的修复提示。"""
    try:
        return dataio.load_dataset()
    except FileNotFoundError as exc:  # pragma: no cover - 正常流程不会走到
        raise AssertionError(
            f"数据集缺失，请先运行 preprocess1.py 生成数据: {exc}"
        ) from exc


# ==========================================================================
# 1. 契约参数
# ==========================================================================
def test_config_contract_self_consistent() -> None:
    """config.py 的参数必须自洽：派生频率与几何公式一致、窗口数可推导。"""
    # 外圈故障特征频率 BPFO = (n/2)·(1 - d/D·cosθ)·f_r，f_r = FS/60? 这里 f_r=FUNC_F
    n, d, D, ang = (
        C.BEARING["n_balls"], C.BEARING["d_ball"],
        C.BEARING["D_pitch"], np.deg2rad(C.BEARING["contact_deg"]),
    )
    bpfo_formula = 0.5 * n * (1.0 - d / D * np.cos(ang)) * C.FUNC_F
    assert abs(bpfo_formula - C.BPFO) < 0.01, f"BPFO 与几何公式不符: {bpfo_formula} vs {C.BPFO}"
    assert C.N_PER_SAMPLE == int(C.FS * C.T_PER_SAMPLE) == 36000
    assert P.n_windows_per_segment() == 137, P.n_windows_per_segment()
    assert 1 + (C.N_PER_SAMPLE - C.WINDOW) // C.STRIDE == 137
    assert C.TRAIN_RATIO == 0.8
    assert C.IMPACT_FREQ_FAULT == C.BPFO
    assert C.CLASS_NAMES == ("normal", "fault")


def test_data_paths_are_under_project_root() -> None:
    """路径常量必须落在项目目录内（避免往仓库外写文件）。"""
    root = Path(C.ROOT).resolve()
    for p in (C.DIR_DATA, C.DIR_FIG, C.DIR_RESULT):
        assert Path(p).resolve().is_relative_to(root), p
    assert Path(C.DATA_PATH).name == "clean_signal1.npy"
    assert Path(C.FIG_CACHE_PATH).name == "fig_preprocess_cache.png"


# ==========================================================================
# 2/3. 数据集存在性 + 契约校验
# ==========================================================================
def test_dataset_exists_and_validates() -> None:
    """load_dataset() 必须通过 validate()（形状/标签/NaN/seg_id 全覆盖）。"""
    path = Path(C.DATA_PATH)
    assert path.exists(), f"数据集不存在: {path}（请先运行 preprocess1.py）"
    payload = require_dataset()
    dataio.validate(payload)          # 不抛异常即通过
    for key in ("X", "y", "seg_id", "raw", "meta"):
        assert key in payload, f"缺少字段 {key}"


def test_dataset_shapes_dtypes_and_counts() -> None:
    """X 形状/样本数/标签分布/seg_id 覆盖率全部对齐契约。"""
    payload = require_dataset()
    X, y, seg, raw = payload["X"], payload["y"], payload["seg_id"], payload["raw"]

    n_win = P.n_windows_per_segment()
    n_total = 2 * C.N_SAMPLES_PER_CLASS

    assert X.ndim == 2 and X.shape[1] == C.WINDOW == 1024
    assert X.shape == (n_total * n_win, C.WINDOW) == (109600, 1024), X.shape
    assert X.dtype == np.float32 and raw.dtype == np.float32
    assert y.dtype == np.int64 and seg.dtype == np.int64
    assert raw.shape == (n_total, C.N_PER_SAMPLE) == (800, 36000), raw.shape
    assert y.shape == seg.shape == (X.shape[0],)

    counts = dataio.class_counts(y)
    assert counts["total"] == 109600
    assert counts["normal"] == 54800 and counts["fault"] == 54800, counts
    assert set(np.unique(y).tolist()) == {C.NORMAL_LABEL, C.FAULT_LABEL}
    assert set(np.unique(seg).tolist()) == set(range(n_total))
    assert np.isfinite(X).all() and np.isfinite(raw).all()


def test_segment_layout_labels_are_consistent() -> None:
    """raw 前 400 条为正常、后 400 条为故障；窗口标签必须与其来源信号一致。"""
    payload = require_dataset()
    y, seg = payload["y"], payload["seg_id"]
    n_win = P.n_windows_per_segment()
    n_cls = C.N_SAMPLES_PER_CLASS

    assert (y[seg < n_cls] == C.NORMAL_LABEL).all(), "正常条目的窗口标签应为 0"
    assert (y[seg >= n_cls] == C.FAULT_LABEL).all(), "故障条目的窗口标签应为 1"
    # 每条信号恰好贡献 n_win 个连续窗口，且是第 0 条、第 400 条边界
    for s in (0, n_cls - 1, n_cls, 2 * n_cls - 1):
        idx = np.flatnonzero(seg == s)
        assert idx.size == n_win and idx[0] == s * n_win, (s, idx[:3], idx.size)


# ==========================================================================
# 4. 滑窗正确性（独立小例子，不依赖大数据集）
# ==========================================================================
def test_sliding_window_matches_manual_extraction() -> None:
    """滑窗结果必须等于逐步手工截取的结果，且相邻窗口按 STRIDE 平移。"""
    rng = np.random.default_rng(1234)
    # 构造 2 条"正常" + 2 条"故障"的假 raw，覆盖两种标签分支
    # （真实布局里 index < N_SAMPLES_PER_CLASS 为正常，故这里用假 raw 直接验证窗口逻辑，
    #   标签规则单独按同一约定断言）
    m = 4
    n_win = P.n_windows_per_segment()
    raw = rng.normal(size=(m, C.N_PER_SAMPLE)).astype(np.float32)

    X, y, seg = P.sliding_window(raw)
    assert X.shape == (m * n_win, C.WINDOW)
    assert X.dtype == np.float32

    for s in range(m):
        for j in (0, 1, 7, n_win - 1):
            expect = raw[s, j * C.STRIDE: j * C.STRIDE + C.WINDOW]
            got = X[s * n_win + j]
            assert np.array_equal(expect, got), f"seg={s} win={j} 与手工截取不一致"
        # 相邻窗口应恰好相差一个 STRIDE
        a, b = X[s * n_win], X[s * n_win + 1]
        assert np.array_equal(a[C.STRIDE:], b[:C.WINDOW - C.STRIDE])
    assert np.array_equal(seg, np.repeat(np.arange(m), n_win))

    # 标签规则：第 s 条信号（s < N_SAMPLES_PER_CLASS）应为正常(0)，否则故障(1)
    for s in range(m):
        expect_label = 0 if s < C.N_SAMPLES_PER_CLASS else 1
        assert (y[s * n_win:(s + 1) * n_win] == expect_label).all(), f"seg={s} 标签错误"
    assert y.dtype == np.int64


# ==========================================================================
# 5. 防泄漏划分
# ==========================================================================
def test_split_by_segment_no_leakage_and_balanced_classes() -> None:
    """按 seg_id 划分：两侧无重叠信号、两侧都含两类、比例接近 TRAIN_RATIO。"""
    payload = require_dataset()
    X, y, seg = payload["X"], payload["y"], payload["seg_id"]

    X_tr, y_tr, X_te, y_te = dataio.split_by_segment(X, y, seg)

    assert X_tr.shape[1] == X_te.shape[1] == C.WINDOW
    assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]

    # 训练集窗口数 / 总窗口数 ≈ TRAIN_RATIO（按信号分组，故非严格整除）
    assert abs(X_tr.shape[0] / X.shape[0] - C.TRAIN_RATIO) < 0.01, X_tr.shape[0] / X.shape[0]

    # 两侧都含两类
    assert set(np.unique(y_tr).tolist()) == {0, 1}, np.unique(y_tr)
    assert set(np.unique(y_te).tolist()) == {0, 1}, np.unique(y_te)
    assert dataio.class_counts(y_tr)["total"] == X_tr.shape[0]
    assert dataio.class_counts(y_te)["total"] == X_te.shape[0]

    # 防泄漏：独立重算一遍划分掩码（按文档算法），再逐块比对返回的窗口。
    # 若两侧共用任何 seg_id，X_tr 必然与 X[m_tr] 不一致。
    rng = np.random.default_rng(C.SEED)
    unique_segs = np.unique(seg)
    rng.shuffle(unique_segs)
    n_train_seg = int(round(len(unique_segs) * C.TRAIN_RATIO))
    train_segs = set(unique_segs[:n_train_seg].tolist())
    test_segs = set(unique_segs[n_train_seg:].tolist())
    assert not (train_segs & test_segs), "划分本身两侧 seg_id 重叠 = 数据泄漏"

    m_tr = np.array([s in train_segs for s in seg])
    m_te = ~m_tr
    assert int(m_tr.sum()) == X_tr.shape[0], (int(m_tr.sum()), X_tr.shape[0])
    assert int(m_te.sum()) == X_te.shape[0]

    # 分块比对，避免一次性再复制 ~360 MB
    step = 20000
    for mask, Xs in ((m_tr, X_tr), (m_te, X_te)):
        idx = np.flatnonzero(mask)
        for a in range(0, idx.size, step):
            sl = idx[a:a + step]
            assert np.array_equal(X[sl], Xs[a:a + sl.size]), "划分结果与按 seg_id 分组不一致"

    # 两侧窗口的来源信号必须分属两份互斥集合
    seg_tr = set(np.unique(seg[m_tr]).tolist())
    seg_te = set(np.unique(seg[m_te]).tolist())
    assert seg_tr == train_segs and seg_te == test_segs, "返回窗口的来源信号集合与划分不符"
    assert not (seg_tr & seg_te), "同一 seg_id 的窗口同时出现在训练与测试 = 数据泄漏"

    # 划分可复现：同种子两次结果一致
    a = dataio.split_by_segment(X, y, seg)
    b = dataio.split_by_segment(X, y, seg)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)


# ==========================================================================
# 6. 非退化
# ==========================================================================
def test_classes_are_not_degenerate() -> None:
    """故障信号必须因为冲击而比正常信号更"尖"（RMS/峰值更大），否则数据无信息量。"""
    payload = require_dataset()
    raw = payload["raw"]
    n_cls = C.N_SAMPLES_PER_CLASS

    rms_n = float(np.mean(np.std(raw[:n_cls], axis=1)))
    rms_f = float(np.mean(np.std(raw[n_cls:], axis=1)))
    pk_n = float(np.mean(np.max(np.abs(raw[:n_cls]), axis=1)))
    pk_f = float(np.mean(np.max(np.abs(raw[n_cls:]), axis=1)))

    assert rms_f > rms_n, f"故障 RMS 未大于正常: {rms_f:.4f} vs {rms_n:.4f}"
    assert pk_f > pk_n, f"故障峰值未大于正常: {pk_f:.4f} vs {pk_n:.4f}"
    # 冲击是"瞬态"而非整体抬升：峰值比应明显大于 RMS 比
    assert pk_f / pk_n > rms_f / rms_n, "故障差异不像冲击瞬态"


# ==========================================================================
# 7. 冲击重复频率 == BPFO
# ==========================================================================
def test_impact_frequency_equals_bpfo() -> None:
    """包络自相关峰应落在 1/BPFO，且正常信号在同一窗口内无峰（对照）。"""
    payload = require_dataset()
    raw = payload["raw"]
    n_cls = C.N_SAMPLES_PER_CLASS

    # 抽查若干条以保证稳定（全量 800 条留给 preprocess1.py 自己打印）
    idx = [0, 1, 2, 137, n_cls - 1]
    for i in idx:
        d = P.impact_autocorr_peak(raw[n_cls + i])
        assert d["freq_rel_err"] < 0.01, f"第 {i} 条故障信号反推频率偏离 BPFO: {d}"
        assert d["peak_value"] > 0.3, f"第 {i} 条故障信号在 1/BPFO 处峰太弱: {d}"

    for i in idx:
        dn = P.impact_autocorr_peak(raw[i])
        assert dn["peak_value"] < 0.2, f"第 {i} 条正常信号意外出现 1/BPFO 自相关峰: {dn}"


def test_envelope_is_required_for_the_peak() -> None:
    """直接对原始信号做自相关不足以定位冲击周期 —— 证明"为什么必须解调"。"""
    payload = require_dataset()
    fault = payload["raw"][C.N_SAMPLES_PER_CLASS]
    lag_exact = C.FS / C.BPFO
    lo, hi = int(round(0.7 * lag_exact)), int(round(1.3 * lag_exact))

    raw_peak = P.autocorrelation(fault)[lo:hi + 1]
    env_peak = P.autocorrelation(P.envelope(fault))[lo:hi + 1]
    assert env_peak.max() > 5.0 * raw_peak.max(), (
        f"包络自相关相对原始信号没有显著优势: env={env_peak.max():.4f} raw={raw_peak.max():.4f}"
    )


# ==========================================================================
# 8. 共振频率条内固定
# ==========================================================================
def test_resonance_frequency_is_fixed_within_segment() -> None:
    """同一条信号前后两半的共振带谱峰应一致；错误实现（每击重抽）必须被判否。"""
    payload = require_dataset()
    rv = P.verify_resonance_per_segment(payload["raw"], n_check=80)
    assert rv["frac_close"] > 0.95, rv
    assert rv["median_hz"] < 5.0, rv
    assert rv["wrong_variant_hz"] > 50.0, f"自检判据无鉴别力: {rv}"


def test_resonance_band_is_actually_sampled() -> None:
    """条间共振频率应覆盖整个 500~4000 Hz 区间（说明真的在随机，而非固定值）。"""
    payload = require_dataset()
    raw = payload["raw"]
    picks = [0, 50, 100, 200, 300, 399]
    freqs = np.fft.rfftfreq(20000, d=1.0 / C.FS)
    m = (freqs >= 500.0) & (freqs <= 4000.0)
    peaks = []
    for i in picks:
        seg = raw[C.N_SAMPLES_PER_CLASS + i][:20000].astype(np.float64)
        peaks.append(float(freqs[m][int(np.argmax(np.abs(np.fft.rfft(seg))[m]))]))
    assert max(peaks) - min(peaks) > 500.0, f"条间共振频率过于集中: {peaks}"


# ==========================================================================
# 9. 可复现性
# ==========================================================================
def test_seed_derivation_is_bitwise_reproducible() -> None:
    """同一派生种子两次生成的信号逐位一致，且与落盘 raw 完全相同。"""
    payload = require_dataset()
    n_cls = C.N_SAMPLES_PER_CLASS
    checks = [("normal", 0), ("normal", 399), ("fault", 0), ("fault", 399)]
    for kind, i in checks:
        a, b = gen_segment(kind, i), gen_segment(kind, i)
        assert np.array_equal(a, b), f"{kind}[{i}] 两次生成结果不同（种子派生有问题）"
        stored = payload["raw"][i if kind == "normal" else n_cls + i]
        assert np.array_equal(a, stored), f"{kind}[{i}] 生成结果与落盘 raw 不一致"


def test_saved_dataset_sha256_matches_regenerated_pipeline() -> None:
    """端到端可复现：用同样的参数重跑一遍生成+滑窗+保存，sha256 必须与磁盘产物一致。

    这是"两次运行 npy 一致"的机器化版本：不需要真的跑两遍 preprocess1.py，
    而是用同一份确定性代码在临时目录重算一次并比对字节。
    """
    disk = Path(C.DATA_PATH)
    assert disk.exists(), f"数据集不存在: {disk}（请先运行 preprocess1.py）"

    raw2, _ = P.generate_raw()
    X2, y2, seg2 = P.sliding_window(raw2)
    # 与 preprocess1.py 共用同一个 meta 构造函数，保证逐字节可比
    meta2 = dataio.build_meta(P.build_extra_meta())
    dataio.validate({"X": X2, "y": y2, "seg_id": seg2, "raw": raw2, "meta": meta2})

    tmpdir = Path(tempfile.mkdtemp(prefix="pp1_repro_"))
    try:
        tmp = tmpdir / C.DATASET_NAME
        dataio.save_dataset(tmp, X2, y2, seg2, raw2, meta2)
        assert sha256_file(tmp) == sha256_file(disk), (
            "重新生成的数据集与磁盘产物 sha256 不一致 —— 可复现性被破坏"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# 直接运行入口（无 pytest 时也能跑）
# ==========================================================================
def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    print("=" * 70)
    print(f"preprocess1.py 自检 —— 共 {len(tests)} 项")
    print("=" * 70)
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - 测试汇总需要捕获一切
            failed.append((name, exc))
            print(f"[FAIL] {name}\n       {type(exc).__name__}: {exc}")
        else:
            print(f"[PASS] {name}")
    print("-" * 70)
    if failed:
        print(f"结果: {len(tests) - len(failed)}/{len(tests)} 通过，{len(failed)} 项失败")
        return 1
    print(f"结果: {len(tests)}/{len(tests)} 全部通过 √")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
