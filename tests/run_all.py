#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键验收：机械故障诊断三步流水线
==================================
依次执行（全部用 ``sys.executable`` —— 即当前解释器，禁止硬编码 python）::

    preprocess1.py            第一步：生成 data/clean_signal1.npy
    fft1.py                   第二步：时域 / 频谱 / 包络谱分析
    cnn1.py                   第三步：纯 NumPy 1D-CNN 训练与评估
    tests/test_preprocess.py  第一步自检（14 项，无 pytest 依赖）
    tests/test_nn1d.py        nn1d 自检（15 项，数值梯度 + 过拟合）

结束时打印 PASS/FAIL 汇总表与总耗时；任一环节失败会给出中文排查建议，
并以非零退出码结束，便于 CI / 人工一眼判断。

用法
----
    python tests/run_all.py                 # 完整跑通（本机实测约 5 分钟，慢机器 10 分钟以上；
                                            # 主要耗在 cnn1.py 的纯 NumPy 训练）
    python tests/run_all.py --timeout 3600  # 自定义单个子进程超时（秒）
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

def _force_utf8_stdout() -> None:
    """把本进程的 stdout/stderr 切到 UTF-8。

    Windows 控制台默认 GBK，而子脚本（已 `stream.reconfigure(encoding="utf-8")`）
    会输出 `√` 等非 ASCII 字符；父进程若仍按 GBK 编码回显，就会抛
    ``UnicodeEncodeError: 'gbk' codec can't encode character '\\u2286'``，
    整个一键验收在第一环就崩掉。这里做三层兜底：reconfigure → 重开 text 流 → 替换字符。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
            continue
        except Exception:  # noqa: BLE001 - 兜底路径，任何失败都继续往下试
            pass
        try:
            import io

            buf = getattr(stream, "buffer", None)
            wrapped = io.TextIOWrapper(buf, encoding="utf-8", errors="replace",
                                       line_buffering=True)
            if stream is sys.stdout:
                sys.stdout = wrapped
            elif stream is sys.stderr:
                sys.stderr = wrapped
        except Exception:  # noqa: BLE001
            pass


_force_utf8_stdout()

# --------------------------------------------------------------------------
# 路径：本文件在 <项目根>/tests/run_all.py，故根目录是上一级
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable  # 与手工运行 run_all.py 的解释器完全一致

# --------------------------------------------------------------------------
# 五个环节的定义
# --------------------------------------------------------------------------
#   name    : 汇总表里显示的名字
#   script  : 相对项目根目录的脚本路径
#   keys    : 从 stdout 里挑出来作为"关键结论"的正则（按顺序取前若干个）
#   tips    : 失败时的中文排查建议（多行）
STAGES = [
    {
        "name": "preprocess1.py",
        "script": "preprocess1.py",
        "desc": "第一步 · 生成仿真数据集（正常/故障 + 滑窗 + 打标签）",
        "keys": [
            r"\[契约\] 采样率",
            r"窗口总数\s*:",
            r"训练侧:",
            r"测试侧:",
            r"sha256\s*:",
            r"第 1 步完成",
        ],
        "tips": [
            "1) 确认用的是装了 numpy/matplotlib 的解释器（见 README「环境要求」）；",
            "2) 磁盘至少留 1.5 GB 空闲（data/clean_signal1.npy 约 540 MB）；",
            "3) 若报 UnicodeEncodeError，设 PYTHONUTF8=1（本脚本已自动设置）；",
            "4) 报『第 N 项自检失败』说明信号机理或参数被改动，先看完整输出再定位。",
        ],
    },
    {
        "name": "fft1.py",
        "script": "fft1.py",
        "desc": "第二步 · 时域波形 / FFT 频谱 / 包络谱分析",
        "keys": [
            r"BPFO 1x @",
            r"fig_time_domain\.png",
            r"analysis_report\.md",
            r"包络谱：BPFO",
        ],
        "tips": [
            "1) 第二步消费 data/clean_signal1.npy，必须先成功跑完 preprocess1.py；",
            "2) 三张图大小若显示 '!! 偏小'，通常是中文字体缺失导致绘图异常，见 README 常见问题；",
            "3) 报 FileNotFoundError 说明数据集缺失或被 --data 指向了错误路径。",
        ],
    },
    {
        "name": "cnn1.py",
        "script": "cnn1.py",
        "desc": "第三步 · 1D-CNN 训练 / 评估 / 混淆矩阵 / 单样本预测",
        "keys": [
            r"trainable_params\s*=",
            r"训练规模:",
            r"训练完成:",
            r"测试集最终结果",
            r"混淆矩阵（行=真实",
            r"指标已落盘",
        ],
        "tips": [
            "1) 这是最慢的一步（canonical 默认约 4–5 分钟），请给足超时时间；",
            "2) 报『单轮耗时超过 30 秒红线』不一定是失败，但机器过慢时可用 "
            "--steps-per-epoch 调小，或先看 nn1d/model.py 顶部列出的缩减方案；",
            "3) 若内存不足（MemoryError），关闭其他占内存程序；本步峰值常驻约 1.5 GB；",
            "4) 训练结果随机性由 config.SEED=42 固定，两次结果应完全一致。",
        ],
    },
    {
        "name": "tests/test_preprocess.py",
        "script": "tests/test_preprocess.py",
        "desc": "第一步自检 · 14 项（契约 / 滑窗 / 防泄漏划分 / 可复现性）",
        "keys": [
            r"^\[FAIL\]",
            r"结果: \d+/\d+",
            r"结果: 全部通过",
        ],
        "tips": [
            "1) 该脚本自带 if __name__ == '__main__' 入口，环境无 pytest，直接运行即可；",
            "2) 失败时上面会打印 [FAIL] 用例名 + 异常，按用例名定位到 tests/test_preprocess.py 对应函数；",
            "3) 若报数据集缺失，先跑 python preprocess1.py。",
        ],
    },
    {
        "name": "tests/test_nn1d.py",
        "script": "tests/test_nn1d.py",
        "desc": "nn1d 自检 · 15 项（数值梯度 / train-eval 模式 / 小样本过拟合）",
        "keys": [
            r"<<< .* FAIL",
            r"结果: \d+/\d+",
            r"结果: 全部通过",
        ],
        "tips": [
            "1) 该脚本自带 if __name__ == '__main__' 入口，直接运行即可；",
            "2) 数值梯度用例失败 → nn1d/layers.py 反向公式有问题，不要靠调大学习率掩盖；",
            "3) 报 UnicodeEncodeError（非 GBK 字符）说明未走 UTF-8 输出，"
            "直接运行时请先设 PYTHONUTF8=1 或 set PYTHONIOENCODING=utf-8。",
        ],
    },
]

# 通用：即使上面 keys 没命中，也把最后几行留下当证据
TAIL_ON_FAIL = 25
MAX_KEY_LINES = 8


def _make_env() -> dict:
    """强制 UTF-8 输出，避免 Windows 控制台 GBK 遇到非 ASCII 字符时报 UnicodeEncodeError。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run_one(script_rel: str, timeout: float | None):
    """跑一个子进程，边跑边回显；返回 (returncode, stdout, stderr, seconds)。"""
    path = ROOT / script_rel
    if not path.exists():
        return None, "", f"脚本不存在: {path}", 0.0

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [PY, str(path)],
        cwd=str(ROOT),
        env=_make_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            print("    " + line, flush=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        lines.append(f"[run_all] 超时（>{timeout:.0f}s），已终止子进程")
        return -9, "\n".join(lines), "", time.perf_counter() - t0
    return proc.returncode, "\n".join(lines), "", time.perf_counter() - t0


def _pick_keys(stdout: str, patterns: list[str]) -> list[str]:
    """按给定正则顺序，从 stdout 里挑出关键结论行（每类最多若干行）。"""
    out: list[str] = []
    for pat in patterns:
        rx = re.compile(pat)
        for line in stdout.splitlines():
            if rx.search(line):
                out.append(line.strip())
                break  # 每类只取第一处命中
    return out[:MAX_KEY_LINES]


def _tail(stdout: str, n: int = TAIL_ON_FAIL) -> list[str]:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    return lines[-n:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="一键跑通三步流水线 + 两个自检脚本")
    ap.add_argument("--timeout", type=float, default=3600.0, metavar="SEC",
                    help="单个子进程超时秒数，默认 3600（cnn1.py 约需 4–5 分钟）")
    ap.add_argument("--no-pause", action="store_true",
                    help="保留参数位（默认就不暂停，兼容交互使用习惯）")
    args = ap.parse_args(argv)

    print("=" * 84, flush=True)
    print("机械设备故障诊断 · 三步流水线一键验收", flush=True)
    print("=" * 84, flush=True)
    print(f"项目根目录 : {ROOT}", flush=True)
    print(f"解释器     : {PY}", flush=True)
    print(f"Python     : {sys.version.split()[0]}", flush=True)
    print(f"流水线     : preprocess1.py -> fft1.py -> cnn1.py -> 两个自检脚本", flush=True)
    print(f"单步超时   : {args.timeout:.0f} s", flush=True)
    print("=" * 84, flush=True)

    results: list[dict] = []
    t_all = time.perf_counter()

    for i, st in enumerate(STAGES, 1):
        print("", flush=True)
        print("-" * 84, flush=True)
        print(f"[{i}/{len(STAGES)}] {st['name']}  ——  {st['desc']}", flush=True)
        print("-" * 84, flush=True)
        rc, out, err, sec = _run_one(st["script"], args.timeout)
        ok = (rc == 0)
        results.append({
            "name": st["name"], "desc": st["desc"], "rc": rc, "ok": ok,
            "seconds": sec, "stdout": out, "stderr": err,
            "keys": _pick_keys(out, st["keys"]) if out else [],
            "tips": st["tips"],
        })
        if rc is None:  # 脚本文件不存在等
            print(f"  >>> {st['name']}: FAIL（{err}）", flush=True)
        else:
            print(f"  >>> {st['name']}: {'PASS' if ok else 'FAIL'}"
                  f"（退出码 {rc}，耗时 {sec:.1f} s）", flush=True)

    total = time.perf_counter() - t_all
    n_pass = sum(1 for r in results if r["ok"])

    # ------------------------------------------------------------------
    # 关键结论
    # ------------------------------------------------------------------
    print("", flush=True)
    print("=" * 84, flush=True)
    print("关键输出摘录", flush=True)
    print("=" * 84, flush=True)
    for r in results:
        print(f"\n[{r['name']}]", flush=True)
        if r["keys"]:
            for line in r["keys"]:
                print(f"  · {line}", flush=True)
        else:
            print("  （无可摘录的关键行）", flush=True)

    # ------------------------------------------------------------------
    # 汇总表
    # ------------------------------------------------------------------
    print("", flush=True)
    print("=" * 84, flush=True)
    print("汇总表", flush=True)
    print("=" * 84, flush=True)
    head = f"{'#':>2}  {'环节':<26} {'结果':<8} {'退出码':>6}  {'耗时':>9}"
    print(head, flush=True)
    print("-" * 84, flush=True)
    for i, r in enumerate(results, 1):
        rc_txt = "-" if r["rc"] is None else str(r["rc"])
        print(f"{i:>2}  {r['name']:<26} {'PASS' if r['ok'] else 'FAIL':<8} "
              f"{rc_txt:>6}  {r['seconds']:>7.1f}s", flush=True)
    print("-" * 84, flush=True)
    print(f"合计: {n_pass}/{len(results)} 通过   总耗时 {total:.1f} s "
          f"（约 {total / 60:.1f} 分钟）", flush=True)
    print("=" * 84, flush=True)

    if n_pass == len(results):
        print("", flush=True)
        print("全部 PASS —— 三步流水线端到端可复现，产物齐备：", flush=True)
        print("  data/clean_signal1.npy      数据集（约 540 MB，可重生成，未入库）", flush=True)
        print("  figures/fig_*.png           8 张结果图", flush=True)
        print("  results/metrics.json        训练指标 + 配置快照 + 环境信息", flush=True)
        print("  analysis_report.md          第二步量化分析报告", flush=True)
        print("", flush=True)
        return 0

    # ------------------------------------------------------------------
    # 失败排查建议
    # ------------------------------------------------------------------
    print("", flush=True)
    print("=" * 84, flush=True)
    print("存在 FAIL 环节 —— 下一步排查建议", flush=True)
    print("=" * 84, flush=True)
    for r in results:
        if r["ok"]:
            continue
        print(f"\n[{r['name']}] 退出码 {r['rc']}，耗时 {r['seconds']:.1f} s", flush=True)
        if r["rc"] is not None and r["rc"] < 0:
            print(f"  · 子进程被超时终止（当前 --timeout={args.timeout:.0f}s），"
                  f"可用 --timeout 3000 放大后重试", flush=True)
        for tip in r["tips"]:
            print("  " + tip, flush=True)
        if r["stdout"]:
            print("  · 该环节最后 25 行输出：", flush=True)
            for line in _tail(r["stdout"]):
                print("      " + line, flush=True)

    print("", flush=True)
    print("其它通用建议：", flush=True)
    print(f"  1) 手工复现失败环节：cd /d \"{ROOT}\" 然后 \"{PY}\" <失败脚本路径>", flush=True)
    print("  2) 确认解释器正确：README「环境要求」里给的解释器才有 numpy/matplotlib；", flush=True)
    print("  3) 数据集损坏时删掉 data/clean_signal1.npy 重新跑 preprocess1.py。", flush=True)
    print("", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
