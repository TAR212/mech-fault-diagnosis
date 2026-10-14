# 机械设备故障诊断 · 操作手册（RUNBOOK）

> **本文面向「要动手跑、要排障、要交接」的人。**
> 想先知道这个项目是什么、为什么这么设计 → 看 [项目说明](PROJECT.md)。
> 想逐条读参数与实现 → 看 [README.md](../README.md)。

**所有命令都在项目根目录执行。** 下文出现的 `python` 一律指 §1.2 自检通过的那个解释器——
这是本项目**最容易踩的坑**：机器上常同时存在多个 Python，只有一个装了 numpy。

---

## 1. 开工前

### 1.1 硬性要求

| 项 | 要求 | 不满足会怎样 |
|---|---|---|
| Python | ≥ 3.9（本仓库验证时为 3.12.14） | 语法不兼容 |
| numpy | ≥ 2.0（实测 2.3.5） | 什么都跑不了 |
| matplotlib | ≥ 3.8（实测 3.11.2） | 8 张图全部画不出来 |
| 磁盘空闲 | **≥ 1.5 GB** | `data/clean_signal1.npy` 约 540 MB，写盘时中途失败 |
| 内存 | **≥ 2 GB 可用** | `cnn1.py` 训练期常驻约 1.5 GB，可能 MemoryError |
| 中文字体 | 至少一个（见 §1.3） | 图上中文变方块，但**不影响数值结果** |

**不要安装** torch / tensorflow / scipy / scikit-learn / pandas / pytest —— 本项目刻意零重型依赖，
1D-CNN 是纯 NumPy 手写的（`nn1d/`）。装了也不会被用到，只会浪费时间与磁盘。

安装依赖（若你还没有）：

```bash
python -m pip install -r requirements.txt
```

### 1.2 解释器自检（**必做第一步**）

```bash
python -c "import sys, numpy, matplotlib; print('解释器:', sys.executable); print('Python:', sys.version.split()[0]); print('numpy:', numpy.__version__, '| matplotlib:', matplotlib.__version__)"
```

期望：打印出**一个绝对路径**和 `numpy: 2.3.5 | matplotlib: 3.11.2` 之类的版本号。

- 报 `ModuleNotFoundError: No module named 'numpy'` → **你选错了解释器**。
  换一个（虚拟环境、conda、绝对路径），或对当前解释器装 numpy。**不要**指望改代码能绕过。
- 同时存在多个解释器时，把自检通过的那个绝对路径记下来，后续命令全用它，例如：
  ```bash
  # 例：用绝对路径显式指定，避免 PATH 里那个没装 numpy 的 python
  "C:\path\to\python.exe" tests/run_all.py
  ```

### 1.3 中文字体（只影响图，不影响数字）

`config.py:132-135` 的候选列表为：
`Microsoft YaHei` → `SimHei` → `Noto Sans CJK SC` → `Source Han Sans SC` → `Arial Unicode MS`。
`config.configure_matplotlib()` 会按顺序挑第一个存在的，并强制 Agg 无 GUI 后端。

Linux / CI 环境通常一个都没有，请装（Debian/Ubuntu）：

```bash
sudo apt-get install -y fonts-noto-cjk
```

自检：跑完第二步后看 `figures/fig_spectrum.png` 标题，中文正常即 OK。

---

## 2. 标准操作流程（SOP）

三步必须**按顺序**跑（后一步消费前一步的产物）。

### 步骤 1 —— 生成数据集

```bash
python preprocess1.py
```

| | |
|---|---|
| 耗时 | 约 7–10 s |
| 产物 | `data/clean_signal1.npy`（约 540 MB）、`figures/fig_preprocess_cache.png` |
| 无参数 | 该脚本没有命令行参数，全部行为由 `config.py` 决定 |

期望看到（节选）：

```
[契约] 采样率=12000 Hz，单条时长=3 s，每条=36000 点，WINDOW=1024，STRIDE=256
[契约] 轴承特征频率: BPFO=158.7532 Hz，BPFI=241.2468 Hz，BSF=116.0658 Hz，FTF=19.8442 Hz
[契约] 冲击间隔 1/BPFO = 6.2991 ms，单条窗口数 = 1+(N-WINDOW)//STRIDE = 137，窗口总数 = 800×137 = 109600
生成完成: raw.shape=(800, 36000), dtype=float32, 占用 109.9 MB
X.shape=(109600, 1024), dtype=float32（428.1 MB）
```

末尾会有 **4 项自检**与一个 `sha256:`；自检失败会明确报第几项失败并**非零退出**。

### 步骤 2 —— 时域 / 频域分析

```bash
python fft1.py
```

| | |
|---|---|
| 耗时 | 约 3–4 s |
| 产物 | `figures/fig_time_domain.png`、`figures/fig_spectrum.png`、`figures/fig_envelope_spectrum.png`、`analysis_report.md` |
| 前置 | `data/clean_signal1.npy` 必须已存在，否则 `FileNotFoundError` |
| 无参数 | 该脚本没有命令行参数 |

期望看到（节选）：

```
X.shape = (109600, 1024)   raw.shape = (800, 36000)   meta.format_version = 1.0
类别计数（窗口级）= {'total': 109600, 'normal': 54800, 'fault': 54800}
```

### 步骤 3 —— 1D-CNN 训练与评估

```bash
python cnn1.py
```

| | |
|---|---|
| 耗时 | canonical 默认 **约 4–5 分钟**（训练约 3.5–5 分钟 + 评估出图） |
| 产物 | `results/metrics.json`、`figures/fig_loss_curve.png`、`figures/fig_accuracy_curve.png`、`figures/fig_confusion_matrix.png`、`figures/fig_single_sample_pred.png` |
| 前置 | 数据集必须已存在 |

期望看到（节选）：

```
trainable_params = 18594
训练规模: 部分轮 90 步 x batch 64 = 5760 个窗口/轮（完整轮为 87680 个），时间预算 900s
测试集最终结果: loss=0.0000  accuracy=1.0000
混淆矩阵（行=真实, 列=预测）: [[10549, 0], [0, 11371]]，四格之和=21920，测试样本数=21920
由混淆矩阵算出的准确率 = 1.0000（与 test_accuracy 一致 √）
```

> ⚠️ **副作用必须知道**：`cnn1.py` 每次运行都会**无条件覆写** `results/metrics.json` 与上表 4 张训练图。
> 用非默认参数做实验（例如 `--steps-per-epoch 1`）会把 canonical 产物覆盖成未收敛形态。
> 想试参数：**先备份** `results/` 与那 4 张图，跑完再用裸 `python cnn1.py` 还原。

### 步骤 4 —— 全部自检

```bash
python tests/test_preprocess.py    # 14 项，约 6–7 s
python tests/test_nn1d.py         # 15 项，约 3.6–4.3 s
```

环境**未安装 pytest**，这两个脚本自带 `if __name__ == "__main__"` 入口，直接运行即可。
期望末行：`结果: 14/14 全部通过 √` / `结果: 15/15 全部通过 √`。

---

## 3. 一键验收（推荐）

```bash
python tests/run_all.py
```

按顺序跑完 **5 个环节**（3 条流水线脚本 + 2 个自检脚本），逐环节回显输出，
最后打出 PASS/FAIL 汇总表；**全 PASS 退出码 0，任一失败退出码 1 并给中文排查建议**。

| # | 环节 | 典型耗时 |
|---|---|---|
| 1 | `preprocess1.py` | 7–9 s |
| 2 | `fft1.py` | 3–4 s |
| 3 | `cnn1.py` | 约 247–290 s |
| 4 | `tests/test_preprocess.py` | 6–7 s |
| 5 | `tests/test_nn1d.py` | 3.6–4.1 s |
| | **合计** | **约 266–314 s（4.4–5.2 分钟）** |

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--timeout SEC` | `3600` | 单个子进程超时；慢机器可放大到 `3000`+ |
| `--no-pause` | — | 保留参数位（默认就不暂停） |

```bash
# 慢机器 / 首次冷启动
python tests/run_all.py --timeout 3000
```

它会把子进程 stdout 强制为 UTF-8（`tests/run_all.py:174-175` 注入 `PYTHONUTF8=1`、
`PYTHONIOENCODING=utf-8`），因此**在 Windows 重定向/CI 下也不会因编码问题假失败**。

---

## 4. 常用操作任务

### 4.1 只想重做分析，不想重训

```bash
python fft1.py          # 3 秒，只重画 3 张分析图 + 报告
```

### 4.2 只想重训（数据集已在）

```bash
python cnn1.py
```

### 4.3 快速冒烟（不想等 4 分钟）

```bash
python cnn1.py --steps-per-epoch 5 --time-budget 60
```

⚠️ **会覆盖 canonical 产物**（见 §2 步骤 3 的警告）。仅用于确认「能跑通」，不要用它的指标做结论。

### 4.4 完整的教科书式 epoch

```bash
python cnn1.py --full-epoch
```

关闭部分轮与时间预算，每轮跑完整 **1370 步 / 87680 窗口**。
**本机约 212 s/轮，40 轮远超 20 分钟** —— 仅供对照，别在验收时用。

### 4.5 调超参

```bash
python cnn1.py --lr 1e-3 --batch-size 128 --epochs 20
```

`--epochs` 默认取 `config.EPOCHS=40`，`--lr` 默认 `config.LR=3e-3`，`--batch-size` 默认 `config.BATCH_SIZE=64`。

> **改默认值请改 `config.py`**，不要在命令行长期堆参数 —— `config.py` 是唯一的超参契约文件。

### 4.6 用别的数据集

```bash
python cnn1.py --data /path/to/other.npy
```

目标文件必须符合 [README §5 数据集契约](../README.md)。
自己造数据时用 `dataio.save_dataset()` 落盘、`dataio.validate()` 自检，别手搓 npy。

### 4.7 完整重建（从零到全套产物）

```bash
rm -rf data figures results analysis_report.md   # 清掉所有产物（源码与文档不动）
python tests/run_all.py                           # 一条命令全部重生成
```

---

## 5. 排障手册

按「症状 → 原因 → 处置」查。

| 症状 | 原因 | 处置 |
|---|---|---|
| `ModuleNotFoundError: No module named 'numpy'` | 用错了解释器 | 回到 §1.2，换自检通过的那个 |
| `FileNotFoundError: ... clean_signal1.npy` | 数据集没生成 | 先 `python preprocess1.py`；或 `--data` 指错了 |
| `UnicodeEncodeError: 'gbk' codec can't encode character` | Windows 下 stdout 被重定向时按 locale（中文系统为 GBK）编码，遇到非 GBK 字符即抛错**且退出码为 1** | 本项目已给 `preprocess1.py` / `fft1.py` / `cnn1.py` / `tests/test_nn1d.py` / `tests/run_all.py` 内置了 UTF-8 守卫，正常不会遇到。⚠️ `tests/test_preprocess.py` 自身**没有**守卫，它靠 `import preprocess1` 触发的传递性副作用而幸免（能工作但依赖导入顺序）。你自己写脚本时请显式加一句 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，或临时 `set PYTHONUTF8=1` |
| 图上中文变 `□□` 方块 | 缺中文字体 | §1.3 装字体；数值结果不受影响 |
| `MemoryError` | 训练期常驻约 1.5 GB | 关掉其他吃内存的程序；或调小 `--batch-size` |
| 训练特别慢 / 超过 30 s 红线 | 机器慢或磁盘慢 | `--steps-per-epoch` 调小；参考 `nn1d/model.py` 顶部列出的缩减方案 |
| `results/metrics.json` 里 `test_accuracy` 只有 0.5 左右 | 被 `--steps-per-epoch 1` 之类的实验跑覆盖了 | 裸跑 `python cnn1.py` 还原（见 §4.3 警告） |
| 两次运行耗时不同 | **正常**。墙钟随机器负载浮动 | 只要 `test_accuracy`/混淆矩阵/`trainable_params` 一致即可；精确值看 `results/metrics.json` |
| 自检报 `[FAIL] <用例名>` | 契约或反向公式被改坏 | 按用例名定位到 `tests/` 对应函数；数值梯度失败**不要**靠调大学习率掩盖 |
| `tests/run_all.py` 报某环节超时 | 默认 3600 s 不够（极慢机器） | `--timeout 3000` 放大后重试 |

### 5.1 一条命令拿到完整现场信息

```bash
python -c "import sys,platform,numpy,matplotlib;print('exe:',sys.executable);print('py:',sys.version);print('np:',numpy.__version__,'mpl:',matplotlib.__version__);print('os:',platform.platform())"
```

排障时把这条的输出连同失败日志一起贴出来，能省一轮往返。

---

## 6. 产物与生命周期

| 路径 | 入库? | 可删? | 重建方式 |
|---|---|---|---|
| `data/clean_signal1.npy`（约 540 MB） | ❌ 被 `.gitignore` 排除 | ✅ 可删 | `python preprocess1.py` |
| `figures/*.png`（8 张） | ✅ 入库 | ✅ 可删 | 步骤 1/2/3 各自重生成 |
| `results/metrics.json` | ✅ 入库 | ⚠️ 覆盖式，建议先备份 | `python cnn1.py` |
| `analysis_report.md` | ✅ 入库 | ✅ 可删 | `python fft1.py` |
| `__pycache__/` | ❌ 忽略 | ✅ 可删 | 自动重建 |

**为什么 `results/metrics.json` 要入库**：它是「这次结果确实跑出来了」的**可追溯证据** ——
含配置快照、划分明细、环境版本、逐轮 history、混淆矩阵自洽校验。
`.gitignore` 里专门写了注释说明这些产物**故意保留**。

**为什么数据集不入库**：540 MB 进 git 历史不可逆；它可由固定种子**逐位复现**，没必要存。

---

## 7. 交接 / 验收检查清单

复制这份清单逐项打勾：

- [ ] §1.2 解释器自检通过，且已记下解释器绝对路径
- [ ] 磁盘 ≥ 1.5 GB、内存 ≥ 2 GB 可用
- [ ] `python tests/run_all.py` → 汇总表 **5/5 PASS**、退出码 **0**
- [ ] `results/metrics.json`：`test_accuracy` 为 `1.0`、混淆矩阵 `[[10549, 0], [0, 11371]]`
- [ ] `results/metrics.json`：`trainable_params` = `18594` 且 `trainable_params_match_manual` 为 `true`
- [ ] `results/metrics.json`：`split` 里 `n_train_segments` = 640、`n_test_segments` = 160
- [ ] `figures/` 下恰好 **8 张** PNG，且中文标题无方块
- [ ] `results/metrics.json` 的 `history` 有 **9 个点**（不是 1 个点）
- [ ] 两次独立跑 `python cnn1.py`，`test_accuracy` 与混淆矩阵一致（耗时不同属正常）

> `test_accuracy = 1.0` 是**预期结果**（数据是机理仿真的合成信号，类间差异明显），
> 它证明**流水线端到端正确**，**不证明**模型在真实工业数据上有效。详见 [项目说明 §7](PROJECT.md)。

---

## 8. 发布与更新（GitHub）

本项目已发布为独立公开仓库：**https://github.com/TAR212/mech-fault-diagnosis**

它**不是**私人工作区仓库的子目录，而是一个独立 git 仓库（有自己的 `.git`）。
更新流程：

```bash
git status                     # 确认改动范围
git add -A
git commit -m "描述这次改了什么"
git push origin master:main    # 本地分支 master → 远端 main
```

### 8.1 推送前请务必自检两件事

**① 不要泄露本机绝对路径。** `results/metrics.json` 会记录数据集/图片路径，
`cnn1.py` 已用 `portable_path()` 保证写入的是**相对路径**。
推送前扫一遍：

```bash
python -c "
import subprocess
needles = ['C:' + chr(92) + 'Users', 'AppData', 'dsh-runtimes', 'deepseek-harness',
           chr(0x767d) + chr(0x6c34), '/home/', '/Users/']
files = subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.split()
bad = 0
for f in files:
    try:
        t = open(f, encoding='utf-8').read().replace(chr(92) * 2, chr(92))
    except (UnicodeDecodeError, OSError):
        continue
    for i, line in enumerate(t.splitlines(), 1):
        if any(n in line for n in needles):
            print(f'{f}:{i}: {line.strip()[:100]}')
            bad += 1
print('suspicious lines =', bad)
raise SystemExit(1 if bad else 0)
"
```

预期输出 `suspicious lines = 0` 且退出码 0。
**为什么用子串而不是正则**：JSON 转义后路径里是 `C:\\Users`（双反斜杠），文档里手写的是 `C:\Users`（单反斜杠）。
先用 `.replace(chr(92)*2, chr(92))` 把双反斜杠压成单反斜杠，再用子串匹配，两种写法都能命中；
只写 `r'C:\\Users'` 这类正则反而会漏掉**手写在文档里的那一半**。
（本命令已用「干净树 + 7 种人工植入的泄漏」做过正反双向对照，均正确。）

**② 确认大数据集没被误提交。**

```bash
git ls-files | grep -c "\.npy$"     # 期望 0
git check-ignore -v data/clean_signal1.npy   # 期望命中 .gitignore
```

### 8.2 网络抖动

`git push` 偶发 `Failed to connect to github.com:443` 多为瞬时问题。重试即可；
必要时放大低速容忍度：

```bash
GIT_HTTP_LOW_SPEED_TIME=120 git push origin master:main
```

---

## 9. 一分钟速查

```bash
# 环境自检
python -c "import sys,numpy,matplotlib;print(sys.executable,numpy.__version__,matplotlib.__version__)"

# 全流程一键验收（推荐）
python tests/run_all.py

# 分步执行
python preprocess1.py    # 数据集      ~7-10 s
python fft1.py           # 分析        ~3-4 s
python cnn1.py           # 训练        ~4-5 min

# 自检
python tests/test_preprocess.py   # 14 项
python tests/test_nn1d.py         # 15 项

# 冒烟（会覆盖 canonical 产物！）
python cnn1.py --steps-per-epoch 5 --time-budget 60
```
