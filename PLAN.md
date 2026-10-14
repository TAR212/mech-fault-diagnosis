# 《机械设备故障诊断技能》三步流水线 —— 成熟实践方案 (PLAN.md)

> 源需求：一张教学幻灯片，标题「机械设备故障诊断技能」，流水线
> `preprocess1.py → fft1.py → cnn1.py`，分三步：数据集生成 / 信号时域+频域分析 / 1D-CNN 训练与评估。
>
> 本文件是**方案与验收基线**，不是原始需求本身；代码实现以仓库为准，二者的偏差必须在本文「偏差登记」一节留痕。

---

## 0. 一句话方案

把幻灯片上的三步"脚本链"，升级为**可复现、可验收、可扩展的工程化教学流水线**：
冻结配置与数据契约 → 三条脚本各自单一职责 → 全程带自检与量化指标 → 一键端到端验收。

```
┌──────────────────────┐   ┌────────────────────────┐   ┌───────────────────────────┐
│ 1. preprocess1.py    │   │ 2. fft1.py             │   │ 3. cnn1.py                │
│ 数据集生成            │──▶│ 信号时域 + 频域分析      │──▶│ 1D-CNN 训练与评估          │
│ 正常/故障仿真振动信号  │   │ FFT 频谱 + 包络谱        │   │ 训练/测试曲线 + 混淆矩阵    │
│ 滑窗截取 → 打标签 0/1 │   │ 定位 BPFO 谐波族         │   │ 单样本预测演示             │
└──────────────────────┘   └────────────────────────┘   └───────────────────────────┘
        data/clean_signal1.npy          figures/fig_*.png          results/metrics.json
        ▲                              ▲                          ▲
        └──────────── config.py（唯一参数源） + dataio.py（唯一 IO 契约） ───────────┘
```

三步之间用**文件契约**解耦：第 1 步产出 `.npy` 数据集，第 2、3 步各自消费，互不 import 对方。
这样任何一步都能单独重跑、单独替换（例如把仿真信号换成真实轴承数据，第 2、3 步代码不改）。

---

## 1. 对原幻灯片的 5 处"成熟化"升级

幻灯片本身是作业级别的三段式描述。真正落地时，下面 5 个坑不处理，"跑通"和"做对"是两件事：

| # | 原始做法（作业级） | 升级后的做法（工程级） | 不升级的后果 |
|---|---|---|---|
| 1 | 三个脚本各自硬编码参数 | 抽出 `config.py` 作为唯一参数源，三个脚本只 import | 改一处参数要改三个文件，必然漂移；`fft1.py` 算出的 BPFO 与 `preprocess1.py` 用的不是同一个数 |
| 2 | 直接对整个数据集随机划分训练/测试 | **先按原始信号（`seg_id`）分组划分，再取窗口** | 滑窗重叠 75%，相邻片段同时进训练/测试 → 测试准确率虚高到 99%+ 却毫无泛化能力（**数据泄漏**） |
| 3 | 只看 Accuracy | Accuracy + 混淆矩阵 + 每类 P/R/F1 + 训练/测试双曲线 | 两类不平衡时 Accuracy 会骗人；曲线分离才能看出过拟合 |
| 4 | 频域图只是"画出来看看" | 图上标注 BPFO/BPFI/BSF/FTF 理论频率线，报告给出**量化 dB 抬升** | 无法判断这是轴承故障还是噪声，分析不可证伪 |
| 5 | 无自检、无复现保证 | 数据校验门 + 固定种子 + 可复现性测试 + 一键 `run_all.py` | 换台机器结果就变，无法交付 |

---

## 2. 分阶段实施计划

### 阶段 0：接口契约冻结（架构负责人，串行，先做）

在任何人写业务代码之前，先把"别人依赖我的东西"定死：

- `config.py`：采样率、转速、窗口、步长、样本量、轴承几何、网络超参、路径常量、绘图风格。
  附带命名兼容层：`import config` / `import config_bp` / `from bp import *` 三种写法都指向**同一个模块对象**（`sys.modules` 别名），
  避免"两份配置漂移"这类最隐蔽的 bug。
- `dataio.py`：数据集字段（`X/y/seg_id/raw/meta`）的形状与语义、`validate()` 校验门、
  `split_by_segment()` 防泄漏划分、指标落盘。

**冻结的含义**：契约文件一旦定稿，子任务只读不写；要改必须走「偏差登记」并由负责人统一改。

### 阶段 1：三条工作流并行（子代理）

| 任务 | 负责 | 写范围 | 依赖 |
|---|---|---|---|
| A 数据生成 `preprocess1.py` | sim-data | 脚本 + `data/*.npy` + 预处理图 + 单测 | 契约冻结 |
| B 信号分析 `fft1.py` | spectral | 脚本 + 3 张分析图 + 分析报告 | 契约冻结；**运行期**依赖 A 的数据 |
| C 模型训练 `cnn1.py` + `nn1d/` | cnn-model | 脚本 + 手写 NN 包 + 4 张训练图 + `metrics.json` + 单测 | 契约冻结；**运行期**依赖 A 的数据 |

写范围互不重叠（唯一共享的是只读的契约文件），因此三条流可以真并行。
B、C 的**代码**不依赖 A，只有**跑数据**依赖 A —— 这个拆分是并行度的关键：
先写代码 + 用临时小数组自测，A 交付后立刻接上正式数据。

### 阶段 2：文档与一键验收

`README.md` + `tests/run_all.py`（一键跑通三步 + 两个单测，输出 PASS/FAIL 汇总表）。
此阶段**只读**他人代码；发现 bug 不自己改，报给负责人 —— 避免验收者顺手改实现导致"验收即污染"。

### 阶段 3：集成验收（负责人，串行，最后做）

见 §5 验收清单。

---

## 3. 技术方案要点

### 3.1 数据生成（第一步）

**信号模型** —— 正常与故障的差异必须落在可解释的物理量上，而不是"随便加点噪声"：

```
正常: x(t) = Σ A_h·sin(2π·h·f_r·t + φ_h) + n(t)          h = 1..3, f_r = 50 Hz
故障: x(t) = 正常成分 + Σ_k A_i·exp(-α·(t-k·T))·sin(2π·f_n·(t-k·T)) + n(t)
                                    T = 1/BPFO ≈ 6.30 ms
```

其中 `f_n ∈ [500, 4000] Hz` 为轴承座共振频率（**条内固定、条间随机**），
`α = IMPULSE_DECAY = 800 /s` 决定冲击衰减快慢，`n(t)` 为高斯白噪声。

**为什么这样设**：
- 冲击串的**重复频率 = BPFO**，这正是第二步包络谱能把它揪出来的物理依据；
- 共振频率条间随机 → 模型不能靠"记住某个频率成分"作弊，必须学**冲击的周期性包络形状**；
- 这也让第三步的准确率有真实含义，而不是学一个恒定偏置。

**滑窗**：`WINDOW=1024`（≈85.3 ms，含约 13.5 个冲击周期，足够 CNN 看出周期性），`STRIDE=256`（75% 重叠，扩样本量）。
每条 3 s 信号 → `(36000-1024)/256 + 1 = 137` 个窗口；800 条 → 109600 个样本。

### 3.2 时域/频域分析（第二步）

- **时域**：波形图 + 20 ms（转频周期）参考线 + 冲击位置红线（间隔 1/BPFO ≈ 6.30 ms）。
  量化指标：RMS、峰值、峰峰值、**峭度 (kurtosis)**、裕度指标。峭度对冲击最敏感，是"早期故障"的经典指征。
- **频域**：`np.fft.rfft` 幅值谱，0–2000 Hz 主视区，标注 BPFO/BPFI/BSF/FTF 与 2×f_r。
  但**频谱本身对早期轴承故障不敏感**（冲击能量被分摊到很宽的共振带上）。
- **包络谱（关键判据）**：`np.abs(x)` 去均值后 FFT → 得到冲击包络的重复频率，BPFO 1x/2x/3x 谱峰清晰可辨。
  这是滚动轴承故障诊断的标准手法，也是"为什么要做第二步"的答案：它给第三步提供了物理依据。

> 环境无 scipy ⇒ 用 `|x|` + 可选 `np.convolve` 滑动平均近似 Hilbert 包络，禁止引入新依赖。

### 3.3 1D-CNN（第三步）

**环境硬约束**：torch / tensorflow / scipy / sklearn **均未安装且不允许安装**。
因此 `nn1d/` 是**纯 NumPy 手写的前向 + 反向传播**（Conv1d、BatchNorm1d、ReLU、MaxPool1d、GlobalAvgPool、Dropout、Linear、softmax-CE、Adam）。

- 这反而更契合教学目标：网络不再是黑盒，每个算子都能看到前向/反向公式。
- 代价：训练慢。因此网络刻意做小（16/32/64 通道）。
- **实测代价（重要）**：完整一轮 = 87680 训练窗 / batch 64 = 1370 步 ≈ **212 s**；`EPOCHS=40` 需 **2.3 h+**。
  而幻灯片第 3 步④要求 Loss/Accuracy **曲线**（单 epoch 只有 1 个点，画不出曲线）。
  故 `cnn1.py` 的 canonical 默认走**部分轮**：`--steps-per-epoch 90`（5760 窗/轮）+ `--time-budget 900 s`，
  实测 9 轮 / 平均 26–35 s 轮（多次运行 27.4 / 27.2 / 26.1 / 34.7 / 28.3 s，随机器负载浮动）/ 总 235–312 s，早停于第 9 轮；要完整轮加 `--full-epoch`。
  **`config.py` 的 `EPOCHS=40` 未被修改**，CLI 覆盖值在 `metrics.json.training_info` 如实留痕。
  详见 §7 局限第 6 条与 §9 偏差登记。
- **已落地的两处算子优化**（Lead 实测定位瓶颈后由 task-3 owner 实施）：
  ① `MaxPool1d` 在 `k == stride` 时走 `reshape(N,C,L//2,2).max(-1)` 快路径（前向 14.32 → 3.26 ms，4.4×，
  与通用实现最大绝对差 `0.000e+00`）；② `Conv1d` 前向输出转 C 连续，避免 stride 布局拖慢下游 BN（1.5×）。
  优化前 195 ms/step，现约 135 ms/step。
- **修掉一个真 bug**：`BatchNorm1d` 反向 `dx` 第三项漏乘 `istd`（数值梯度 `rel_err ≈ 1` 量级）。
  该 bug 会被「合成数据太易分」掩盖 —— 指标正常但梯度是错的，已由数值梯度检验兜住。

结构：

```
输入 (B, 1, 1024)
  Conv1d(1→16,  k=9)  → BN → ReLU → MaxPool(2)     # 1024 → 512
  Conv1d(16→32, k=7)  → BN → ReLU → MaxPool(2)     # 512  → 256
  Conv1d(32→64, k=5)  → BN → ReLU → MaxPool(2)     # 256  → 128
  GlobalAvgPool → Dropout(0.3) → Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→2)
```

**为什么用 GlobalAvgPool 而不是 Flatten**：Flatten 会把参数量推到 10 万级，在 10 万样本的合成数据上秒过拟合；
GAP 把参数量压到 **18594**（`metrics.json` 实测值，与手算一致），同时保留"哪段波形有冲击"的定位能力，
并且对输入相位轻微平移更鲁棒。

**为什么全局平均池化后还要接两层 FC**：给模型一个非线性的类别判别头，而不是让 Softmax 直接读 64 维特征。

### 3.4 三条正确性红线（本次集成验收重点）

1. **防数据泄漏**：`dataio.split_by_segment()` 按 `seg_id` 划分；任何标准化统计量只从训练集拟合。
2. **train/eval 模式**：BatchNorm 与 Dropout 推理时必须切换（BN 用滑动平均，Dropout 关闭）。
3. **反向传播正确**：中心差分数值梯度校验 `Conv1d / BatchNorm1d / Linear` 的 `dW`、`db`；
   外加"32 样本训练 200 步 → 训练准确率 100%"的过拟合测试来证明梯度真的在下降。

---

## 4. 目录与产物契约

```
mech-fault-diagnosis/
├── config.py                 # [冻结] 唯一参数源 + 路径常量 + 绘图风格 + config/config_bp/bp 别名
├── dataio.py                 # [冻结] 数据集契约：save/load/validate/split_by_segment/save_metrics
├── preprocess1.py            # 第一步：生成仿真信号 + 滑窗 + 打标签
├── fft1.py                   # 第二步：时域 / 频谱 / 包络谱
├── cnn1.py                   # 第三步：1D-CNN 训练与评估
├── nn1d/                     # 纯 NumPy 手写神经网络内核（grad / layers / model）
├── data/clean_signal1.npy    # 数据集（图片指定的文件名）
├── figures/*.png             # 全部图表
├── results/metrics.json      # 指标 + 配置快照 + 环境信息（可追溯）
├── tests/                    # 单测 + run_all.py 一键验收
├── analysis_report.md        # 第二步产出：量化分析报告
├── README.md                 # 使用手册
└── PLAN.md                   # 本文件
```

**数据集字段契约**（`data/clean_signal1.npy`，`np.save` 的字典）

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| `X` | `(N, 1024) float32` | 滑窗截取后的信号片段，CNN 输入 |
| `y` | `(N,) int64` | 标签，`0`=正常，`1`=故障 |
| `seg_id` | `(N,) int64` | 该窗口来自第几条原始信号（**防泄漏划分的依据**） |
| `raw` | `(800, 36000) float32` | 原始长信号，供第二步时域/频域分析 |
| `meta` | `dict` | 采样率、窗口、轴承频率、标签映射、种子等自描述信息 |

---

## 5. 验收清单（Definition of Done）

**可运行**
- [x] 具备 numpy/matplotlib 的解释器下 `python preprocess1.py`、`python fft1.py`、`python cnn1.py` 均退出码 0。
- [x] `python tests/run_all.py` 汇总表全 PASS。

**数据正确**
- [ ] `dataio.load_dataset()` 通过 `validate()`（形状、标签、NaN/Inf、`seg_id` 全覆盖）。
- [ ] 训练集与测试集**两侧都含两类**。
- [ ] 同一 `seg_id` 的窗口只出现在一侧（防泄漏，可脚本核查）。
- [ ] 两次运行 `preprocess1.py`，数据集 sha256 一致（可复现）。

**信号分析正确**
- [ ] 故障信号自相关在 `1/BPFO` 处出现显著峰值（证明冲击重复频率落在设计值）。
- [ ] 包络谱在 BPFO 1x/2x/3x 处的谱峰显著高于正常样本，且报告给出量化 dB 值。
- [ ] 峭度：故障 > 正常（冲击性指征）。

**模型正确**
- [ ] 数值梯度校验：`Conv1d / BatchNorm1d / Linear` 最大相对误差 < 1e-4。
- [ ] 小样本过拟合测试：32 样本 200 步 → 训练准确率 100%。
- [ ] `results/metrics.json` 存在且自洽：混淆矩阵四格之和 == 测试样本总数，
      由混淆矩阵反算的 accuracy == 文件中的 `test_accuracy`。
- [ ] `test_accuracy ≥ 0.90`（合成数据上应接近 1.0；若偏低先查泄漏与 train/eval 模式，而不是加大网络）。
- [ ] 四张训练图（Loss / Accuracy / 混淆矩阵 / 单样本预测）非空白。

**教学可用**
- [ ] 每个脚本运行时逐步打印中文说明；图表中文不乱码、带物理量与单位。
- [ ] `README.md` 中每条命令都真实执行过，引用的参数值与 `config.py` 一致。
- [ ] `cnn1.py` 结尾明确写出**本实验的已知局限**。

### 5.1 验收执行结果（Lead 独立复验，非子代理自述）

> 下列结论由架构负责人**亲自重跑或独立重算**得出；关键项刻意不复用被检实现的输出，避免循环论证。

| 判据 | 结果 | 证据（Lead 实测） |
|---|---|---|
| 三脚本退出码 0 | ✅ | 具备 numpy/matplotlib 的解释器下逐个实跑；`-W error` / `-X dev` 下 task-1 同样干净 |
| 数据集 `validate()` | ✅ | 形状全对：`X (109600,1024) f32`、`y`/`seg_id (109600,) i64`、`raw (800,36000) f32` |
| 两侧都含两类 | ✅ | train 87680 窗(640 段) normal 44251/fault 43429；test 21920 窗(160 段) normal 10549/fault 11371 |
| **防泄漏（红线①）** | ✅ | `np.intersect1d(np.unique(seg[train]), np.unique(seg[test])).size == 0` —— 同一 `seg_id` 不跨两侧 |
| `preprocess1.py` 可复现 | ✅ | 两次独立整跑 sha256 均为 `8c57a018f70d035e43496216ae84725452ebbfb24bbd45c7ed0d954b9a28ff27` |
| 冲击重复频率 = 设计值 | ✅ | 独立重算：检出 **477 个/3 s**（理论 476.25），平均间隔 **6.300 ± 0.064 ms**（理论 1/BPFO=6.299 ms，误差 **+0.54%**） |
| 包络谱 BPFO 谐波抬升 | ✅ | 1x/2x/3x：−52.1/−53.8/−53.6 dB → −18.2/−20.8/−23.4 dB，**+33.8/+33.0/+30.2 dB** |
| 峭度 故障 > 正常 | ✅ | 1.665 → 2.658（1.60×） |
| BN/Conv/Linear 数值梯度 | ✅ | 修复 BN 反向漏乘 `istd` 的 bug 后全部通过（修复前 `rel_err ≈ 1` 量级） |
| 小样本过拟合测试 | ✅ | 32 样本 200 步 → 训练准确率 100% |
| `metrics.json` 自洽 | ✅ | 混淆矩阵 `[[10549,0],[0,11371]]` 四格和 21920 == 测试样本数；反算 accuracy 1.0 == `test_accuracy` |
| `test_accuracy ≥ 0.90` | ✅ | **1.0**；per_class P/R/F1 全 1.0 |
| 参数量自证 | ✅ | `trainable_params = 18594 == trainable_params_manual`，`match_manual = true` |
| 四张训练图非空白 | ✅ | Loss 曲线 0.044→~0、Accuracy 曲线 0.982→1.0（**9 个真实点**）、混淆矩阵、单样本预测图均经目视核对 |
| 中文不乱码 | ✅ | 8 张图全部使用 `Microsoft YaHei`，轴标签带物理量与单位（时间(ms)/幅值(g)/准确率…） |
| A/B 单测 | ✅ | `tests/test_preprocess.py` **14/14** 全绿（Lead 实跑，exit=0） |
| C 单测 | ✅ | `tests/test_nn1d.py` **15/15** 全绿 |
| `tests/run_all.py` 一键验收 | ✅ | Lead 亲自执行 `tests/run_all.py --timeout 1800`（多次，含修复前后各一轮），每次都汇总表 **5/5 PASS**、退出码 0；最终验收跑总耗时 313.5 s（preprocess1 8.8 / fft1 3.8 / cnn1 289.9 / test_preprocess 6.9 / test_nn1d 4.1） |
| README 与代码一致 | ✅ | 逐条对照 `config.py`（39 项参数 + 行号）与目录清单；机器化校验 README + PLAN 的 `文件:行号` 引用**逐条命中、零越界**（脚本 `_research_tmp/final_verify_all.py` 实测 60 条 / bad=0），目录树行数 688/917/664/94 与 `splitlines()` 实测一致 |
| 纯净 GBK 重定向下可运行（**回归缺陷修复**） | ✅ | 修复前 `cnn1.py` 在 `python cnn1.py`（README 首推命令、无真实控制台）于训练前即 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2714'`、退出码 1；修复后 `cnn1.py` / `tests/test_nn1d.py` 在 `PYTHONIOENCODING=gbk` + 重定向下退出码均 0（`tests/test_preprocess.py` 同条件亦为 0，但靠 `import preprocess1` 的传递性副作用，未自行加守卫），全库 `.py` 字符均 GBK 可编码 |

**红线②（train/eval 模式）**：BatchNorm 推理用滑动平均、Dropout 只在训练期生效，已由 `tests/test_nn1d.py` 的
模式切换用例 + 训练/测试双曲线行为覆盖。

---

## 6. 关键决策记录（Decision Log）

| 决策 | 选择 | 理由 | 备选与为何不选 |
|---|---|---|---|
| 神经网络实现 | 纯 NumPy 手写 | 环境无 torch/scipy/sklearn 且不允许安装；教学价值更高 | 用 torch → 环境不具备，无法交付 |
| 配置组织 | 单 `config.py` + `sys.modules` 别名 | 一份参数无漂移；兼容历史 `config_bp`/`bp` 写法 | 三份配置 → 必然漂移 |
| 数据集容器 | `np.save` 字典（含 meta） | 单文件、自描述、可追溯；`npz` 亦可但 npy 更贴合幻灯片描述 | 裸数组 → 丢失 seg_id，无法防泄漏 |
| 划分策略 | 按 `seg_id` 分组划分 | 75% 重叠窗口若随机划分 = 严重泄漏 | 窗口级随机划分 → 指标虚高、结论无效 |
| 池化策略 | 三层 Conv + MaxPool + GlobalAvgPool | 参数量 18594（实测），抗过拟合、抗相位平移 | Flatten (~10 万参数) → 秒过拟合 |
| 正则化 | Dropout 0.3 + weight_decay 1e-4 | 合成数据极易过拟合；回归测试需要稳定的高准确率 | 无正则 → 训练/测试曲线大幅分离 |
| 故障判据 | 包络谱 BPFO 谐波族 | 频谱本身对早期轴承故障不敏感，包络解调是行业标准 | 只看频谱 → 判据不成立 |
| 中文字体 | `Microsoft YaHei` / `SimHei` 候选列表 + `Agg` 后端 | 无 GUI 环境必须用 `Agg`；字体缺失要有回退 | 默认字体 → 中文方框乱码 |

---

## 7. 已知局限与风险

1. **合成数据过于规整**：准确率接近 100% 是**预期结果**，不代表算法能力强。
   真实价值在于证明"流水线正确"，而非"模型优秀"。换 CWRU 等真实轴承数据，指标会明显下降。
2. **仿真模型的简化**：空载、恒转速、单一故障模式（外圈为主）、理想高斯噪声。
   真实工况存在变转速、多故障耦合、冲击与噪声混叠、传感器传递函数影响。
3. **纯 NumPy 的性能天花板**：无 BLAS 级卷积优化，训练耗时远高于 torch；因此网络规模刻意保守。
4. **包络谱近似**：`|x|` 不是严格 Hilbert 包络，在信噪比极低时判据会变弱；本数据集 SNR 足够，可接受。
5. **无跨工况验证**：仅在同一仿真分布内划分训练/测试，没有做"变转速/变负载"的域偏移测试。
6. **canonical 默认走「部分轮」**：`--steps-per-epoch 90`（共 1370 步/完整轮）是为在纯 NumPy 下同时满足
   「曲线可绘」与「20 分钟内跑完」而做的工程妥协。因此横轴 `epoch` **不等于**扫完整个训练集一遍，
   loss/accuracy 曲线上的"第 2 轮即收敛"比真实完整轮训练更快出现。要做严格收敛分析请用 `--full-epoch`
   （约 212 s/轮 × 40 轮，2.3 h+）。`config.py` 的 `EPOCHS=40` 等超参未被修改，CLI 覆盖值在
   `metrics.json.training_info` 中如实留痕（`steps_per_epoch` / `full_steps_per_epoch`）。
7. **准确率 1.0 的解读**：本数据集两类样本（50 Hz 转频 vs BPFO 冲击族）在包络谱上差异达 +30 dB 量级，
   1D-CNN 极易可分。这**证明流水线端到端正确**（信号模型 → 滑窗 → 防泄漏划分 → 网络 → 评估 → 出图闭环成立），
   **不证明**模型或方法在真实工业数据上有效。真实数据见 P1/P2。

---

## 8. 进阶路线（从教学 Demo 到工程可用）

| 阶段 | 动作 | 预期收益 |
|---|---|---|
| P1 | 换真实数据：CWRU / Paderborn / 自采数据集 | 暴露合成数据掩盖的问题（域偏移、不平衡、标注噪声） |
| P2 | 加噪鲁棒性矩阵：SNR 从 −5 到 +20 dB 扫描，画"准确率-信噪比"曲线 | 得到工程可用的性能边界，而非单点指标 |
| P3 | 频域输入对比：原始时域 vs FFT 幅值谱 vs 包络谱作为 CNN 输入，做消融 | 量化"特征工程 vs 端到端"的收益 |
| P4 | 变工况域自适应（DANN / CORAL） | 解决换转速即失效的问题 |
| P5 | 可解释性：Grad-CAM 映射回时域波形，看模型关注冲击还是注意力在别处 | 让结论可被工程师采信 |
| P6 | 边缘部署：参数量 < 10 万的模型量化后上 MCU | 让"诊断技能"落到设备端，而非实验室 |
| P7 | 从分类到剩余寿命预测（RUL） | 从"有没有故障"升级到"还能用多久" |

---

## 9. 偏差登记（Deviation Log）

> 契约文件或本方案与最终实现不一致时，必须在此登记：**谁、改了什么、为什么、影响范围**。

| 日期 | 项 | 变更 | 原因 | 影响 |
|---|---|---|---|---|
| 2026-10-13 | 初始冻结 | `config.py` / `dataio.py` 定稿 | 契约先行，避免三份配置漂移 | 三条工作流按本方案执行 |
| 2026-10-13 | `cnn1.py` CLI（**未改 `config.py`**） | 新增 `--steps-per-epoch`（默认 90）与 `--time-budget`（默认 900 s）；完整轮由 `--full-epoch` 显式开启 | 纯 NumPy 下完整轮 = 1370 步 ≈ 212 s，`config.EPOCHS=40` 需 2.3 h+，超出「单轮 > 30 s 必须给缩减方案」红线；但幻灯片第 3 步④要求 Loss/Accuracy **曲线**，单点不成立 | canonical 默认 9 轮 / 平均 26–35 s 轮 / 总 235–312 s（数值结果逐位一致，仅墙钟浮动），`history` 有 9 个真实点；**代价：1 个 epoch 不再等于扫完整个训练集**（90/1370 步）。`config_snapshot.epochs` 仍如实记录 40，`training_info` 另记 `steps_per_epoch`/`full_steps_per_epoch` 以自证 |
| 2026-10-13 | `nn1d/layers.py` MaxPool1d | 前向在 `k == stride` 时走 `reshape(N,C,L//2,2).max(-1)` 快路径，保留通用 `sliding_window_view` 回退；反向在 reshape 视图上 `put_along_axis` | 实测前向 14.32 ms → 3.26 ms（4.4×）；三个 MaxPool 合计占前向 49.3 ms（比三个卷积的 21.0 ms 还贵） | 两实现最大绝对差 **0.000e+00**（完全等价）；仅在 `k==stride` 成立，故保留回退分支 |
| 2026-10-13 | `nn1d/layers.py` Conv1d | 前向输出 `ascontiguousarray`，避免把 stride 布局传给下游 BN/ReLU/MaxPool | 非连续视图使下游归约变慢：strided BN 10.16 ms vs contig 6.67 ms（1.5×） | 一次拷贝换后续三层加速；数值语义不变 |
| 2026-10-13 | `nn1d/layers.py` BatchNorm1d 反向（**真 bug 修复**） | `dx` 第三项补乘 `istd`（原实现漏乘） | 数值梯度对比 `rel_err ≈ 1` 量级，属确定性错误而非精度问题 | 修复后所有层数值梯度检验通过；此前该 bug 被「合成数据太易分」掩盖，指标看不出异常 |
| 2026-10-13 | `nn1d/model.py` docstring | 训练规模描述由「6400 窗口 / 2–4 s 每轮」更正为「87680 训练窗口（1370 步）/ 完整轮 ≈ 212 s」 | 原文与实际相差约 100 倍，属文档性错误 | 无功能影响；避免后续读者据错误注释做扩容决策 |
| 2026-10-13 | `results/metrics.json` `reproduce.command` | 由 `python cnn1.py --epochs 1`（冒烟跑）改为 `python cnn1.py`（canonical 默认） | 原文件记录的是一次 1 epoch 冒烟跑，与 DoD（曲线 + 早停）不符 | 现行 `metrics.json` 为 canonical 默认档产物：9 轮 history、`test_accuracy=1.0`、混淆矩阵 `[[10549,0],[0,11371]]` 自洽、`trainable_params=18594` 与手算一致 |
| 2026-10-13 | `cnn1.py` 产物覆写副作用（**验收期发现，未改代码**） | 无代码变更；仅在 `README.md` §3 与本节如实披露 | 验收期 Lead 用 `cnn1.py --steps-per-epoch 1 --time-budget 120` 做 GBK 复验，该次实验跑**无条件覆写**了 canonical 工件：`results/metrics.json` 变成 `steps_per_epoch=1`、`test_accuracy=0.51875`、混淆矩阵 `[[0,10549],[0,11371]]`，4 张训练图同步变成未收敛形态 | 已用裸 `python cnn1.py` 还原，数值逐位一致（`test_accuracy=1.0`、`cm=[[10549,0],[0,11371]]`、9 轮、90/1370 步）。README §3 新增 ⚠️ 副作用提醒；**根因是「实验跑」与「canonical 跑」共用同一输出路径**，属设计取舍，未改脚本行为 |
| 2026-10-13 | `cnn1.py` / `tests/test_nn1d.py`（**GBK 崩溃回归修复**） | 两个入口补 `_force_utf8_stdout()` 守卫（`stream.reconfigure(encoding="utf-8", errors="replace")`，失败静默跳过）；同时把 `cnn1.py`、`tests/test_nn1d.py`、`tests/test_preprocess.py` 中的 U+2714/U+2718/U+2705 全部降级为 GBK 可编码的 `√`/`×` | Windows 在 stdout **被重定向**（CI / `> log` / `| tee`，无真实控制台）时按 locale 编码（中文 = GBK），非 GBK 字符直接抛 `UnicodeEncodeError` 并以**退出码 1** 结束。此前被 `run_all.py` 向子进程注入 `PYTHONUTF8=1` 完全掩盖，只有手工直跑才暴露。**注**：`tests/test_preprocess.py` 自身**未加**守卫，它能通过纯属 `import preprocess1` 触发了 `preprocess1.py:67` 模块级 `_force_utf8_stdout()` 的**传递性副作用** —— 这不可靠（依赖导入顺序），已在 README §8 Q1 如实说明 | `cnn1.py` 与 `tests/test_nn1d.py` 在 `PYTHONIOENCODING=gbk` + 重定向下退出码均 0（`test_preprocess.py` 同条件下亦为 0，但依赖上述副作用）；全库 `.py` 字符均 GBK 可编码；`run_all.py` 回归 5/5 PASS；**纯互补变更，无数值语义改动** |

> 契约文件 `config.py` / `dataio.py` 自冻结起**一字未改**；上表所有变更均发生在实现层、CLI 层或文档层。
