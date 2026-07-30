# test_direction_speed.py 实验方案（rebuttal 纯文本版）

## 0. 实验目的与汇报约束

`sensitivity_analysis.py` 证明的是 **embedding 层面**：LeGO-CLIP 对 motion-related keyword 的替换更敏感。
本实验要证明 **motion 层面**：这种敏感度**真正传导到了生成的动作上** —— 换掉关键词，生成动作的物理运动方向/速度真的跟着反过来。

> **汇报约束（决定了整个指标设计）**：当前处于 rebuttal 阶段，**只能回复纯文本，不能放表格和图**。
> 因此所有指标必须满足：① 是**标量**；② 有**物理单位**，一句话能解释清楚；③ 审稿人扫一眼数字就能判断哪个模型更好。
>
> 由此放弃：ROC-AUC（解释成本太高）、俯视轨迹图、多窗口对照表、Cohen's d。
> 只保留下面 §2 的四个「每秒 XX」标量 + 每类一个 gap + 可选的一个百分比。

---

## 1. 训练集词频调研（模板设计的依据）

在 `dataset/HumanML3D/texts` 的 43692 条 caption（排除 M 前缀镜像文件）上统计：

| 关键词 | caption 数 | 占比 | 结论 |
|---|---|---|---|
| left / right | 8620 / 11286 | 19.7% / 25.8% | 充足 |
| forward | 10182 | 23.3% | 充足 |
| backward / backwards | 1948 / 1525 | 4.5% / 3.5% | 充足 |
| slow | 2185 | 5.0% | 充足 |
| quick | 802 | 1.8% | 偏少，可用 |
| fast | 266 | 0.6% | 少，仅作辅助模板 |
| clockwise（纯，不含 counter/anti） | 641 | 1.5% | 偏少，有风险 |
| counterclockwise / counter-clockwise / anticlockwise | 449 | 1.0% | 偏少，有风险 |
| **leftward / rightward** | **10 / 11** | **0.02%** | **训练集中基本不存在，禁用** |
| rapidly / briskly / hurry / stroll | 47 / 76 / 20 / 17 | <0.2% | 禁用 |

**两条硬性结论**：

1. `leftward/rightward`、`briskly/stroll/rapidly` 不能用作模板 —— 用训练集里不存在的词，测的是「模型没见过的词」而非「是否遵循关键词」，两个模型都会掉到随机水平，实验失去区分度。
2. `clockwise/counterclockwise` 词频只有约 1%，两模型都可能接近随机。这一类**必须先跑 GT Oracle**（§7）：若 GT 上判据成立但两模型都无区分度，说明是 HumanML3D 数据本身不支持该概念，应在 rebuttal 中如实说明，不能算方法缺陷。

---

## 2. 汇报指标：四类统一为「每秒的 XX」

这是纯文本 rebuttal 最好的结构 —— 四类用同一个句式，审稿人读一遍就懂全部。

| 类别 | 指标 | 单位 | 符号约定 | 参考系 |
|---|---|---|---|---|
| left / right | 每秒**横向**位移 | m/s | **+ = 向左** | 第 0 帧朝向（固定） |
| slowly / quickly | 每秒**位移大小** | m/s | 恒正 | 无关 |
| forward / backward | 每秒**沿自身朝向**的位移 | m/s | **+ = 前进，− = 倒退** | **每帧当前朝向** |
| clockwise / ccw | 每秒**朝向转角** | deg/s | **+ = 逆时针（俯视）** | 每帧当前朝向 |

每类在 100 个 motion 上取平均，给出 4 个数（2 模型 × 2 文本）。

### 2.1 让审稿人一眼看出谁赢的两个抓手

**抓手 A：符号有没有翻转。** left/right 和 forward/backward 的指标是有符号的。如果 baseline 对两条文本给出**同号**数值，就直接说明它压根没区分关键词 —— 这比任何百分比都有力。

**抓手 B：gap（两条文本的差值）。** 一个带单位的标量，直接比较两个模型的区分能力：

```
gap = mean(指标 | 文本A) − mean(指标 | 文本B)
```

speed 类因为都是正数没有符号翻转，改用**倍率** `mean(quickly)/mean(slowly)` 更直观。

**可选补充（每类一个百分比）**：PPA = 100 个配对样本中方向正确的比例，chance = 50%。
一句话即可："LeGO reaches 91% vs. MDM 58% (chance 50%)"。与均值互补：均值说幅度，PPA 说一致性。

---

## 3. 固定参考坐标系（本实验的地基）

`recover_from_ric` 输出世界坐标 `J ∈ R^{T×22×3}`，`y` 是高度，`x/z` 是地面。

`recover_root_rot_pos` 中 `r_rot_ang[0]=0`（`utils/motion_process.py:50`），根节点旋转在第 0 帧恒为单位四元数；
但身体实际朝向由模型生成的 ric 决定，第 0 帧可能略微侧身几度。**朝向必须从关节显式计算，不能假定为 +Z。**

关节索引依据 `face_joint_indx = [2, 1, 17, 16]`（`utils/motion_process.py:116`，即 r_hip, l_hip, sdr_r, sdr_l），是解剖学定义：

```
up = (0, 1, 0)
hproj(v) = (v_x, 0, v_z)                          # 投影到地面

r_t_raw = hproj( (J[t,2] - J[t,1]) + (J[t,17] - J[t,16]) )   # 指向人体右侧
r_t     = normalize( smooth(r_t_raw, w=5) )       # 先滑动平均再归一化，抑制生成噪声
f_t     = normalize( cross(up, r_t) )             # 逐帧朝向
l_t     = -r_t                                    # 逐帧左向量

R0 = r_s,  F0 = f_s,  L0 = -r_s                   # 以窗口起始帧 s 为基准的固定参考系
```

### 3.1 ⚠ 坐标轴方向的实测纠正

**`+X` 指向人体左侧，`−X` 指向人体右侧，`+Z` 是正前方。**

GT 实测：`r_t`（指向人体右侧的向量）在 HumanML3D 的 `new_joints` 上**恒等于 `[-1,0,0]`**，`F0` 恒为 `[0,0,+1]`。
GT 交叉验证一致：真正 "walks to the right" 的样本 x 位移全为负。

> 这与「x 轴指向人体右边」的直觉相反。若按直觉写，rebuttal 里的正负号会**整体反向**。
> **建议 rebuttal 正文里不要提 x 轴**，直接写 "lateral displacement (positive = leftward)"，把坐标约定藏起来。

### 3.2 通用底层量

```
p_t = hproj( J[t,0] )                # 根节点水平位置
Δp_t = p_{t+1} - p_t                 # 逐帧位移
v_t = Δp_t * fps                     # 水平速度, m/s, fps = 20
d   = p_{e-1} - p_s                  # 窗口内净位移
P   = Σ_t |Δp_t|                     # 窗口内路径长度（走过的总里程）
Δt  = (e - s) / fps                  # 窗口时长, 秒
```

**静止样本**：`P < 0.3 m` 标记为 `no-move`，方向判据对其无意义，从 PPA 中剔除并单独汇报占比
（否则原地不动的样本会随机贡献 50% 准确率，虚高结果）。均值仍照常统计（静止样本贡献接近 0，不会造假）。

---

## 4. 四类指标的计算方法

### 4.1 left / right —— 每秒横向位移（+ = 向左）

```
m_lat = < d , L0 > / Δt              # m/s
```

用**第 0 帧的固定左轴 `L0`**，因为"向左走"的语义是相对**起始朝向**而言的。
（若改用每帧当前的 `l_t`，测到的是"侧身平移 strafe"；人若先左转再直走，逐帧 `l_t` 投影会接近 0，语义就错了。）

符号已用 GT 严格校准（§7）：真正"走向左/左转"的 GT 样本全为正，"走向右/右转"的全为负。

**⚠ 这一类唯一的坑：净位移会被 U 型/弧线回转自我抵消**（GT 里 `000093 "walks in an arc to the left"` 净位移反而偏右）。
因为用固定轴时 `Σ(Δp_t · L0) ≡ (p_e − p_s) · L0`，逐帧累加和净位移严格相等，换算法解决不了。

**唯一解法是缩短统计窗口。** 代码同时计算三个窗口，**rebuttal 默认汇报前 6 秒（0–120 帧）**：
动作开头意图最纯净，且叙述简单（"measured over the first 6 seconds"）。三个窗口的数都存进 CSV 供选择。

### 4.2 slowly / quickly —— 每秒位移大小（恒正）

```
m_speed = P / Δt                     # m/s
```

用**路径长度 P** 而非净位移：绕圈/曲线走也是在走，净位移会低估。固定生成 196 帧使两组时长严格相同，速度差异直接体现为里程差异。

**辅助量（可选，写进 rebuttal 一句话很有力）**：

```
cadence = (左脚上升沿数 + 右脚上升沿数) / Δt        # 步/秒
contact[t, side] = ( J[t, side_idx, 1] < 0.05 )   # 脚高 < 5cm；side_idx = 10(左) / 11(右)
```

步频是最强的生物力学证据（"LeGO 的 quickly 是 1.9 步/秒、slowly 是 0.8 步/秒"）。
沿用 `utils/metrics.py:489` 的脚部关节 `[10, 11]`，与 skating ratio 同一套触地定义。

> 263 特征的最后 4 维本身就是 foot contact，但那是**模型生成的通道**，可能不可靠；这里统一**从关节坐标重算**。

### 4.3 forward / backward —— 每秒沿自身朝向的位移（+ = 前进）

```
m_fwd = Σ_t < Δp_t , f_t > / Δt      # m/s，有符号
```

含义：**人往哪走**和**人朝哪看**是否一致。正 = 顺着自己面朝的方向走，负 = 背着自己面朝的方向走（倒退）。

**关键在于用每帧当前的 `f_t`，而不是第 0 帧的 `F0`。** 文本是 `faces forward and walks backward`，
存在一个作弊解：模型**转身 180° 后往前走** —— 此时 `<d, F0>` 仍是负的，用初始朝向会误判为"成功倒走"。
而用逐帧 `f_t`：转完身后速度与朝向仍同向，`m_fwd > 0`，自动归类为 forward。
**这一个式子就同时考察了朝向与速度方向的一致性，不需要再单列「转身作弊率」。**

附带好处：逐帧投影不会被回转抵消（倒着走绕圈时每帧投影都是负的），所以这一类**没有 §4.1 的窗口问题**，可以放心用全程 9.8 秒。

**辅助量**：净转身角 `|Ψ|`（见 §4.4）。文本明确说 "faces forward"，理想应接近 0°，用于说明朝向是否被保持。

### 4.4 clockwise / counterclockwise —— 每秒朝向转角（+ = 逆时针）

```
δ_t   = atan2( < cross(f_t, f_{t+1}) , up > ,  < f_t , f_{t+1} > )    # 逐帧有符号增量角
Ψ     = Σ_{t=s}^{e-2} δ_t                                            # 累计转向角
m_yaw = degrees(Ψ) / Δt                                              # deg/s
```

**⚠ 必须逐帧增量累加，绝不能用首末朝向的夹角 `signed_angle(f_s → f_{e-1})`。**
首末夹角只落在 (−180°, 180°]，无法区分「转 +350°」与「转 −10°」，也无法区分「转 370°」与「转 10°」。
而 `walks in a circle clockwise` 的样本恰恰会转满一圈甚至多圈 —— 用首末夹角会把这类样本**系统性判错**。
逐帧累加没有 wrap-around 歧义。

**符号约定**：由 `< cross(f_t, f_{t+1}), up >` 的正负决定。绕 +Y 的正旋转把 `+Z` 转向 `+X`，而 `+X` 是人体左侧（§3.1），
所以 `Ψ > 0 = 左转 = 俯视逆时针 = counterclockwise`，与 `L0 = +X` 自洽。
**此符号不靠推导定稿，由 §7 的 GT 校准强制验证**（用 `000879 "walks and turns left"` 应为正、`000834 "walks forwards and turns right"` 应为负来交叉验证）。

**数值细节**：
- `f_t` 必须先平滑（§3 的 `w=5`），否则生成噪声会让 `δ_t` 乱跳。
- `|δ_t| > 30°/frame` 视为异常跳变并裁剪（20fps 下相当于 600°/s，超出人体合理转速）。
- 求和时正负噪声本身相互抵消，平滑 + 裁剪已足够。

**辅助量：轨迹绕转角 `Ψ_path`** —— 把 `f_t` 换成速度方向 `u_t = v_t/|v_t|`，只在 `|v_t| > 0.1` 的帧上累加。
用于区分「原地转身」（`Ψ` 大而 `Ψ_path` 小）与「沿圆周走」（两者都大）。

### 4.5 汇总

| 类别 | 指标 | 期望更大的文本 | 单位 | 参考系 | 窗口建议 |
|---|---|---|---|---|---|
| left / right | `<d, L0> / Δt` | left | m/s | 第 0 帧固定 | **前 6 秒** |
| slowly / quickly | `P / Δt` | quickly | m/s | — | 全程 |
| forward / backward | `Σ<Δp_t, f_t> / Δt` | forward | m/s | 每帧当前 | 全程 |
| clockwise / ccw | `Σδ_t / Δt` | counterclockwise | deg/s | 每帧当前 | 全程 |

---

## 5. 文本模板（4 类 × 5 对）

依据 §1 词频筛选，**已剔除 leftward/rightward/briskly/stroll 等训练集中不存在的词**。

### 5.1 left / right（A = left）
| # | A | B |
|---|---|---|
| 1 | A person walks toward left | A person walks toward right |
| 2 | A person walks to the left | A person walks to the right |
| 3 | A person turns left and keeps walking | A person turns right and keeps walking |
| 4 | A man steps to his left side | A man steps to his right side |
| 5 | A person is walking to the left direction | A person is walking to the right direction |

### 5.2 slowly / quickly（A = quickly）
| # | A | B |
|---|---|---|
| 1 | A person walks quickly | A person walks slowly |
| 2 | A person walks forward quickly | A person walks forward slowly |
| 3 | A person is walking at a fast pace | A person is walking at a slow pace |
| 4 | A man walks very quickly | A man walks very slowly |
| 5 | A person quickly moves forward | A person slowly moves forward |

### 5.3 forward / backward（A = forward）
| # | A | B |
|---|---|---|
| 1 | A person faces forward and walks forward | A person faces forward and walks backward |
| 2 | A person walks forward | A person walks backward |
| 3 | A man is walking forwards | A man is walking backwards |
| 4 | A person steps forward | A person steps backward |
| 5 | A person moves forward in a straight line | A person moves backward in a straight line |

### 5.4 clockwise / counterclockwise（A = counterclockwise）
| # | A | B |
|---|---|---|
| 1 | A person walks in a circle counterclockwise | A person walks in a circle clockwise |
| 2 | A person turns counterclockwise | A person turns clockwise |
| 3 | A person walks counterclockwise | A person walks clockwise |
| 4 | A man rotates his body counterclockwise | A man rotates his body clockwise |
| 5 | A person moves along a counter-clockwise path | A person moves along a clockwise path |

**样本量**：每类 5 模板 × 20 = **n = 100 / 类别 / 模型**。既满足"每对生成 20 个"，
又把 PPA 的标准误从 n=20 的 ≈11% 压到 ≈5%。同时输出逐模板（n=20）的数，用于说明结论不依赖某个句式。

---

## 6. 实验设置

| 项 | 取值 |
|---|---|
| LeGO | `output/0814_MDMCLIPlora_cl10_tcl2_0716_scratch_ricglobal1/net_best.pth`，`add_clip_lora=True` |
| Baseline MDM | `output/0814_MDMCLIP_b128/net_best.pth`，`add_clip_lora=False` |
| 网络 | `modeltype='mdm_bert'`，`text_encoder_type='clip'`，`dataset_name='t2m'` |
| diffusion steps | 50 |
| 生成长度 | 固定 196 帧，fps=20（9.8 秒） |
| 每条文本 | 20 samples（`batch_size=20`，一 batch 一条文本） |
| 反归一化 | `dataset/HumanML3D/Mean.npy` / `Std.npy`，再 `recover_from_ric(·, 22)` |
| 统计窗口 | `[(0,60), (0,120), (0,196)]` = 3s / 6s / 9.8s，一次生成算三遍 |

**总生成量**：4 类 × 5 对 × 2 文本 × 20 = 800 motion / 模型；两模型共 1600（80 batch/模型）。

### 6.1 配对噪声的实现方式

`GaussianDiffusionSimple.p_sample_loop`（`diffusion/gaussian_diffusion_simple.py:794`）**不接受 `noise` 参数**，
它在第 829 行内部 `torch.randn`；且 `p_sample` 每步还有 `torch.randn_like`（`no_random` 只作用于 dataset，不影响采样）。

因此**不修改 diffusion 代码**，改为在每次 `p_sample_loop` 调用前重设种子：

```
torch.manual_seed(seed_k); torch.cuda.manual_seed_all(seed_k)
```

`seed_k` 由 (类别, 模板 id) 决定，对内两条文本 + 两个模型共用同一个 `seed_k`。
这样不仅初始噪声相同，**整条采样轨迹的每步噪声都相同**，配对比 `noise=` 更彻底。
唯一变量就是关键词（和模型）。

---

## 7. 符号校准与 GT Oracle（`--calibrate`）

整个实验最容易出错的是**符号翻转** —— 一旦左右或顺逆搞反，rebuttal 的结论会整体反向。
内置自检模式，用 HumanML3D **GT motion** 跑完全相同的判据函数并断言符号。

**校准结果（已全部完成，24/24 通过）**。严格文本筛选：要求文件内所有 caption 都含关键词、都不含反义词，且排除 `left hand/arm/leg` 等肢体用法。运行 `python test_direction_speed.py --calibrate` 复现（只读 GT npy，不需要 GPU 和模型）：

| 判据 | GT 样本（值为该判据的实测数） | 结果 |
|---|---|---|
| 参考系 `R0 ≈ [-1,0,0]`, `F0 ≈ [0,0,+1]` | 全部样本 | **恒成立** |
| `m_lat > 0` for left | 000365 (+0.58), 000879 (+0.21), 002141 (+0.18), 002427 (+0.22) m/s | **4/4** |
| `m_lat < 0` for right | 000390 (−0.25), 000407 (−0.24), 000463 (−0.40), 000834 (−0.27), 001081 (−0.18) m/s | **5/5** |
| `m_fwd < 0` for backward | 000028 (−0.43), 000109 (−0.46), 000144 (−0.21), 000178 (−0.77), 000267 (−0.06), 000282 (−0.07) m/s | **6/6** |
| `m_yaw > 0` for turns-left | 000879 (+16.06) deg/s | **1/1** |
| `m_yaw < 0` for turns-right | 000834 (−16.56) deg/s | **1/1** |
| `m_yaw > 0` for counterclockwise | 000212 (+50.8), 003456 (+15.7), 009648 (+30.6), 010002 (+45.5), 011492 (+54.2) deg/s | **5/5** |
| `m_yaw < 0` for clockwise | 001236 (−40.7), 002448 (−38.4), 003329 (−46.2), 006926 (−55.1), 007662 (−42.8), 008872 (−47.4), 010378 (−42.2) deg/s | **7/7** |

**关键确认**：`m_fwd` 在 backward 组 6/6 为负，说明 GT 里的 "walks backward" 确实是**真倒着走**（速度与自身朝向相反），而不是转身后往前走 —— 这既验证了判据符号，也验证了它的语义。

**两个印证辅助量价值的案例**：
- `010020 "stumbles in a clockwise circular motion"`：`m_yaw = −0.37`（身体朝向几乎没转）但 `yaw_path = −36.6`（轨迹在绕圈）—— 人是踉跄侧身绕圈。因其贴近 0，不适合做断言样本，已从校准集排除。
- `009648 "turns around 180 degrees anti-clockwise"`：反过来，`m_yaw = +30.6` 而 `yaw_path = +0.42` —— 原地转身，没有位移。

两者说明 `m_yaw`（朝向转）与 `yaw_path`（轨迹绕）确实互补，缺一会漏判。

### 7.1 GT Oracle 上界

上表的实测值即 Oracle：**左右 ±0.18~0.58 m/s，前后 −0.06~−0.77 m/s，顺逆 ±16~55 deg/s**。

rebuttal 里一句话即可："on ground-truth motions the same measurement gives +0.30 / −0.27 m/s for left/right,
so the measurement is reliable; LeGO's +0.42 / −0.39 matches it while MDM's +0.11 / +0.04 does not."
这一句同时说明**判据可信**且 **LeGO 接近上界**。对 clockwise/ccw 这一类尤其关键（§1 风险 2）。

---

## 8. 质量守护：Foot Skating Ratio

必须能回答"为了让方向词更敏感，是否牺牲了动作质量"，尤其要证明"倒着走"是真的走而非滑过去。

直接复用 `utils/metrics.py:480` 的 `calculate_skating_ratio(motions)`，输入 `(bs, 22, 3, T)`，
参数与主表一致（`thresh_height=0.05`, `thresh_vel=0.50`, `avg_window=5`, `fps=20`），保证可比。
按 模型 × 类别 汇报 mean，并单列 backward 组与 clockwise 组（最易出现滑步）。

---

## 9. 代码结构（`test_direction_speed.py`）

按"同功能必须抽成函数"的要求组织；四类判据共用同一套底层几何函数，只是投影轴 / 累加量不同。

```
# ---- 配置 ----
TEXT_PAIRS            # 4 类 × 5 对，含 category / 主判据 key / A-B 语义
EVAL_WINDOWS, N_SAMPLES, FPS, BASE_SEED, MODELS

# ---- 模型与生成 ----
build_model(args, ckpt_path, add_clip_lora)      # 复用 sample-lora.py:54 逻辑
load_norm()                     -> mean, std
generate(diffusion, text, n, length, seed)       # 调用前 manual_seed 实现配对
denorm_to_joints(sample, mean, std)  -> (n,196,22,3)

# ---- 通用几何（§3）----
hproj / smooth_unit / body_frames / ref_frame
root_horizontal / frame_disp / path_length
signed_yaw_increments / cumulative_yaw
foot_contact_from_joints / cadence

# ---- 四类主判据（§4）----
metric_lateral(joints, s, e)     -> m_lat  (+ 辅助)
metric_speed(joints, s, e)       -> m_speed, cadence
metric_facing_disp(joints, s, e) -> m_fwd  (+ |Ψ|)
metric_turn_rate(joints, s, e)   -> m_yaw  (+ Ψ_path)
compute_all_metrics(joints, s, e)-> dict   # 统一入口，一次算全部

# ---- 汇总（§2）----
summarize_pair(mA, mB)  -> mean_A, mean_B, gap, ratio, ppa, no_move_rate

# ---- 质量 ----
skating(joints)   # 复用 utils/metrics.py:480

# ---- 自检与主流程 ----
calibrate_on_gt()                # §7，只读 GT，不需要 GPU
main()                           # 模型 × 类别 × 模板 × 窗口 -> CSV + rebuttal 文本块
```

**不修改任何现有文件**；`calculate_skating_ratio`、`recover_from_ric`、`generate_src_mask`、
`load_lora_mdm_for_eval`、`load_ckpt`、`create_gaussian_diffusion_simple` 全部按现状 import 复用。

---

## 10. 产出

| 文件 | 内容 |
|---|---|
| `test_direction_speed_samples.csv` | 逐样本一行：model, category, template_id, side(A/B), sample_idx, window, 主判据, 辅助量, skating |
| `test_direction_speed_summary.csv` | 汇总：model × category × window × (mean_A, mean_B, gap, ratio, PPA, no-move 率, skating) |
| 控制台 | **直接可粘贴进 rebuttal 的英文文本块** |

### rebuttal 段落形态（数字待填）

> We further verify that the improved keyword sensitivity actually transfers to the generated motion.
> For each prompt we generate 100 motions (5 paraphrases × 20 samples) with **identical sampling noise**
> shared across the two prompts and the two models, so the only variable is the keyword.
> We report the **per-second root displacement** of the generated motion.
>
> **Left/right** (lateral displacement, positive = leftward, first 6 s): LeGO gives **+0.42** for "left"
> and **−0.39** for "right" — a clean sign flip — whereas MDM gives **+0.11** and **+0.04**, both positive,
> i.e. it does not respond to the keyword at all (gap **0.81** vs **0.07** m/s).
>
> **Forward/backward** (displacement projected onto the character's *own facing direction*, positive = forward):
> LeGO **+1.05** vs **−0.71**; MDM **+0.98** vs **+0.32** — MDM stays positive for "backward", meaning it keeps
> walking forward regardless of the word (gap **1.76** vs **0.66** m/s). Note this measure uses the per-frame
> facing direction, so "turning around and walking forward" is correctly counted as *forward*, not backward.
>
> **Speed** (per-second travelled distance): LeGO **1.62** ("quickly") vs **0.54** ("slowly"), a **3.0×** ratio;
> MDM **1.21** vs **0.97**, only **1.2×**. Step frequency shows the same trend (**1.9** vs **0.8** steps/s for LeGO).
>
> **Clockwise/counterclockwise** (turning rate of the facing direction, positive = counterclockwise):
> LeGO **+31.2** vs **−28.7** deg/s; MDM **+8.1** vs **−2.3** deg/s.
>
> Motion quality is not sacrificed: foot skating ratio is **0.081** for LeGO vs **0.086** for MDM.

---

## 11. 已知风险

1. **clockwise/counterclockwise 词频仅约 1%**：两模型都可能接近随机。GT Oracle 已确认判据在真实数据上完全可靠（12/12），所以若两模型都无区分度，那是数据限制而非判据问题，如实报告。
2. **quickly (1.8%) / fast (0.6%) 词频偏低**：若这一类效果差，需说明是数据分布限制。
3. ~~`m_yaw` 与 `m_fwd` 的符号尚未 GT 校准~~ → **已完成，24/24 通过（§7）**。
4. **left/right 的净位移会被 U 型回转抵消**（§4.1），只能靠缩短窗口缓解，默认汇报前 6 秒。
5. **no-move 样本**必须从 PPA 中剔除并单独汇报占比，否则准确率虚高。
6. **配对噪声依赖调用前设种子**（§6.1）：要求两次 `p_sample_loop` 调用消耗随机数的顺序完全一致。同一模型下对内两条文本满足；跨模型因网络结构不同（lora 层）不消耗额外随机数，也满足。已在代码中通过缓存机制隔离，重跑不会改变已生成的样本。
