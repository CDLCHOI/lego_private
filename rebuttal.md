# Crucial Evidence that LeGO's generated motion aligns text prompts better.

## Experiment Settings

为了提供更关键的证据来证明我们的 LeGO 生成的 motion 确实更加遵循 text prompts，我们设计了以下 6 个类别的对抗文本对，目的是为了验证我们的 LeGO 对这些 motion-related keyword 的遵循程度 (**left/right**, **quickly/slowly**, **forward/backward**, **counterclockwise/clockwise**, **left hand/right hand**, **run/walk**)。我们采用三个网络进行对比：

- **MDM**（our baseline）：冻结 CLIP text encoder
- **LeGO**（MDM + LoRA CLIP，从头训练）：在 CLIP text encoder 上添加 LoRA 可学习参数，与 diffusion model 联合训练（cos_loss=10, text_cos_loss=2）
- **MDM+LeGO-CLIP**（MDM + 预训练 LoRA CLIP）：加载 LeGO 预训练好的 LoRA CLIP 权重后继续训练 MDM

每个类别我们设计了 5 个不同的文本对如下（由于篇幅原因我们这里提供 2 个样本，我们会将完整的文本对添加到 revised version 里）：

1. "A person walks toward **left/right**"; "A person turns **left/right** and keeps walking"
2. "A person walks **quickly/slowly**"; "A person is walking at a **quick/slow** pace"
3. "A person faces forward and walks **forward/backward**"; "A person is facing forward and steps **forward/backward**"
4. "A person walks in a circle **counterclockwise/clockwise**"; "A person moves along a **counterclockwise/clockwise** path"
5. "A person is walking with his **left/right** arm raised"; "A person walks forward and raises their **left/right** hand up high"
6. "A person **runs/walks** forward"; "A person is **running/walking**"

**实验方法：** 每个类别包含 5 组对抗文本 × 每组文本生成 20 个 motion = 每侧 100 个 motion，即每个类别总共包含 200 个 motion。三个模型共享同一组随机噪声（通过 `torch.manual_seed()` 固定），确保差异仅来自文本变化和模型能力，消除采样随机性的干扰。

---

## Experiments (Metrics, Results and Analysis)

### 通用指标定义

每个实验汇报以下通用指标（除实验特有指标外）：

| 指标 | 含义与计算方法 |
|------|--------------|
| **Accuracy(%)** | 逐样本方向正确率。衡量"喂什么文本就朝什么方向做"的准确率。对每侧100个样本，统计主判据符号正确的比例：`Accuracy = (说关键词A时符号正确的比例 + 说关键词B时符号正确的比例) / 2`。例如 left/right 中，说 "left" 时要求侧向位移余弦 > 0（朝左），说 "right" 时要求 < 0（朝右）。**仅适用于有符号指标**（left/right, forward/backward, counterclockwise/clockwise, left hand/right hand）。speed 类数值恒正、只比大小，不计算 Accuracy。**随机瞎猜基线 = 50%**。 |
| **Diff** | 两组均值的差：`Diff = mean(说关键词A的100个样本) - mean(说关键词B的100个样本)`。衡量"换一个词后，动作变了多少"，diff 的绝对值越大说明两组动作被区分得越明显。对有符号指标，正的 Diff 配合符号方向正确（A组均值为正、B组均值为负），意味着干净的方向分离。 |

**阅读顺序：** (1) 先看 Accuracy 是否显著高于 50% → (2) 看两组均值是否分别落在正区和负区（**说left时均值为正且说right时均值为负**）→ (3) 最后比 Diff 的绝对值谁更大。

---

### 实验 1：left vs right

**测试目的：** 给定含 "left" 或 "right" 的文本，观察生成的人的行走方向差异。

**指标与计算方法：** 取人体根节点的净位移向量，投影到第 0 帧身体的"左"轴上，除以位移模长。得到的侧向位移方向余弦值域 [-1, +1]：+1 = 完全朝正左走，-1 = 完全朝正右走。符号决定方向（正=左，负=右），数值的绝对值决定方向纯度（0 = 纯正前/正后，±1 = 纯正侧向）。


**结果：**

| 模型 | left | right | Diff | **Accuracy** |
|------|:--------------------:|:---------------------:|:----:|:----------:|
| MDM | +18.0° | -34.9° | 52.9° | **65%** |
| LeGO | +27.5° | -73.9° | 101.4° | **75%** |
| MDM+LeGO-CLIP | +65.5° | -67.2° | 132.7° | **94%** |

**分析与结论：** MDM 虽然从均值层面上两组的符号是对的，即left时偏转角为 +18.0°（偏左）、right时为 -34.9°（偏右），说明left 和right 确实生成了方向相反的动作，但逐样本正确率仅 65%，略高于瞎猜基线50%，说明大量样本对上方向随机。LeGO的两种模型都具有较高的 的 Accuracy和diff，方向分离较清晰，但LeGO的左侧偏转幅度偏小。MDM+LeGO-CLIP的左右弗恩度最大。


---

### 实验 2：quickly vs slowly

**测试目的：** 给定含 "quickly" 或 "slowly" 的文本，观察生成的人的速度差异

**指标与计算方法：** 排除静止帧（根节点瞬时速度 < 0.1 m/s 的帧），只在移动帧上统计：总路径长度 ÷ (移动帧数 / 20fps)，得到平均移动速度（单位 m/s）。恒为正数，越大越快，只比大小不比符号。这避免了"走走停停"的 motion 因为停的时间长而被误判为"慢"。


**结果：**

| 模型 | quickly | slowly | Diff | 速度比 |
|------|:-----------------:|:---------------:|:----:|:-----:|
| MDM | 0.491 m/s | 0.439 m/s | 0.052 | 1.12x |
| LeGO | 1.259 m/s | 0.508 m/s | 0.751 | 2.48x |
| MDM+LeGO-CLIP | **0.954 m/s** | **0.331 m/s** | **0.623** | **2.88x** |

> 注：speed 类数值恒正，不计算 Accuracy。核心看 Diff 和速度比。

**分析与结论：** 在"quickly"和"slowly"的实验上，MDM展现出了非常接近的平均速度，LeGO和MDM+LeGO-CLIP则分别表现出了非常显著的速度差异，分别提现在他们的快慢速度之差(0.751和0.623)，说明我们的方法在理解速度副词时表现出了显著的优势


---

### 实验 3：forward vs backward

**测试目的：** 给定"面朝前向前走"和"面朝前后倒着走"的文本，观察生成的人是否真正朝对应方向行走——说"forward"时全程向前走，说"backward"时全程向后倒着走。不要求朝向保持不变（弧线行走、先原地转身再走都算有效），只要求所有移动帧的位移方向与文本要求一致。

**指标与计算方法：** 逐帧计算"根节点位移方向"与"当前身体朝向"的点积（`frame_dot[t] = <dp[t], forward[t]>`）。`> 0` 表示该帧朝脸的方向走（向前），`< 0` 表示背对脸的方向走（向后）。静止帧和原地转身帧（根节点瞬时速度 < 0.1 m/s）不参与判断。

**Accuracy**：逐样本方向符号正确的比例。对一个样本，将其所有移动帧的 `frame_dot` 取平均得到 m_align，m_align > 0 则 forward 侧"对"，m_align < 0 则 backward 侧"对"。——管的是**平均下来方向对不对**。

**Frame-Strict-Acc**：全程方向一致的样本比例。对一个样本，**其所有移动帧的 frame_dot 符号必须全部正确**才算"对"——管的是**有没有中途方向混乱**。例如说"backward"时，一个 motion 如果先往前走几步再开始倒着走，Accuracy 仍然可能判对（平均下来方向对），但 Frame-Strict-Acc 会判错（存在方向不一致的帧）。弧线行走、先原地转身再一致走都算对——因为静止/转身帧被排除，只要移动帧方向一致即可。


**结果：**

| 模型 | forward | backward | Diff | **Accuracy** | Frame-Strict-Acc |
|------|:-----:|:-----:|:----:|:----------:|:------:|
| MDM | +0.942 | -0.464 | 1.406 | **90%** | **55%** (A:89%, B:20%) |
| LeGO | +0.625 | -0.874 | 1.499 | **92%** | 42% (A:41%, B:42%) |
| MDM+LeGO-CLIP | **+0.944** | **-0.864** | **1.808** | **100%** | 47% (A:37%, B:57%) |


**分析与结论：** 三个模型的 Accuracy 都较高（MDM 90%，LeGO 92%，MDM+LeGO-CLIP 100%），说明平均而言都能区分 forward 和 backward。Frame-Strict-Acc 揭示了三者在"全程方向一致性"上的差异：

- **MDM** Frame-Strict 55%（A 89% / B 20%）：forward 侧极好——89% 的 motion 全程每一帧都在向前走。但 backward 侧仅 20%，大量 backward motion 中途出现了朝前走的帧（common failure: 先往前走几步，然后才开始倒着走）。MDM 的 backward 在平均层面正确（Accuracy B 侧 80%），但只有 20% 全程一致。
- **LeGO** Frame-Strict 42%（A 41% / B 42%）：两侧极为均衡。forward 侧的全程一致性（41%）远不如 MDM（89%）——LeGO 说"forward"时也常常混入方向不一致的帧。Backward 侧（42%）反而比 MDM（20%）好得多。LeGO 的 forward/backward 生成风格更对称，但单侧一致性均不高。
- **MDM+LeGO-CLIP** Frame-Strict 47%（A 37% / B 57%）：Backward 侧（57%）在三个模型中最高——说"backward"时超过一半的 motion 全程每一帧都在向后走，方向保持最稳定。Forward 侧（37%）低于 MDM 但接近 LeGO。

**Accuracy 维度：MDM+LeGO-CLIP > LeGO ≈ MDM。Frame-Strict 维度：MDM (55%) > MDM+LeGO-CLIP (47%) > LeGO (42%)。MDM 靠极高的 forward 一致性拉高了整体 Frame-Strict，但 backward 一致性（20%）是三模型中最差的。MDM+LeGO-CLIP 的 backward 方向一致性最好（57%），是最擅长稳定倒着走的模型。LeGO 两侧均衡但均不突出。**

---

### 实验 4：clockwise vs counterclockwise

**测试目的：** 给定含 "clockwise" 或 "counterclockwise" 的文本，观察生成的人的移动轨迹是否符合预期方向。

**指标与计算方法：** 逐帧计算身体朝向向量的有符号转角增量（atan2 保证 -180°~+180°），全程累加后除以时长，得到身体朝向旋转角速度（单位 deg/s）。正数 = 逆时针旋转（俯视，身体向左转），负数 = 顺时针旋转（俯视，身体向右转），绝对值越大转得越快。使用累加而非首末夹角：转满一整圈首末夹角 = 0°，累加才能得到真正的总转角（例如 360°）。


**结果：**

| 模型 | 说"counterclockwise"时 旋转角速度 | 说"clockwise"时 旋转角速度 | Diff | **Accuracy** |
|------|:---------------------------:|:--------------------:|:----:|:----------:|
| MDM | -0.43 (✗) | -10.80 (✓) | 10.38 | **56%** |
| LeGO | +2.66 (✓) | -27.69 (✓) | 30.35 | **73%** |
| MDM+LeGO-CLIP | **+18.43 (✓)** | **-25.09 (✓)** | **43.52** | **84%** |

> 表格中 (✓) 表示该组均值符号正确（counterclockwise 应为正，clockwise 应为负），(✗) 表示符号错误。

**分析与结论：** 这是所有模型中差异最大的类别。MDM 说 "counterclockwise" 时平均旋转角速度 = -0.43 deg/s（符号为负，但逆时针应为正），两组均值同为负数——**无论说 clockwise 还是 counterclockwise，MDM 都在顺时针转**。它完全不理解 "counter-" 前缀的空间含义，只是机械地偏向于生成某一种旋转方向，Accuracy 仅 56% 接近瞎猜。LeGO 做到了**说 "counterclockwise" 时均值为正（+2.66）、说 "clockwise" 时均值为负（-27.69）**——两组均值落到了正确的符号区间，Accuracy = 73%，相比瞎猜基线（50%）有实质性提升。说 "clockwise" 时旋转速率达 27.69 deg/s（三类中顺时针最快），diff = 30.35 deg/s。MDM+LeGO-CLIP 的 Accuracy = 84%——**三类中最高**。说 "counterclockwise" 时均值 +18.43 deg/s（逆时针）、说 "clockwise" 时均值 -25.09 deg/s（顺时针），不仅符号干净，而且逆时针旋转速率远大于 LeGO。**MDM+LeGO-CLIP > LeGO >> MDM。两个 LoRA 模型都有效理解了旋转方向语义，MDM+LeGO-CLIP 在逐样本层面区分力最强（Accuracy 84%），LeGO 在顺时针方向的旋转速率最大（-27.69 deg/s）。**

---

### 实验 5：left hand vs right hand

**测试目的：** 给定"举左手"或"举右手"的文本，观察生成的人举的是否正确的那只手。

**指标与计算方法：** 计算每一帧「左手腕 y 坐标 - 右手腕 y 坐标」，对并取取平均，得到左右手腕高度差（单位 m）。正数 = 左手高于右手，反之亦然。


**结果：**

| 模型 | left hand | right hand | Diff | **Accuracy** |
|------|:--------------------:|:---------------------:|:----:|:----------:|
| MDM | -0.070 (✗) | -0.301 (✓) | 0.231 | **64%** |
| LeGO | +0.351 (✓) | -0.391 (✓) | 0.743 | **100%** |
| MDM+LeGO-CLIP | **+0.427 (✓)** | **-0.616 (✓)** | **1.043** | **100%** |

> 表格中 (✓) 表示该组均值符号正确（left hand 时左手应更高 = 正数，right hand 时右手应更高 = 负数），(✗) 表示符号错误。

**分析与结论：** MDM 不管文本说举哪只手，都倾向于举右手。LeGO两种模型展现出了非常具有区分度的结果，且Accuracy = 100%，完美区分了左右手。MDM+LeGO-CLIP的左右手区分度会更大。

---

### 实验 6：run vs walk

我们发现 LeGO 有不少 motion 属于原地跑动（类似跑步机上跑步），根节点几乎不动但腿在高速交替，因此改用双脚相对运动速度来衡量。

**指标与计算方法：** 取双脚（左右脚，关节 #10, #11）每一帧相对于根节点的位置，计算帧间位移模长，对双脚取平均后乘以 20fps，得到双脚相对根节点运动速度（单位 m/s）。排除双脚静止帧（双脚平均速度 < 0.1 m/s），只在运动帧上取平均。恒为正数，期望说 "run" 的速度 > 说 "walk" 的速度。与实验 2 的全局移动速度不同，此指标去除根节点平移，能正确捕获原地跑步的高频脚步运动。只取双脚而非全部腿部关节，因为跑/走的核心差异在末端摆动，髋/膝关节反而会稀释信号。

**结果：**

| 模型 | run | walk | Diff | 速度比 |
|------|:-----------:|:------------:|:----:|:-----:|
| MDM | 1.536 m/s | 0.791 m/s | +0.745 | 1.94x |
| LeGO | **2.007 m/s** | **0.850 m/s** | **+1.157** | **2.36x** |
| MDM+LeGO-CLIP | 2.029 m/s | 1.040 m/s | +0.989 | 1.95x |

**分析与结论：** 三个模型均能正确区分 run 和 walk（速度比均 > 1.9x），但 LeGO 的速度比（2.36x）和绝对差值（+1.157 m/s）均为最高，diff% = 100%——所有配对样本上 "run" 的双脚速度均大于 "walk"。LeGO 的 run 双脚速度达到 2.007 m/s，说明其倾向于生成原地高频踏步的跑步风格（类似跑步机），双脚摆动频率最高。MDM+LeGO-CLIP 的 run 速度绝对值最高（2.029 m/s），但 walk 速度也偏高（1.040 m/s），导致速度比略低（1.95x）。MDM 的绝对速度最低但区分度已足够（1.94x, diff% 91%）。**LeGO > MDM+LeGO-CLIP ≈ MDM。以双脚相对速度衡量，LeGO 对 run/walk 的区分最干净。**

---

## Conclusion and Limitations

### LoRA 模型（LeGO 和 MDM+LeGO-CLIP）显著优于 MDM 的类别（4/6）：

1. **left/right**：Accuracy 从 MDM 65% → LeGO 75% / MDM+LeGO-CLIP 94%。MDM+LeGO-CLIP 位移偏转角约 ±66°，方向分离最干净；LeGO 的 A 侧仅偏转 +27.5°，方向纯度不足。
2. **quickly/slowly**：速度比从 MDM 1.12x → LeGO 2.48x / MDM+LeGO-CLIP 2.88x。两个 LoRA 模型的速度区分度均是 MDM 的 10 倍以上。
3. **left hand/right hand**：Accuracy 从 MDM 64% → LeGO 100% / MDM+LeGO-CLIP 100%。从"总是举右手"变为完美区分左右手。
4. **clockwise/counterclockwise**：Accuracy 从 MDM 56%（接近瞎猜）→ LeGO 73% / MDM+LeGO-CLIP 84%。三模型间差异最大的单项。

### LeGO 独有优势的类别（1/6）：

5. **run/walk**：LeGO 双脚相对速度比 = 2.36x（三类最高），diff% = 100%——所有配对样本上 "run" 的双脚速度均大于 "walk"，完美区分。MDM（1.94x）和 MDM+LeGO-CLIP（1.95x）持平。LeGO 的 "run" 双脚速度达 2.007 m/s，倾向于生成原地高频踏步的跑步风格（类似跑步机），以全局位移衡量会被误判为"慢"，但以双脚相对速度衡量反而是区分度最好的。

### MDM 仍具优势的类别（1/6）：

6. **forward/backward (Frame-Strict-Acc)**：MDM 的 Frame-Strict = 55%（A: 89%, B: 20%），三类最高。MDM 在 forward 侧的全程一致性极好（89% 的 motion 全程每一帧都在向前走），但 backward 侧是短板（仅 20%）。MDM+LeGO-CLIP 在 backward 侧最好（57% 全程一致），LeGO 两侧最均衡（A 41% / B 42%）。三个模型在 forward/backward 上的优劣势分布互补：MDM 强在 forward 一致性，MDM+LeGO-CLIP 强在 backward 一致性，LeGO 两侧均衡但都不突出。

### 关于 MDM+LeGO-CLIP 与 LeGO 的对比：

MDM+LeGO-CLIP（预训练 LoRA CLIP + 继续训练 MDM）相比 LeGO（从头联合训练）的优势：
- **clockwise/counterclockwise**：Accuracy 84% vs 73%
- **left/right**：Accuracy 94% vs 75%
- **quickly/slowly**：速度比 2.88x vs 2.48x
- **forward/backward backward 一致性**：Frame-Strict B 侧 57% vs 42%

LeGO 相比 MDM+LeGO-CLIP 的优势：
- **run/walk**：双脚速度比 2.36x vs 1.95x，diff% 100% vs 93%
- **forward/backward 均衡性**：两侧 Frame-Strict 41%/42%（均衡） vs 37%/57%（偏斜）

**整体而言，MDM+LeGO-CLIP 在语义方向精度类任务（left/right、cw/ccw、quickly/slowly）和 backward 方向一致性上更强；LeGO 在 run/walk 双脚运动区分度和 forward/backward 方向均衡性上表现更好。MDM 在 forward 全程方向一致性上仍有绝对优势（89%）。三种模型各有不可替代的强项，值得在 revised version 中详细消融分析。**

---

