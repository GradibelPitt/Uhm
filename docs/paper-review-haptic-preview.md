# 两篇论文对 UHM 的启发与本次落地

阅读与验证日期：2026-10-09。

结论：保留现有 Python / PyTorch / FastAPI / TypeScript 技术栈，优先补齐“视觉判断 → 可追溯的触觉语义 → 时序渲染”。本次已实现独立的触觉包络预览，可以消费已有接触分析和 BodyForceEvent，不需要重新训练。以下架构选择是结合本项目的工程判断，不是论文已经验证过的 UHM 结果。

| 论文 | 原文方法与适用范围 | 对项目的启发 |
| --- | --- | --- |
| [Vis2Hap，ICRA 2023](https://arxiv.org/abs/2301.06826)；[IEEE 原链接](https://ieeexplore.ieee.org/document/10160373) | cGAN 从视觉生成高度图、摩擦系数频谱，再转换为摩擦图用于 TanvasTouch。训练用到 GelSight 和力传感器数据；“视觉输入”指推理阶段。实验对象是 15 种织物，用户研究有 10 人。 | 应保留可解释的中间表示，再由独立渲染器使用。纹理粗糙度、摩擦感和撞击时序需要不同的监督与输出，不能由一个强度标量包办。 |
| [VLH，2025 arXiv 预印本](https://arxiv.org/html/2508.01361v1) | 方法部分使用对齐的真实/VR 俯视画面和语言指令；OpenVLA-7B 经 LoRA 微调，输出速度、方向触觉和振动的 7D 向量。450 种组合，INT8 在 RTX 4090 上报告 4–5 Hz。 | 借鉴结构化输出、跨模态对齐与分维度评估。OpenVLA 的飞行控制动作空间不能直接替代我们的玩家身体事件。 |

IEEE 页面有访问验证；Vis2Hap 的全文阅读使用作者的 arXiv 版本，并通过[作者机构论文页](https://kclpure.kcl.ac.uk/portal/en/publications/vis2hap-vision-based-haptic-rendering-by-cross-modal-generation/)核对标题与会议信息。

VLH 摘要中的“100% texture discrimination”不能理解成通用触觉准确率：正文另报联合形状/振动模式识别平均 57%，语义泛化 35%，这些是不同任务。4–5 Hz 也不等于项目设想的 80/150 ms 端到端延迟目标。[正文实验](https://arxiv.org/html/2508.01361v1)

现在直接训练这两个模型不具备数据条件：仓库现有合成受击数据和本地 VLM 接触候选，没有与真实高度图、摩擦信号或经过人体标定的方向力配对。把候选结果当作真实触觉标签会把视觉误判固化到模型里。

本次实现包括：

- `src/uhm/runtime/haptic_preview.py`：独立 HapticCue / HapticPreview 数据结构；保存源视频、模型、规范化输入 JSON 的 SHA-256、来源事件、证据时间、对象与推断依据。输入指纹用于追溯，不是对源视频内容的重新验证。
- 接触候选必须与 `ok` / `partial` 窗口中的同类正标签相交。错误、未知、拒判、未分析区间保持空白；未知区域不定位。运动推断默认关闭，需要显式纳入。第三人称对象保持 `observed_subject`，不会自动升级为 `player`。
- 相邻同类、同区域、同依据的正区间先合并，再渲染。落地/撞击使用最长 220 ms 的短包络，蹬地 180 ms；持续支撑与承载使用带淡入淡出的持续包络。这里的时长和形状是版本化的设计规则，不是从论文复制的实验参数或模型预测。
- 所有包络峰值统一为 1；概率不参与幅度计算。接触结果的强度保持 `null`；BodyForceEvent 的感知显著性单独保存，也不改变等幅预览。没有牛顿值、纹理参数、硬件载波或硬件输出。
- 工作台新增独立预览区、脉冲/持续轨道、当前 6 秒放大、全片视图、当前包络值与点击定位。采样只依赖视频时间，暂停和跳转不会继续推进旧反馈。
- `GET /api/haptic-preview?include_inferred=false` 与 `uhm preview-haptics` 提供同一编译结果；原始 BodyForceEvent 协议、模型 checkpoint 和原有接触结果不变。

编译的是离线候选。源标签边界仍约为 500 ms；将包络平滑绘制出来，不会提高接触定位精度，也不构成实时视觉推理。一个长正区间被保守地解释为一个事件，不能恢复该区间中模型没有分辨出的多次撞击。

在项目目录运行：

```powershell
# 现有跑酷分析 → 独立预览 JSON；显式允许运动推断
.\.venv\Scripts\python.exe -m uhm preview-haptics runs/pov_demo.json --output runs/pov_demo.haptic-preview.json --include-inferred

# 同步工作台；在界面勾选“纳入运动推断”查看 POV 足部曲线
.\.venv\Scripts\python.exe -m uhm serve-actions data/raw/pov/prague_demo_4k.mp4 runs/pov_demo.json --port 8002

# 兼容原有模型事件流，显式提供视频时长
.\.venv\Scripts\python.exe -m uhm preview-haptics runs/evaluation_events.jsonl --source-kind events --duration-ms 20000 --output runs/baseline.haptic-preview.json
```

`runs/` 和上述媒体是本机已有、忽略提交的产物；干净克隆需要先生成或提供自己的分析文件。命令会拒绝把预览写回输入文件。

验证结果：全量 Python 测试 152 项通过；前端测试 31 项通过；TypeScript / Vite 构建通过；修改的 Python 文件 Ruff 检查通过。用已有真实录像的候选结果导出，跑酷得到 12 条曲线，跳高得到 5 条。跑酷原分析尚未覆盖全片，未分析区间不补信号；条数不是精度指标。浏览器已验证推断开关、曲线显示和点击跳转至 2.00 秒，控制台无错误。

下一步优先级：先对落地、起跳、支撑和困难负例做独立人工复核，保留来源录像与证据帧；按源录像隔离训练/验证/测试。然后分别检验事件检测与时刻、位置与拒判、主观触觉匹配、端到端延迟。取得可信配对样本后，再尝试适配现有 VLM 的 LoRA 或向较小的时序模型蒸馏；纹理生成应另建有接触表面与触觉监督的数据分支。
