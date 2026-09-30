# Universal Haptic Model · V0

**视觉视频 → 真实模型推理 → 玩家相对的 `BodyForceEvent` → 人体区域图。**

这是可训练、可替换模型的研究基础。默认示例是代码生成的简化人体接触动画，**不是 gameplay，也不是通用游戏理解能力的验证**。模拟事件只存在于界面的 **Manual mock** 模式；模型模式读取视频像素和 checkpoint，不读取标注或预先编写的事件时间轴。

## 一条命令运行

在本项目目录执行（需要安装 uv 和 Node.js/npm）：

```powershell
uv run uhm demo
```

打开 **http://127.0.0.1:8000**，点击 **Play video**。左侧播放视频，右侧人体接收真实推理事件。首次运行会安装锁定的 Python 依赖、生成示例数据、下载 ImageNet 预训练权重并训练小模型、构建前端。已有数据和 checkpoint 会复用。按 `Ctrl+C` 停止服务。

本次实现已经完成实际训练，工作区的 `checkpoints/baseline.pt` 可直接使用。CPU 版本是默认配置；没有依赖 GPU、VLM、游戏 API 或外部推理服务。首次准备需要网络下载依赖及预训练权重，后续推理可离线运行。

GitHub 仓库仅包含代码、配置、测试、文档和依赖锁文件，不包含实际数据集、标注、模型权重或运行日志。从仓库克隆后，使用上面的 `uv run uhm demo` 在本地生成合成示例并训练；接入真实数据需自行准备标注。

```powershell
# 重新训练默认示例模型，再启动
uv run uhm demo --retrain

# 仅启动已有构建和 checkpoint
uv run uhm serve

# 前端开发（另一个终端，后端仍使用8000端口）
cd frontend
npm run dev
```

## 已交付的六项

| 内容 | 实现 |
| --- | --- |
| Human Body Renderer | React + TypeScript + SVG，8个独立区域，整身映射、未知区域、强度红色梯度、持续及淡出 |
| Canonical Event Schema | Pydantic → JSON Schema → 自动生成TypeScript，前端Ajv校验 |
| Dataset + Annotation | 视频导入、静音H.264切片、JSONL标注、因果采样、源视频划分和范围校验 |
| Trainable Baseline | ImageNet预训练MobileNet空间特征 + GRU + 多任务预测头，训练/验证/保存/重载/导出 |
| Inference Pipeline | 视频像素 → checkpoint → 置信度门控、确认、合并、冷却 → JSONL事件流 |
| End-to-End Demo | FastAPI + WebSocket，以视频播放时间驱动后端推理及人体高亮，事件和诊断日志落盘 |

## 默认素材从哪里来

由 `src/uhm/data/example.py` 使用OpenCV绘制并通过FFmpeg编码。**当前生成的全部训练、验证和演示视频均为第三人称、正面视角（`viewpoint: third_person`），没有第一人称数据。** 画面包括简化的正面人体、移动物体、实际可见的接触点、NPC、环境物体和遮挡；没有把类别名、区域ID或标签文字写进像素。

- 8个原始源视频，切成288个1秒片段：216训练、72验证，按源视频划分。
- 默认训练直接从这8个源视频取过去1000ms的因果窗口，每125ms一个；1728个训练窗口、576个验证窗口，按窗口当前时刻监督。精确接触及负例区间保存在 `data/annotations/example_stream.jsonl`；原始弱片段标注保留供对照。
- 50%为非玩家反馈样本：普通运动、菜单、UI效果、NPC受击、环境事件及未知受力对象。
- 8个可见身体区域，以及身体位置不可观察的玩家样本。
- 独立种子的20秒测试视频，另有精确接触区间的测试标注。
- 当前示例只训练 `no_event` 和 `impact`；`collision`、`explosion` 是保留词汇，尚无示例训练支持。

青色外观表示这个合成世界里的玩家；这种约定不适用于任意游戏。训练与验证共享同一个生成器，只有源视频和种子独立。详细来源及限制见 `data/README.md`、`data/example_manifest.json`。生成的媒体和权重不纳入Git，可由命令重建。

每条 `ClipAnnotation` 包含 `viewpoint`，可取 `first_person`、`third_person` 或 `unknown`。未声明视角的外部数据默认 `unknown`。后续第一人称训练集应使用独立标注文件、明确 `--viewpoint first_person` 并独立验证；当前第三人称成绩不能代表第一人称表现。checkpoint 会记录训练/验证来源的视角，视角标签不进入推理事件，也不会凭视角强行推断受击部位。

## 模型与事件语义

默认骨干使用 torchvision 的真实 ImageNet 预训练权重，保留空间特征后送入可训练GRU，预测事件、受力对象、身体区域、感知强度和区域可观察性。视觉骨干默认冻结，特征在每次训练时从当前视频像素重新计算。可通过注册接口替换骨干和时序模块；已提供GRU、均值池化和因果Conv1D。

默认特征缓存包含原始画面及3个固定增强视图：同一窗口内各帧共享平移、亮度、对比度和RGB偏移，不翻转人体左右。增强仅用于训练，验证和测试保留原画面；这些有限视图在每个epoch复用，并非每轮重新抽样。`data.sampling: clip` 保留旧的弱片段训练方式，默认 `causal` 要求整段录像具有完整、不重叠的接触/负例标注。

`ontology/body_regions.yaml` 是区域名称及SVG路径的共享来源。`ontology/events.yaml` 定义事件及受力对象。改动ontology后必须重新生成schema并重新训练；旧checkpoint会因ontology不一致被拒绝。

```json
{
  "event_id": "evt_unique_id",
  "timestamp_ms": 3625,
  "event_class": "impact",
  "recipient": "player",
  "body_region": "left_torso",
  "intensity": 0.75,
  "duration_ms": 220,
  "direction": "unknown",
  "confidence": {"event": 0.85, "recipient": 0.99, "region": 0.97, "intensity": 0.0}
}
```

- `timestamp_ms` 使用源视频时间；强度是 `[0,1]` 的感知显著性，不是牛顿、伤害或疼痛。
- V0运行时只发出已确认的玩家事件。NPC、环境、未知对象及被门控的预测仍保存在诊断日志。
- 身体位置必须同时满足区域置信度、可观察性和连续预测一致性；否则输出 `unknown`，人体保持中性。
- 左右是人体的解剖学左右；正面图上玩家左侧显示在观看者右侧。屏幕攻击方向不会直接转换成身体部位。
- 连续事件只发出一次，直到足够的负证据释放，再经过冷却。默认220ms持续时间是显示策略，并非模型学到的物理量。
- `direction=unknown`；分类置信度尚未校准，`confidence.intensity=0` 明确表示没有可靠的强度置信度估计。
- 暂停按媒体时间冻结高亮；seek/reset清空时序状态，前端拒绝过期推理回复。模拟模式使用独立时钟。

## 独立运行各阶段

```powershell
uv run uhm prepare-data
uv run uhm schema
uv run uhm validate-data
uv run uhm train --epochs 40
uv run uhm validate
uv run uhm export
uv run uhm infer --output runs/events.jsonl
uv run uhm evaluate-stream
uv run python scripts/evaluate_head_regression.py --checkpoint checkpoints/baseline.pt --assert-head
uv run pytest -q
uv run ruff check src tests
npm --prefix frontend test
npm --prefix frontend run build
```

`infer` 默认尽快处理完整视频；加入 `--realtime` 按视频速度输出。输出包括标准事件JSONL、逐窗口预测JSONL和性能报告。`evaluate-stream` **先运行推理，再加载测试标注做独立匹配**。

`configs/baseline.yaml` 配置训练，`configs/runtime.yaml` 配置置信度、连续确认、释放、冷却和强度平滑。不要用测试视频反复调门限来包装测试成绩。

单元测试外，还用实际checkpoint运行头部漏检回归；详细修复及验证见 `docs/head-impact-fix.md`，初版记录见 `docs/verification.md`。TorchScript导出按sidecar中的时序长度和图像尺寸使用，修改输入配置后重新导出。

## 接入自己的视频和标注

当前解码基线假设恒定帧率（CFR）；浏览器播放请使用H.264 MP4。可变帧率录像先转成CFR再做精确标注。导入不会修改原文件；切片输出不含音轨。模型始终不读取音频。

```powershell
uv run uhm import-video "C:\path\gameplay.mp4"
uv run uhm split-video data/raw/gameplay.mp4 --destination data/clips/gameplay

# 以下假设1秒片段的首个接触时刻为417ms；实际录像必须人工判断
# video、source_id、start_ms、end_ms 是位置参数，区间为闭区间
uv run uhm annotate data/clips/gameplay/gameplay_000.mp4 session_a 0 416
uv run uhm annotate data/clips/gameplay/gameplay_000.mp4 session_a 417 999 --event-class impact --recipient player --region left_torso --intensity 0.7

# 负例：默认 no_event / unknown / 强度0
uv run uhm annotate data/clips/gameplay/gameplay_001.mp4 session_a 0 999

# 从另一段独立录像建立验证集，不能把同一源录像的相邻片段放进验证集
uv run uhm annotate data/clips/other/other_000.mp4 session_b 0 999 --split val

uv run uhm validate-data --annotations data/annotations/custom.jsonl
uv run uhm train --annotations data/annotations/custom.jsonl --checkpoint checkpoints/gameplay.pt
uv run uhm demo --video data/raw/gameplay.mp4 --checkpoint checkpoints/gameplay.pt
```

以上标注命令是格式示例，需要根据实际可见证据标注，并准备足够的正负例及独立验证录像。无法确定位置时保留 `--region unknown`；受力对象无法确定时使用 `--recipient unknown`。标注schema在 `schemas/clip_annotation.schema.json`，支持直接编辑JSONL。标注工具会拒绝不可读媒体、越界区间和跨split的同源视频。

推荐直接对完整源录像标注，以便训练窗口包含片段切点之前的历史。默认因果训练要求每段媒体从0到duration_ms−1被完整覆盖，缺失区间不能隐式视为负例。只有弱整片段标签时应另建 `data.sampling: clip` 配置，并承认训练与连续流时序监督的差异。

## 实测与限制

完整40轮训练、参数变化、checkpoint重载、独立像素验证、TorchScript导出均已执行。下表来自默认20秒**合成测试视频**的固定125ms采样，事件匹配容差250ms，统计单位是事件，不是视频窗口：

| 指标 | 本次测量 |
| --- | --- |
| 玩家事件 precision / recall | 9/9 / 9/9 |
| 误报 | 此20秒视频为0个；样本很小 |
| 匹配且可定位事件的区域准确率 | 8/8 |
| 对全部可观察参考事件的区域覆盖率 | 8/8 |
| 不可观察事件的错误定位率 | 0/1；仅一个遮挡事件 |
| 已匹配事件的媒体时间检测延迟中位数 | 208ms；不含浏览器传输和显示 |
| 解码+预处理+模型 p50 / p95 | 当前checkpoint实测约37 / 44ms；不含采样/确认等待和浏览器显示 |

初版漏掉头部、左腿接触且遮挡事件提前触发。修复了时序监督与对背景/位置的过度依赖后，原视频、不同采样偏移/跳转起点和三个独立种子的全流回归均通过；另外32个低显著性头部样本也全部通过原门限。上述测试仍共享同一合成生成器。当前结果保存在 `runs/evaluation_events.metrics.json`，初版checkpoint及结果保留在 `runs/head_fix/before`。详见 `docs/head-impact-fix.md`。

**80ms p50 / 150ms p95仍是未来端到端目标，本项目不声称已经达到。** 当前使用CPU，GPU利用率、VRAM和完整的浏览器显示延迟在未测量时报告为 `null`。重复上下文帧的处理吞吐不等于输入视频的独立帧率。还未验证任何真实游戏的泛化或人类触觉强度标定。

## 文件和日志

```text
ontology/                 共享身体及事件词汇
schemas/                  生成的跨语言JSON Schema
src/uhm/data/             导入、切片、标注、样本、合成示例
src/uhm/models/           视觉骨干、时序模块、多头模型、checkpoint
src/uhm/training/         训练、损失、特征缓存、导出
src/uhm/evaluation/       窗口指标及事件匹配指标
src/uhm/inference/        真实像素推理、离线事件流、性能测量
src/uhm/runtime/          时序事件合并、HTTP/WebSocket服务
frontend/src/body-renderer/  仅做可视化的SVG人体
configs/                  训练与运行参数
tests/                    合约、因果性、abstention、模型及服务验证
checkpoints/              可重建的本地权重与导出模型
runs/                     训练历史、评估报告、每个WebSocket会话的事件/预测/性能日志
```

实时日志使用 `runs/session-<id>.events.jsonl`；完整预测使用同会话的 `.predictions.jsonl`。seek会写入reset记录，便于区分同一视频的多次播放；人体图不做预测。

没有实现音频、游戏API、Mod、进程内存读取、特定游戏Hook、硬件输出、电刺激、温度或每帧VLM。接下来应优先收集和标注范围明确的真实游戏视频，包含困难负例与不可观察事件，再评估事件层面的效果。

技术参考：[torchvision MobileNetV3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)、[FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)。完整原始需求见 `docs/project-brief.md`，验收定义见 `docs/acceptance.md`。
