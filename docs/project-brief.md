# Universal Haptic Model
## Real-Time Visual-to-Haptics Inference Layer

### 1. Project Vision

Universal Haptic Model 的长期目标，是构建一个位于数字内容与触觉设备之间的通用智能中间层。

它不要求游戏、视频播放器或内容开发者主动提供 haptic API，也不依赖特定游戏的 Mod、内存读取、事件 Hook 或人工编写的触觉轨道。

系统直接观察用户本来就能看到的数字内容：

**Game / Video → Visual Perception → Physical Event Understanding → Body-relative Haptic Representation → Device Adapter**

长期版本可以进一步引入 audio，并连接 vibration、pressure、force、temperature 等不同类型的触觉设备。

核心思想不是：

> 给每一个游戏单独写震动配置。

而是：

> 让模型理解“画面里发生了什么，以及这个事件对体验者的身体意味着什么”，然后输出统一的、设备无关的触觉语义。

因此，Universal Haptic Model 更接近一个：

**Universal perception-to-haptics translation layer**

而不是某一个具体的触觉背心、游戏插件或者振动生成器。

---

# 2. Current V1 Scope

当前版本必须大幅收窄。

## V1 = Real-Time Visual Impact Localization

第一阶段只解决：

**视频画面 → 事件 → 受力对象 → 身体区域 → 强度 → 人体可视化**

当前明确：

**只使用视觉。**

V1 不做 audio。

V1 不做真实硬件输出。

V1 不做温度。

V1 不做电刺激。

V1 不做完整物理模拟。

V1 不试图理解所有游戏和所有事件。

V1 的目的只是证明：

> 一个模型是否可以仅从连续视频画面中判断“玩家是否正在受到某种明显物理作用”，并将这个事件实时映射成人体上的结构化区域和强度。

第一版应优先使用 prerecorded gameplay/video，以保证实验可重复。

实时 screen capture 可以在 pipeline 成功之后接入。

---

# 3. Core Demo

Demo UI 分为左右两个区域。

### Left

播放原始 gameplay/video。

### Right

显示一个简化的人体模型。

人体由多个独立区域组成，每个区域能够单独高亮。

当模型判断玩家某个身体区域受到作用时：

对应人体区域产生颜色变化。

颜色表示 `intensity_norm`：

低强度：

浅红

中等强度：

红色

高强度：

深红

极高强度：

接近黑红 / 黑色

第一版只表达：

**Force / Impact**

不要加入之前讨论过的：

蓝—橙温度系统、冲击波扩散动画、材质感、疼痛、压力、电刺激等。

这些全部属于后续扩展。

人体模型本身只是：

**visualization layer**

它不是最终硬件设计。

---

# 4. Core Semantic Representation

模型最终不应该直接输出：

“震动马达 3 使用 70% 功率”。

这是错误的 abstraction。

模型应该输出设备无关的物理/触觉语义。

V1 建议统一使用：

```text
BodyForceEvent
```

基础 schema：

```json
{
  "event_id": "evt_001",

  "timestamp_ms": 18320,

  "event_class": "impact",

  "recipient": "player",

  "body_region": "left_torso",

  "intensity": 0.74,

  "duration_ms": 160,

  "direction": "front_left",

  "confidence": {
    "event": 0.93,
    "recipient": 0.88,
    "region": 0.72,
    "intensity": 0.67
  }
}
```

其中：

`event_class`

表示视觉事件类别。

`recipient`

表示事件究竟作用于：

```text
player
other_character
environment
unknown
```

`body_region`

表示模型认为作用发生在哪个身体区域。

如果画面证据不足：

```text
body_region = unknown
```

必须允许 abstention。

绝对不要为了让 Demo “看起来聪明”而强制猜一个身体区域。

`intensity`

范围：

```text
0.0 - 1.0
```

这里必须理解为：

**perceptual impact / haptic salience**

而不是真正意义上的 Newton、伤害值或医学意义上的受伤严重程度。

模型是在预测“这个事件应该产生多强的触觉反馈”。

不是从像素精确恢复真实物理力。

---

# 5. Most Important Reasoning Problem

这个项目真正困难的地方不是检测：

“画面发生了爆炸。”

而是：

**这个爆炸和玩家是什么关系？**

例如：

远处 NPC 被炸飞：

不应该给玩家胸口产生强烈反馈。

玩家自己被击中：

应该产生 player-relative event。

玩家自己开枪：

可能产生 weapon recoil，但不是 torso impact。

车辆撞击：

可能对应 whole-body / torso force。

镜头 shake：

不意味着玩家一定受到直接身体接触。

因此模型至少应该逐步学习：

```text
What happened?
        ↓
Who did it happen to?
        ↓
Is the player the recipient?
        ↓
Can the affected body region be inferred?
        ↓
How strong should the tactile event be?
```

这也是项目相对于简单 audio-reactive vibration 的核心研究价值之一。

---

# 6. Observability Rule

这是整个项目必须遵守的一条原则：

**Do not hallucinate physical state that cannot be observed.**

屏幕左边出现攻击：

不意味着玩家身体左边被击中。

画面方向：

不等于身体接触位置。

镜头 shake：

不等于某个具体 body region。

因此：

```text
unknown
```

必须是合法输出。

模型应该同时学习：

**什么时候输出**

以及：

**什么时候不输出。**

这比强迫模型永远产生一个人体位置更重要。

---

# 7. Initial Human Body Representation

Agent 首先需要实现一个初步人体结构示例。

不要使用复杂的医学人体模型。

目标是建立一个：

**simple, extensible, region-addressable body map**

推荐使用：

```text
React
TypeScript
SVG
```

人体应该由独立 SVG region 构成。

例如：

```text
body_regions/
    head
    upper_torso
    lower_torso
    left_arm
    right_arm
    left_leg
    right_leg
    ...
```

具体 V1 region ontology 放到独立配置文件：

```text
ontology/body_regions.yaml
```

不要让 UI 和模型代码 hardcode 同一套 region。

这样以后可以从：

8 zones

升级成：

16 zones

32 zones

甚至根据具体 wearable 动态映射。

人体 renderer 接收到：

```json
{
  "body_region": "left_torso",
  "intensity": 0.74,
  "duration_ms": 160
}
```

以后，只负责：

1. 找到对应 body region；
2. 将 intensity 映射成颜色；
3. 保持对应 duration；
4. 平滑 fade out。

Renderer 不负责推理。

---

# 8. Architecture

建议 V1 architecture：

```text
Video
  ↓
Frame / Clip Sampler
  ↓
Visual Encoder
  ↓
Temporal Model
  ↓
Event Detection
  ↓
Recipient Attribution
  ↓
Body Region Prediction
  ↓
Intensity Estimation
  ↓
BodyForceEvent
  ↓
Runtime
  ↓
WebSocket
  ↓
Human Body Renderer
```

模型层和体验层必须分开。

不要把模型预测逻辑写进 frontend。

Frontend 只接受标准化事件。

---

# 9. Training Baseline

现在不要尝试训练所谓真正的 “Universal Haptic Foundation Model”。

Agent 的任务只是搭建：

**一个可以训练、验证、替换模型并执行实时 inference 的完整 baseline。**

第一版应该使用 pretrained visual backbone。

不要从零训练视觉模型。

建议把 architecture 做成 pluggable。

基础结构可以是：

```text
frames
 ↓
pretrained visual encoder
 ↓
temporal aggregation
 ↓
shared feature
 ├── event classification head
 ├── recipient classification head
 ├── body-region classification head
 └── intensity regression head
```

例如 temporal aggregation 可以先采用简单：

```text
Temporal Conv
GRU
Transformer Encoder
```

任何一种。

重点不是一开始选择“最先进模型”。

重点是保证整个 training pipeline 真正跑通。

最终应该能够方便地替换 visual backbone 和 temporal model，而不需要重写 dataset/runtime。

---

# 10. Training Dataset Format

创建统一 annotation format。

例如：

```json
{
  "video_id": "clip_0021",
  "start_ms": 18100,
  "end_ms": 18400,

  "event_class": "impact",

  "recipient": "player",

  "body_region": "left_torso",

  "intensity": 0.74,

  "observable_region": true,

  "ambiguous": false
}
```

数据必须包含大量 negative clips。

比如：

NPC 打 NPC；

远处爆炸；

菜单；

过场动画；

单纯镜头运动；

玩家攻击敌人；

环境物体运动；

UI 特效。

否则模型很容易学成：

**“看到爆炸就震”。**

这恰好是我们最不希望出现的结果。

---

# 11. Initial Event Vocabulary

V1 不追求大规模 ontology。

先限制在几个非常明显的视觉事件。

例如：

```text
impact
collision
explosion
fall
melee_hit
projectile_hit
```

但真正训练时可以进一步收窄。

宁愿：

3 个 event class 做得稳定

也不要：

30 个 class 全部不可用。

---

# 12. Training Pipeline

Agent 需要建立完整 pipeline：

```text
Raw Video
    ↓
Clip Extraction
    ↓
Annotation
    ↓
Dataset Loader
    ↓
Preprocessing / Augmentation
    ↓
Model
    ↓
Training
    ↓
Validation
    ↓
Checkpoint
    ↓
Export
    ↓
Streaming Inference
    ↓
BodyForceEvent
```

代码应明确分离：

```text
dataset
model
training
evaluation
runtime
visualization
```

建议 repo：

```text
universal-haptic-model/

    data/
        raw/
        clips/
        annotations/

    ontology/
        body_regions.yaml
        events.yaml

    src/
        data/
        models/
        training/
        evaluation/
        inference/
        runtime/
        schemas/

    frontend/
        body-renderer/

    scripts/

    checkpoints/

    configs/

    tests/

    README.md
```

Python 环境和 dependencies 优先使用：

```text
uv
```

不要默认使用散乱的全局 pip 环境。

---

# 13. Inference Runtime

训练 pipeline 与实时 runtime 必须分开。

Runtime 输入：

连续 frames。

输出：

```text
BodyForceEvent stream
```

事件应该支持：

deduplication、confidence gating、duration、cooldown 和 temporal smoothing。

不要让连续 10 帧检测到同一个 impact 就产生 10 个独立事件。

例如：

```text
frame 100: impact
frame 101: impact
frame 102: impact
```

应该 merge 成：

```text
one temporal event
```

而不是三次反馈。

---

# 14. VLM Strategy

大型视觉语言模型可以用于：

**offline teacher / annotation assistance / semantic exploration**

例如未来可以用 MOSS-VL 或其他视频/VLM 模型帮助生成 pseudo labels。

但是：

VLM 不应该成为 V1 每帧实时 inference 的必要路径。

长期架构更合理的是：

```text
Slow semantic system
        +
Fast realtime perception system
```

slow path 理解：

```text
player identity
viewpoint
weapon
scene
semantic context
```

fast path 负责：

```text
motion
impact
collision
event onset
tracking
temporal change
```

V1 暂时先把 fast visual path 跑通。

---

# 15. Performance Metrics

不要只报告“准确率”。

至少记录：

```text
Event precision
Event recall
False events / minute
Body-region accuracy
Body-region coverage
False localization rate
Latency
FPS
VRAM usage
GPU utilization
Temporal stability
```

特别注意：

body-region accuracy 必须只在：

```text
body location actually observable
```

的事件上计算。

否则模型应该输出：

```text
unknown
```

---

# 16. Real-Time Goal

最终目标是实时运行。

但是 Agent 第一阶段不要过早优化 latency。

先做到：

```text
prerecorded video
        ↓
model inference
        ↓
BodyForceEvent
        ↓
human visualization
```

完整闭环。

然后才优化：

```text
streaming
GPU inference
ONNX / TensorRT
frame sampling
async pipeline
buffering
latency
```

之前设想的 prototype engineering target 可以保留：

```text
p50 event latency ≤ ~80 ms
p95 ≤ ~150 ms
```

但这目前只是目标。

不是已经实现的性能。

---

# 17. What the Agent Should Build Now

本轮 Agent 的目标不是完成 Universal Haptic Model。

而是完成一个：

**technically clean V0 training and visualization foundation**

最终必须至少交付以下六项：

1. **Human Body Renderer**

左边视频、右边人体区域图。

能够通过模拟 BodyForceEvent 单独点亮任意区域。

强度从浅红逐渐到深红/近黑。

2. **Canonical Event Schema**

实现 `BodyForceEvent`。

训练、runtime、frontend 都使用同一个 schema。

3. **Dataset + Annotation Pipeline**

能够：

```text
import video
→ split clips
→ load annotations
→ construct training samples
```

并提供 example dataset。

4. **Trainable Baseline Model**

即使初期 accuracy 很差，也必须是真正可以：

```text
train
validate
save checkpoint
reload
infer
```

的模型。

禁止只做 fake demo。

5. **Inference Pipeline**

能够：

```text
video.mp4
→ model
→ BodyForceEvent stream
```

6. **End-to-End Demo**

最终运行一个 command 后：

```text
video
 ↓
model
 ↓
events
 ↓
body renderer
```

完整工作。

---

# 18. Development Order

严格按照以下顺序完成：

```text
Phase 1
Human Body Renderer
+
mock BodyForceEvent generator

Phase 2
schemas
+
ontology
+
video loader
+
dataset structure

Phase 3
baseline model
+
training loop
+
validation loop

Phase 4
checkpoint inference
+
temporal event aggregation

Phase 5
model → BodyForceEvent → frontend integration

Phase 6
profiling
+
tests
+
documentation
```

不要一开始就优化大模型。

不要一开始接真实 wearable。

不要加入 audio。

不要扩展 temperature。

不要尝试支持几十种游戏。

---

# 19. Definition of Done

项目只有满足下面这个场景才算当前阶段完成：

运行一个命令，加载一个测试视频。

界面左侧正常播放视频。

系统逐帧或逐 clip 运行真实模型 inference。

当模型检测到玩家发生 impact 时，生成：

```json
{
  "recipient": "player",
  "body_region": "...",
  "intensity": 0.0,
  "duration_ms": 0
}
```

前端收到 event。

右侧人体对应区域实时变红。

强度不同产生明显颜色差异。

事件结束后逐渐恢复。

同时 console / log 保存完整 event stream。

即：

```text
VIDEO
  ↓
VISUAL MODEL
  ↓
PHYSICAL EVENT
  ↓
BODY FORCE EVENT
  ↓
HUMAN BODY MAP
```

整个链条真实运行。

---

# 20. Hard Constraints

当前版本禁止偏离以下约束：

**Video only. No audio.**

**No game API.**

**No mod.**

**No process-memory reading.**

**No title-specific event hook.**

**No wearable hardware implementation.**

**No electrical stimulation.**

**No temperature system.**

**No fake event detection disguised as model inference.**

**No requirement that every event has a body region.**

**Unknown / abstain must remain first-class outputs.**

**Do not equate screen direction with physical body location.**

**Do not put a large VLM in the per-frame critical path.**

**Do not train a vision foundation model from scratch.**

---

# 21. Long-Term Architecture

如果 V1 被证明可行，项目最终可以扩展成：

```text
             VIDEO
               │
               ▼
       Visual Perception
               │
AUDIO ───────► Multimodal Fusion
               │
               ▼
       Physical Reasoning
               │
               ▼
       Participant Model
               │
               ▼
        Haptic Semantics
               │
               ▼
        HapticIntent API
               │
      ┌────────┼─────────┐
      ▼        ▼         ▼
 vibration   pressure   force
 device      device     device
```

Universal Haptic Model 不应该关心下面具体接的是：

背心、手套、腕带、座椅、全身贴片还是未来新的 tactile hardware。

它负责的是：

> What happened, who experienced it, where it should be felt, how strong it should feel, and for how long.

硬件 adapter 再负责：

> How this specific device physically reproduces that intent.

这才是整个项目真正具有扩展性的 architecture。

---

# 22. Core Thesis

这个项目最终要验证的不是：

**“AI 能不能看到爆炸。”**

这个问题已经没有太大研究意义。

真正要验证的是：

> **Can a visual system transform arbitrary digital content into a reliable, participant-aware, body-relative physical event representation quickly enough to serve as a universal haptic interface?**

如果答案成立，那么 Universal Haptic Model 就可以成为：

**digital content → embodied experience**

之间的一层通用 translation layer。

当前 V1 要做的，就是实现这条路线中最小、最困难、同时也是最值得验证的第一步：

> **video → player-relative impact → body region → intensity → real-time human visualization.**