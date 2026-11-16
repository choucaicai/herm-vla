# Qwen3.5 + GR00T：终版记忆架构的 RMBench 版本

基于已发布终版的 Qwen3.5-4B、64Q Gate memory 和单图 Top-4 历史 KV，
将 MLP 动作头替换为 [starVLA QwenGR00T](https://github.com/starVLA/starVLA/blob/13d8ad43eb4128554c57227507055ecf9c3a62a2/starVLA/model/framework/VLM4A/QwenGR00T.py)
的 Flow-Matching / DiT-B 动作头。框架名为 `Qwen35Memory64GateTop4GR00T`。

| 设置 | 本版本 |
| --- | --- |
| 视觉语言模型 | Qwen3.5-4B，全参数训练 |
| 观测 | `[0,t-15,t] × 3 cameras`，9 张 224×224 图像 |
| 跨决策记忆 | 64 个零初始化槽，首次全写，之后逐槽 Gate；TBPTT=2 |
| 历史 KV | 只缓存当前三视角图像，detach，单图 Top-4；每次决策重置原生 S/conv |
| 动作条件 | Qwen 最后一层完整有效 token，包含记忆和原有 50 个 ◆ token |
| 动作专家 | DiT-B，16 层、宽度 768、12 heads，交替 cross/self attention |
| 专家输出投影 | 1024，32 个 future tokens，学习式动作位置编码 |
| Flow matching | Beta(1.5,1.0)，1000 时间桶，每个观测 8 次噪声采样 |
| 推理 | 4 步 Euler，输出 `[B,50,14]` |
| 状态输入 | 关闭，`state_dim=0` |
| 数据 | `rmbench9`，绝对关节动作，沿用终版归一化和 episode 采样 |
| 默认训练 | 80,000 updates，8 GPU × B2，ZeRO-1，warmup 5,000，seed 42 |
| 学习率 | Qwen 和记忆 1e-5，动作专家 1e-4 |

一次决策只运行一次 Qwen、写一次 64Q 记忆并暂存一次历史 KV。8 次噪声采样和
4 步推理都只执行无状态动作专家。训练仅对动作专家启用 activation checkpointing，
不会重放有状态的 Qwen 和记忆写入。

动作专家的 cross attention 屏蔽 prompt padding 和冷启动空记忆；episode 尾部的无效
动作在构造噪声轨迹前清零，损失只统计有效动作元素。噪声重复不会把该观测在全局
TBPTT 损失中的权重乘以 8。

默认从 Qwen3.5-4B 基座初始化，记忆和动作专家随机初始化；这是 GR00T 架构训练，
没有加载 NVIDIA GR00T 的预训练权重。原终版 MLP checkpoint 不能直接作为本框架 checkpoint。

## 训练

在仓库根目录运行（路径按环境设置）：

```bash
export STARVLA_ENV=/path/to/envs/starvla
export QWEN35_BASE_MODEL=/path/to/Qwen3.5-4B
export RMBENCH_DATA_ROOT=/path/to/RMBench/data/lerobot_2.1
RUN_ID=rmbench9_qwen35_groot_80k \
  bash qwen_35_version/qwen35_memory64_gate_top4_groot/scripts/train_groot.sh
```

默认每 10,000 步保存，保留最近 4 个 checkpoint，最终权重位于
`results/qwen35-runs/<RUN_ID>/final_model/pytorch_model.pt`。
每次训练保存配置、归一化统计、源码快照、runtime/input validation 和 metrics。
脚本拒绝复用已有运行目录。

## 验证与推理

```bash
python -m unittest qwen_35_version.qwen35_memory64_gate_top4_groot.tests.test_groot -v
python -m qwen_35_version.qwen35_memory64_gate_top4_groot.tests.validate_real \
  --config qwen_35_version/qwen35_memory64_gate_top4_groot/config/train.yaml \
  --output /tmp/groot_real_validation.json
python -m qwen_35_version.qwen35_memory64_gate_top4_groot.eval.validate_checkpoint \
  --checkpoint /path/to/final_model/pytorch_model.pt
bash qwen_35_version/qwen35_memory64_gate_top4_groot/scripts/eval_groot.sh \
  --checkpoint /path/to/final_model/pytorch_model.pt
```

服务端严格加载本框架权重。每个环境拥有独立记忆；相同 decision 重试返回相同结果，
新 episode 重置记忆。RMBench 接口沿用终版归一化、相机顺序和动作执行约定。

## 来源

starVLA 固定 commit：`13d8ad43eb4128554c57227507055ecf9c3a62a2`。
原文件及哈希见 `docs/source_provenance.json`，上游许可证见 `docs/STARVLA_LICENSE`。
NVIDIA 文件的 Apache-2.0 头保留，完整许可证见 `docs/APACHE-2.0.txt`。
本地改动限定为包内导入、无状态 block checkpointing 和独立 masked-loss 适配层。
终版公共模块的逐文件一致性记录见 `docs/backbone_preservation.json`。
