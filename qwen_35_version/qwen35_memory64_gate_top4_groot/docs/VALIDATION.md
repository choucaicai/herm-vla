# 验证记录

环境：`zjc-hope-8-2`，H100 80GB；Python 3.10.20、PyTorch 2.6.0 / CUDA 12.4、
Transformers 5.12.1、diffusers 0.38.0、DeepSpeed 0.16.9。

## 动作头回归测试

`python -m unittest qwen_35_version.qwen35_memory64_gate_top4_groot.tests.test_groot -v`
通过 6 项测试：

1. 固定噪声、时间、权重时，对齐固定版本 starVLA 的训练损失及 4 步推理。
2. 改变 masked 条件 token、向 episode 尾部无效动作填 NaN，不改变有效预测和损失；梯度有限。
3. 含 dropout 时，动作专家 checkpointing 保持相同损失和梯度。
4. 8 次相同噪声重复的平均损失与单次相同，观测权重不被重复放大。
5. 动作头权重严格保存/加载，预测一致；Euler 方向和调用次数正确。
6. 全局有效动作元素归一化与分片加权平均一致，无效动作梯度为零。

## 真实 Qwen3.5-4B 与 RMBench

`tests.validate_real` 在真实九任务数据上通过。B1、TBPTT2，配置为完整 16 层
GR00T DiT-B，8 次噪声重复。检查涵盖：

- 9 张图像，Qwen 最后一层 2560 维条件，64 个有效记忆槽，50 个原有动作 query。
- 下一决策损失传回上一决策记忆：gradient L1 = `78.77714538574219`。
- 记忆、动作专家及视觉参数梯度有效；优化器参数覆盖完整且无重复。
- 原生 S/conv 每次重置，历史图像 KV 保持 detach，单图 Top-4 路由正确。
- 4 步推理只调用一次 Qwen、一次记忆写入和一次 KV stage。
- 输出 `[1,50,14]`，重复请求一致，切换 episode 重置。
- 峰值 PyTorch GPU allocation `27,179,414,528` 字节（约 25.3 GiB）。

以上真实模型验证未进行 optimizer update。

## 8 卡训练与保存

8×H100、B2/GPU、TBPTT2、ZeRO-1，使用正式架构和真实九任务数据完成 2 次
optimizer update，退出码 0。两步 loss 为 `1.4147849`、`1.4583800`，
梯度范数 `24.364426`、`13.217430`，均无 optimizer skip。

每张卡输入为 2×2 的时序窗口、9 图/观测、50×14 动作。第一次决策冷记忆有效槽数 0，
第二次为 64；条件只有最后一层 2560 维特征；噪声重复数 8；窗口边界正确 detach。
保存的完整 checkpoint 为 `10,698,241,282` 字节。
checkpoint 结构检查（980 tensors）和 `strict=True` 完整重新加载均通过；
重新加载后真实数据推理输出 `[1,50,14]`，重复请求输出完全相同，记忆只保留一次写入。

## 源码一致性

23 个公共文件与已提交终版逐字节一致，详见 `backbone_preservation.json`。
服务器上 64 个共享 `eventvla` / `deployment` Python 文件与本地发布仓库一致。
原终版源码保留；本实验通过独立包提供模型、训练、checkpoint 验证与服务入口。
