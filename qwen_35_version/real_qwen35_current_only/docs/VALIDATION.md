# 无记忆基线验证

验证主机：`zjc-hope-8-2` / `psxl1wbxru396rfe-worker-0`。

- 4 项回归测试通过：历史通路配置拒绝、无需历史的当前帧观测、状态传递拒绝、连续夹爪反归一化。
- Easy + Hard 共 180 条轨迹、165,118 帧；两路相机全部分片边界和动作归一化往返验证通过。
- 实际 Qwen processor + DataLoader worker 在连续 8 个批次内覆盖两份数据；每个决策只有 2 张当前图像，输出目标为 50×7。
- 真实 Qwen3.5-4B 模型不含 QueryMemory 或 MoBA 模块，优化器参数分组完整且无重复。
- 读 memory token、写 memory token、历史 KV 均为 0；24 层原生 S/卷积状态逐决策初始化，跨决策返回状态为 `None`。
- 同一当前帧在其他决策前后得到完全一致的 normalized actions；serving 的连续位置 actions 也完全一致，最大误差为 0。
- B2、两个独立决策联合前后向通过，745 个梯度张量有限；视觉、语言和动作头梯度均非零。
- 输出形状为 `[2,50,7]`；单环境 serving 输出 `[1,50,7]`。
- 验证峰值显存为 23,831,165,952 字节；模型验证约 53 秒。此验证未执行 optimizer update。

机器可读记录保存在远程仓库的 `results/real-qwen35-launches/current_only_prepare_20261116/` 下。
