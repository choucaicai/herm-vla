# 验证记录

验证主机：zjc-hope-8-3 / psxlgg0xru4vuomj-worker-0。

- 5 项接口回归测试通过：旧维度拒绝、两路相机历史与 KV 选择、相机独立视频分片、连续夹爪反归一化、14→7 维权重迁移边界。
- 实际数据核对：90 episodes、82,280 frames、3 tasks；逐条轨迹检查帧索引、语言和视频路径。
- 真实视频解码：两个 MP4 分片各自首末轨迹的首帧、50 帧及尾帧；每样本 6 图、50×7 动作；尾部 action mask 与连续归一化往返通过。
- 真实 Qwen processor + DataLoader worker：B1、TBPTT2，每步 6 图；动作均为 50×7。
- 真实 Qwen3.5-4B：用 episode 60 的两个决策执行前向和反向；首步历史图数 0，第二步 2；M 为 1×64×2560，梯度有限，动作头及 memory 写入参数有非零梯度。
- 推理：输出 1×50×7 连续位置，重复请求返回一致结果，episode 重置成功。
- 训练启动器 bash 语法检查与 DRY_RUN 通过。
- 原终版源码 78 个文件 SHA256 核对无变化；原始数据文件仍为 9 个。
- 上述模型验证仅执行前后向；正式训练结果另列如下。

运行验证命令后生成机器可读记录：`validation/dataset_validation.json`、`validation/model_validation.json`（本地生成文件不纳入 Git）。

## 已完成的 20k 正式训练

- 主机：`zjc-hope-8-3`，8×H100；B2/GPU、TBPTT2、ZeRO-1、seed42。
- 初始化：Qwen3.5-4B 基座；warmup 1,250 步，每 5,000 步保存。
- Run ID：`real_piper_qwen35_64q_gate_top4_20k_b2_8gpu_hope8_3_20261113_230219_EST_retry2`。
- Run 目录：仓库根目录下 `results/real-qwen35-runs/<run_id>/`。
- 完成步数：20,000 次 optimizer update；DeepSpeed global steps 同为 20,000。
- 最后一步 action L1 loss：`0.00502579053863883`；记录时间为 `2026-11-14T16:05:19.336778+08:00`。
- 训练循环 elapsed：`13123.672702683136` 秒（约 3 小时 39 分钟）。
- `exit_code` 为 `0`；日志记录 `Training complete`。
- `checkpoints/steps_{5000,10000,15000,20000}_pytorch_model.pt` 均存在。
- `final_model/pytorch_model.pt` 已保存，大小为 `10,508,026,262` 字节。
- 成功 run 使用本包的提前初始化进程组和独立 classic TCPStore 启动修复；训练源码快照保存在 run 的 `source/` 中。

## Easy + Hard 双数据集验证

在 `zjc-hope-8-2`（`psxl1wbxru396rfe-worker-0`）完成：

- 6 项回归测试通过，新增多数据集路径、重复别名和缺失数据集检查。
- 两份数据各 90 条轨迹，共 180 条、165,118 帧；相机、25 Hz 采样率、7 维位置字段一致。
- 两份数据各两个视频分片的边界解码、任务语言、50×7 动作和尾部 mask 检查通过。
- 使用共享 q01/q99 区间，两份数据的连续动作归一化往返均通过；7 个 action mask 均为 true。
- 合并后 180 个 episode ID 无碰撞；第一个采样 epoch 的 easy/hard 决策数为 1671/1653。
- Qwen processor + DataLoader worker 连续 8 个 B1/TBPTT2 批次覆盖两个数据集，每个样本 6 张图。
- 正式训练配置为 `config/train_two_tasks.yaml`：从基座初始化，40k steps，warmup 2,500，每 5k 保存。

## Easy + Hard 无固定首帧输入

在 `zjc-hope-8-2` 的独立源码目录中完成 CPU 与真实数据验证：

- 8 项接口回归测试通过，覆盖旧 6 图配置及新 4 图配置、冷启动历史截断、首帧退出缓存、两路当前图像 KV 选择及错误配置拒绝。
- 新配置 `train_two_tasks_no_first.yaml` 使用 `absolute_indices=[]`、`delta_indices=[-15,0]`，真实数据覆盖两份数据共 180 条轨迹、165,118 帧。
- 各视频分片首末轨迹在第 0 帧、第 50 帧和末帧解码通过，检查实际帧索引、4 图顺序、50×7 动作、尾部 mask 与归一化往返。
- 真实 Qwen processor + DataLoader worker 连续 8 个 B1/TBPTT2 批次覆盖两个数据源，每个样本均为 4 图。
- 原双数据集 6 图配置的数据验证也通过。
- 此轮未加载 4B 模型进行 GPU 前反向，未启动无首帧版本的正式训练。

## Hard-only 40k 配置验证

在 `zjc-hope-8-3` 验证 `config/train_hard.yaml`：

- 仅一个数据源 `Pick_block_hard`，90 条轨迹、82,838 帧，三条原始颜色指令。
- 保留首帧：`absolute_indices=[0]`、`delta_indices=[-15,0]`；每样本 6 图，动作 50×7。
- 真实视频分片边界、episode 元数据、动作 mask 和仅由 Hard 计算的归一化往返通过。
- 真实 Qwen processor + B1/TBPTT2 worker 读取到 Hard，输出每样本 6 图。
- 本次使用的 17 个模型／训练 Python 文件与已完成的 Easy+Hard 40k run 源码快照完全一致。
- 训练预算 40k、warmup 2,500、每 5k 保存，与双数据集 40k 配置一致。
