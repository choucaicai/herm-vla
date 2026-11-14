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
