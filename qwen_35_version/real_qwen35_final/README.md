# real_qwen35_final：Piper 真机适配

基于远程 repo-version 已发布的单组 64Q 零初态 Gate Memory＋单图 Top-4 历史 KV 终版源码创建。原源码 SHA256 见 `docs/original_final_sha256.json`。适配全部位于本包内。

## 数据与模型接口

- 直接读取 LeRobot v3.0 `Pick_block_easy`，不生成或改写原始数据文件。
- 相机固定顺序：`cam_high`、`cam_follower_wrist`，RGB。每次决策使用 `[0,t-15,t]`，共 6 张图，训练时缩放到 224×224。
- 动作固定顺序：`joint_1.pos` 至 `joint_6.pos`，然后 `gripper.pos`，共 7 维；50 个 action query 输出 `[B,50,7]`。
- 使用 `action_mode: abs`，保留采集值，不做角度／距离单位转换，不计算位置差。按本数据的位置字段建立绝对位置监督；物理单位以采集端控制接口为准。
- 7 维均使用本数据重新计算的 q01/q99 连续归一化；夹爪不二值化。保存的 `dataset_statistics.json` 中 7 个 action mask 均为 true。
- 视频按每路相机各自的 chunk/file 编号和 episode 起始时间定位。
- `observation.state` 用于完整性检查和统计，模型保持终版 `include_state: false`。
- 保留 TBPTT2、首次完整 memory 写入、episode 边界重置、每决策重置原生 S、仅缓存当前两张图的 detached KV、单图 Top-4 路由和 episode 尾部动作 mask。
- 默认动作块和决策 stride 均为 50；25 Hz 下每块覆盖 2 秒，历史偏移 15 帧对应 0.6 秒。

## 代码入口

- `data/piper_dataset.py`：独立数据适配器。
- `model/`：终版记忆机制、7 维动作头和连续反归一化。
- `training/`：训练及原终版权重迁移。
- `eval/`：策略服务与两路相机观测缓存。
- `tests/`：数据、接口、模型前后向验证。

## 环境与训练

在远程仓库根目录运行：

```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhoujiacong/code/RMBench/repo-version
export STARVLA_ENV=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhoujiacong/envs/starvla
export PIPER_DATASET="$PWD/data/Pick_block_easy"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$STARVLA_ENV/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1

# 仅显示命令，不创建 run、不启动训练。
DRY_RUN=1 bash qwen_35_version/real_qwen35_final/scripts/train_real.sh

# 默认从 Qwen3.5-4B 基座初始化。
bash qwen_35_version/real_qwen35_final/scripts/train_real.sh
```

支持 `NUM_PROCESSES`、`CUDA_VISIBLE_DEVICES`、`PER_DEVICE_BATCH_SIZE`、`MAX_TRAIN_STEPS`、`RUN_ID`、`RUN_ROOT_DIR`、`QWEN35_BASE_MODEL`。默认继承终版的 8 卡、B2/GPU、80k update、每 10k 保存和 seed42。输出默认在仓库内 `results/real-qwen35-runs/`。启动器拒绝复用已有 run 目录。

已在 `zjc-hope-8-3` 使用 8×H100 完成 20,000 次 optimizer update，B2/GPU、TBPTT2、ZeRO-1、seed42，从 Qwen3.5-4B 基座初始化。复现该训练预算：

```bash
MAX_TRAIN_STEPS=20000 bash qwen_35_version/real_qwen35_final/scripts/train_real.sh \
  --trainer.num_warmup_steps 1250 \
  --trainer.save_interval 5000
```

本次训练循环耗时约 3 小时 39 分钟，最后一步 action L1 loss 为 `0.00502579053863883`，退出码为 0；5k、10k、15k、20k checkpoint 和 `final_model/pytorch_model.pt` 均已保存。记录见 [验证与训练记录](docs/VALIDATION.md)。

单机多卡启动在导入创建 `accelerate.PartialState` 的模块前初始化进程组，使用独立的 classic TCPStore（`use_libuv=False`）。可通过 `MAIN_PROCESS_PORT` 和 `REAL_PROCESS_GROUP_PORT` 指定两个不同端口，默认分别为 38985 和 38986。

可选从原 RMBench 终版迁移：设置 `INIT_CHECKPOINT=/absolute/path/to/original/checkpoints/steps_80000_pytorch_model.pt` 后运行同一启动脚本。迁移要求参数名称完全一致；仅跳过原 14 维 `action_model.model.fc2.weight/bias`，保留新 7 维输出层初始化，其他形状不匹配会报错。

`trainer.resume_checkpoint` 用于本包 7 维模型的严格权重恢复，不能与 `INIT_CHECKPOINT` 同时设置。继承的恢复入口不恢复 optimizer 状态。

## 验证

```bash
python -m unittest qwen_35_version.real_qwen35_final.tests.test_piper -v
python -m qwen_35_version.real_qwen35_final.tests.validate_dataset --output qwen_35_version/real_qwen35_final/validation --workers
```

数据验证包括两段 MP4 首末轨迹解码、语言对应、连续动作归一化往返、尾部 mask、真实 Qwen processor 的 DataLoader worker 批处理。可选 `tests.validate_model` GPU 验证只执行前后向，不进行 optimizer update，不保存模型权重。

## 推理

```bash
python -m qwen_35_version.real_qwen35_final.eval.server_policy --ckpt_path /absolute/path/to/real/run/checkpoints/steps_20000_pytorch_model.pt --port 7402 --use_bf16
```

服务严格加载本包 7 维 checkpoint 和同 run 的归一化统计。原 14 维终版 checkpoint 应先通过训练初始化入口迁移，不能直接作为真机推理 checkpoint。

`eval.observations.PiperObservationBuffer` 每 25 Hz 接收一帧两相机 RGB，通过 `append(timestep, images)` 累积，在决策时调用 `example(instruction)` 构造 6 图输入。每条 episode 从 timestep 0 开始，新 episode 调用 `reset()`。默认在 0、50、100…帧发起决策。

WebSocket infer payload：

```python
payload = {
    "examples": [buffer.example(instruction)],
    "episode_id": "episode-unique-id",
    "decision_id": 0,  # 同一 episode 内连续递增
}
```

返回 `normalized_actions` 和 `actions`，形状均为 `[1,50,7]`；`actions` 已恢复为数据记录的绝对位置单位，包含连续夹爪位置。客户端依据采集端 Piper 控制接口发送动作；本包不连接机械臂、不发送电机命令。
