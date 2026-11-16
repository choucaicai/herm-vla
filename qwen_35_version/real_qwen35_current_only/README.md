# Piper 当前帧无记忆基线

输入仅为当前时刻 `[t]` 的 `cam_high`、`cam_follower_wrist` 两张 RGB 图像和语言指令，输出 `50×7` 绝对关节／连续夹爪位置。基座为 Qwen3.5-4B，动作头与 `real_qwen35_final` 相同。

- 不输入首帧或历史帧，不创建 QueryMemory 参数或读写 token，不启用历史 KV/MoBA。
- 每个决策都创建新的原生 DeltaNet S、卷积和 full-attention cache；位置编号在决策内独立计算。模型返回的跨决策状态始终为 `None`。
- 保留两决策一组的数据打包和 loss 聚合，用于匹配有记忆版本的训练样本量；两个决策之间没有状态传递或跨决策梯度。8 GPU × B2 × 2，每次 optimizer update 最多监督 32 个决策。
- 使用 Easy + Hard 两份 LeRobot v3 数据，180 条轨迹、165,118 帧；按数据集均衡决策步数，使用与有记忆双任务版本相同的连续动作归一化。
- 默认从基座重新训练 40k steps，warmup 2,500，seed42，ZeRO-1，每 5k 保存，保留最近 4 个 checkpoint。

## 训练

在仓库根目录、已安装的训练环境中运行：

```bash
export STARVLA_ENV=/path/to/envs/starvla
export QWEN35_BASE_MODEL=/path/to/Qwen3.5-4B
export PIPER_DATA_ROOT="$PWD/data"
bash qwen_35_version/real_qwen35_current_only/scripts/train_real.sh
```

支持 `DRY_RUN=1`、`RUN_ID`、`RUN_ROOT_DIR`、`MAIN_PROCESS_PORT`、`REAL_PROCESS_GROUP_PORT`。输出默认位于 `results/real-qwen35-runs/`。`INIT_CHECKPOINT` 不适用于此无记忆基线；显式 resume 只接受本模型的严格匹配权重。

## 验证

```bash
python -m unittest qwen_35_version.real_qwen35_current_only.tests.test_current_only -v
python -m qwen_35_version.real_qwen35_current_only.tests.validate_dataset \
  --output qwen_35_version/real_qwen35_current_only/validation/data --workers
python -m qwen_35_version.real_qwen35_current_only.validate_real \
  --config qwen_35_version/real_qwen35_current_only/config/train.yaml \
  --output qwen_35_version/real_qwen35_current_only/validation/model.json
```

GPU 验证检查参数中无记忆模块、拒绝传入状态、同一输入在其他决策前后得到完全一致的动作、两个独立决策前后向及视觉／语言／动作头梯度；验证阶段不进行 optimizer update。

## 推理

```bash
python -m qwen_35_version.real_qwen35_current_only.eval.server_policy \
  --ckpt_path /path/to/run/checkpoints/steps_40000_pytorch_model.pt --port 7403 --use_bf16
```

使用 `eval.observations.current_observation(images, instruction, timestep)` 构造输入；`images` 包含两路当前 RGB uint8 HWC 图像，无需历史图像缓存。WebSocket payload 为 `{"examples": [example]}`，可附加 `episode_id`、`decision_id` 用于日志。返回 `normalized_actions` 和已恢复数据单位的 `actions`，均为 `[1,50,7]`。

推理须使用同 run 的 `config.yaml` 和 `dataset_statistics.json`。源代码来源记录见 `docs/source_provenance.json`。
