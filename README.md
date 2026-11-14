# HERM-VLA

终版：**64Q 零初态 Gate memory＋单图 Top-4 历史 KV**。此仓库包含数据处理、模型、训练、检查点验证与 RMBench 九任务评测代码。模型、数据和训练核心来自正式 80k run 的冻结源码。

## 1. 终版设置与结果

- 基座：Qwen3.5-4B；全参数训练 80,000 次 optimizer update。
- 输入：首帧 `0`、相对帧 `t−15`、当前帧 `t`，每帧三个相机，共 9 张图。
- 跨决策 memory：64 槽，零初态，首次完整写入，之后按槽 Gate 更新；原生 DeltaNet S 每次决策重置。
- 历史 KV：仅写入当前帧的三张图；历史 KV detached；每个 full-attention query/head 独立选最多 4 张历史图；所有有效 token 可读。
- 训练：8 GPU × B2，TBPTT=2，ZeRO-1，seed=42，stride=50，action chunk 为 50×14。
- 评测：9 任务 × 25 局，seed=0，`demo_clean`、`unseen`；仅 `cover_blocks` 对三个相机做 R/B 交换，与正式结果协议一致。

| 任务 | 成功数 | 成功率 | R/B 交换 |
|---|---:|---:|:---:|
| battery_try | 19/25 | 76% | 否 |
| blocks_ranking_try | 25/25 | 100% | 否 |
| cover_blocks | 25/25 | 100% | 是 |
| observe_and_pickup | 4/25 | 16% | 否 |
| press_button | 20/25 | 80% | 否 |
| put_back_block | 25/25 | 100% | 否 |
| rearrange_blocks | 25/25 | 100% | 否 |
| swap_T | 20/25 | 80% | 否 |
| swap_blocks | 25/25 | 100% | 否 |
| **合计** | **188/225** | **83.56%** | |

结果与源码来源见 `docs/results.json`、`docs/source_provenance.json`、`docs/frozen_source_sha256.json`。

## 2. 目录

```text
qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/
  data/       时间采样、TBPTT窗口、worker预处理
  model/      Qwen3.5、64Q Gate memory、单图Top-4路由及缓存
  training/   训练循环、状态管理、kernel预热
  eval/       检查点验证、在线策略、服务端、RMBench适配器
  config/     终版train.yaml、ZeRO配置、已有参考测试配置
  tests/      memory、routing、cache等测试；reference_gate仅供回归测试
eventvla/     共享框架、LeRobot数据读取与训练工具
deployment/   图像工具与websocket运行时
examples/RMBench/train_files/  LeRobot元数据转换与modality定义
scripts/      终版训练和九任务评测入口
```

保留原 Python 包名以兼容已有 checkpoint。默认训练入口只有 `scripts/train_final.sh`。

## 3. 环境与路径

已核对当前工作环境：Python **3.10.20**、PyTorch **2.6.0 / CUDA 12.4**、Transformers **5.12.1**、Accelerate **1.5.2**、DeepSpeed **0.16.9**。Python依赖见 `requirements.txt`，CUDA kernels见 `requirements-cuda.txt`；数据变换还需要匹配当前 PyTorch/CUDA 的 `pytorch3d==0.7.9`。

```bash
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-cuda.txt
pip install --no-deps -e .
```

训练数据、基座权重、checkpoint 和 RoboTwin/RMBench 仿真资产通过外部路径提供。训练配置支持：

| 环境变量 | 含义 |
|---|---|
| `RMBENCH_DATA_ROOT` | 转换后的九任务 LeRobot 根目录；默认实验室现有数据路径 |
| `QWEN35_BASE_MODEL` | Qwen3.5-4B 本地目录；也用于迁移 checkpoint 后覆盖旧基座路径 |
| `STARVLA_ENV` | 训练 Python 环境目录 |
| `RUN_ROOT_DIR` | 训练输出目录，默认仓库内 `results/qwen35-runs` |

## 4. 数据处理

现有转换器输入是已经准备好的 **StarVLA LeRobot-v2** 数据，不是原始仿真轨迹。它映射三路相机及14维动作字段，重写 EventVLA 元数据/统计，并通过符号链接复用 parquet 与 MP4。

```bash
python examples/RMBench/train_files/convert_rmbench_to_lerobot.py \
  --source /path/to/starvla_rmbench_9tasks \
  --destination /path/to/lerobot_2.1
export RMBENCH_DATA_ROOT=/path/to/lerobot_2.1
```

九任务混合定义：`eventvla/dataloader/gr00t_lerobot/mixtures.py` 中 `rmbench9`。模型内的 `data/` 负责首帧/历史帧选择、episode slot保持、两步TBPTT打包和worker图像处理。

## 5. 训练

```bash
export STARVLA_ENV=/path/to/envs/starvla
export QWEN35_BASE_MODEL=/path/to/Qwen3.5-4B
export RMBENCH_DATA_ROOT=/path/to/lerobot_2.1
DRY_RUN=1 ./scripts/train_final.sh
./scripts/train_final.sh
```

可覆盖 `NUM_PROCESSES`、`CUDA_VISIBLE_DEVICES`、`PER_DEVICE_BATCH_SIZE`、`ZERO_STAGE`、`MAX_TRAIN_STEPS`、`RUN_ID`、`MAIN_PROCESS_PORT`、`RUN_ROOT_DIR`。默认使用冻结终版 `config/train.yaml`；其他 `train_*.yaml` 为已有参考配置。

zjc-hope-8 现有环境示例：

```bash
export STARVLA_ENV=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhoujiacong/envs/starvla
./scripts/train_final.sh
```

## 6. 验证与评测

checkpoint应与原run的 `config.yaml` 和 `dataset_statistics.json` 保持原相对布局。

```bash
export PATH="$STARVLA_ENV/bin:$PATH"
./scripts/validate_checkpoint.sh --checkpoint /path/to/run/checkpoints/steps_80000_pytorch_model.pt
./scripts/eval_final.sh \
  --checkpoint /path/to/run/checkpoints/steps_80000_pytorch_model.pt \
  --robotwin-mem-root /path/to/RMBench \
  --eventvla-python /path/to/envs/starvla/bin/python \
  --robotwin-mem-python /path/to/envs/RMBench/bin/python \
  --gpus 0,1,2,3 --run-tag herm_final --dry-run
```

去掉 `--dry-run` 即运行评测。调度器每张GPU同时运行一个任务，记录server/eval日志、状态和 `summary.csv`。用 `--tasks cover_blocks` 可单独补跑；默认九任务且仅 cover_blocks 做颜色修正。底层完整参数见 `bash scripts/eval_rmbench_v2.sh --help`。

单独启动服务：

```bash
./scripts/run_policy_server.sh --ckpt_path /path/to/run/checkpoints/steps_80000_pytorch_model.pt --port 7402 --use_bf16
```

## 7. 检查

```bash
python scripts/verify_release.py
python -m unittest discover -s scripts/tests -p 'test_*.py'
PYTHONPATH="$PWD" python -m unittest discover \
  -s qwen_35_version/raw_qwen35_mix_current_kv_action_memory64_all_tokens_zero_state_gate_image_topk/tests -t . -p 'test_*.py'
```

后者需要模型运行环境，使用小型随机模型测试 memory、图像路由、padding、KV缓存和TBPTT，不加载4B权重。完整真实模型验证入口为模型包内 `tests/validate_real_memory64.py`。

## 8. 终版 M-only 消融

`./scripts/train_m_only.sh` 保留终版64Q零初态Gate query memory，原生DeltaNet S/conv每次重置；唯一通路开关是 `cache_delta_indices: []`，使跨决策历史KV始终为空。输入、80k预算、8GPU×B2及TBPTT2不变。详见 `docs/M_ONLY_ABLATION.md`。旧原生S-only不作为这组的替代结果。

Historical KV budget ablations: [Top-6 / Top-8](docs/TOPK_ABLATIONS.md), launched with `bash scripts/train_topk.sh 6` or `8`.

## 9. Piper 真机数据适配

[`qwen_35_version/real_qwen35_final`](qwen_35_version/real_qwen35_final/README.md) 基于单组 64Q Gate＋单图 Top-4 终版，直接读取 `data/Pick_block_easy` 的 LeRobot v3 数据：两路相机、7 维绝对关节／夹爪位置、连续夹爪归一化，输出动作块为 `50×7`。模型保持 `include_state: false`。

已在 `zjc-hope-8-3` 用 8×H100 完成 20,000 steps，退出码 0，最终 action L1 loss 为 `0.005026`。训练入口为 `bash qwen_35_version/real_qwen35_final/scripts/train_real.sh`；20k 参数、数据约定、推理接口及验证记录见包内说明。原始数据和训练权重保存在运行环境中。
