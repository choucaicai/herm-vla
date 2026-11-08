#!/usr/bin/env bash
# Unified RMBench entry point. Scheduling uses only Python's standard library.
# Models and simulator run in their separate Python environments.
exec python3 - "$0" "$@" <<'PYTHON'
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
from pathlib import Path
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

SCRIPT = Path(sys.argv[1]).resolve()
ROOT = SCRIPT.parent.parent
TASKS = ["battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup",
         "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks"]
OWNER = Path("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hlsc/hadoop-mlm/zhoujiacong")
SCALARS = [
    "TASK_CONFIG", "SEED", "HOST", "UNNORM_KEY", "ACTION_MODE", "INSTRUCTION_TYPE",
    "TEST_NUM", "USE_BF16", "POLICY_NAME", "SERVER_SCRIPT_PATH", "DEPLOY_POLICY_PATH",
    "EVENTVLA_PYTHON", "ROBOTWIN_MEM_PYTHON", "ROBOTWIN_MEM_ROOT", "SAMPLING_INTERVAL",
    "SERVER_READY_TIMEOUT", "SERVER_START_MAX_RETRIES", "SERVER_RETRY_SLEEP",
    "SERVER_START_STAGGER_SECONDS", "SERVER_STARTUP_STALL_TIMEOUT",
    "KEYFRAME_CLUSTER_TIMESTEP_WINDOW", "KEYFRAME_COMMIT_CONFIDENCE_THRESHOLD",
    "FIRST_CHUNK_RANDOM_REPLAN", "FIRST_CHUNK_RANDOM_REPLAN_SEED",
    "FIRST_CHUNK_FIXED_REPLAN_STEP", "REPLAN_AFTER_KEYFRAME_COMMIT",
    "SWAP_RB", "SWAP_RB_TASKS",
]
ARRAYS = ["TASKS", "GPU_SLOTS", "PORT_SLOTS", "EVAL_EXTRA_OVERRIDES", "SERVER_EXTRA_ARGS"]


def shell_config(path, checkpoint=None, ckpt_setting=None):
    # Only source an explicitly supplied, trusted Bash config. Preserve array
    # elements and spaces with NUL records; never eval generated command strings.
    shell = r'''
set -eo pipefail
source "$1" >&2
shift
names=("$@")
for ((i=1; i<=${#TASKS[@]}; i++)); do
  names+=("WEIGHT_TASK$i" "CKPT_SETTING_TASK$i" "KEYFRAME_CLUSTER_TIMESTEP_WINDOW_TASK$i")
done
for name in "${names[@]}"; do
  if declare -p "$name" >/dev/null 2>&1; then
    declare -n ref="$name"
    printf '%s\0%d\0' "$name" "${#ref[@]}"
    if ((${#ref[@]})); then printf '%s\0' "${ref[@]}"; fi
    unset -n ref
  fi
done
'''
    environment = os.environ.copy()
    if checkpoint:
        environment["LONGMEM_CKPT_PATH"] = str(existing(checkpoint))
    if ckpt_setting:
        environment["LONGMEM_CKPT_SETTING"] = ckpt_setting
    try:
        output = subprocess.check_output(
            ["bash", "-c", shell, "eval-config", str(path), *SCALARS, *ARRAYS], env=environment)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Weights config failed (exit {error.returncode}): {path}") from error
    pieces, config, i = output.decode().split("\0"), {}, 0
    while i < len(pieces) - 1:
        key, count = pieces[i], int(pieces[i + 1])
        values = pieces[i + 2:i + 2 + count]
        config[key] = values if key in ARRAYS else (values[0] if values else "")
        i += 2 + count
    return config


def existing(value, directory=False):
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Path not found: {path}")
    return path


def python_bin(value):
    path = shutil.which(str(value))
    if path is None: raise ValueError(f"Python is not executable: {value}")
    return str(Path(path).absolute())


def csv_list(value):
    parts = [p.strip() for p in value.split(",")]
    if any(not p for p in parts): raise ValueError("Comma-separated lists cannot contain empty entries")
    return parts


def boolean(value):
    if isinstance(value, bool): return value
    if str(value).strip().lower() in ("1", "true", "yes", "on"): return True
    if str(value).strip().lower() in ("0", "false", "no", "off"): return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def label(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "eval"


def arguments():
    p = argparse.ArgumentParser(prog=str(SCRIPT), description="RMBench: dynamic queue, one active task per GPU.")
    p.add_argument("--model-class", "--model", help="Optional compatibility label; must match the model_root package name")
    p.add_argument("--model-root", required=True, help="Model package directory owning eval/server_policy.py and the client")
    p.add_argument("--server-script", help="Server accepting --ckpt_path and --port")
    p.add_argument("--deploy-policy", help="RoboTwin deployment YAML")
    p.add_argument("--policy-name", help="Client module exposing get_model/reset_model/eval")
    p.add_argument("--checkpoint", "--ckpt", help="Checkpoint file for all selected tasks")
    p.add_argument("--weights-config", help="Existing trusted per-task Bash weights config")
    p.add_argument("--ckpt-setting", help="Result label; otherwise derives from model/run/checkpoint")
    p.add_argument("--tasks", help="Comma-separated tasks; defaults to config or 9 RMBench tasks")
    color = p.add_mutually_exclusive_group()
    color.add_argument("--swap-rb", dest="swap_rb", action="store_true", default=None,
                       help="Swap R/B on all three cameras before the policy records history; default off")
    color.add_argument("--no-swap-rb", dest="swap_rb", action="store_false",
                       help="Explicitly disable evaluation R/B swapping")
    p.add_argument("--swap-rb-tasks", help="Tasks affected by --swap-rb; comma-separated, default cover_blocks")
    p.add_argument("--gpus", help="Unique GPU pool, e.g. 0,1,2,3; default 0..7")
    p.add_argument("--ports", help="One distinct port per GPU")
    p.add_argument("--port-base", type=int, default=6902)
    p.add_argument("--test-num", type=int)
    p.add_argument("--sampling-interval", type=int, help="Replanning override; otherwise checkpoint setting")
    p.add_argument("--seed", type=int)
    p.add_argument("--task-config")
    p.add_argument("--host")
    p.add_argument("--unnorm-key")
    p.add_argument("--action-mode", choices=["abs", "delta"])
    p.add_argument("--instruction-type")
    p.add_argument("--eventvla-python", help="Model server Python")
    p.add_argument("--robotwin-mem-python", help="Simulator Python")
    p.add_argument("--robotwin-mem-root", help="RMBench/RoboTwin checkout")
    p.add_argument("--python-path", action="append", default=[], help="Additional import directory; repeatable")
    p.add_argument("--server-arg", action="append", default=[], help="Extra server token; e.g. --server-arg=--idle_timeout")
    p.add_argument("--eval-arg", action="append", default=[], help="Extra simulator override token; repeatable")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--server-ready-timeout", type=int)
    p.add_argument("--server-retries", type=int)
    p.add_argument("--run-tag")
    p.add_argument("--log-root")
    p.add_argument("--runtime-log-root")
    p.add_argument("--dry-run", action="store_true", help="Validate/print only; no files or model/simulator processes")
    return p.parse_args(sys.argv[2:])


def prepare(args):
    config = shell_config(existing(args.weights_config), args.checkpoint, args.ckpt_setting) if args.weights_config else {}
    def option(attr, name, default):
        value = getattr(args, attr, None)
        return value if value is not None else config.get(name, os.environ.get(name, default))

    if not args.model_root:
        raise ValueError("eval_rmbench_v2 requires --model-root")
    package = existing(args.model_root, True)
    package_name = package.name
    if not re.fullmatch(r"[A-Za-z_]\w*", package_name):
        raise ValueError(f"model_root basename must be a Python package name: {package_name}")
    if args.model_class and args.model_class not in (package_name, "custom"):
        raise ValueError(f"--model-class={args.model_class} conflicts with model_root package {package_name}")

    # Every model package owns its serving protocol and RMBench adapter.
    eval_dir = existing(package / "eval", True)
    for required in ("__init__.py", "server_policy.py", "model2robotwin_interface.py", "deploy_policy.yml"):
        existing(eval_dir / required)
    family = package_name
    server = eval_dir / "server_policy.py"
    deploy = eval_dir / "deploy_policy.yml"
    policy = f"{package_name}.eval.model2robotwin_interface"
    server, deploy, policy = args.server_script or server, args.deploy_policy or deploy, args.policy_name or policy
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", policy):
        raise ValueError("Invalid client module")
    server, deploy = existing(server), existing(deploy)

    robot = existing(option("robotwin_mem_root", "ROBOTWIN_MEM_ROOT", ROOT.parent.parent), True)
    eval_script = existing(robot / "script/eval_policy.py")
    server_python = python_bin(option("eventvla_python", "EVENTVLA_PYTHON", OWNER / "envs/starvla/bin/python"))
    eval_python = python_bin(option("robotwin_mem_python", "ROBOTWIN_MEM_PYTHON", OWNER / "miniconda3/envs/RMBench/bin/python"))
    config_tasks = config.get("TASKS", TASKS)
    tasks = csv_list(args.tasks) if args.tasks else list(config_tasks)
    if len(set(tasks)) != len(tasks) or not tasks:
        raise ValueError("Tasks must be nonempty and unique")
    swap_rb = boolean(option("swap_rb", "SWAP_RB", False))
    swap_rb_tasks = csv_list(str(option("swap_rb_tasks", "SWAP_RB_TASKS", "cover_blocks")))
    if len(set(swap_rb_tasks)) != len(swap_rb_tasks):
        raise ValueError("--swap-rb-tasks must be unique")
    for task in swap_rb_tasks:
        if not re.fullmatch(r"[A-Za-z_]\w*", task):
            raise ValueError(f"Invalid color-swap task: {task}")
        if swap_rb:
            existing(robot / "envs" / (task + ".py"))
    if swap_rb:
        existing(SCRIPT.parent / "rmbench_color_swap_interface.py")

    jobs = []
    for index, task in enumerate(tasks):
        if not re.fullmatch(r"[A-Za-z_]\w*", task):
            raise ValueError(f"Invalid task: {task}")
        existing(robot / "envs" / (task + ".py"))
        source_index = config_tasks.index(task) + 1 if task in config_tasks else index + 1
        if not args.checkpoint and task not in config_tasks:
            raise ValueError(f"No checkpoint configured for {task}")
        checkpoint = args.checkpoint or config.get(f"WEIGHT_TASK{source_index}")
        if not checkpoint:
            raise ValueError("Provide --checkpoint or --weights-config with per-task checkpoints")
        checkpoint = existing(checkpoint)
        run_dir = checkpoint.parent.parent
        existing(run_dir / "config.yaml")
        existing(run_dir / "dataset_statistics.json")
        default_tag = f"{family}_{run_dir.name}_{checkpoint.stem}"
        tag = args.ckpt_setting or (None if args.checkpoint else config.get(f"CKPT_SETTING_TASK{source_index}")) or default_tag
        task_swap = swap_rb and task in swap_rb_tasks
        jobs.append({"task": task, "task_index": index + 1, "checkpoint": str(checkpoint),
                     "ckpt_setting": label(tag) + ("_swap_rb" if task_swap else ""), "source_index": source_index,
                     "swap_rb": task_swap, "base_policy_name": policy,
                     "policy_name": "rmbench_color_swap_interface" if task_swap else policy})

    legacy_gpus = config.get("GPU_SLOTS", [])
    gpus = csv_list(args.gpus) if args.gpus else list(dict.fromkeys(legacy_gpus)) or [str(i) for i in range(8)]
    if len(set(gpus)) != len(gpus) or any(not re.fullmatch(r"\d+|(?:MIG-|GPU-)[A-Za-z0-9/-]+", g) for g in gpus):
        raise ValueError("--gpus must be a unique GPU pool")
    if args.ports:
        ports = [int(p) for p in csv_list(args.ports)]
    elif not args.gpus and legacy_gpus and config.get("PORT_SLOTS"):
        ports = [int(config["PORT_SLOTS"][legacy_gpus.index(g)]) for g in gpus]
    else:
        ports = [args.port_base + i for i in range(len(gpus))]
    if len(ports) != len(gpus) or len(set(ports)) != len(ports) or any(p < 1 or p > 65535 for p in ports):
        raise ValueError("Provide one distinct port in 1..65535 per GPU")

    test_num = int(option("test_num", "TEST_NUM", 50))
    sampling = option("sampling_interval", "SAMPLING_INTERVAL", None)
    if test_num <= 0 or (sampling is not None and int(sampling) <= 0):
        raise ValueError("Episode count/interval must be positive")
    host = str(option("host", "HOST", "127.0.0.1"))
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Local servers require a loopback --host")
    run_tag = label(args.run_tag or f"{family}_{dt.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}")
    logs = Path(args.log_root or ROOT / "scripts/logs/eval_batch" / run_tag).expanduser().resolve()
    runtime = Path(args.runtime_log_root or Path("/tmp/eventvla_rmbench_eval") / run_tag).expanduser().resolve()

    paths = [str(existing(p, True)) for p in args.python_path]
    paths.append(str(package.parent))
    paths += [str(SCRIPT.parent), str(ROOT), str(ROOT / "examples/RMBench/eval_files"), str(robot)]
    paths += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    for key, value in {"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "2", "RMBENCH_CUROBO_LBFGS": "torch",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
        environment.setdefault(key, value)
    environment["PATH"] = str(robot / ".setup/bin") + os.pathsep + environment.get("PATH", "")

    # Optional model-local validation runs before any GPU server is started.
    validator = eval_dir / "validate_checkpoint.py"
    if validator.is_file():
        for checkpoint in sorted({job["checkpoint"] for job in jobs}):
            subprocess.run([server_python, str(validator), "--checkpoint", checkpoint],
                           env=environment, check=True)

    plan = {
        "model_class": family, "model_root": str(package),
        "server_script": str(server), "deploy_policy": str(deploy), "policy_name": policy,
        "server_python": server_python, "eval_python": eval_python, "robot_root": str(robot),
        "eval_script": str(eval_script), "host": host, "jobs": jobs,
        "slots": list(zip(gpus, ports))[:len(jobs)], "test_num": test_num,
        "seed": int(option("seed", "SEED", 0)),
        "task_config": str(option("task_config", "TASK_CONFIG", "demo_clean")),
        "instruction_type": str(option("instruction_type", "INSTRUCTION_TYPE", "unseen")),
        "unnorm_key": str(option("unnorm_key", "UNNORM_KEY", "new_embodiment")),
        "action_mode": str(option("action_mode", "ACTION_MODE", "abs")),
        "sampling_interval": int(sampling) if sampling is not None else None,
        "bf16": not args.no_bf16 and str(config.get("USE_BF16", os.environ.get("USE_BF16", "1"))) == "1",
        "server_ready_timeout": int(option("server_ready_timeout", "SERVER_READY_TIMEOUT", 1800)),
        "server_retries": int(option("server_retries", "SERVER_START_MAX_RETRIES", 2)),
        "server_retry_sleep": float(config.get("SERVER_RETRY_SLEEP", 20)),
        "server_stagger": float(config.get("SERVER_START_STAGGER_SECONDS", 8)),
        "server_stall_timeout": float(config.get("SERVER_STARTUP_STALL_TIMEOUT", 1200)),
        "eval_extra": list(config.get("EVAL_EXTRA_OVERRIDES", [])) + args.eval_arg,
        "server_extra": list(config.get("SERVER_EXTRA_ARGS", [])) + args.server_arg,
        "log_root": str(logs), "runtime_log_root": str(runtime), "run_tag": run_tag,
        "python_path": paths, "config": config,
        "swap_rb": swap_rb, "swap_rb_tasks": swap_rb_tasks,
    }
    if plan["server_ready_timeout"] <= 0 or plan["server_retries"] <= 0:
        raise ValueError("Timeout/retry count must be positive")
    if swap_rb and any(token in ("--policy_name", "--base_policy_name", "--swap_rb") for token in plan["eval_extra"]):
        raise ValueError("Color-swap overrides are managed by --swap-rb/--swap-rb-tasks; use --policy-name for the base client")
    return plan, environment


def commands(plan, job, gpu, port):
    server = [plan["server_python"], plan["server_script"], "--ckpt_path", job["checkpoint"], "--port", str(port)]
    if plan["bf16"]: server.append("--use_bf16")
    server += plan["server_extra"]
    overrides = {
        "task_name": job["task"], "task_config": plan["task_config"], "ckpt_setting": job["ckpt_setting"],
        "seed": plan["seed"], "policy_name": job["policy_name"], "instruction_type": plan["instruction_type"],
        "host": plan["host"], "port": port, "policy_ckpt_path": job["checkpoint"],
        "unnorm_key": plan["unnorm_key"], "action_mode": plan["action_mode"],
    }
    if job["swap_rb"]:
        overrides.update(base_policy_name=job["base_policy_name"], swap_rb=True)
    optional = []
    for key in ("KEYFRAME_CLUSTER_TIMESTEP_WINDOW", "KEYFRAME_COMMIT_CONFIDENCE_THRESHOLD",
                "FIRST_CHUNK_RANDOM_REPLAN", "FIRST_CHUNK_RANDOM_REPLAN_SEED",
                "FIRST_CHUNK_FIXED_REPLAN_STEP", "REPLAN_AFTER_KEYFRAME_COMMIT"):
        value = plan["config"].get(f"{key}_TASK{job['source_index']}", plan["config"].get(key))
        if value is not None: optional += ["--" + key.lower(), str(value)]
    eval_cmd = [plan["eval_python"], plan["eval_script"], "--config", plan["deploy_policy"], "--overrides"]
    for name, value in overrides.items():
        eval_cmd += ["--" + name, repr(value) if isinstance(value, str) else str(value)]
    eval_cmd += optional + plan["eval_extra"]
    eval_cmd += ["--test_num", str(plan["test_num"])]
    if plan["sampling_interval"] is not None: eval_cmd += ["--sampling_interval", str(plan["sampling_interval"])]
    return server, eval_cmd


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.3): return True
    except OSError: return False


def result_metrics(plan, job, started):
    folder = Path(plan["robot_root"]) / "eval_result" / job["task"] / job["policy_name"] / plan["task_config"] / job["ckpt_setting"]
    files = [p for p in folder.glob("*/_result.txt") if p.stat().st_mtime >= started - 1]
    if not files: return {"success_rate": "NA", "result_file": "NA", "failed_examples": "NA"}
    latest = max(files, key=lambda p: p.stat().st_mtime)
    match = re.search(r"^Success Rate:\s*(.*)$", latest.read_text(), re.M)
    detail = latest.parent / "eval_log.txt"
    failures = [s for s in detail.read_text().splitlines() if "result=Fail" in s] if detail.is_file() else []
    return {"success_rate": match.group(1).strip() if match else "NA",
            "result_file": str(latest), "failed_examples": " | ".join(failures) or "none"}


def evaluate(plan, environment):
    logs, runtime = Path(plan["log_root"]), Path(plan["runtime_log_root"])
    if logs.exists() or runtime.exists(): raise ValueError("Output directory exists; choose a new --run-tag or output path")
    for gpu, port in plan["slots"]:
        if port_open(plan["host"], port): raise ValueError(f"Port {port} is occupied; choose --port-base/--ports. No process stopped.")
    logs.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (logs / "resolved_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    stop, lock = threading.Event(), threading.RLock()
    children, mirrors, rows = set(), [], []
    pending = queue.Queue()
    for job in plan["jobs"]: pending.put(job)

    def terminate(process):
        # Only process groups created by this invocation may be terminated.
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
        with lock: children.discard(process)

    def on_signal(signum, frame):
        stop.set()
        with lock: active = list(children)
        for process in active:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def launch(command, cwd, env, output):
        with lock:
            if stop.is_set(): raise InterruptedError("Evaluation interrupted")
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            children.add(process)
            return process

    def mirror_logs():
        with lock: pairs = list(mirrors)
        for src, dest in pairs:
            try: shutil.copyfile(src, dest)
            except OSError: pass

    def run_job(job, gpu, port):
        started = time.time()
        taskdir, localdir = logs / "tasks" / job["task"], runtime / job["task"]
        taskdir.mkdir(parents=True)
        localdir.mkdir()
        serverlog, evallog = localdir / "server.log", localdir / "eval.log"
        row = {**job, "gpu": gpu, "port": port, "seed": plan["seed"],
               "server_ready": 0, "eval_exit": None, "final_exit": 1,
               "server_log": str(taskdir / "server.log"), "eval_log": str(taskdir / "eval.log"),
               "runtime_server_log": str(serverlog), "runtime_eval_log": str(evallog),
               "started_at": dt.datetime.now().astimezone().isoformat()}
        with lock: mirrors.extend([(serverlog, taskdir / "server.log"), (evallog, taskdir / "eval.log")])
        server_cmd, eval_cmd = commands(plan, job, gpu, port)
        (taskdir / "commands.json").write_text(json.dumps({"server": server_cmd, "eval": eval_cmd}, indent=2) + "\n")
        (taskdir / "status.json").write_text(json.dumps(row, indent=2) + "\n")
        child_env = environment.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        server_env, eval_env = child_env.copy(), child_env.copy()
        server_env["PATH"] = str(Path(plan["server_python"]).parent) + os.pathsep + environment["PATH"]
        eval_env["PATH"] = str(Path(plan["eval_python"]).parent) + os.pathsep + environment["PATH"]
        processes = []
        try:
            if port_open(plan["host"], port): raise RuntimeError(f"Port {port} became occupied")
            print(f"[start] {job['task']} gpu={gpu} port={port}", flush=True)
            with serverlog.open("a") as output:
                for attempt in range(plan["server_retries"]):
                    output.write(f"\nServer attempt {attempt + 1}\n")
                    output.flush()
                    server = launch(server_cmd, ROOT, server_env, output)
                    processes.append(server)
                    deadline = time.monotonic() + plan["server_ready_timeout"]
                    last_size, last_progress = -1, time.monotonic()
                    while not stop.is_set() and time.monotonic() < deadline:
                        if server.poll() is not None: break
                        if port_open(plan["host"], port):
                            row["server_ready"] = 1
                            break
                        size = serverlog.stat().st_size
                        if size != last_size: last_size, last_progress = size, time.monotonic()
                        if plan["server_stall_timeout"] > 0 and time.monotonic() - last_progress > plan["server_stall_timeout"]: break
                        stop.wait(1)
                    if row["server_ready"]: break
                    terminate(server)
                    processes.remove(server)
                    if stop.is_set(): raise InterruptedError("Evaluation interrupted")
                    if attempt + 1 < plan["server_retries"]: stop.wait(plan["server_retry_sleep"])
                if not row["server_ready"]: raise RuntimeError("Server did not become ready; inspect server.log")
                with evallog.open("w") as eval_output:
                    evaluation = launch(eval_cmd, plan["robot_root"], eval_env, eval_output)
                    processes.append(evaluation)
                    while evaluation.poll() is None and not stop.is_set():
                        if server.poll() is not None: raise RuntimeError("Server exited during evaluation")
                        stop.wait(1)
                    if stop.is_set(): raise InterruptedError("Evaluation interrupted")
                    row["eval_exit"] = row["final_exit"] = evaluation.returncode
        except Exception as error:
            row["error"], row["final_exit"] = str(error), 130 if stop.is_set() else 1
            print(f"[error] {job['task']}: {error}", flush=True)
        finally:
            for process in reversed(processes): terminate(process)
            row.update(result_metrics(plan, job, started))
            if row["final_exit"] == 0 and row["result_file"] == "NA":
                row["final_exit"] = 1
                row["error"] = "Simulator exited without a result file from this evaluation"
            row["finished_at"] = dt.datetime.now().astimezone().isoformat()
            row["elapsed_seconds"] = round(time.time() - started, 1)
            (taskdir / "status.json").write_text(json.dumps(row, indent=2) + "\n")
            (taskdir / "exit_code").write_text(str(row["final_exit"]) + "\n")
            with lock:
                rows.append(row)
                with (logs / "summary.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            print(f"[done] {job['task']} exit={row['final_exit']} success={row['success_rate']}", flush=True)

    def worker(index, gpu, port):
        if stop.wait(index * plan["server_stagger"]): return
        while not stop.is_set():
            try: job = pending.get_nowait()
            except queue.Empty: return
            run_job(job, gpu, port)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan["slots"])) as pool:
            futures = [pool.submit(worker, i, gpu, port) for i, (gpu, port) in enumerate(plan["slots"])]
            last_copy = 0.0
            while not all(f.done() for f in futures):
                if time.monotonic() - last_copy >= 60:
                    mirror_logs()
                    last_copy = time.monotonic()
                time.sleep(1)
            for future in futures: future.result()
    finally:
        with lock: active = list(children)
        for process in active: terminate(process)
        mirror_logs()
    fields = ["task_index", "task", "checkpoint", "ckpt_setting", "gpu", "port", "seed", "swap_rb", "policy_name", "base_policy_name",
              "server_ready", "eval_exit", "final_exit", "success_rate", "failed_examples",
              "result_file", "server_log", "eval_log", "elapsed_seconds", "error"]
    with (logs / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["task_index"]))
    code = 130 if stop.is_set() else int(len(rows) != len(plan["jobs"]) or any(r["final_exit"] for r in rows))
    (logs / "finished.txt").write_text(f"finished_at={dt.datetime.now().astimezone().isoformat()}\noverall_exit={code}\n")
    print(f"Summary: {logs / 'summary.csv'}", flush=True)
    return code


def main():
    args = arguments()
    plan, environment = prepare(args)
    print(f"Model: {plan['model_class']}\nServer: {plan['server_script']}\nClient: {plan['policy_name']}", flush=True)
    print(f"GPU slots: {plan['slots']}; tasks: {len(plan['jobs'])}; logs: {plan['log_root']}", flush=True)
    print(f"R/B swap: {'enabled' if plan['swap_rb'] else 'disabled'}; affected selected tasks: "
          f"{[job['task'] for job in plan['jobs'] if job['swap_rb']]}", flush=True)
    if args.dry_run:
        for i, job in enumerate(plan["jobs"]):
            gpu, port = plan["slots"][i % len(plan["slots"])]
            server_cmd, eval_cmd = commands(plan, job, gpu, port)
            print(f"[dry-run] {job['task']} gpu={gpu} port={port} checkpoint={job['checkpoint']}")
            print("  server: " + shlex.join(server_cmd))
            print("  eval:   " + shlex.join(eval_cmd))
        print("[dry-run] Slots shown are illustrative; actual tasks use the next free GPU. No files or model/simulator processes created.")
        return 0
    return evaluate(plan, environment)


try:
    raise SystemExit(main())
except (ValueError, OSError, subprocess.CalledProcessError) as error:
    print(f"[eval-rmbench][error] {error}", file=sys.stderr)
    raise SystemExit(2)
PYTHON
