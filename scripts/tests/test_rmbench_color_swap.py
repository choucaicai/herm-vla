"""Exercise task-selective command planning and policy-boundary color handling."""
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rmbench_color_swap_interface", SCRIPTS / "rmbench_color_swap_interface.py")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        top = Path(self.temp.name)
        self.root, self.robot = top / "EventVLA", top / "RMBench"
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(SCRIPTS / "rmbench_color_swap_interface.py", scripts / "rmbench_color_swap_interface.py")
        (self.robot / "script").mkdir(parents=True)
        (self.robot / "script/eval_policy.py").touch()
        (self.robot / "envs").mkdir()
        for task in ["cover_blocks", "put_back_block"]:
            (self.robot / f"envs/{task}.py").touch()
        self.model_root = self.root / "plain_model"
        eval_dir = self.model_root / "eval"
        eval_dir.mkdir(parents=True)
        for name in ["__init__.py", "server_policy.py", "model2robotwin_interface.py", "deploy_policy.yml"]:
            (eval_dir / name).touch()
        run = self.root / "run"
        (run / "checkpoints").mkdir(parents=True)
        self.checkpoint = run / "checkpoints/state.pt"
        self.checkpoint.touch()
        (run / "config.yaml").touch()
        (run / "dataset_statistics.json").write_text("{}")
        body = (SCRIPTS / "eval_rmbench_v2.sh").read_text().split("<<'PYTHON'\n", 1)[1].rsplit("\nPYTHON", 1)[0]
        tree = ast.parse(body)
        self.assertIsInstance(tree.body[-1], ast.Try)
        tree.body.pop()  # Import the entry point without invoking main.
        self.ns = {"__name__": "eval_rmbench_test"}
        with mock.patch.object(sys, "argv", ["python", str(scripts / "eval_rmbench_v2.sh")]):
            exec(compile(tree, str(SCRIPTS / "eval_rmbench_v2.sh"), "exec"), self.ns)
        self.cli = ["python", str(scripts / "eval_rmbench_v2.sh"),
                    "--model-root", str(self.model_root), "--checkpoint", str(self.checkpoint),
                    "--robotwin-mem-root", str(self.robot), "--eventvla-python", sys.executable,
                    "--robotwin-mem-python", sys.executable, "--tasks", "cover_blocks,put_back_block", "--gpus", "0"]

    def prepare(self, *tokens, env=None):
        with mock.patch.dict(os.environ, {"SWAP_RB": "0", "SWAP_RB_TASKS": "cover_blocks", **(env or {})}), \
             mock.patch.object(sys, "argv", self.cli + list(tokens)):
            return self.ns["prepare"](self.ns["arguments"]())[0]

    def overrides(self, plan, job):
        _, command = self.ns["commands"](plan, job, "0", 7000)
        values = command[command.index("--overrides") + 1:]
        return {values[i]: ast.literal_eval(values[i + 1]) for i in range(0, len(values), 2)}

    def test_default_preserves_original_policy(self):
        plan = self.prepare()
        for job in plan["jobs"]:
            self.assertFalse(job["swap_rb"])
            self.assertEqual(job["policy_name"], "plain_model.eval.model2robotwin_interface")
            self.assertNotIn("--swap_rb", self.overrides(plan, job))

    def test_enabled_only_for_cover_blocks_and_results_are_read_from_wrapper(self):
        plan = self.prepare("--swap-rb")
        cover, other = plan["jobs"]
        self.assertTrue(cover["swap_rb"])
        self.assertFalse(other["swap_rb"])
        args = self.overrides(plan, cover)
        self.assertEqual(args["--policy_name"], "rmbench_color_swap_interface")
        self.assertEqual(args["--base_policy_name"], "plain_model.eval.model2robotwin_interface")
        self.assertIs(args["--swap_rb"], True)
        folder = self.robot / "eval_result" / cover["task"] / cover["policy_name"] / plan["task_config"] / cover["ckpt_setting"] / "episode"
        folder.mkdir(parents=True)
        (folder / "_result.txt").write_text("Success Rate: 1.0\n")
        metrics = self.ns["result_metrics"](plan, cover, time.time() - 2)
        self.assertEqual(metrics["success_rate"], "1.0")

    def test_explicit_disable_overrides_environment(self):
        plan = self.prepare("--no-swap-rb", "--swap-rb-tasks", "cover_blocks", env={"SWAP_RB": "1"})
        self.assertFalse(plan["swap_rb"])
        self.assertFalse(any(job["swap_rb"] for job in plan["jobs"]))

    def test_explicit_task_selection(self):
        plan = self.prepare("--swap-rb", "--swap-rb-tasks", "put_back_block")
        self.assertFalse(plan["jobs"][0]["swap_rb"])
        self.assertTrue(plan["jobs"][1]["swap_rb"])

    def test_task_names_and_unsafe_override_conflicts_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare("--swap-rb", "--swap-rb-tasks", "../cover_blocks")
        with self.assertRaises(ValueError):
            self.prepare("--swap-rb", "--eval-arg=--swap_rb", "--eval-arg=False")

    def test_disabled_does_not_require_cover_environment(self):
        (self.robot / "envs/cover_blocks.py").unlink()
        self.cli[self.cli.index("--tasks") + 1] = "put_back_block"
        self.assertFalse(self.prepare()["jobs"][0]["swap_rb"])


class PolicyBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("fake_color_base")
        self.client = types.SimpleNamespace(history=[], resets=0)
        self.module.get_model = lambda args: self.client

        def evaluate(task, client, observation):
            client.history.append(observation)

        def reset(client):
            client.resets += 1
            client.history.clear()

        self.module.eval, self.module.reset_model = evaluate, reset
        self.patch = mock.patch.dict(sys.modules, {self.module.__name__: self.module})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def observation(self, delta=0):
        return {"observation": {camera: {"rgb": np.full((3, 4, 3), [10 + delta, 20, 90 + delta], dtype=np.uint8)}
                               for camera in adapter.CAMERAS}, "joint_action": {"vector": [1, 2]}}

    def test_three_camera_swap_precedes_history_and_reset_is_delegated(self):
        model = adapter.get_model({"base_policy_name": self.module.__name__, "swap_rb": True})
        for delta in [0, 1]:
            original = self.observation(delta)
            adapter.eval(None, model, original)
            for camera in adapter.CAMERAS:
                rgb = self.client.history[-1]["observation"][camera]["rgb"]
                np.testing.assert_array_equal(rgb[0, 0], [90 + delta, 20, 10 + delta])
                self.assertTrue(rgb.flags.c_contiguous)
                np.testing.assert_array_equal(original["observation"][camera]["rgb"][0, 0], [10 + delta, 20, 90 + delta])
        self.assertEqual(len(self.client.history), 2)
        adapter.reset_model(model)
        self.assertEqual(self.client.resets, 1)
        self.assertEqual(self.client.history, [])

    def test_disabled_passes_original_observation_through(self):
        model = adapter.get_model({"base_policy_name": self.module.__name__, "swap_rb": False})
        obs = self.observation()
        adapter.eval(None, model, obs)
        self.assertIs(self.client.history[0], obs)

    def test_bad_shape_fails_before_delegate(self):
        model = adapter.get_model({"base_policy_name": self.module.__name__, "swap_rb": True})
        obs = self.observation()
        obs["observation"]["head_camera"]["rgb"] = np.zeros((3, 4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            adapter.eval(None, model, obs)
        self.assertEqual(self.client.history, [])


if __name__ == "__main__":
    unittest.main()
