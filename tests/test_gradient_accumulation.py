import ast
import unittest
from pathlib import Path

import torch


def _load_scale_microbatch_loss():
    path = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_scale_microbatch_loss"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function.name]


_scale_microbatch_loss = _load_scale_microbatch_loss()


def _load_train_main_ast():
    path = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


class GradientAccumulationTests(unittest.TestCase):
    def test_eight_serial_microbatches_match_one_batch_mean(self):
        serial = torch.nn.Linear(2, 1, bias=False)
        batched = torch.nn.Linear(2, 1, bias=False)
        batched.load_state_dict(serial.state_dict())
        serial_optimizer = torch.optim.SGD(serial.parameters(), lr=0.1)
        batched_optimizer = torch.optim.SGD(batched.parameters(), lr=0.1)
        inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10
        targets = torch.arange(8, dtype=torch.float32).reshape(8, 1) / 10

        serial_optimizer.zero_grad()
        for input_, target in zip(inputs, targets):
            loss = torch.nn.functional.mse_loss(serial(input_[None]), target[None])
            _scale_microbatch_loss(loss, 8).backward()
        serial_optimizer.step()

        batched_optimizer.zero_grad()
        torch.nn.functional.mse_loss(batched(inputs), targets).backward()
        batched_optimizer.step()

        torch.testing.assert_close(serial.weight, batched.weight)

    def test_rejects_zero_accumulation_steps(self):
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            _scale_microbatch_loss(torch.tensor(1.0), 0)

    def test_optimizer_update_occurs_after_full_accumulation_guard(self):
        main = _load_train_main_ast()
        calls = {}
        for node in ast.walk(main):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = ast.unparse(node.func.value)
            calls.setdefault(f"{owner}.{node.func.attr}", []).append(node)

        for name in (
            "optimizer.zero_grad",
            "fabric.backward",
            "fabric.clip_gradients",
            "optimizer.step",
            "scheduler.step",
        ):
            self.assertEqual(len(calls[name]), 1, name)

        guard = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "microbatches_accumulated < gradient_accumulation_steps"
        )
        self.assertTrue(any(isinstance(node, ast.Continue) for node in guard.body))

        backward_line = calls["fabric.backward"][0].lineno
        guard_line = guard.lineno
        clip_line = calls["fabric.clip_gradients"][0].lineno
        optimizer_line = calls["optimizer.step"][0].lineno
        scheduler_line = calls["scheduler.step"][0].lineno
        self.assertLess(backward_line, guard_line)
        self.assertLess(guard_line, clip_line)
        self.assertLess(clip_line, optimizer_line)
        self.assertLess(optimizer_line, scheduler_line)

        backward_argument = ast.unparse(calls["fabric.backward"][0].args[0])
        self.assertEqual(
            backward_argument,
            "_scale_microbatch_loss(loss, gradient_accumulation_steps)",
        )

    def test_expensive_diagnostics_use_optimizer_step_interval(self):
        main_source = ast.unparse(_load_train_main_ast())

        self.assertIn(
            "total_steps % expensive_diagnostics_interval == 0",
            main_source,
        )
        self.assertIn(
            "run_expensive_diagnostics=run_expensive_diagnostics",
            main_source,
        )
        self.assertIn("gradient_diagnostics.begin()", main_source)


if __name__ == "__main__":
    unittest.main()
