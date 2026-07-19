import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

import vocab as V
from grpo_all import (ArithmeticExample, allocate_weighted_counts,
                      checkpoint_gate, exact_rewards,
                      load_arithmetic_examples, load_arithmetic_sources,
                      replay_anchor_losses,
                      shaped_rewards, supervised_replay_loss,
                      weighted_prompt_batches)
from model import build_model


class AllOperationGrpoTests(unittest.TestCase):
    def setUp(self):
        self.stoi, _ = V.build_vocab()

    def example(self, operation, prompt, truth):
        return ArithmeticExample(
            prompt_ids=tuple(
                [self.stoi[V.BOS]] + V.encode(prompt, self.stoi)),
            truth_ids=tuple(V.encode(truth, self.stoi)),
            text=prompt + truth,
            operation=operation,
            scenario="test",
            tier=4,
            stratum=("3x3", "test"),
        )

    def test_weight_allocation_is_exact(self):
        counts = allocate_weighted_counts(
            100, {"*": 0.45, "/": 0.35, "+": 0.10, "-": 0.10})
        self.assertEqual(counts, {"*": 45, "/": 35, "+": 10, "-": 10})
        self.assertEqual(sum(counts.values()), 100)

    def test_weighted_batches_do_not_mix_operations(self):
        examples = []
        prompts = {
            "+": ("321+654=", "777"),
            "-": ("654-321=", "333"),
            "*": ("321*654=", "83025"),
            "/": ("123/456=", "000.270"),
        }
        for operation, (prompt, truth) in prompts.items():
            examples.extend(self.example(operation, prompt, truth)
                            for _ in range(10))
        batches, counts = weighted_prompt_batches(
            examples, 20, {operation: 1 for operation in prompts}, 3,
            random.Random(42))
        self.assertEqual(sum(counts.values()), 20)
        self.assertTrue(all(len({row.operation for row in batch}) == 1
                            for batch in batches))

    def test_weighted_batches_cover_every_micro_stratum(self):
        examples = [
            dataclasses.replace(
                self.example("*", "12*34=", "804"),
                stratum=("1x2", "dense")),
            dataclasses.replace(
                self.example("*", "123*456=", "88065"),
                stratum=("3x3", "dense")),
            dataclasses.replace(
                self.example("*", "100*200=", "00002"),
                stratum=("3x3", "zero")),
        ]
        batches, _ = weighted_prompt_batches(
            examples, 3, {"*": 1.0}, 2, random.Random(7))
        observed = {
            example.stratum for batch in batches for example in batch}
        self.assertEqual(observed, {example.stratum for example in examples})

    def test_priority_sampling_favors_requested_sign_pattern(self):
        examples = []
        for index in range(100):
            sign = "negative_negative" if index < 50 else "positive_positive"
            examples.append(dataclasses.replace(
                self.example("*", f"{index + 100}*2=", "0"),
                stratum=("3x1", "dense", sign)))
        batches, _ = weighted_prompt_batches(
            examples, 30, {"*": 1.0}, 5, random.Random(11),
            priority={"*": {
                "sign_pattern_weights": {"negative_negative": 5.0}}})
        selected = [example for batch in batches for example in batch]
        negative = sum(
            example.stratum[2] == "negative_negative" for example in selected)
        self.assertGreater(negative, 20)

    def test_exact_reward_supports_reversed_and_division_answers(self):
        eos = self.stoi[V.EOS]
        pad = self.stoi[V.PAD]
        answers = [
            V.encode("861", self.stoi),
            V.encode("000.333", self.stoi),
        ]
        width = 8
        ids = torch.tensor([
            answers[0] + [eos] + [pad] * (width - len(answers[0]) - 1),
            answers[1] + [eos],
        ])
        mask = torch.tensor([
            [True] * (len(answers[0]) + 1)
            + [False] * (width - len(answers[0]) - 1),
            [True] * width,
        ])
        rewards = exact_rewards(
            ids, mask, [tuple(answers[0]), tuple(answers[1])], 1, eos)
        self.assertEqual(rewards.tolist(), [1.0, 1.0])

    def test_shaped_reward_ranks_canonical_token_matches_below_exact(self):
        eos = self.stoi[V.EOS]
        truth = V.encode("000.333", self.stoi)
        samples = [
            truth,
            V.encode("000.332", self.stoi),
            V.encode("999.999", self.stoi),
        ]
        ids = torch.tensor([sample + [eos] for sample in samples])
        mask = torch.ones_like(ids, dtype=torch.bool)
        rewards = shaped_rewards(
            ids, mask, [tuple(truth)], 3, eos, 0.25, 100.0)
        self.assertEqual(float(rewards[0]), 1.0)
        self.assertGreater(float(rewards[1]), float(rewards[2]))
        self.assertLess(float(rewards[1]), 0.25)

    def test_numeric_shaping_rewards_closer_fixed_width_division(self):
        eos = self.stoi[V.EOS]
        truth = V.encode("012.345", self.stoi)
        samples = [
            V.encode("012.344", self.stoi),
            V.encode("012.245", self.stoi),
            V.encode("12.344", self.stoi),
        ]
        width = max(len(sample) for sample in samples) + 1
        pad = self.stoi[V.PAD]
        ids = torch.tensor([
            sample + [eos] + [pad] * (width - len(sample) - 1)
            for sample in samples])
        mask = torch.tensor([
            [True] * (len(sample) + 1)
            + [False] * (width - len(sample) - 1)
            for sample in samples])
        rewards = shaped_rewards(
            ids, mask, [tuple(truth)], 3, eos, 0.25, 100.0)
        self.assertGreater(float(rewards[0]), float(rewards[1]))
        self.assertGreater(float(rewards[0]), float(rewards[2]))

    def test_nan_shaping_remains_exact_only(self):
        eos = self.stoi[V.EOS]
        nan = V.encode("NAN", self.stoi)
        numeric = V.encode("000.000", self.stoi)
        ids = torch.tensor([numeric + [eos]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        rewards = shaped_rewards(
            ids, mask, [tuple(nan)], 1, eos, 0.25, 100.0)
        self.assertEqual(float(rewards[0]), 0.0)

    def test_loader_filters_operations_and_reservoir_size(self):
        rows = []
        for value in range(5):
            rows.append({
                "text": f"{value}+1={value + 1}", "split": "train",
                "operation": "+", "tier": 1,
                "representation": "abacus-v1",
            })
        rows.append({
            "text": "1/2=000.500", "split": "train", "operation": "/",
            "tier": 2, "representation": "abacus-v1",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            examples = load_arithmetic_examples(
                path, "train", self.stoi, 20, ["+"], "abacus-v1",
                max_per_operation=2, seed=42)
        self.assertEqual(len(examples), 2)
        self.assertTrue(all(example.operation == "+" for example in examples))

    def test_multi_source_loader_deduplicates_prompts(self):
        rows = [{
            "text": "1+2=3", "split": "train", "operation": "+",
            "tier": 1, "representation": "abacus-v1",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with path.open("w") as f:
                f.write(json.dumps(rows[0]) + "\n")
            sources = [
                {"data_path": str(path), "split": "train"},
                {"data_path": str(path), "split": "train"},
            ]
            examples = load_arithmetic_sources(
                sources, self.stoi, 20, ["+"], "abacus-v1")
        self.assertEqual(len(examples), 1)

    def test_supervised_replay_loss_backpropagates(self):
        model = build_model({
            "vocab_size": V.VOCAB_SIZE,
            "max_seq_len": 20,
            "d_model": 16,
            "attention_heads": 4,
            "n_layers": 1,
            "n_loops": 1,
            "dropout": 0.0,
        })
        examples = [
            self.example("+", "1+2=", "3"),
            self.example("-", "5-2=", "3"),
        ]
        loss = supervised_replay_loss(
            model, examples, self.stoi, torch.device("cpu"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()))

    def test_replay_distillation_detects_policy_drift(self):
        config = {
            "vocab_size": V.VOCAB_SIZE,
            "max_seq_len": 20,
            "d_model": 16,
            "attention_heads": 4,
            "n_layers": 1,
            "n_loops": 1,
            "dropout": 0.0,
        }
        reference = build_model(config)
        policy = build_model(config)
        policy.load_state_dict(reference.state_dict())
        examples = [self.example("+", "1+2=", "3")]
        _, initial_kl = replay_anchor_losses(
            policy, reference, examples, self.stoi, torch.device("cpu"))
        with torch.no_grad():
            policy.projection.weight[0, 0] += 1.0
        _, drifted_kl = replay_anchor_losses(
            policy, reference, examples, self.stoi, torch.device("cpu"))
        self.assertAlmostEqual(float(initial_kl.detach()), 0.0, places=6)
        self.assertGreater(float(drifted_kl.detach()), 0.0)

    def test_checkpoint_gate_rejects_protected_regression(self):
        baseline = {
            "accuracy": 0.70,
            "operation": {"+": 0.985, "-": 0.976,
                          "*": 0.692, "/": 0.125},
        }
        candidate = {
            "accuracy": 0.70,
            "operation": {"+": 0.980, "-": 0.976,
                          "*": 0.700, "/": 0.130},
        }
        gate = checkpoint_gate(candidate, baseline, {
            "protected_operations": {"+": 0.002, "-": 0.002},
            "overall_regression_tolerance": 0.001,
            "target_weights": {"*": 0.55, "/": 0.45},
        })
        self.assertFalse(gate["eligible"])
        self.assertTrue(any(message.startswith("+=")
                            for message in gate["violations"]))


if __name__ == "__main__":
    unittest.main()
