import json
import tempfile
import unittest
from pathlib import Path

import torch

import vocab as V
from grpo import (active_completion_ids, exact_division_rewards,
                  grpo_loss, group_advantages, load_division_examples)


class GrpoTests(unittest.TestCase):
    def setUp(self):
        self.stoi, _ = V.build_vocab()

    def test_group_advantages_are_centered_and_constant_groups_are_zero(self):
        rewards = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        advantages = group_advantages(rewards, group_size=4).view(2, 4)
        self.assertAlmostEqual(float(advantages[0].mean()), 0.0, places=6)
        self.assertTrue(torch.equal(advantages[1], torch.zeros(4)))
        self.assertLess(float(advantages[0, 0]), 0.0)
        self.assertGreater(float(advantages[0, 1]), 0.0)

    def test_exact_reward_uses_canonical_internal_division_answer(self):
        correct = V.encode("000.333", self.stoi)
        normalized_but_not_canonical = V.encode("0.333", self.stoi)
        width = len(correct) + 1
        rows = [
            correct + [self.stoi[V.EOS]],
            normalized_but_not_canonical
            + [self.stoi[V.PAD]] * (width - len(normalized_but_not_canonical)),
        ]
        mask = torch.tensor([
            [True] * width,
            [True] * len(normalized_but_not_canonical)
            + [False] * (width - len(normalized_but_not_canonical)),
        ])
        rewards = exact_division_rewards(
            torch.tensor(rows), mask, [tuple(correct)], group_size=2,
            eos_id=self.stoi[V.EOS])
        self.assertEqual(rewards.tolist(), [1.0, 0.0])

    def test_terminal_eos_is_excluded_but_generated_padding_is_not(self):
        eos = self.stoi[V.EOS]
        pad = self.stoi[V.PAD]
        one = self.stoi["1"]
        self.assertEqual(
            active_completion_ids([one, eos, pad], [True, True, False], eos),
            (one,))
        self.assertEqual(
            active_completion_ids([one, pad], [True, True], eos),
            (one, pad))

    def test_grpo_objective_backpropagates_group_preferences(self):
        new = torch.zeros((2, 2), requires_grad=True)
        old = torch.zeros_like(new)
        reference = torch.zeros_like(new)
        mask = torch.ones_like(new, dtype=torch.bool)
        advantages = torch.tensor([-1.0, 1.0])
        loss, metrics = grpo_loss(
            new, old, reference, mask, advantages,
            clip_epsilon=0.2, beta=0.01)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(new.grad[0].mean()), 0.0)
        self.assertLess(float(new.grad[1].mean()), 0.0)
        self.assertEqual(metrics["reference_kl"], 0.0)

    def test_loader_filters_to_division_and_validates_representation(self):
        rows = [
            {
                "text": "1/3=000.333", "split": "train", "operation": "/",
                "scenario": "rounded", "division_sign": "positive",
                "operand_digits": "1x1", "tier": 2,
                "representation": "abacus-v1",
            },
            {
                "text": "1+2=3", "split": "train", "operation": "+",
                "representation": "abacus-v1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            examples = load_division_examples(
                path, "train", self.stoi, max_seq_len=20,
                expected_representation="abacus-v1")
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].truth_ids,
                         tuple(V.encode("000.333", self.stoi)))

    def test_loader_can_focus_a_division_stratum(self):
        rows = [
            {
                "text": "1/3=000.333", "split": "train", "operation": "/",
                "operand_digits": "1x1", "tier": 2,
                "representation": "abacus-v1",
            },
            {
                "text": "819/942=000.869", "split": "train",
                "operation": "/", "operand_digits": "3x3", "tier": 4,
                "representation": "abacus-v1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            with path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            examples = load_division_examples(
                path, "train", self.stoi, max_seq_len=20,
                expected_representation="abacus-v1",
                operand_digits="3x3", min_tier=4)
        self.assertEqual([example.text for example in examples],
                         ["819/942=000.869"])


if __name__ == "__main__":
    unittest.main()
