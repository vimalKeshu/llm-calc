import tempfile
import unittest
from pathlib import Path

import torch

import vocab as V
from generate_data import format_division_answer
from generate_data_v2 import (build_rows, division_capability_group,
                              length_targets, load_spec, operation_targets,
                              scenario_targets)
from model import RandomizedAbacusEmbedding


class CompactDivisionDataTests(unittest.TestCase):
    def sample(self, a, b, scenario):
        return {
            "a": a,
            "b": b,
            "op": "/",
            "scenario": scenario,
            "primary": ("length", len(str(abs(a))), len(str(abs(b))),
                        "negative" if a < 0 else "positive"),
        }

    def test_fixed_width_remains_the_default(self):
        self.assertEqual(format_division_answer(0.64), "000.640")
        self.assertEqual(format_division_answer(-4.014), "-004.014")

    def test_compact_format_keeps_three_decimal_places(self):
        self.assertEqual(
            format_division_answer(0.64, "compact_fixed_precision"),
            "0.640")
        self.assertEqual(
            format_division_answer(-4.014, "compact_fixed_precision"),
            "-4.014")
        self.assertEqual(
            format_division_answer(294, "compact_fixed_precision"),
            "294.000")
        self.assertEqual(
            format_division_answer("NAN", "compact_fixed_precision"),
            "NAN")

    def test_compact_rows_round_trip_and_mark_focus_capability(self):
        rows = build_rows(
            [self.sample(596, 931, "rounded_below_medium")],
            "train", reverse=True, max_seq_len=20,
            division_answer_format="compact_fixed_precision",
            representation="abacus-v2-compact-division",
            division_focus_scenarios=["rounded_below_medium"])
        row = rows[0]
        self.assertEqual(row["text"], "596/931=0.640")
        self.assertEqual(row["representation"],
                         "abacus-v2-compact-division")
        self.assertEqual(row["division_answer_format"],
                         "compact_fixed_precision")
        self.assertEqual(row["division_capability_group"],
                         "rounded_below_one")
        self.assertTrue(row["division_capability_focus"])

    def test_compact_rows_support_negative_and_nan_answers(self):
        rows = build_rows(
            [
                self.sample(-289, 72, "rounded_ge1_q1digit"),
                self.sample(-15, 0, "division_by_zero"),
            ],
            "train", reverse=True, max_seq_len=20,
            division_answer_format="compact_fixed_precision")
        self.assertEqual(rows[0]["text"], "-289/72=-4.014")
        self.assertEqual(rows[1]["text"], "-15/0=NAN")

    def test_v3_spec_inherits_v2_and_targets_measured_weaknesses(self):
        spec = load_spec("config/data_distribution_v3_compact_division.yaml")
        operations = operation_targets(spec["dataset"]["train_rows"], spec)
        self.assertEqual(
            operations, {"+": 100000, "-": 100000,
                         "*": 100000, "/": 200000})
        division_lengths = length_targets("/", operations["/"], spec)
        self.assertEqual(division_lengths["3x3"], 80000)
        division_scenarios = scenario_targets("/", operations["/"], spec)
        focus = spec["operations"]["/"]["capability_focus_scenarios"]
        self.assertEqual(sum(division_scenarios[name] for name in focus), 142000)
        self.assertEqual(
            spec["operations"]["+"]["scenario_weights"]["no_carry"],
            0.14)

    def test_distribution_spec_inheritance_rejects_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.yaml"
            path.write_text("extends: cycle.yaml\n")
            with self.assertRaises(ValueError):
                load_spec(path)

    def test_capability_groups_cover_measured_failure_regimes(self):
        self.assertEqual(
            division_capability_group("exact_q_2digit"),
            "exact_two_digit_quotient")
        self.assertEqual(
            division_capability_group("near_integer_above"),
            "near_integer_boundary")
        self.assertEqual(
            division_capability_group("terminating_at_least_one_3dp"),
            "terminating_at_least_one")

    def test_compact_answer_digits_do_not_receive_fixed_width_places(self):
        stoi, _ = V.build_vocab()
        embedding = RandomizedAbacusEmbedding(
            d_model=8,
            digit_token_ids=[stoi[digit] for digit in V.DIGITS],
            division_token_id=stoi['/'], equals_token_id=stoi['='],
            decimal_token_id=stoi['.'], max_offset=0,
            division_answer_format="compact_fixed_precision")
        text = "596/931=0.640"
        ids = torch.tensor([[stoi[V.BOS], *V.encode(text, stoi)]])
        positions = embedding.position_ids(ids, beta=0)[0]
        equals_index = text.index('=') + 1  # account for BOS
        answer_ids = ids[0, equals_index + 1:]
        answer_positions = positions[equals_index + 1:]
        digit_mask = torch.isin(
            answer_ids, torch.tensor([stoi[digit] for digit in V.DIGITS]))
        self.assertTrue(torch.equal(
            answer_positions[digit_mask],
            torch.zeros_like(answer_positions[digit_mask])))
        operand_positions = positions[1:equals_index]
        self.assertTrue(bool(operand_positions.gt(0).any()))


if __name__ == "__main__":
    unittest.main()
