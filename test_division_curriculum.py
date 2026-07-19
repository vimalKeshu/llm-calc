import dataclasses
import random
import tempfile
import unittest
from pathlib import Path

import vocab as V
from division_curriculum import (filter_stage_examples,
                                 sample_stratified_examples,
                                 validate_output_paths)
from grpo_all import ArithmeticExample


class DivisionCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.stoi, _ = V.build_vocab()

    def example(self, cell, scenario, detail="positive"):
        return ArithmeticExample(
            prompt_ids=tuple(
                [self.stoi[V.BOS]] + V.encode("12/3=", self.stoi)),
            truth_ids=tuple(V.encode("004.000", self.stoi)),
            text="12/3=004.000",
            operation="/",
            scenario=scenario,
            tier=2,
            stratum=(cell, scenario, detail),
        )

    def test_stage_filter_selects_requested_operand_cells(self):
        examples = [
            self.example("1x1", "exact"),
            self.example("2x2", "rounded"),
            self.example("3x3", "rounded"),
        ]
        selected = filter_stage_examples(examples, {
            "name": "fundamentals",
            "include_operand_cells": ["1x1", "2x2"],
        })
        self.assertEqual(
            {example.stratum[0] for example in selected}, {"1x1", "2x2"})

    def test_stratified_sample_covers_all_micro_strata(self):
        examples = [
            self.example("3x3", "rounded", "positive"),
            self.example("3x3", "rounded", "negative"),
            self.example("3x3", "exact", "positive"),
        ]
        sampled = sample_stratified_examples(
            examples, 6, random.Random(42))
        self.assertEqual(
            {example.stratum for example in sampled},
            {example.stratum for example in examples})

    def test_output_validation_never_allows_input_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.touch()
            with self.assertRaises(ValueError):
                validate_output_paths(checkpoint, checkpoint)

    def test_output_validation_refuses_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source.pt"
            output = Path(directory) / "curriculum.pt"
            checkpoint.touch()
            output.touch()
            with self.assertRaises(FileExistsError):
                validate_output_paths(checkpoint, output)


if __name__ == "__main__":
    unittest.main()
