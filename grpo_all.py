# type: ignore
"""Operation-weighted GRPO with replay and regression-gated checkpoints.

All rewards are exact matches against the calculator's canonical internal
representation: reversed magnitudes for +, -, and *, and fixed ``DDD.ddd`` (or
the atomic ``<nan>`` token) for division.  Addition/subtraction supervised
replay and baseline-relative checkpoint gates protect already-solved skills.
"""

import argparse
import collections
import dataclasses
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml

import vocab as V
from grpo import (active_completion_ids, completion_log_probs, grpo_loss,
                  group_advantages, load_state_dict, pick_device)
from model import build_model


@dataclasses.dataclass(frozen=True)
class ArithmeticExample:
    prompt_ids: tuple[int, ...]
    truth_ids: tuple[int, ...]
    text: str
    operation: str
    scenario: str
    tier: int
    stratum: tuple = ()


@dataclasses.dataclass
class OperationRollout:
    sequences: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    exact_rewards: torch.Tensor
    advantages: torch.Tensor
    prompt_length: int
    group_size: int
    operation: str

    @property
    def mixed_groups(self):
        grouped = self.rewards.view(-1, self.group_size)
        return int((grouped.max(dim=1).values
                    - grouped.min(dim=1).values > 1e-6).sum().item())

    @property
    def exact_mixed_groups(self):
        grouped = self.exact_rewards.view(-1, self.group_size)
        return int(((grouped.min(dim=1).values == 0)
                    & (grouped.max(dim=1).values == 1)).sum().item())


def _validate_operations(operations):
    operations = tuple(operations)
    unknown = set(operations) - set(V.OPERATORS)
    if unknown:
        raise ValueError(f"unsupported operations: {sorted(unknown)}")
    if not operations:
        raise ValueError("at least one operation is required")
    return operations


def load_arithmetic_examples(data_path, split, stoi, max_seq_len, operations,
                             expected_representation=None,
                             operand_digits=None, min_tier=None,
                             max_per_operation=0, seed=42):
    """Load canonical examples, optionally reservoir-sampling each operation."""
    operations = _validate_operations(operations)
    reservoirs = {operation: [] for operation in operations}
    seen = collections.Counter()
    rng = random.Random(seed)
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            operation = row.get("operation")
            if row.get("split") != split or operation not in reservoirs:
                continue
            if (operand_digits is not None
                    and row.get("operand_digits") != operand_digits):
                continue
            tier = int(row.get("tier", -1))
            if min_tier is not None and tier < min_tier:
                continue
            if (expected_representation is not None
                    and row.get("representation") != expected_representation):
                raise ValueError(
                    f"{data_path} uses representation "
                    f"{row.get('representation', 'missing')!r}, expected "
                    f"{expected_representation!r}")
            prompt, truth = row["text"].split("=", 1)
            prompt_ids = tuple(
                [stoi[V.BOS]] + V.encode(prompt + "=", stoi))
            truth_ids = tuple(V.encode(truth, stoi))
            # Replay includes EOS, so validate the complete trainable sequence.
            if len(prompt_ids) + len(truth_ids) + 1 > max_seq_len:
                raise ValueError(
                    f"row {row['text']!r} exceeds max_seq_len={max_seq_len}")
            example = ArithmeticExample(
                prompt_ids=prompt_ids,
                truth_ids=truth_ids,
                text=row["text"],
                operation=operation,
                scenario=row.get("scenario", "unclassified"),
                tier=tier,
                stratum=_example_stratum(row),
            )
            seen[operation] += 1
            reservoir = reservoirs[operation]
            if not max_per_operation or len(reservoir) < max_per_operation:
                reservoir.append(example)
            else:
                replacement = rng.randrange(seen[operation])
                if replacement < max_per_operation:
                    reservoir[replacement] = example

    missing = [operation for operation, values in reservoirs.items()
               if not values]
    if missing:
        raise ValueError(
            f"no examples for operations {missing} in split {split!r} of "
            f"{data_path}")
    return [example for operation in operations
            for example in reservoirs[operation]]


def load_arithmetic_sources(sources, stoi, max_seq_len, operations,
                            expected_representation=None, seed=42):
    """Load and deduplicate multiple GRPO sources with per-source filters."""
    combined = []
    seen = set()
    for index, source in enumerate(sources):
        examples = load_arithmetic_examples(
            source["data_path"], source.get("split", "train"),
            stoi, max_seq_len, operations, expected_representation,
            operand_digits=source.get("operand_digits"),
            min_tier=(int(source["min_tier"])
                      if "min_tier" in source else None),
            max_per_operation=int(source.get("max_per_operation", 0)),
            seed=seed + index)
        for example in examples:
            key = (example.operation, example.prompt_ids)
            if key not in seen:
                seen.add(key)
                combined.append(example)
    counts = collections.Counter(
        example.operation for example in combined)
    missing = [operation for operation in operations if not counts[operation]]
    if missing:
        raise ValueError(f"combined GRPO sources lack operations {missing}")
    return combined


def _example_stratum(row):
    """Micro-stratum used to guarantee broad coverage in every GRPO epoch."""
    operation = row["operation"]
    common = (row.get("operand_digits", "unknown"),
              row.get("scenario", "unclassified"))
    if operation in ("+", "-"):
        detail = (row.get("first_operand_negative", False),
                  row.get("operand_digit_pattern", "unknown"))
    elif operation == "*":
        detail = (row.get("multiplication_sign_pattern", "unknown"),)
    else:
        detail = (row.get("division_sign", "unknown"),
                  row.get("rounding_direction", "not_applicable"))
    return common + detail


def exact_rewards(completion_ids, completion_mask, truths, group_size, eos_id):
    rewards = []
    for index in range(completion_ids.shape[0]):
        sampled = active_completion_ids(
            completion_ids[index].tolist(),
            completion_mask[index].tolist(), eos_id)
        rewards.append(float(sampled == truths[index // group_size]))
    return torch.tensor(
        rewards, dtype=torch.float32, device=completion_ids.device)


_TOKEN_ID_TO_TEXT = dict(enumerate(V.TOKENS))


def _canonical_division_milli(token_ids):
    """Parse canonical ``DDD.ddd`` tokens into signed thousandths."""
    text = V.decode(list(token_ids), _TOKEN_ID_TO_TEXT)
    negative = text.startswith("-")
    magnitude = text[1:] if negative else text
    if (len(magnitude) != 7 or magnitude[3] != "."
            or not (magnitude[:3] + magnitude[4:]).isdigit()):
        return None
    value = int(magnitude[:3]) * 1000 + int(magnitude[4:])
    return -value if negative else value


def shaped_rewards(completion_ids, completion_mask, truths, group_size, eos_id,
                   partial_credit_weight, numeric_distance_scale=0.0):
    """Exact reward plus bounded canonical-token credit for wrong answers.

    Exact answers always receive 1.0.  A wrong answer receives at most
    ``partial_credit_weight``.  With a positive numeric scale, credit combines
    position-wise token agreement (50%), canonical fixed-width format (10%),
    and numerical closeness in thousandths (40%).  Otherwise it uses token
    agreement only. Atomic ``NAN`` targets remain exact-only.
    """
    weight = float(partial_credit_weight)
    if not 0.0 <= weight < 1.0:
        raise ValueError("partial_credit_weight must be in [0, 1)")
    numeric_scale = float(numeric_distance_scale)
    if numeric_scale < 0.0:
        raise ValueError("numeric_distance_scale must be non-negative")
    rewards = []
    for index in range(completion_ids.shape[0]):
        sampled = active_completion_ids(
            completion_ids[index].tolist(),
            completion_mask[index].tolist(), eos_id)
        truth = truths[index // group_size]
        if sampled == truth:
            rewards.append(1.0)
            continue
        width = max(len(sampled), len(truth), 1)
        matches = sum(left == right for left, right in zip(sampled, truth))
        token_score = matches / width
        score = token_score
        if numeric_scale > 0.0:
            truth_value = _canonical_division_milli(truth)
            sampled_value = _canonical_division_milli(sampled)
            # NAN and other non-numeric targets receive no partial credit.
            if truth_value is None:
                score = 0.0
            else:
                format_score = float(sampled_value is not None)
                numeric_score = 0.0
                if sampled_value is not None:
                    distance = abs(sampled_value - truth_value)
                    numeric_score = math.exp(-distance / numeric_scale)
                score = (
                    0.50 * token_score
                    + 0.10 * format_score
                    + 0.40 * numeric_score)
        rewards.append(weight * score)
    return torch.tensor(
        rewards, dtype=torch.float32, device=completion_ids.device)


@torch.no_grad()
def rollout_operation(model, examples, group_size, stoi, max_seq_len, device,
                      temperature=1.0, sample=True,
                      partial_credit_weight=0.0,
                      numeric_distance_scale=0.0):
    """Generate one same-operation, same-prompt-length rollout batch."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    operation = examples[0].operation
    prompt_length = len(examples[0].prompt_ids)
    if any(example.operation != operation for example in examples):
        raise ValueError("a rollout batch cannot mix operations")
    if any(len(example.prompt_ids) != prompt_length for example in examples):
        raise ValueError("rollout prompts must have equal encoded lengths")

    prompts = torch.tensor(
        [example.prompt_ids for example in examples],
        dtype=torch.long, device=device)
    tokens = prompts.repeat_interleave(group_size, dim=0)
    sample_count = tokens.shape[0]
    finished = torch.zeros(sample_count, dtype=torch.bool, device=device)
    numeric_count = torch.zeros(
        sample_count, dtype=torch.long, device=device)
    digit_ids = torch.tensor(
        [stoi[digit] for digit in V.DIGITS], device=device)
    decimal_id = stoi["."]
    eos_id = stoi[V.EOS]
    nan_id = stoi[V.NAN]
    pad_id = stoi[V.PAD]
    sampled_tokens = []
    sampled_masks = []
    sampled_log_probs = []

    model.eval()
    while tokens.shape[1] < max_seq_len and not bool(finished.all()):
        seq_len = tokens.shape[1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device,
                       dtype=torch.bool)).unsqueeze(0)
        logits = model(tokens, causal_mask)[:, -1, :] / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        if sample:
            next_tokens = torch.multinomial(
                torch.softmax(logits, dim=-1), 1).squeeze(1)
        else:
            next_tokens = logits.argmax(dim=-1)

        active = ~finished
        next_log_probs = log_probs.gather(
            1, next_tokens[:, None]).squeeze(1)
        next_tokens = torch.where(
            active, next_tokens, torch.full_like(next_tokens, pad_id))
        next_log_probs = torch.where(
            active, next_log_probs, torch.zeros_like(next_log_probs))
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=1)
        sampled_tokens.append(next_tokens)
        sampled_masks.append(active)
        sampled_log_probs.append(next_log_probs)

        is_numeric = (
            torch.isin(next_tokens, digit_ids)
            | next_tokens.eq(decimal_id)) & active
        numeric_count += is_numeric.long()
        newly_finished = next_tokens.eq(eos_id) | next_tokens.eq(nan_id)
        if operation == "/":
            newly_finished |= numeric_count.ge(7)
        finished |= active & newly_finished

    completion_ids = torch.stack(sampled_tokens, dim=1)
    completion_mask = torch.stack(sampled_masks, dim=1)
    old_log_probs = torch.stack(sampled_log_probs, dim=1)
    truths = [example.truth_ids for example in examples]
    exact = exact_rewards(
        completion_ids, completion_mask, truths, group_size, eos_id)
    rewards = shaped_rewards(
        completion_ids, completion_mask, truths, group_size, eos_id,
        partial_credit_weight, numeric_distance_scale)
    return OperationRollout(
        sequences=tokens,
        completion_mask=completion_mask,
        old_log_probs=old_log_probs,
        rewards=rewards,
        exact_rewards=exact,
        advantages=group_advantages(rewards, group_size),
        prompt_length=prompt_length,
        group_size=group_size,
        operation=operation,
    )


def allocate_weighted_counts(total, weights):
    """Largest-remainder allocation preserving an exact integer total."""
    if total < 1:
        raise ValueError("total must be >= 1")
    weights = {key: float(value) for key, value in weights.items()}
    if not weights or any(value < 0 for value in weights.values()):
        raise ValueError("operation weights must be non-negative and non-empty")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one operation weight must be positive")
    raw = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(
        weights, key=lambda key: (raw[key] - result[key], weights[key]),
        reverse=True)
    for key in order[:remainder]:
        result[key] += 1
    return result


def _priority_weight(example, priority):
    operation_config = (priority or {}).get(example.operation, {})
    weight = float(operation_config.get("default", 1.0))
    weight *= float(operation_config.get(
        "scenario_weights", {}).get(example.scenario, 1.0))
    if example.operation == "*" and len(example.stratum) >= 3:
        weight *= float(operation_config.get(
            "sign_pattern_weights", {}).get(example.stratum[2], 1.0))
    if weight <= 0:
        raise ValueError("priority weights must be positive")
    return weight


def weighted_prompt_batches(examples, total, weights, batch_size, rng,
                            priority=None):
    """Sample the operation mix while covering every observed micro-stratum."""
    by_operation = collections.defaultdict(list)
    for example in examples:
        by_operation[example.operation].append(example)
    counts = allocate_weighted_counts(total, weights)
    batches = []
    for operation, count in counts.items():
        if count == 0:
            continue
        pool = by_operation.get(operation, [])
        if not pool:
            raise ValueError(f"no GRPO examples for operation {operation!r}")
        strata = collections.defaultdict(list)
        for example in pool:
            strata[example.stratum].append(example)
        if count < len(strata):
            raise ValueError(
                f"operation {operation!r} receives {count} prompts but has "
                f"{len(strata)} strata; increase prompts_per_epoch")
        # Seed the epoch with one row from every length/scenario/sign stratum,
        # then fill the remaining quota from unused rows to approximately retain
        # the configured distribution instead of flattening it completely.
        selected = [rng.choice(bucket) for bucket in strata.values()]
        selected_ids = {id(example) for example in selected}
        remaining = count - len(selected)
        unused = [example for example in pool if id(example) not in selected_ids]
        if remaining <= len(unused):
            # Efraimidis-Spirakis weighted sampling without replacement. Equal
            # weights reduce to ordinary random sampling; larger priority
            # weights make a row more likely without duplicating it in an epoch.
            ranked = sorted(
                unused,
                key=lambda example: rng.random() ** (
                    1.0 / _priority_weight(example, priority)),
                reverse=True)
            selected.extend(ranked[:remaining])
        else:
            selected.extend(unused)
            selected.extend(rng.choices(
                pool,
                weights=[_priority_weight(example, priority)
                         for example in pool],
                k=remaining - len(unused)))
        buckets = collections.defaultdict(list)
        for example in selected:
            buckets[len(example.prompt_ids)].append(example)
        for bucket in buckets.values():
            rng.shuffle(bucket)
            batches.extend(
                bucket[start:start + batch_size]
                for start in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches, counts


def sample_balanced_replay(pools, operations, batch_size, rng):
    counts = allocate_weighted_counts(
        batch_size, {operation: 1.0 for operation in operations})
    examples = []
    for operation, count in counts.items():
        pool = pools[operation]
        examples.extend(rng.choices(pool, k=count))
    rng.shuffle(examples)
    return examples


def _replay_batch_tensors(examples, stoi, device):
    eos_id = stoi[V.EOS]
    pad_id = stoi[V.PAD]
    rows = [
        list(example.prompt_ids) + list(example.truth_ids) + [eos_id]
        for example in examples]
    max_length = max(len(row) for row in rows)
    padded = torch.tensor(
        [row + [pad_id] * (max_length - len(row)) for row in rows],
        dtype=torch.long, device=device)
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    answer_mask = torch.zeros_like(targets, dtype=torch.bool)
    for index, (example, row) in enumerate(zip(examples, rows)):
        start = len(example.prompt_ids) - 1
        answer_mask[index, start:len(row) - 1] = True
    seq_len = inputs.shape[1]
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=device,
                   dtype=torch.bool)).unsqueeze(0)
    return inputs, targets, answer_mask, causal_mask


def supervised_replay_loss(model, examples, stoi, device):
    """Answer-only teacher-forced loss, including the terminal EOS token."""
    inputs, targets, answer_mask, causal_mask = _replay_batch_tensors(
        examples, stoi, device)
    model.eval()
    log_probs = torch.log_softmax(model(inputs, causal_mask), dim=-1)
    token_nll = -log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)
    lengths = answer_mask.sum(dim=1).clamp_min(1)
    return ((token_nll * answer_mask).sum(dim=1) / lengths).mean()


def replay_anchor_losses(model, reference, examples, stoi, device):
    """Return hard-label replay CE and forward KL from the initial policy."""
    inputs, targets, answer_mask, causal_mask = _replay_batch_tensors(
        examples, stoi, device)
    model.eval()
    policy_log_probs = torch.log_softmax(
        model(inputs, causal_mask), dim=-1)
    with torch.no_grad():
        reference_log_probs = torch.log_softmax(
            reference(inputs, causal_mask), dim=-1)
        reference_probabilities = reference_log_probs.exp()
    token_nll = -policy_log_probs.gather(
        2, targets.unsqueeze(2)).squeeze(2)
    token_kl = (
        reference_probabilities
        * (reference_log_probs - policy_log_probs)).sum(dim=-1)
    lengths = answer_mask.sum(dim=1).clamp_min(1)
    supervised = ((token_nll * answer_mask).sum(dim=1) / lengths).mean()
    distillation = ((token_kl * answer_mask).sum(dim=1) / lengths).mean()
    return supervised, distillation


def stratified_limit(examples, limit, operations, seed):
    if not limit or limit >= len(examples):
        return list(examples)
    counts = allocate_weighted_counts(
        limit, {operation: 1.0 for operation in operations})
    rng = random.Random(seed)
    by_operation = collections.defaultdict(list)
    for example in examples:
        by_operation[example.operation].append(example)
    selected = []
    for operation, count in counts.items():
        if count > len(by_operation[operation]):
            raise ValueError(
                f"eval limit requests {count} {operation!r} rows, only "
                f"{len(by_operation[operation])} are available")
        selected.extend(rng.sample(by_operation[operation], count))
    return selected


def evaluate_operations(model, examples, stoi, max_seq_len, device,
                        batch_size=128):
    totals = collections.Counter()
    hits = collections.Counter()
    scenario_totals = collections.Counter()
    scenario_hits = collections.Counter()
    buckets = collections.defaultdict(list)
    for example in examples:
        buckets[(example.operation, len(example.prompt_ids))].append(example)
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start:start + batch_size]
            rollout = rollout_operation(
                model, batch, 1, stoi, max_seq_len, device,
                temperature=1.0, sample=False)
            batch_hits = rollout.exact_rewards.detach().cpu().tolist()
            for example, hit in zip(batch, batch_hits):
                totals[example.operation] += 1
                hits[example.operation] += int(hit)
                key = f"{example.operation}|{example.scenario}"
                scenario_totals[key] += 1
                scenario_hits[key] += int(hit)
    total = sum(totals.values())
    correct = sum(hits.values())
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "operation": {
            operation: hits[operation] / totals[operation]
            for operation in V.OPERATORS if totals[operation]},
        "operation_counts": {
            operation: {"correct": hits[operation],
                        "total": totals[operation]}
            for operation in V.OPERATORS if totals[operation]},
        "scenario": {
            key: scenario_hits[key] / scenario_totals[key]
            for key in sorted(scenario_totals)},
    }


def print_evaluation(label, metrics):
    print(
        f"{label}: {metrics['correct']}/{metrics['total']} "
        f"= {100 * metrics['accuracy']:.2f}%")
    for operation in V.OPERATORS:
        if operation in metrics["operation"]:
            print(
                f"  {operation:<2} {100 * metrics['operation'][operation]:6.2f}%")


def checkpoint_gate(metrics, baseline, gate_config):
    """Return target score and eligibility under regression constraints."""
    protected = gate_config.get(
        "protected_operations", {"+": 0.002, "-": 0.002})
    violations = []
    for operation, tolerance in protected.items():
        floor = baseline["operation"][operation] - float(tolerance)
        actual = metrics["operation"][operation]
        if actual < floor:
            violations.append(
                f"{operation}={actual:.4f} below floor {floor:.4f}")
    overall_tolerance = float(gate_config.get("overall_regression_tolerance", 0.001))
    overall_floor = baseline["accuracy"] - overall_tolerance
    if metrics["accuracy"] < overall_floor:
        violations.append(
            f"overall={metrics['accuracy']:.4f} below floor {overall_floor:.4f}")
    target_weights = gate_config.get(
        "target_weights", {"*": 0.55, "/": 0.45})
    denominator = sum(float(value) for value in target_weights.values())
    score = sum(
        metrics["operation"][operation] * float(weight)
        for operation, weight in target_weights.items()) / denominator
    return {
        "eligible": not violations,
        "target_score": score,
        "violations": violations,
    }


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gc = cfg["grpo"]
    rc = cfg["replay"]
    ec = cfg["eval"]
    device = pick_device(args.device)
    seed = int(gc.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    operations = _validate_operations(gc["operation_weights"].keys())
    operation_weights = {
        operation: float(weight)
        for operation, weight in gc["operation_weights"].items()}
    group_sizes = {
        operation: int(gc.get("group_sizes", {}).get(
            operation, gc.get("group_size", 8)))
        for operation in operations}
    if any(size < 2 for size in group_sizes.values()):
        raise ValueError("all GRPO group sizes must be >= 2")
    replay_operations = _validate_operations(rc["operations"])
    checkpoint_path = args.checkpoint or gc["checkpoint_path"]
    output_path = Path(args.output or gc["output_path"])
    epochs = args.epochs if args.epochs is not None else int(gc["epochs"])
    prompts_per_epoch = (
        args.prompts_per_epoch if args.prompts_per_epoch is not None
        else int(gc["prompts_per_epoch"]))
    model_cfg = cfg["model"]
    max_seq_len = int(model_cfg["max_seq_len"])
    stoi, _ = V.build_vocab()
    expected_representation = (
        "abacus-v1" if model_cfg.get("use_abacus", False) else None)

    if "data_sources" in gc:
        train_examples = load_arithmetic_sources(
            gc["data_sources"], stoi, max_seq_len, operations,
            expected_representation, seed=seed)
    else:
        train_examples = load_arithmetic_examples(
            gc["data_path"], gc.get("split", "train"), stoi, max_seq_len,
            operations, expected_representation,
            operand_digits=gc.get("operand_digits"),
            min_tier=(int(gc["min_tier"]) if "min_tier" in gc else None),
            seed=seed)
    replay_examples = load_arithmetic_examples(
        rc["data_path"], rc.get("split", "train"), stoi, max_seq_len,
        replay_operations, expected_representation,
        max_per_operation=int(rc.get("pool_per_operation", 10000)),
        seed=seed + 1)
    eval_examples = load_arithmetic_examples(
        ec["data_path"], ec.get("split", "test"), stoi, max_seq_len,
        operations, expected_representation, seed=seed + 2)
    eval_limit = (
        args.eval_limit if args.eval_limit is not None
        else int(ec.get("limit", 0)))
    eval_examples = stratified_limit(
        eval_examples, eval_limit, operations, seed + 3)
    eval_prompts = {
        (example.operation, example.prompt_ids) for example in eval_examples}
    grpo_overlap = eval_prompts & {
        (example.operation, example.prompt_ids) for example in train_examples}
    replay_overlap = eval_prompts & {
        (example.operation, example.prompt_ids) for example in replay_examples}
    if grpo_overlap or replay_overlap:
        raise ValueError(
            "training/evaluation overlap: "
            f"GRPO={len(grpo_overlap)}, replay={len(replay_overlap)}")

    replay_pools = collections.defaultdict(list)
    for example in replay_examples:
        replay_pools[example.operation].append(example)
    policy = build_model(model_cfg).to(device)
    policy.load_state_dict(load_state_dict(checkpoint_path, device))
    reference = build_model(model_cfg).to(device)
    reference.load_state_dict(load_state_dict(checkpoint_path, device))
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy.eval()

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print("GRPO prompt pool:", dict(collections.Counter(
        example.operation for example in train_examples)))
    print("replay pool:", dict(collections.Counter(
        example.operation for example in replay_examples)))
    print("operation weights:", operation_weights)
    print("group sizes:", group_sizes)
    baseline = evaluate_operations(
        policy, eval_examples, stoi, max_seq_len, device,
        batch_size=int(ec.get("batch_size", 128)))
    print_evaluation("baseline greedy evaluation", baseline)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(gc["lr"]),
        weight_decay=float(gc.get("weight_decay", 0.0)))
    prompt_batch_size = int(gc.get("prompt_batch_size", 4))
    temperature = float(gc.get("temperature", 1.0))
    partial_credit_weights = {
        operation: float(gc.get("partial_credit_weights", {}).get(
            operation, 0.0))
        for operation in operations}
    numeric_distance_scales = {
        operation: float(gc.get("numeric_distance_scales", {}).get(
            operation, 0.0))
        for operation in operations}
    clip_epsilon = float(gc.get("clip_epsilon", 0.2))
    beta = float(gc.get("beta", 0.01))
    max_grad_norm = float(gc.get("max_grad_norm", 1.0))
    optimization_epochs = int(gc.get("optimization_epochs", 1))
    replay_batch_size = int(rc.get("batch_size", 16))
    replay_supervised_weight = float(rc.get("supervised_weight", 0.2))
    replay_distillation_weight = float(rc.get("distillation_weight", 1.0))
    eval_every = int(ec.get("every_epochs", 1))
    log_every = int(gc.get("log_every", 10))
    if (prompt_batch_size < 1 or optimization_epochs < 1
            or replay_batch_size < 1 or eval_every < 1):
        raise ValueError("batch sizes, optimization epochs, and eval interval must be >= 1")
    if (not 0.0 < clip_epsilon < 1.0 or beta < 0
            or replay_supervised_weight < 0 or replay_distillation_weight < 0):
        raise ValueError("invalid clip/KL/replay loss configuration")
    if any(not 0.0 <= weight < 1.0
           for weight in partial_credit_weights.values()):
        raise ValueError("partial-credit weights must be in [0, 1)")
    if any(scale < 0.0 for scale in numeric_distance_scales.values()):
        raise ValueError("numeric-distance scales must be non-negative")

    gate_config = ec.get("checkpoint_gate", {})
    baseline_gate = checkpoint_gate(baseline, baseline, gate_config)
    best_score = baseline_gate["target_score"]
    best_state = _cpu_state_dict(policy)
    best_metrics = baseline
    best_epoch = 0
    rng = random.Random(seed)
    history = []
    evaluations = [{"epoch": 0, "metrics": baseline, "gate": baseline_gate}]
    update = 0
    start_time = time.monotonic()
    stop = False

    for epoch in range(1, epochs + 1):
        batches, allocated = weighted_prompt_batches(
            train_examples, prompts_per_epoch, operation_weights,
            prompt_batch_size, rng, priority=gc.get("priority"))
        operation_samples = collections.Counter()
        operation_hits = collections.Counter()
        operation_reward_sum = collections.Counter()
        operation_groups = collections.Counter()
        operation_mixed = collections.Counter()
        operation_exact_mixed = collections.Counter()
        sampled_strata = collections.defaultdict(set)
        for batch in batches:
            if args.max_updates and update >= args.max_updates:
                stop = True
                break
            operation = batch[0].operation
            sampled_strata[operation].update(
                example.stratum for example in batch)
            rollout = rollout_operation(
                policy, batch, group_sizes[operation], stoi, max_seq_len,
                device, temperature=temperature, sample=True,
                partial_credit_weight=partial_credit_weights[operation],
                numeric_distance_scale=numeric_distance_scales[operation])
            completion_steps = rollout.completion_mask.shape[1]
            with torch.no_grad():
                ref_log_probs = completion_log_probs(
                    reference, rollout.sequences, rollout.prompt_length,
                    completion_steps, temperature)
            replay_batch = sample_balanced_replay(
                replay_pools, replay_operations, replay_batch_size, rng)
            last_metrics = None
            replay_supervised_value = 0.0
            replay_distillation_value = 0.0
            grad_norm = 0.0
            for _ in range(optimization_epochs):
                new_log_probs = completion_log_probs(
                    policy, rollout.sequences, rollout.prompt_length,
                    completion_steps, temperature)
                policy_loss, last_metrics = grpo_loss(
                    new_log_probs, rollout.old_log_probs, ref_log_probs,
                    rollout.completion_mask, rollout.advantages,
                    clip_epsilon, beta)
                replay_supervised, replay_distillation = replay_anchor_losses(
                    policy, reference, replay_batch, stoi, device)
                total_loss = (
                    policy_loss
                    + replay_supervised_weight * replay_supervised
                    + replay_distillation_weight * replay_distillation)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_grad_norm).item())
                optimizer.step()
                policy.eval()
                replay_supervised_value = float(
                    replay_supervised.detach().item())
                replay_distillation_value = float(
                    replay_distillation.detach().item())

            update += 1
            samples = rollout.rewards.numel()
            operation_samples[operation] += samples
            operation_hits[operation] += float(
                rollout.exact_rewards.sum().item())
            operation_reward_sum[operation] += float(
                rollout.rewards.sum().item())
            operation_groups[operation] += len(batch)
            operation_mixed[operation] += rollout.mixed_groups
            operation_exact_mixed[operation] += rollout.exact_mixed_groups
            record = {
                "epoch": epoch,
                "update": update,
                "operation": operation,
                "sample_exact_rate": float(
                    rollout.exact_rewards.mean().item()),
                "sample_reward_mean": float(rollout.rewards.mean().item()),
                "mixed_groups": rollout.mixed_groups,
                "exact_mixed_groups": rollout.exact_mixed_groups,
                "groups": len(batch),
                "replay_supervised_loss": replay_supervised_value,
                "replay_distillation_kl": replay_distillation_value,
                "total_loss": (
                    last_metrics["loss"]
                    + replay_supervised_weight * replay_supervised_value
                    + replay_distillation_weight * replay_distillation_value),
                "grad_norm": grad_norm,
                **last_metrics,
            }
            history.append(record)
            if update == 1 or update % log_every == 0:
                print(
                    f"epoch {epoch}/{epochs} update {update:>4} op={operation}  "
                    f"exact={record['sample_exact_rate']:.3f}  "
                    f"reward={record['sample_reward_mean']:.3f}  "
                    f"signal={rollout.mixed_groups}/{len(batch)}  "
                    f"exact_mixed={rollout.exact_mixed_groups}/{len(batch)}  "
                    f"replay_kl={replay_distillation_value:.5f}  "
                    f"ref_kl={last_metrics['reference_kl']:.5f}  "
                    f"grad={grad_norm:.3f}")

        print(f"epoch {epoch} allocated prompts: {allocated}")
        available_strata = collections.defaultdict(set)
        for example in train_examples:
            available_strata[example.operation].add(example.stratum)
        for operation in operations:
            if operation_samples[operation]:
                print(
                    f"  {operation} rollout exact="
                    f"{operation_hits[operation]/operation_samples[operation]:.3f}  "
                    f"reward="
                    f"{operation_reward_sum[operation]/operation_samples[operation]:.3f}  "
                    f"signal={operation_mixed[operation]}/"
                    f"{operation_groups[operation]}  strata="
                    f"{len(sampled_strata[operation])}/"
                    f"{len(available_strata[operation])}")
                if partial_credit_weights[operation] > 0:
                    print(
                        f"    exact-mixed={operation_exact_mixed[operation]}/"
                        f"{operation_groups[operation]}  partial-credit="
                        f"{partial_credit_weights[operation]:.3f}  "
                        f"numeric-scale="
                        f"{numeric_distance_scales[operation]:.1f}")
        if epoch % eval_every == 0 or stop or epoch == epochs:
            metrics = evaluate_operations(
                policy, eval_examples, stoi, max_seq_len, device,
                batch_size=int(ec.get("batch_size", 128)))
            gate = checkpoint_gate(metrics, baseline, gate_config)
            evaluations.append({"epoch": epoch, "metrics": metrics, "gate": gate})
            print_evaluation(f"epoch {epoch} greedy evaluation", metrics)
            print(
                f"  checkpoint eligible={gate['eligible']}  "
                f"target_score={gate['target_score']:.4f}  "
                f"violations={gate['violations'] or 'none'}")
            if gate["eligible"] and gate["target_score"] > best_score:
                best_score = gate["target_score"]
                best_state = _cpu_state_dict(policy)
                best_metrics = metrics
                best_epoch = epoch
                print(f"  new best eligible checkpoint at epoch {epoch}")
        if stop:
            break

    final_metrics = evaluations[-1]["metrics"]
    last_path = output_path.with_name(
        f"{output_path.stem}.last{output_path.suffix}")
    last_state = _cpu_state_dict(policy)
    last_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(last_state, last_path)
    policy.load_state_dict(best_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    report_path = output_path.with_suffix(".metrics.json")
    report = {
        "config": args.config,
        "initial_checkpoint": checkpoint_path,
        "output_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "device": str(device),
        "seed": seed,
        "updates": update,
        "elapsed_seconds": time.monotonic() - start_time,
        "baseline": baseline,
        "final_trained": final_metrics,
        "selected": best_metrics,
        "selected_epoch": best_epoch,
        "history": history,
        "evaluations": evaluations,
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print_evaluation(f"selected epoch {best_epoch}", best_metrics)
    print(f"saved checkpoint -> {output_path}")
    print(f"saved last trained checkpoint -> {last_path}")
    print(f"saved metrics -> {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/grpo_all_operations_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prompts-per-epoch", type=int, default=None)
    parser.add_argument(
        "--max-updates", type=int, default=0,
        help="stop after this many optimizer updates; 0 uses the full run")
    parser.add_argument(
        "--eval-limit", type=int, default=None,
        help="stratified evaluation subset size; default comes from config")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
