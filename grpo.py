# type: ignore
"""Division-only Group Relative Policy Optimization for LLMCalc.

The trainer deliberately uses the calculator's canonical internal division
answers as the verifier: numeric quotients must exactly match ``DDD.ddd`` and
division by zero must emit the atomic ``<nan>`` token.  This keeps the reward
fully deterministic and avoids teaching to a permissive string parser.
"""

import argparse
import collections
import dataclasses
import json
import random
import time
from pathlib import Path

import torch
import yaml

import vocab as V
from model import build_model


@dataclasses.dataclass(frozen=True)
class DivisionExample:
    prompt_ids: tuple[int, ...]
    truth_ids: tuple[int, ...]
    text: str
    scenario: str
    sign: str


@dataclasses.dataclass
class Rollout:
    sequences: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    prompt_length: int
    group_size: int

    @property
    def mixed_groups(self) -> int:
        grouped = self.rewards.view(-1, self.group_size)
        return int(((grouped.min(dim=1).values == 0)
                    & (grouped.max(dim=1).values == 1)).sum().item())


def pick_device(name="auto"):
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def load_division_examples(data_path, split, stoi, max_seq_len,
                           expected_representation=None,
                           operand_digits=None, min_tier=None):
    """Load only division rows and retain their exact internal token targets."""
    examples = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != split or row.get("operation") != "/":
                continue
            if (operand_digits is not None
                    and row.get("operand_digits") != operand_digits):
                continue
            if min_tier is not None and int(row.get("tier", -1)) < min_tier:
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
            if len(prompt_ids) + len(truth_ids) > max_seq_len:
                raise ValueError(
                    f"division row {row['text']!r} needs more than "
                    f"max_seq_len={max_seq_len} tokens")
            examples.append(DivisionExample(
                prompt_ids=prompt_ids,
                truth_ids=truth_ids,
                text=row["text"],
                scenario=row.get("scenario", "unclassified"),
                sign=row.get(
                    "division_sign",
                    "negative" if prompt.startswith("-") else "positive"),
            ))
    if not examples:
        raise ValueError(
            f"no division examples in split {split!r} of {data_path}")
    return examples


def group_advantages(rewards, group_size, eps=1e-4):
    """Standard GRPO reward normalization within each prompt's sample group."""
    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return (centered / (std + eps)).reshape(-1)


def active_completion_ids(token_ids, mask, eos_id):
    """Return sampled completion IDs, excluding a terminal EOS only."""
    result = []
    for token_id, active in zip(token_ids, mask):
        if not bool(active):
            continue
        value = int(token_id)
        if value == eos_id:
            break
        result.append(value)
    return tuple(result)


def exact_division_rewards(completion_ids, completion_mask, truths,
                           group_size, eos_id):
    rewards = []
    for index in range(completion_ids.shape[0]):
        sampled = active_completion_ids(
            completion_ids[index].tolist(),
            completion_mask[index].tolist(), eos_id)
        rewards.append(float(sampled == truths[index // group_size]))
    return torch.tensor(
        rewards, dtype=torch.float32, device=completion_ids.device)


@torch.no_grad()
def rollout_division(model, examples, group_size, stoi, max_seq_len, device,
                     temperature=1.0, sample=True):
    """Sample equally sized GRPO groups for same-length division prompts.

    Generation stops after the seven numeric/decimal slots in ``DDD.ddd``.
    This mirrors the repository decoder and prevents a malformed eighth digit
    from requesting an unsupported fractional Abacus position.
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    prompt_length = len(examples[0].prompt_ids)
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

    model.eval()  # beta=0 Abacus positions and no dropout during RL rollouts.
    while tokens.shape[1] < max_seq_len and not bool(finished.all()):
        seq_len = tokens.shape[1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device,
                       dtype=torch.bool)).unsqueeze(0)
        logits = model(tokens, causal_mask)[:, -1, :] / temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        if sample:
            probabilities = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probabilities, 1).squeeze(1)
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
        finished |= active & (
            next_tokens.eq(eos_id)
            | next_tokens.eq(nan_id)
            | numeric_count.ge(7))

    completion_ids = torch.stack(sampled_tokens, dim=1)
    completion_mask = torch.stack(sampled_masks, dim=1)
    old_log_probs = torch.stack(sampled_log_probs, dim=1)
    truths = [example.truth_ids for example in examples]
    rewards = exact_division_rewards(
        completion_ids, completion_mask, truths, group_size, eos_id)
    advantages = group_advantages(rewards, group_size)
    return Rollout(
        sequences=tokens,
        completion_mask=completion_mask,
        old_log_probs=old_log_probs,
        rewards=rewards,
        advantages=advantages,
        prompt_length=prompt_length,
        group_size=group_size,
    )


def completion_log_probs(model, sequences, prompt_length, completion_steps,
                         temperature=1.0):
    """Score sampled completion tokens with gradients under ``model``."""
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    seq_len = inputs.shape[1]
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=sequences.device,
                   dtype=torch.bool)).unsqueeze(0)
    logits = model(inputs, causal_mask) / temperature
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(
        2, targets.unsqueeze(2)).squeeze(2)
    start = prompt_length - 1
    return token_log_probs[:, start:start + completion_steps]


def grpo_loss(new_log_probs, old_log_probs, ref_log_probs, completion_mask,
              advantages, clip_epsilon, beta):
    """Clipped GRPO objective plus the positive reference-KL estimator."""
    mask = completion_mask.to(new_log_probs.dtype)
    lengths = mask.sum(dim=1).clamp_min(1.0)
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    advantage_tokens = advantages[:, None]
    unclipped = ratio * advantage_tokens
    clipped = ratio.clamp(
        1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_tokens
    policy_tokens = -torch.minimum(unclipped, clipped)

    ref_delta = ref_log_probs - new_log_probs
    kl_tokens = torch.exp(ref_delta.clamp(-20.0, 20.0)) - ref_delta - 1.0
    per_sequence = (
        ((policy_tokens + beta * kl_tokens) * mask).sum(dim=1) / lengths)
    loss = per_sequence.mean()

    with torch.no_grad():
        ppo_kl = ((torch.exp(log_ratio.clamp(-20.0, 20.0))
                   - 1.0 - log_ratio) * mask).sum() / mask.sum().clamp_min(1.0)
        ref_kl = (kl_tokens * mask).sum() / mask.sum().clamp_min(1.0)
        clipped_fraction = (
            ((ratio - 1.0).abs() > clip_epsilon).to(mask.dtype)
            * mask).sum() / mask.sum().clamp_min(1.0)
    metrics = {
        "loss": float(loss.detach().item()),
        "ppo_kl": float(ppo_kl.item()),
        "reference_kl": float(ref_kl.item()),
        "clip_fraction": float(clipped_fraction.item()),
    }
    return loss, metrics


def prompt_batches(examples, limit, batch_size, rng, shuffle=True):
    if limit and limit < len(examples):
        selected = rng.sample(examples, limit)
    else:
        selected = list(examples)
    buckets = collections.defaultdict(list)
    for example in selected:
        buckets[len(example.prompt_ids)].append(example)
    batches = []
    for bucket in buckets.values():
        if shuffle:
            rng.shuffle(bucket)
        batches.extend(
            bucket[start:start + batch_size]
            for start in range(0, len(bucket), batch_size))
    if shuffle:
        rng.shuffle(batches)
    return batches


def evaluate_division(model, examples, stoi, max_seq_len, device,
                      batch_size=128):
    """Greedy exact internal-answer accuracy, safely stopped at DDD.ddd."""
    rewards = []
    scenario_counts = collections.Counter()
    scenario_hits = collections.Counter()
    sign_counts = collections.Counter()
    sign_hits = collections.Counter()
    buckets = collections.defaultdict(list)
    for example in examples:
        buckets[len(example.prompt_ids)].append(example)
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            batch = bucket[start:start + batch_size]
            rollout = rollout_division(
                model, batch, 1, stoi, max_seq_len, device,
                temperature=1.0, sample=False)
            hits = rollout.rewards.detach().cpu().tolist()
            rewards.extend(hits)
            for example, hit in zip(batch, hits):
                scenario_counts[example.scenario] += 1
                scenario_hits[example.scenario] += int(hit)
                sign_counts[example.sign] += 1
                sign_hits[example.sign] += int(hit)
    return {
        "accuracy": sum(rewards) / len(rewards),
        "correct": int(sum(rewards)),
        "total": len(rewards),
        "sign": {
            name: sign_hits[name] / count
            for name, count in sorted(sign_counts.items())},
        "scenario": {
            name: scenario_hits[name] / count
            for name, count in sorted(scenario_counts.items())},
    }


def print_evaluation(label, metrics):
    print(
        f"{label}: {metrics['correct']}/{metrics['total']} "
        f"= {100 * metrics['accuracy']:.2f}%")
    for sign, accuracy in metrics["sign"].items():
        print(f"  {sign:<8} {100 * accuracy:6.2f}%")


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gc = cfg["grpo"]
    ec = cfg.get("eval", {})
    device = pick_device(args.device)
    seed = int(gc.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    checkpoint_path = args.checkpoint or gc["checkpoint_path"]
    output_path = Path(args.output or gc["output_path"])
    epochs = args.epochs if args.epochs is not None else int(gc["epochs"])
    prompts_per_epoch = (
        args.prompts_per_epoch if args.prompts_per_epoch is not None
        else int(gc["prompts_per_epoch"]))
    max_updates = args.max_updates
    model_cfg = cfg["model"]
    max_seq_len = int(model_cfg["max_seq_len"])
    stoi, _ = V.build_vocab()
    expected_representation = (
        "abacus-v1" if model_cfg.get("use_abacus", False) else None)
    train_examples = load_division_examples(
        gc["data_path"], gc.get("split", "train"), stoi, max_seq_len,
        expected_representation,
        operand_digits=gc.get("operand_digits"),
        min_tier=(int(gc["min_tier"]) if "min_tier" in gc else None))
    eval_examples = load_division_examples(
        ec.get("data_path", gc["data_path"]), ec.get("split", "val"),
        stoi, max_seq_len, expected_representation)
    overlap = (
        {example.prompt_ids for example in train_examples}
        & {example.prompt_ids for example in eval_examples})
    if overlap:
        raise ValueError(
            f"GRPO and evaluation data overlap on {len(overlap)} prompts")
    eval_limit = (
        args.eval_limit if args.eval_limit is not None
        else int(ec.get("limit", 0)))
    if eval_limit and eval_limit < len(eval_examples):
        eval_examples = random.Random(seed + 1).sample(
            eval_examples, eval_limit)

    policy = build_model(model_cfg).to(device)
    policy.load_state_dict(load_state_dict(checkpoint_path, device))
    reference = build_model(model_cfg).to(device)
    reference.load_state_dict(load_state_dict(checkpoint_path, device))
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    # eval() still permits gradients. It is intentional here: GRPO must score
    # the same beta=0, dropout-free policy used to produce each rollout.
    policy.eval()

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"division train prompts available: {len(train_examples):,}")
    print(f"division eval prompts: {len(eval_examples):,}")
    baseline = evaluate_division(
        policy, eval_examples, stoi, max_seq_len, device,
        batch_size=int(ec.get("batch_size", 128)))
    print_evaluation("baseline greedy division", baseline)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(gc["lr"]),
        weight_decay=float(gc.get("weight_decay", 0.0)))
    group_size = int(gc.get("group_size", 8))
    prompt_batch_size = int(gc.get("prompt_batch_size", 4))
    temperature = float(gc.get("temperature", 1.0))
    clip_epsilon = float(gc.get("clip_epsilon", 0.2))
    beta = float(gc.get("beta", 0.01))
    max_grad_norm = float(gc.get("max_grad_norm", 1.0))
    optimization_epochs = int(gc.get("optimization_epochs", 1))
    log_every = int(gc.get("log_every", 10))
    if group_size < 2:
        raise ValueError("GRPO training requires group_size >= 2")
    if prompt_batch_size < 1 or optimization_epochs < 1:
        raise ValueError(
            "prompt_batch_size and optimization_epochs must be >= 1")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if beta < 0.0:
        raise ValueError("beta must be >= 0")
    rng = random.Random(seed)
    history = []
    update = 0
    start_time = time.monotonic()
    stop = False

    for epoch in range(1, epochs + 1):
        batches = prompt_batches(
            train_examples, prompts_per_epoch, prompt_batch_size, rng)
        epoch_rewards = 0.0
        epoch_samples = 0
        epoch_mixed = 0
        epoch_groups = 0
        for batch in batches:
            if max_updates and update >= max_updates:
                stop = True
                break
            rollout = rollout_division(
                policy, batch, group_size, stoi, max_seq_len, device,
                temperature=temperature, sample=True)
            completion_steps = rollout.completion_mask.shape[1]
            with torch.no_grad():
                ref_log_probs = completion_log_probs(
                    reference, rollout.sequences, rollout.prompt_length,
                    completion_steps, temperature)

            last_metrics = None
            grad_norm = 0.0
            for _ in range(optimization_epochs):
                new_log_probs = completion_log_probs(
                    policy, rollout.sequences, rollout.prompt_length,
                    completion_steps, temperature)
                loss, last_metrics = grpo_loss(
                    new_log_probs, rollout.old_log_probs, ref_log_probs,
                    rollout.completion_mask, rollout.advantages,
                    clip_epsilon, beta)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_grad_norm).item())
                optimizer.step()
                policy.eval()

            update += 1
            exact_rate = float(rollout.rewards.mean().item())
            mixed = rollout.mixed_groups
            epoch_rewards += float(rollout.rewards.sum().item())
            epoch_samples += rollout.rewards.numel()
            epoch_mixed += mixed
            epoch_groups += len(batch)
            record = {
                "epoch": epoch,
                "update": update,
                "sample_exact_rate": exact_rate,
                "mixed_groups": mixed,
                "groups": len(batch),
                "grad_norm": grad_norm,
                **last_metrics,
            }
            history.append(record)
            if update == 1 or update % log_every == 0:
                print(
                    f"epoch {epoch}/{epochs} update {update:>4}  "
                    f"reward={exact_rate:.3f}  mixed={mixed}/{len(batch)}  "
                    f"loss={last_metrics['loss']:.4f}  "
                    f"ref_kl={last_metrics['reference_kl']:.5f}  "
                    f"grad={grad_norm:.3f}")
        print(
            f"epoch {epoch} rollout exact={epoch_rewards/max(1, epoch_samples):.3f}  "
            f"mixed_groups={epoch_mixed}/{epoch_groups}")
        if stop:
            break

    final = evaluate_division(
        policy, eval_examples, stoi, max_seq_len, device,
        batch_size=int(ec.get("batch_size", 128)))
    print_evaluation("post-GRPO greedy division", final)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    report_path = output_path.with_suffix(".metrics.json")
    report = {
        "config": args.config,
        "initial_checkpoint": checkpoint_path,
        "output_checkpoint": str(output_path),
        "device": str(device),
        "seed": seed,
        "updates": update,
        "elapsed_seconds": time.monotonic() - start_time,
        "baseline": baseline,
        "final": final,
        "history": history,
    }
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"saved checkpoint -> {output_path}")
    print(f"saved metrics -> {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/grpo_division_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prompts-per-epoch", type=int, default=None)
    parser.add_argument(
        "--max-updates", type=int, default=0,
        help="stop after this many optimizer updates; 0 uses the full run")
    parser.add_argument(
        "--eval-limit", type=int, default=None,
        help="deterministic held-out subset size; default comes from config")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
