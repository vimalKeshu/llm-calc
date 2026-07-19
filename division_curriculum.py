# type: ignore
"""Regression-gated supervised division curriculum with operation replay.

The input checkpoint is always treated as immutable. Training writes stage,
last, selected, and metrics artifacts only under a distinct output stem.
"""

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch
import yaml

import vocab as V
from grpo import load_state_dict, pick_device
from grpo_all import (checkpoint_gate, evaluate_operations,
                      load_arithmetic_examples, replay_anchor_losses,
                      sample_balanced_replay, stratified_limit,
                      supervised_replay_loss)
from model import build_model


def filter_stage_examples(examples, stage):
    """Filter division examples by configured operand-width cells/scenarios."""
    include_cells = set(stage.get("include_operand_cells", []))
    exclude_cells = set(stage.get("exclude_operand_cells", []))
    include_scenarios = set(stage.get("include_scenarios", []))
    exclude_scenarios = set(stage.get("exclude_scenarios", []))
    selected = []
    for example in examples:
        cell = example.stratum[0]
        if include_cells and cell not in include_cells:
            continue
        if cell in exclude_cells:
            continue
        if include_scenarios and example.scenario not in include_scenarios:
            continue
        if example.scenario in exclude_scenarios:
            continue
        selected.append(example)
    if not selected:
        raise ValueError(f"curriculum stage {stage['name']!r} has no examples")
    return selected


def sample_stratified_examples(examples, count, rng):
    """Sample while guaranteeing and then balancing observed micro-strata."""
    if count < 1:
        raise ValueError("examples_per_epoch must be >= 1")
    strata = collections.defaultdict(list)
    for example in examples:
        strata[example.stratum].append(example)
    if count < len(strata):
        raise ValueError(
            f"examples_per_epoch={count} cannot cover {len(strata)} strata")
    keys = list(strata)
    selected = [rng.choice(strata[key]) for key in keys]
    selected.extend(
        rng.choice(strata[rng.choice(keys)])
        for _ in range(count - len(selected)))
    rng.shuffle(selected)
    return selected


def iter_batches(examples, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(examples), batch_size):
        yield examples[start:start + batch_size]


def validate_output_paths(checkpoint_path, output_path, allow_overwrite=False):
    """Reject any path layout that could overwrite the source checkpoint."""
    checkpoint = Path(checkpoint_path).resolve()
    output = Path(output_path).resolve()
    last = output.with_name(f"{output.stem}.last{output.suffix}")
    metrics = output.with_suffix(".metrics.json")
    candidates = [output, last.resolve(), metrics.resolve()]
    if checkpoint in candidates:
        raise ValueError("curriculum output paths must differ from checkpoint")
    if not allow_overwrite:
        existing = [path for path in candidates if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite curriculum artifacts: "
                + ", ".join(str(path) for path in existing))
    return output, last, metrics


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}


def _print_evaluation(label, metrics):
    print(
        f"{label}: {metrics['correct']}/{metrics['total']} "
        f"= {100 * metrics['accuracy']:.2f}%")
    for operation in V.OPERATORS:
        if operation in metrics["operation"]:
            print(
                f"  {operation:<2} "
                f"{100 * metrics['operation'][operation]:6.2f}%")


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    cc = cfg["curriculum"]
    rc = cfg["replay"]
    ec = cfg["eval"]
    device = pick_device(args.device)
    seed = int(cc.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    checkpoint_path = Path(args.checkpoint or cc["checkpoint_path"])
    output_path, last_path, metrics_path = validate_output_paths(
        checkpoint_path, args.output or cc["output_path"],
        allow_overwrite=args.allow_overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_seq_len = int(model_cfg["max_seq_len"])
    stoi, _ = V.build_vocab()
    expected_representation = (
        "abacus-v1" if model_cfg.get("use_abacus", False) else None)

    division_examples = load_arithmetic_examples(
        cc["data_path"], cc.get("split", "train"), stoi, max_seq_len,
        ["/"], expected_representation, seed=seed)
    replay_operations = tuple(rc["operations"])
    replay_examples = load_arithmetic_examples(
        rc["data_path"], rc.get("split", "train"), stoi, max_seq_len,
        replay_operations, expected_representation,
        max_per_operation=int(rc.get("pool_per_operation", 10000)),
        seed=seed + 1)
    eval_operations = tuple(ec.get("operations", V.OPERATORS))
    eval_examples = load_arithmetic_examples(
        ec["data_path"], ec.get("split", "test"), stoi, max_seq_len,
        eval_operations, expected_representation, seed=seed + 2)
    eval_examples = stratified_limit(
        eval_examples, int(ec.get("limit", 0)), eval_operations, seed + 3)

    train_prompts = {
        (example.operation, example.prompt_ids)
        for example in [*division_examples, *replay_examples]}
    eval_prompts = {
        (example.operation, example.prompt_ids) for example in eval_examples}
    overlap = train_prompts & eval_prompts
    if overlap:
        raise ValueError(f"training/evaluation overlap: {len(overlap)}")

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
    print(f"checkpoint (read-only): {checkpoint_path}")
    print(f"parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"division pool: {len(division_examples):,}")
    print("replay pool:", dict(collections.Counter(
        example.operation for example in replay_examples)))
    baseline = evaluate_operations(
        policy, eval_examples, stoi, max_seq_len, device,
        batch_size=int(ec.get("batch_size", 128)))
    _print_evaluation("baseline greedy evaluation", baseline)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(cc["lr"]),
        weight_decay=float(cc.get("weight_decay", 0.0)))
    replay_batch_size = int(rc.get("batch_size", 48))
    replay_supervised_weight = float(rc.get("supervised_weight", 0.20))
    replay_distillation_weight = float(rc.get("distillation_weight", 1.0))
    max_grad_norm = float(cc.get("max_grad_norm", 1.0))
    log_every = int(cc.get("log_every", 20))
    gate_config = ec.get("checkpoint_gate", {})
    baseline_gate = checkpoint_gate(baseline, baseline, gate_config)
    best_score = baseline_gate["target_score"]
    best_state = _cpu_state_dict(policy)
    best_metrics = baseline
    best_stage = "baseline"
    rng = random.Random(seed)
    history = []
    evaluations = [{
        "stage": "baseline", "metrics": baseline, "gate": baseline_gate}]
    update = 0
    start_time = time.monotonic()

    for stage_index, stage in enumerate(cc["stages"], start=1):
        stage_pool = filter_stage_examples(division_examples, stage)
        stage_lr = float(stage.get("lr", cc["lr"]))
        for group in optimizer.param_groups:
            group["lr"] = stage_lr
        stage_strata = {example.stratum for example in stage_pool}
        print(
            f"stage {stage_index}/{len(cc['stages'])} {stage['name']}: "
            f"pool={len(stage_pool):,} strata={len(stage_strata)} "
            f"lr={stage_lr:.2e}")
        for epoch in range(1, int(stage.get("epochs", 1)) + 1):
            selected = sample_stratified_examples(
                stage_pool, int(stage["examples_per_epoch"]), rng)
            epoch_loss = 0.0
            division_loss_sum = 0.0
            replay_loss_sum = 0.0
            replay_kl_sum = 0.0
            epoch_updates = 0
            for division_batch in iter_batches(
                    selected, int(stage.get(
                        "batch_size", cc.get("batch_size", 64)))):
                replay_batch = sample_balanced_replay(
                    replay_pools, replay_operations, replay_batch_size, rng)
                division_loss = supervised_replay_loss(
                    policy, division_batch, stoi, device)
                replay_loss, replay_kl = replay_anchor_losses(
                    policy, reference, replay_batch, stoi, device)
                total_loss = (
                    division_loss
                    + replay_supervised_weight * replay_loss
                    + replay_distillation_weight * replay_kl)
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), max_grad_norm).item())
                optimizer.step()
                policy.eval()

                update += 1
                epoch_updates += 1
                division_value = float(division_loss.detach().item())
                replay_value = float(replay_loss.detach().item())
                replay_kl_value = float(replay_kl.detach().item())
                total_value = float(total_loss.detach().item())
                epoch_loss += total_value
                division_loss_sum += division_value
                replay_loss_sum += replay_value
                replay_kl_sum += replay_kl_value
                history.append({
                    "stage": stage["name"],
                    "stage_index": stage_index,
                    "epoch": epoch,
                    "update": update,
                    "division_loss": division_value,
                    "replay_supervised_loss": replay_value,
                    "replay_distillation_kl": replay_kl_value,
                    "total_loss": total_value,
                    "grad_norm": grad_norm,
                    "lr": stage_lr,
                })
                if update == 1 or update % log_every == 0:
                    print(
                        f"  update {update:>4} division={division_value:.4f} "
                        f"replay={replay_value:.4f} "
                        f"replay_kl={replay_kl_value:.5f} "
                        f"grad={grad_norm:.3f}")
            print(
                f"  epoch {epoch}: total={epoch_loss/epoch_updates:.4f} "
                f"division={division_loss_sum/epoch_updates:.4f} "
                f"replay={replay_loss_sum/epoch_updates:.4f} "
                f"replay_kl={replay_kl_sum/epoch_updates:.5f}")

        metrics = evaluate_operations(
            policy, eval_examples, stoi, max_seq_len, device,
            batch_size=int(ec.get("batch_size", 128)))
        gate = checkpoint_gate(metrics, baseline, gate_config)
        evaluations.append({
            "stage": stage["name"], "metrics": metrics, "gate": gate})
        _print_evaluation(f"stage {stage['name']} evaluation", metrics)
        print(
            f"  checkpoint eligible={gate['eligible']} "
            f"target_score={gate['target_score']:.4f} "
            f"violations={gate['violations'] or 'none'}")
        if bool(cc.get("save_stage_checkpoints", True)):
            stage_path = output_path.with_name(
                f"{output_path.stem}.stage-{stage_index}-{stage['name']}"
                f"{output_path.suffix}")
            if stage_path.resolve() == checkpoint_path.resolve():
                raise ValueError("stage checkpoint would overwrite input")
            torch.save(policy.state_dict(), stage_path)
            print(f"  saved stage snapshot -> {stage_path}")
        if gate["eligible"] and gate["target_score"] > best_score:
            best_score = gate["target_score"]
            best_state = _cpu_state_dict(policy)
            best_metrics = metrics
            best_stage = stage["name"]
            print(f"  new best eligible checkpoint at stage {stage['name']}")

    torch.save(_cpu_state_dict(policy), last_path)
    policy.load_state_dict(best_state)
    torch.save(policy.state_dict(), output_path)
    report = {
        "config": args.config,
        "initial_checkpoint": str(checkpoint_path),
        "output_checkpoint": str(output_path),
        "last_checkpoint": str(last_path),
        "device": str(device),
        "seed": seed,
        "updates": update,
        "elapsed_seconds": time.monotonic() - start_time,
        "baseline": baseline,
        "selected": best_metrics,
        "selected_stage": best_stage,
        "history": history,
        "evaluations": evaluations,
    }
    with metrics_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    _print_evaluation(f"selected stage {best_stage}", best_metrics)
    print(f"saved selected checkpoint -> {output_path}")
    print(f"saved last trained checkpoint -> {last_path}")
    print(f"saved metrics -> {metrics_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/division_curriculum_v2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto")
    parser.add_argument(
        "--allow-overwrite", action="store_true",
        help="allow replacing existing curriculum artifacts (never the input)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
