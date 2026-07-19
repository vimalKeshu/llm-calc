import yaml
import torch
import argparse
import json
import collections
import bisect
import math
import random
import re
from functools import lru_cache
import vocab as V
from generate_data import (TIER_NAMES, compute, decode_internal_answer,
                           encode_prompt, make_text, parse_expression,
                           unreverse_magnitude)
from model import build_model
 

@torch.no_grad()
def generate(model, prompt, stoi, itos, max_seq_len, device, temp=0.0,
             reverse=True, prompt_is_internal=False) -> str:
    """Generate from the internal representation and return a natural answer.

    ``reverse`` is retained as a compatibility name for the broader Abacus
    internal-format switch. Stored validation prompts are already internal;
    human-facing demo/probe prompts are converted here.
    """
    internal_prompt = (prompt if prompt_is_internal
                       else encode_prompt(prompt, internal_format=reverse))
    tokens = [stoi[V.BOS]] + V.encode(internal_prompt, stoi)
    prompt_token_count = len(tokens)
    lhs = internal_prompt.rstrip('=')
    try:
        _, _, op, _ = parse_expression(lhs)
    except ValueError:
        op = None
    digit_ids = {stoi[digit] for digit in V.DIGITS}
    decimal_id = stoi['.']

    while len(tokens) < max_seq_len:
        input = torch.tensor([tokens]).to(device)
        mask = torch.tril(torch.ones(input.shape[1], input.shape[1])).unsqueeze(0).to(device)
        logits = model(input, mask)[:, -1, :]
        if temp == 0.0:
            next_token = logits.argmax(dim=-1)
        else:
            output = torch.nn.functional.softmax(logits/temp, dim=-1)
            next_token: torch.Tensor = torch.multinomial(output, 1)

        if next_token == stoi[V.EOS]:
            break

        tokens.append(next_token.item())

        # Division has a fixed DDD.ddd internal answer. Stop after its seven
        # numeric/decimal tokens so malformed continuations cannot create
        # unsupported fractional place IDs on the following model call.
        if reverse and op == '/':
            answer_tokens = tokens[prompt_token_count:]
            numeric_count = sum(
                token_id in digit_ids or token_id == decimal_id
                for token_id in answer_tokens)
            if numeric_count >= 7:
                break

    ans = V.decode(tokens, itos)
    if reverse and '=' in ans:
        _, rhs = ans.split('=', 1)
        if op is not None:
            rhs = decode_internal_answer(op, rhs, internal_format=True)
        return prompt + rhs

    return ans

def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(cfg, checkpoint, device):
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def true_answer(text, reverse=True):
    """Decode a stored validation answer into its natural user-facing form."""
    lhs, stored = text.split('=', 1)
    try:
        _, _, op, _ = parse_expression(lhs)
    except ValueError:
        op = None
    return (decode_internal_answer(op, stored, internal_format=reverse)
            if op is not None else stored)


@lru_cache(maxsize=None)
def load_split_rows(data_path, split="val", expected_representation=None):
    """Parse and retain one JSONL split once per dataset path."""
    rows = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == split:
                if (expected_representation is not None and
                        row.get("representation") != expected_representation):
                    actual = row.get("representation", "missing")
                    raise ValueError(
                        f"{data_path} uses representation {actual!r}, expected "
                        f"{expected_representation!r}; regenerate the dataset")
                rows.append(row)
    return tuple(rows)


def load_val_rows(data_path, expected_representation=None):
    """Backward-compatible validation-split loader."""
    return load_split_rows(data_path, "val", expected_representation)


@lru_cache(maxsize=None)
def load_train_prompts(data_path):
    """Return internal-format prompts present in the configured train split."""
    prompts = set()
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "train":
                prompts.add(row["text"].split('=', 1)[0] + '=')
    return frozenset(prompts)


@lru_cache(maxsize=None)
def load_dataset_prompts(data_path):
    """Return prompts from every split, for genuinely fresh diagnostics."""
    prompts = set()
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            prompts.add(row["text"].split('=', 1)[0] + '=')
    return frozenset(prompts)


@lru_cache(maxsize=None)
def load_train_unsigned_three_digit_multiplications(
        data_path, internal_format=True):
    """Return natural-order (a, b) pairs, retaining training repetition."""
    pairs = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            lhs = row["text"].split('=', 1)[0]
            operator_index = lhs.find('*', 1)
            if operator_index == -1:
                continue
            left = lhs[:operator_index]
            right = lhs[operator_index + 1:]
            try:
                a = int(unreverse_magnitude(left) if internal_format else left)
                b = int(unreverse_magnitude(right) if internal_format else right)
            except ValueError:
                continue
            if 100 <= a <= 999 and 100 <= b <= 999:
                pairs.append((a, b))
    return tuple(pairs)


@torch.no_grad()
def batched_row_predictions(model, rows, stoi, itos, max_seq_len, device,
                            reverse=True, batch_size=128,
                            return_internal=False):
    """Greedily decode stored internal prompts, batching equal prompt lengths."""
    eos_id = stoi[V.EOS]
    grouped = collections.defaultdict(list)
    for index, row in enumerate(rows):
        prompt = row["text"].split("=", 1)[0] + "="
        token_ids = [stoi[V.BOS]] + V.encode(prompt, stoi)
        grouped[len(token_ids)].append((index, token_ids))
    predictions = [None] * len(rows)
    internal_predictions = [None] * len(rows)
    for group in grouped.values():
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            prompt_length = len(batch[0][1])
            tokens = torch.tensor(
                [token_ids for _, token_ids in batch],
                dtype=torch.long, device=device)
            finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
            while tokens.shape[1] < max_seq_len and not bool(finished.all()):
                seq_len = tokens.shape[1]
                mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
                next_tokens = model(tokens, mask)[:, -1, :].argmax(dim=-1)
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, eos_id), next_tokens)
                tokens = torch.cat([tokens, next_tokens[:, None]], dim=1)
                finished |= next_tokens.eq(eos_id)
            generated = tokens[:, prompt_length:].detach().cpu().tolist()
            for (index, _), token_ids in zip(batch, generated):
                if eos_id in token_ids:
                    token_ids = token_ids[:token_ids.index(eos_id)]
                internal_answer = V.decode(token_ids, itos)
                internal_predictions[index] = internal_answer
                op = rows[index].get("operation")
                if op is None:
                    lhs = rows[index]["text"].split("=", 1)[0]
                    _, _, op, _ = parse_expression(lhs)
                predictions[index] = decode_internal_answer(
                    op, internal_answer, internal_format=reverse)
    if return_internal:
        return predictions, internal_predictions
    return predictions


def tier_report(model, cfg, device, stoi, itos, reverse=True,
                data_path=None, split="val", batch_size=128):
    """Greedy exact accuracy over a dataset split, grouped by tier/op/scenario."""
    max_seq_len = int(cfg["model"]["max_seq_len"])
    expected_representation = cfg["model"].get(
        "expected_representation",
        "abacus-v1" if cfg["model"].get("use_abacus", False) else None)
    rows = load_split_rows(
        data_path or cfg["train"]["data_path"], split,
        expected_representation)
    if not rows:
        raise ValueError(f"no {split!r} rows found in {data_path}")
    predictions, internal_predictions = batched_row_predictions(
        model, rows, stoi, itos, max_seq_len, device, reverse, batch_size,
        return_internal=True)

    by_tier, ok_tier = collections.Counter(), collections.Counter()
    by_op, ok_op = collections.Counter(), collections.Counter()
    by_scenario, ok_scenario = collections.Counter(), collections.Counter()
    strict_stored_hits = 0
    for r, pred, internal_prediction in zip(
            rows, predictions, internal_predictions):
        lhs = r["text"].split('=')[0]
        stored_truth = r["text"].split('=', 1)[1]
        strict_stored_hits += internal_prediction == stored_truth
        truth = true_answer(r["text"], reverse)
        hit = (pred == truth)
        by_tier[r["tier"]] += 1; ok_tier[r["tier"]] += hit
        try:
            _, _, op, _ = parse_expression(lhs)
        except ValueError:
            op = None
        if op:
            by_op[op] += 1; ok_op[op] += hit
            scenario = r.get("scenario", "unclassified")
            scenario_key = (op, scenario)
            by_scenario[scenario_key] += 1
            ok_scenario[scenario_key] += hit

    print("\naccuracy by tier:")
    for t in range(5):
        if by_tier[t]:
            print(f"  {t} {TIER_NAMES[t]:<8} {ok_tier[t]:>4}/{by_tier[t]:<4} "
                  f"= {100*ok_tier[t]/by_tier[t]:5.1f}%")
    print("accuracy by op:")
    for op in V.OPERATORS:
        if by_op[op]:
            print(f"  {op:<10} {ok_op[op]:>4}/{by_op[op]:<4} "
                  f"= {100*ok_op[op]/by_op[op]:5.1f}%")
    print(f"accuracy by {split} scenario:")
    for op in V.OPERATORS:
        operation_scenarios = sorted(
            key for key in by_scenario if key[0] == op)
        for key in operation_scenarios:
            _, scenario = key
            print(
                f"  {op} {scenario:<38} "
                f"{ok_scenario[key]:>4}/{by_scenario[key]:<4} "
                f"= {100*ok_scenario[key]/by_scenario[key]:5.1f}%")
    tot, okt = sum(by_tier.values()), sum(ok_tier.values())
    print(f"  overall    {okt:>4}/{tot:<4} = {100*okt/tot:5.1f}%")
    scenario_accuracies = {
        key: ok_scenario[key] / by_scenario[key]
        for key in by_scenario}
    macro_scenario = sum(scenario_accuracies.values()) / len(scenario_accuracies)
    minimum_scenario = min(scenario_accuracies.values())
    print(f"  macro scenario accuracy   = {100*macro_scenario:5.1f}%")
    print(f"  minimum scenario accuracy = {100*minimum_scenario:5.1f}%")
    strict_stored_accuracy = strict_stored_hits / len(rows)
    print(
        f"  strict stored-format accuracy = "
        f"{strict_stored_hits}/{len(rows)} = "
        f"{100*strict_stored_accuracy:5.1f}%")
    return {
        "tier": {t: ok_tier[t] / by_tier[t] for t in by_tier},
        "op": {op: ok_op[op] / by_op[op] for op in by_op},
        "scenario": scenario_accuracies,
        "overall": okt / tot,
        "macro_scenario": macro_scenario,
        "minimum_scenario": minimum_scenario,
        "strict_stored_format": strict_stored_accuracy,
    }


def _pass_at_k(correct_count, sample_count, k):
    """Unbiased pass@k estimate from sample_count draws and correct_count hits."""
    if correct_count <= 0:
        return 0.0
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - math.comb(sample_count - correct_count, k) / math.comb(
        sample_count, k)


def _valid_natural_answer(op, answer):
    if op in ('+', '-', '*'):
        return re.fullmatch(r'-?(?:0|[1-9][0-9]*)', answer) is not None
    return re.fullmatch(
        r'(?:NAN|-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,3})?)',
        answer) is not None


@torch.no_grad()
def stochastic_row_samples(model, rows, stoi, itos, max_seq_len, device,
                           k=64, temp=1.0, reverse=True,
                           sample_batch_size=512, seed=42):
    """Return k independently sampled natural-format answers for each row."""
    if k < 1 or temp <= 0:
        raise ValueError("stochastic sampling requires k >= 1 and temp > 0")
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    eos_id = stoi[V.EOS]
    digit_ids = torch.tensor(
        [stoi[digit] for digit in V.DIGITS], device=device)
    decimal_id = stoi['.']
    grouped = collections.defaultdict(list)
    for index, row in enumerate(rows):
        prompt = row["text"].split("=", 1)[0] + "="
        token_ids = [stoi[V.BOS]] + V.encode(prompt, stoi)
        grouped[(row["operation"], len(token_ids))].append((index, token_ids))

    answers = [None] * len(rows)
    prompts_per_batch = max(1, sample_batch_size // k)
    for (op, _), group in grouped.items():
        for start in range(0, len(group), prompts_per_batch):
            prompt_batch = group[start:start + prompts_per_batch]
            prompt_length = len(prompt_batch[0][1])
            base = torch.tensor(
                [token_ids for _, token_ids in prompt_batch],
                dtype=torch.long, device=device)
            tokens = base.repeat_interleave(k, dim=0)
            finished = torch.zeros(tokens.shape[0], dtype=torch.bool,
                                   device=device)
            while tokens.shape[1] < max_seq_len and not bool(finished.all()):
                seq_len = tokens.shape[1]
                mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
                probabilities = torch.softmax(
                    model(tokens, mask)[:, -1, :] / temp, dim=-1)
                next_tokens = torch.multinomial(probabilities, 1).squeeze(1)
                next_tokens = torch.where(
                    finished, torch.full_like(next_tokens, eos_id), next_tokens)
                tokens = torch.cat([tokens, next_tokens[:, None]], dim=1)
                newly_finished = next_tokens.eq(eos_id)
                if reverse and op == '/':
                    generated = tokens[:, prompt_length:]
                    numeric = (
                        torch.isin(generated, digit_ids)
                        | generated.eq(decimal_id)).sum(dim=1)
                    newly_finished |= numeric.ge(7)
                finished |= newly_finished

            generated_rows = tokens[:, prompt_length:].detach().cpu().tolist()
            decoded = []
            for token_ids in generated_rows:
                if eos_id in token_ids:
                    token_ids = token_ids[:token_ids.index(eos_id)]
                internal_answer = V.decode(token_ids, itos)
                decoded.append(decode_internal_answer(
                    op, internal_answer, internal_format=reverse))
            for batch_index, (row_index, _) in enumerate(prompt_batch):
                offset = batch_index * k
                answers[row_index] = decoded[offset:offset + k]
    return answers


def reachability_report(model, cfg, device, stoi, itos, reverse, data_path,
                        split='test', k=64, temp=1.0,
                        sample_batch_size=512, seed=42,
                        report_out=None, group_size=8, operations=None):
    """Measure exact-answer reachability and DR-GRPO group usefulness."""
    expected_representation = cfg["model"].get(
        "expected_representation",
        "abacus-v1" if cfg["model"].get("use_abacus", False) else None)
    rows = load_split_rows(data_path, split, expected_representation)
    if operations:
        operation_set = set(operations)
        rows = tuple(row for row in rows
                     if row["operation"] in operation_set)
    if not rows:
        raise ValueError(f"no {split!r} rows found in {data_path}")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    samples = stochastic_row_samples(
        model, rows, stoi, itos, int(cfg["model"]["max_seq_len"]), device,
        k=k, temp=temp, reverse=reverse,
        sample_batch_size=sample_batch_size, seed=seed)
    truths = [true_answer(row["text"], reverse) for row in rows]
    correct = [
        [answer == truth for answer in row_samples]
        for row_samples, truth in zip(samples, truths)
    ]
    valid = [
        [_valid_natural_answer(row["operation"], answer)
         for answer in row_samples]
        for row, row_samples in zip(rows, samples)]

    categories = collections.defaultdict(list)
    categories[("all", "all")] = list(range(len(rows)))
    for index, row in enumerate(rows):
        op = row["operation"]
        categories[("operation", op)].append(index)
        categories[("scenario", f"{op}|{row['scenario']}")].append(index)
        if op == '*':
            categories[("multiplication_sign",
                        row["multiplication_sign_pattern"])].append(index)
            categories[("multiplication_central_band",
                        str(row["central_total_band"]))].append(index)
        if op == '/':
            categories[("division_sign", row["division_sign"])].append(index)

    pass_ks = [value for value in (1, 8, 16, 64) if value <= k]
    result_categories = collections.defaultdict(dict)
    for (family, name), indices in categories.items():
        counts = [sum(correct[index]) for index in indices]
        total_samples = len(indices) * k
        metrics = {
            "prompts": len(indices),
            "sample_exact_rate": sum(counts) / total_samples,
            "valid_output_rate": sum(
                sum(valid[index]) for index in indices) / total_samples,
            "pass_at_k": {
                str(value): sum(
                    _pass_at_k(count, k, value) for count in counts
                ) / len(indices)
                for value in pass_ks},
        }
        observed_group_size = min(group_size, k)
        group_states = collections.Counter()
        group_total = 0
        for index in indices:
            for start in range(
                    0, k - observed_group_size + 1, observed_group_size):
                group = correct[index][start:start + observed_group_size]
                hits = sum(group)
                state = ("all_wrong" if hits == 0 else
                         "all_correct" if hits == observed_group_size else "mixed")
                group_states[state] += 1
                group_total += 1
        metrics["observed_group_size"] = observed_group_size
        metrics["group_state_rate"] = {
            state: group_states[state] / group_total
            for state in ("all_wrong", "mixed", "all_correct")}
        result_categories[family][name] = metrics

    print(
        f"stochastic exact-answer reachability: temp={temp}, "
        f"samples/prompt={k}, seed={seed}\n")
    for family in (
            "all", "operation", "multiplication_sign",
            "multiplication_central_band", "division_sign", "scenario"):
        if family not in result_categories:
            continue
        print(f"{family}:")
        print(
            f"  {'category':<42}{'n':>6}{'sample':>9}"
            + ''.join(f"{'p@'+str(value):>9}" for value in pass_ks)
            + f"{('wrong'+str(min(group_size, k))):>9}"
              f"{('mixed'+str(min(group_size, k))):>9}"
              f"{('right'+str(min(group_size, k))):>9}{'valid':>9}")
        for name, metrics in sorted(result_categories[family].items()):
            pass_values = metrics["pass_at_k"]
            states = metrics["group_state_rate"]
            print(
                f"  {name:<42}{metrics['prompts']:>6}"
                f"{100*metrics['sample_exact_rate']:>8.1f}%"
                + ''.join(
                    f"{100*pass_values[str(value)]:>8.1f}%"
                    for value in pass_ks)
                + f"{100*states['all_wrong']:>8.1f}%"
                  f"{100*states['mixed']:>8.1f}%"
                  f"{100*states['all_correct']:>8.1f}%"
                  f"{100*metrics['valid_output_rate']:>8.1f}%")
        print()
    report = {
        "data_path": data_path,
        "split": split,
        "temperature": temp,
        "samples_per_prompt": k,
        "operations": list(operations) if operations else None,
        "group_size": min(group_size, k),
        "seed": seed,
        "categories": dict(result_categories),
    }
    if report_out:
        with open(report_out, 'w') as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
        print(f"wrote reachability report -> {report_out}")
    return report


# prompts whose greedy answer is currently wrong (mult / division hard cases) --
# the exact cases where we want to know if the right answer is REACHABLE by
# sampling (i.e. whether RLVR/GRPO has any signal to sharpen).
PROBE_PROMPTS = [
    "999*999=", "111*999=", "888*999=", "777*777=", "997*998=", "123*456=",
    "10/3=", "2/3=", "100/7=", "1/7=", "999/999=", "500*200=",
]


def parse_prompt(prompt):
    """Parse a natural prompt into integer operands and its operator."""
    if not isinstance(prompt, str) or not prompt.endswith('='):
        raise ValueError(f"prompt must be a string ending in '=': {prompt!r}")
    left, right, op, _ = parse_expression(prompt)
    a, b = int(left), int(right)
    return a, b, op


def prompt_truth(prompt):
    """Ground-truth natural-order answer string for a prompt like '999*999='."""
    a, b, op = parse_prompt(prompt)
    r = compute(a, b, op)
    return None if r is None else str(r)


def algorithmic_report(model, cfg, device, stoi, itos, reverse, data_path,
                       max_failures=40):
    """Greedy exact-match report over held-out, principle-driven groups."""
    with open(data_path) as f:
        suite = yaml.safe_load(f)
    meta = suite.get("meta", {})
    groups = suite.get("groups")
    if groups is None:
        groups = [{"name": "all", "scope": "unspecified",
                   "data": suite.get("data", [])}]
    if not groups:
        raise ValueError(f"no evaluation groups found in {data_path}")

    max_abs_operand = meta.get("max_abs_operand")
    require_unseen = bool(meta.get("require_unseen_from_training", False))
    train_prompts = (load_train_prompts(cfg["train"]["data_path"])
                     if require_unseen else frozenset())

    max_seq_len = int(cfg["model"]["max_seq_len"])
    scope_totals = collections.Counter()
    scope_hits = collections.Counter()
    op_totals = collections.Counter()
    op_hits = collections.Counter()
    length_totals = collections.Counter()
    length_hits = collections.Counter()
    all_failures = []
    overall_hits = overall_total = 0
    group_accuracies = {}
    suite_prompts = set()

    print("algorithmic greedy exact-match report\n")
    if meta.get("name"):
        print(f"suite: {meta['name']}")
    if max_abs_operand is not None:
        print(f"operand domain: |a|, |b| <= {max_abs_operand}")
    if require_unseen:
        print("training overlap: forbidden (checked against configured train split)")
    print()
    print(f"{'group':<38}{'scope':<14}{'correct':<12}{'accuracy'}")
    for group in groups:
        name = group["name"]
        scope = group.get("scope", "unspecified")
        prompts = group.get("data", [])
        if not prompts:
            raise ValueError(f"algorithmic group {name!r} has no prompts")

        hits = 0
        for prompt in prompts:
            if prompt in suite_prompts:
                raise ValueError(f"duplicate algorithmic prompt: {prompt!r}")
            suite_prompts.add(prompt)
            a, b, op = parse_prompt(prompt)
            if (max_abs_operand is not None and
                    (abs(a) > max_abs_operand or abs(b) > max_abs_operand)):
                raise ValueError(
                    f"{prompt!r} exceeds max_abs_operand={max_abs_operand}")
            result = compute(a, b, op)
            if result is None:
                raise ValueError(
                    f"undefined arithmetic prompt in group {name!r}: {prompt}")
            truth = str(result)
            internal_prompt = encode_prompt(prompt, internal_format=reverse)
            if internal_prompt in train_prompts:
                raise ValueError(
                    f"algorithmic prompt {prompt!r} occurs in the training "
                    f"split as {internal_prompt!r}")
            internal_answer = make_text(
                a, b, op, result, reverse=reverse).split('=', 1)[1]
            required_tokens = (
                1 + len(V.encode(internal_prompt, stoi))
                + len(V.encode(internal_answer, stoi)))
            if required_tokens > max_seq_len:
                raise ValueError(
                    f"{prompt!r} needs {required_tokens} tokens to express its "
                    f"answer, but max_seq_len={max_seq_len}")

            prediction = generate(
                model, prompt, stoi, itos, max_seq_len, device,
                temp=0.0, reverse=reverse).split('=')[-1]
            hit = prediction == truth
            hits += hit
            operand_length = max(len(str(abs(a))), len(str(abs(b))))
            op_totals[op] += 1
            op_hits[op] += hit
            length_totals[operand_length] += 1
            length_hits[operand_length] += hit
            if not hit:
                all_failures.append((name, prompt, truth, prediction))

        total = len(prompts)
        overall_hits += hits
        overall_total += total
        scope_hits[scope] += hits
        scope_totals[scope] += total
        group_accuracies[name] = hits / total
        print(f"{name:<38}{scope:<14}{hits:>3}/{total:<8}"
              f"{100 * hits / total:6.1f}%")

    print("\naccuracy by scope:")
    for scope in scope_totals:
        hits, total = scope_hits[scope], scope_totals[scope]
        print(f"  {scope:<14}{hits:>3}/{total:<4} = {100 * hits / total:5.1f}%")
    print(f"  {'overall':<14}{overall_hits:>3}/{overall_total:<4} = "
          f"{100 * overall_hits / overall_total:5.1f}%")
    macro_group = sum(group_accuracies.values()) / len(group_accuracies)
    minimum_group = min(group_accuracies.values())
    print(f"  {'macro group':<14}{100 * macro_group:5.1f}%")
    print(f"  {'minimum group':<14}{100 * minimum_group:5.1f}%")

    print("\naccuracy by operation:")
    for op in V.OPERATORS:
        if op_totals[op]:
            print(f"  {op:<10}{op_hits[op]:>3}/{op_totals[op]:<4} = "
                  f"{100 * op_hits[op] / op_totals[op]:5.1f}%")

    print("\naccuracy by maximum operand length:")
    for length in sorted(length_totals):
        print(f"  {length} digit{'s' if length != 1 else ' ':<5}"
              f"{length_hits[length]:>3}/{length_totals[length]:<4} = "
              f"{100 * length_hits[length] / length_totals[length]:5.1f}%")

    if all_failures and max_failures > 0:
        print(f"\nfailures (showing {min(len(all_failures), max_failures)} of "
              f"{len(all_failures)}):")
        for name, prompt, truth, prediction in all_failures[:max_failures]:
            print(f"  [{name}] {prompt} expected {truth}, got {prediction or '<empty>'}")
    return {
        "overall": overall_hits / overall_total,
        "macro_group": macro_group,
        "minimum_group": minimum_group,
        "group": group_accuracies,
        "scope": {
            scope: scope_hits[scope] / scope_totals[scope]
            for scope in scope_totals
        },
    }


@torch.no_grad()
def multiplication_predictions(model, pairs, stoi, itos, max_seq_len, device,
                               reverse=True, batch_size=128):
    """Greedily generate answers for same-length multiplication prompts.

    The diagnostic only uses unsigned three-digit operands, so every encoded
    prompt has the same length. This makes batched autoregressive evaluation
    exact: no padding tokens can leak into another prompt's attention context.
    """
    unique_pairs = list(dict.fromkeys(pairs))
    eos_id = stoi[V.EOS]
    predictions = {}
    for start in range(0, len(unique_pairs), batch_size):
        batch_pairs = unique_pairs[start:start + batch_size]
        natural_prompts = [f"{a}*{b}=" for a, b in batch_pairs]
        internal_prompts = [
            encode_prompt(prompt, internal_format=reverse)
            for prompt in natural_prompts
        ]
        token_rows = [
            [stoi[V.BOS]] + V.encode(prompt, stoi)
            for prompt in internal_prompts
        ]
        prompt_length = len(token_rows[0])
        if any(len(row) != prompt_length for row in token_rows):
            raise ValueError("multiplication diagnostic prompts must be equal length")

        tokens = torch.tensor(token_rows, dtype=torch.long, device=device)
        finished = torch.zeros(len(batch_pairs), dtype=torch.bool, device=device)
        while tokens.shape[1] < max_seq_len and not bool(finished.all()):
            seq_len = tokens.shape[1]
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
            next_tokens = model(tokens, mask)[:, -1, :].argmax(dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, eos_id), next_tokens)
            tokens = torch.cat([tokens, next_tokens[:, None]], dim=1)
            finished |= next_tokens.eq(eos_id)

        generated = tokens[:, prompt_length:].detach().cpu().tolist()
        for pair, token_ids in zip(batch_pairs, generated):
            if eos_id in token_ids:
                token_ids = token_ids[:token_ids.index(eos_id)]
            internal_answer = V.decode(token_ids, itos)
            predictions[pair] = decode_internal_answer(
                '*', internal_answer, internal_format=reverse)
    return predictions


def sample_fresh_three_digit_pairs(data_path, reverse, samples_per_cell, seed):
    """Uniformly sample each 1xx..9xx by 1xx..9xx operand region.

    Both a*b and b*a must be absent from train and validation. Requiring the
    swapped expression to be unseen makes the later commutativity check fair.
    """
    seen_prompts = load_dataset_prompts(data_path)
    rng = random.Random(seed)
    selected = set()
    pairs = []
    for left_hundreds in range(1, 10):
        for right_hundreds in range(1, 10):
            cell = []
            attempts = 0
            while len(cell) < samples_per_cell:
                attempts += 1
                if attempts > 100000:
                    raise RuntimeError(
                        "could not find enough train/val-disjoint pairs for "
                        f"{left_hundreds}xx*{right_hundreds}xx")
                a = rng.randint(left_hundreds * 100, left_hundreds * 100 + 99)
                b = rng.randint(right_hundreds * 100, right_hundreds * 100 + 99)
                pair = (a, b)
                if pair in selected:
                    continue
                prompt = encode_prompt(f"{a}*{b}=", internal_format=reverse)
                swapped = encode_prompt(f"{b}*{a}=", internal_format=reverse)
                if prompt in seen_prompts or swapped in seen_prompts:
                    continue
                selected.add(pair)
                cell.append(pair)
            pairs.extend(cell)
    return pairs


def multiplication_trace(a, b):
    """Return the five schoolbook column totals, including incoming carry."""
    left = [int(digit) for digit in reversed(f"{a:03d}")]
    right = [int(digit) for digit in reversed(f"{b:03d}")]
    carry = 0
    totals = []
    carries = []
    for column in range(5):
        partial = sum(
            left[i] * right[column - i]
            for i in range(3)
            if 0 <= column - i < 3)
        total = partial + carry
        totals.append(total)
        carry = total // 10
        carries.append(carry)
    return totals, carries


def nearest_training_distances(pairs, training_pairs):
    """Manhattan distance to the closest train pair, allowing operand swap."""
    by_left = collections.defaultdict(set)
    for a, b in training_pairs:
        by_left[a].add(b)
        by_left[b].add(a)
    sorted_rows = [(a, sorted(values)) for a, values in by_left.items()]

    distances = {}
    for a, b in pairs:
        best = 1800
        for train_a, train_bs in sorted_rows:
            left_distance = abs(a - train_a)
            if left_distance >= best:
                continue
            index = bisect.bisect_left(train_bs, b)
            for candidate in train_bs[max(0, index - 1):index + 1]:
                best = min(best, left_distance + abs(b - candidate))
        distances[(a, b)] = best
    return distances


def _accuracy_rows(records, key, order=None):
    totals = collections.Counter()
    hits = collections.Counter()
    for record in records:
        group = key(record)
        totals[group] += 1
        hits[group] += record["hit"]
    groups = order or sorted(totals, key=str)
    return [
        (group, hits[group], totals[group], hits[group] / totals[group])
        for group in groups if totals[group]
    ]


def _print_accuracy_groups(title, rows):
    print(f"\n{title}:")
    for name, hits, total, accuracy in rows:
        print(f"  {str(name):<22}{hits:>4}/{total:<4} = {100 * accuracy:5.1f}%")


def _pearson(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y))
    if denominator == 0:
        return float('nan')
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def multiplication_behavior_report(model, cfg, device, stoi, itos, reverse,
                                   samples_per_cell=50, batch_size=128,
                                   seed=42, max_failures=20):
    """Measure which parts of three-by-three multiplication generalized."""
    if samples_per_cell < 1:
        raise ValueError("samples_per_cell must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    data_path = cfg["train"]["data_path"]
    max_seq_len = int(cfg["model"]["max_seq_len"])
    pairs = sample_fresh_three_digit_pairs(
        data_path, reverse, samples_per_cell, seed)
    print("three-digit multiplication behavior audit")
    print(f"fresh cases: {len(pairs)} ({samples_per_cell} in each of 81 regions)")
    print("overlap: both operand orders excluded from train and validation")
    print("decoding: greedy exact match")

    predictions = multiplication_predictions(
        model, pairs, stoi, itos, max_seq_len, device,
        reverse=reverse, batch_size=batch_size)
    training_pairs = load_train_unsigned_three_digit_multiplications(
        data_path, reverse)
    distances = nearest_training_distances(pairs, training_pairs)

    records = []
    for a, b in pairs:
        truth = str(a * b)
        prediction = predictions[(a, b)]
        totals, carries = multiplication_trace(a, b)
        digits = str(a) + str(b)
        if '0' in digits:
            pattern = "contains zero"
        elif len(set(str(a))) == 1 or len(set(str(b))) == 1:
            pattern = "repeated operand"
        else:
            pattern = "no-zero mixed"
        records.append({
            "a": a,
            "b": b,
            "truth": truth,
            "prediction": prediction,
            "hit": prediction == truth,
            "cell": (a // 100, b // 100),
            "central_total": totals[2],
            "max_carry": max(carries),
            "pattern": pattern,
            "distance": distances[(a, b)],
        })

    hits = sum(record["hit"] for record in records)
    print(f"\noverall exact accuracy: {hits}/{len(records)} "
          f"= {100 * hits / len(records):.1f}%")

    cell_total = collections.Counter()
    cell_hits = collections.Counter()
    for record in records:
        cell_total[record["cell"]] += 1
        cell_hits[record["cell"]] += record["hit"]
    print("\naccuracy by operand hundreds region:")
    print("         " + " ".join(f"{column}xx" for column in range(1, 10)))
    for row in range(1, 10):
        values = [
            100 * cell_hits[(row, column)] / cell_total[(row, column)]
            for column in range(1, 10)
        ]
        print(f"  {row}xx  " + " ".join(f"{value:4.0f}" for value in values))

    train_cell_rows = collections.Counter(
        (a // 100, b // 100) for a, b in training_pairs)
    print("\ntraining rows by operand hundreds region:")
    print("         " + " ".join(f"{column}xx" for column in range(1, 10)))
    for row in range(1, 10):
        print(f"  {row}xx  " + " ".join(
            f"{train_cell_rows[(row, column)]:4d}"
            for column in range(1, 10)))

    exposure = []
    cell_accuracy = []
    for row in range(1, 10):
        for column in range(1, 10):
            exposure.append(train_cell_rows[(row, column)])
            cell_accuracy.append(
                cell_hits[(row, column)] / cell_total[(row, column)])
    correlation = _pearson(exposure, cell_accuracy)
    print("\ntraining-exposure/accuracy correlation across 81 regions: "
          f"r={correlation:.3f}")

    column_names = [
        "units", "tens", "hundreds", "thousands",
        "ten-thousands", "hundred-thousands",
    ]
    digit_total = collections.Counter()
    digit_hits = collections.Counter()
    prefix_hits = collections.Counter()
    first_wrong = collections.Counter()
    length_hits = 0
    for record in records:
        truth_reversed = record["truth"][::-1]
        prediction = record["prediction"]
        prediction_reversed = prediction[::-1] if prediction.isdigit() else ""
        length_hits += len(prediction_reversed) == len(truth_reversed)
        wrong_column = None
        for column, truth_digit in enumerate(truth_reversed):
            digit_total[column] += 1
            digit_hits[column] += (
                column < len(prediction_reversed)
                and prediction_reversed[column] == truth_digit)
            prefix_hits[column] += (
                prediction_reversed[:column + 1]
                == truth_reversed[:column + 1])
            if (wrong_column is None and
                    (column >= len(prediction_reversed)
                     or prediction_reversed[column] != truth_digit)):
                wrong_column = column
        if wrong_column is None and len(prediction_reversed) != len(truth_reversed):
            wrong_column = len(truth_reversed)
        first_wrong["exact" if wrong_column is None else wrong_column] += 1

    print("\naccuracy by result column (units-first computation order):")
    print(f"  {'column':<21}{'digit correct':<18}{'all columns through here'}")
    for column in range(6):
        if digit_total[column]:
            print(
                f"  {column_names[column]:<21}"
                f"{digit_hits[column]:>4}/{digit_total[column]:<4} "
                f"({100 * digit_hits[column] / digit_total[column]:5.1f}%)   "
                f"{prefix_hits[column]:>4}/{digit_total[column]:<4} "
                f"({100 * prefix_hits[column] / digit_total[column]:5.1f}%)")
    print(f"  exact output length: {length_hits}/{len(records)} "
          f"({100 * length_hits / len(records):.1f}%)")

    print("\nfirst wrong result column:")
    for column in range(6):
        if first_wrong[column]:
            print(f"  {column_names[column]:<21}{first_wrong[column]:>4} "
                  f"({100 * first_wrong[column] / len(records):5.1f}%)")
    print(f"  {'none (exact)':<21}{first_wrong['exact']:>4} "
          f"({100 * first_wrong['exact'] / len(records):5.1f}%)")

    central_order = ["0-79", "80-119", "120-159", "160-199", "200+"]
    def central_bin(record):
        total = record["central_total"]
        if total < 80:
            return "0-79"
        if total < 120:
            return "80-119"
        if total < 160:
            return "120-159"
        if total < 200:
            return "160-199"
        return "200+"
    _print_accuracy_groups(
        "accuracy by central three-part column total (including carry-in)",
        _accuracy_rows(records, central_bin, central_order))

    carry_order = ["0-9", "10-14", "15-19", "20+"]
    def carry_bin(record):
        carry = record["max_carry"]
        if carry < 10:
            return "0-9"
        if carry < 15:
            return "10-14"
        if carry < 20:
            return "15-19"
        return "20+"
    _print_accuracy_groups(
        "accuracy by largest carry value",
        _accuracy_rows(records, carry_bin, carry_order))
    _print_accuracy_groups(
        "accuracy by operand digit pattern",
        _accuracy_rows(
            records, lambda record: record["pattern"],
            ["contains zero", "repeated operand", "no-zero mixed"]))

    distance_order = ["1-2", "3-5", "6-10", "11-25", "26+"]
    def distance_bin(record):
        distance = record["distance"]
        if distance <= 2:
            return "1-2"
        if distance <= 5:
            return "3-5"
        if distance <= 10:
            return "6-10"
        if distance <= 25:
            return "11-25"
        return "26+"
    _print_accuracy_groups(
        "accuracy by distance to nearest train multiplication (swap allowed)",
        _accuracy_rows(records, distance_bin, distance_order))

    commutative_candidates = [pair for pair in pairs if pair[0] != pair[1]]
    rng = random.Random(seed + 1)
    commutative_pairs = rng.sample(
        commutative_candidates, min(500, len(commutative_candidates)))
    swapped_pairs = [(b, a) for a, b in commutative_pairs]
    missing_swaps = [pair for pair in swapped_pairs if pair not in predictions]
    predictions.update(multiplication_predictions(
        model, missing_swaps, stoi, itos, max_seq_len, device,
        reverse=reverse, batch_size=batch_size))
    same_answer = both_correct = one_correct = 0
    for a, b in commutative_pairs:
        forward = predictions[(a, b)]
        backward = predictions[(b, a)]
        truth = str(a * b)
        same_answer += forward == backward
        both_correct += forward == truth and backward == truth
        one_correct += (forward == truth) != (backward == truth)
    commutative_total = len(commutative_pairs)
    print(f"\ncommutativity on {commutative_total} fresh paired expressions:")
    print(f"  same prediction: {same_answer}/{commutative_total} "
          f"({100 * same_answer / commutative_total:.1f}%)")
    print(f"  both correct:    {both_correct}/{commutative_total} "
          f"({100 * both_correct / commutative_total:.1f}%)")
    print(f"  only one correct:{one_correct:>5}/{commutative_total} "
          f"({100 * one_correct / commutative_total:.1f}%)")

    numeric_errors = []
    invalid_outputs = 0
    offsets = collections.Counter()
    failures = []
    for record in records:
        if record["hit"]:
            continue
        prediction = record["prediction"]
        if prediction.isdigit():
            error = int(prediction) - int(record["truth"])
            numeric_errors.append(abs(error))
            offsets[error] += 1
        else:
            invalid_outputs += 1
        failures.append(record)
    if failures:
        error_bands = [
            ("1-10", sum(error <= 10 for error in numeric_errors)),
            ("11-100", sum(10 < error <= 100 for error in numeric_errors)),
            ("101-1000", sum(100 < error <= 1000 for error in numeric_errors)),
            (">1000", sum(error > 1000 for error in numeric_errors)),
        ]
        print("\nwrong-answer distance from the truth:")
        for name, count in error_bands:
            print(f"  {name:<10}{count:>4}/{len(failures):<4} "
                  f"({100 * count / len(failures):5.1f}%)")
        print(f"  {'invalid':<10}{invalid_outputs:>4}/{len(failures):<4} "
              f"({100 * invalid_outputs / len(failures):5.1f}%)")
        common_offsets = ", ".join(
            f"{offset:+d}×{count}" for offset, count in offsets.most_common(8))
        print(f"  most common signed offsets: {common_offsets or 'none'}")

    if failures and max_failures > 0:
        print(f"\nrepresentative failures (showing "
              f"{min(max_failures, len(failures))} of {len(failures)}):")
        step = max(1, len(failures) // max_failures)
        for record in failures[::step][:max_failures]:
            print(
                f"  {record['a']}*{record['b']}={record['truth']}, "
                f"got {record['prediction'] or '<empty>'}; "
                f"central={record['central_total']}, "
                f"train_distance={record['distance']}")

    return {
        "overall": hits / len(records),
        "exposure_accuracy_correlation": correlation,
        "commutative_same": same_answer / commutative_total,
        "commutative_both_correct": both_correct / commutative_total,
        "cases": len(records),
    }


def probe(model, cfg, device, stoi, itos, reverse, k, temp):
    """Sample each hard prompt k times at `temp`; report pass@k and the most
    common sampled answers. pass@k > 0 means the correct answer is reachable,
    so GRPO would have signal to reinforce; pass@k == 0 means it isn't."""
    import collections
    max_seq_len = int(cfg["model"]["max_seq_len"])
    print(f"pass@{k} at temp={temp} (correct answer reachable by sampling?)\n")
    print(f"{'prompt':<12}{'truth':<10}{'pass@k':<9}{'top sampled answers'}")
    for prompt in PROBE_PROMPTS:
        truth = prompt_truth(prompt)
        hits, counter = 0, collections.Counter()
        for _ in range(k):
            out = generate(model, prompt, stoi, itos, max_seq_len, device,
                           temp=temp, reverse=reverse).split('=')[-1]
            counter[out] += 1
            if out == truth:
                hits += 1
        top = ", ".join(f"{a}×{c}" for a, c in counter.most_common(3))
        flag = "  <-- reachable" if hits else ""
        print(f"{prompt:<12}{truth:<10}{hits/k:<9.3f}{top}{flag}")


def eval_model(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = pick_device()
    print(f'device: {device}')
    checkpoint = args.checkpoint or cfg["eval"]["checkpoint_path"]
    print(f'checkpoint: {checkpoint}')
    model = load_model(cfg, checkpoint, device)
    stoi, itos = V.build_vocab()
    max_seq_len = int(cfg["model"]["max_seq_len"])
    reverse = cfg["eval"].get(
        "internal_format", cfg["eval"].get("reverse", True))

    if args.mode in ('demo', 'both'):
        with open(args.data_path or cfg["eval"]['data_path']) as f:
            prompts = yaml.safe_load(f)['data']
        temp = cfg["eval"]['temp']
        for prompt in prompts:
            print(generate(model, prompt, stoi, itos, max_seq_len, device, temp,
                           reverse=reverse))

    if args.mode in ('report', 'both'):
        tier_report(model, cfg, device, stoi, itos, reverse)

    if args.mode == 'dataset':
        if not args.data_path:
            raise ValueError("--mode dataset requires --data-path")
        tier_report(
            model, cfg, device, stoi, itos, reverse,
            data_path=args.data_path, split=args.split,
            batch_size=args.batch_size)

    if args.mode == 'reachability':
        if not args.data_path:
            raise ValueError("--mode reachability requires --data-path")
        reachability_report(
            model, cfg, device, stoi, itos, reverse,
            data_path=args.data_path, split=args.split,
            k=args.k, temp=args.temp,
            sample_batch_size=args.batch_size, seed=args.seed,
            report_out=args.report_out, group_size=args.group_size,
            operations=args.operations)

    if args.mode == 'probe':
        probe(model, cfg, device, stoi, itos, reverse, args.k, args.temp)

    if args.mode == 'algorithmic':
        data_path = (args.data_path
                     or cfg["eval"].get("algorithmic_data_path")
                     or "config/pretrain_algorithmic_eval.yaml")
        algorithmic_report(
            model, cfg, device, stoi, itos, reverse, data_path,
            max_failures=args.max_failures)

    if args.mode == 'multiplication':
        multiplication_behavior_report(
            model, cfg, device, stoi, itos, reverse,
            samples_per_cell=args.samples_per_cell,
            batch_size=args.batch_size,
            seed=args.seed,
            max_failures=args.max_failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pretrain.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="override the checkpoint in the config (e.g. ckp/40.pt)")
    parser.add_argument(
        "--mode", choices=[
            'demo', 'report', 'both', 'probe', 'algorithmic', 'multiplication',
            'dataset', 'reachability'],
                        default='both',
        help="demo=prompt list, report=val tiers, probe=sampling, "
             "algorithmic=grouped greedy rule-transfer suite, "
             "multiplication=stratified three-digit behavior audit, "
             "dataset=grouped exact-match report for a JSONL split, "
             "reachability=stochastic pass@k and RLVR group diagnostics")
    parser.add_argument("--k", type=int, default=200, help="samples per prompt (probe)")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temp (probe)")
    parser.add_argument(
        "--data-path", default=None,
        help="override the YAML prompt suite for demo or algorithmic mode")
    parser.add_argument(
        "--split", default="test",
        help="JSONL split used by --mode dataset (default: test)")
    parser.add_argument(
        "--report-out", default=None,
        help="optional JSON output path for --mode reachability")
    parser.add_argument(
        "--operations", nargs="+", choices=V.OPERATORS, default=None,
        help="optional operation filter for dataset reachability")
    parser.add_argument(
        "--group-size", type=int, default=8,
        help="sample group size for all-wrong/mixed/all-correct diagnostics")
    parser.add_argument(
        "--max-failures", type=int, default=40,
        help="maximum failed prompts printed by diagnostic modes")
    parser.add_argument(
        "--samples-per-cell", type=int, default=50,
        help="fresh cases per hundreds-by-hundreds region (multiplication)")
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="greedy generation batch size (multiplication)")
    parser.add_argument("--seed", type=int, default=42,
                        help="diagnostic sampling seed")
    args = parser.parse_args()
    eval_model(args)


if '__main__' == __name__:
    main()
