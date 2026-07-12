import yaml
import torch
import argparse
import json
import collections
from functools import lru_cache
import vocab as V
from generate_data import (TIER_NAMES, compute, decode_internal_answer,
                           encode_prompt)
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
    op = next((candidate for candidate in V.OPERATORS
               if lhs.find(candidate, 1) != -1), None)
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
    op = next((o for o in V.OPERATORS if lhs.find(o, 1) != -1), None)
    return (decode_internal_answer(op, stored, internal_format=reverse)
            if op is not None else stored)


@lru_cache(maxsize=None)
def load_val_rows(data_path, expected_representation=None):
    """Parse and retain the small validation subset once per dataset path."""
    rows = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            if row["split"] == "val":
                if (expected_representation is not None and
                        row.get("representation") != expected_representation):
                    actual = row.get("representation", "missing")
                    raise ValueError(
                        f"{data_path} uses representation {actual!r}, expected "
                        f"{expected_representation!r}; regenerate the dataset")
                rows.append(row)
    return tuple(rows)


def tier_report(model, cfg, device, stoi, itos, reverse=True):
    """Greedy accuracy over the held-out val split, broken down by tier and op."""
    max_seq_len = int(cfg["model"]["max_seq_len"])
    expected_representation = (
        "abacus-v1" if cfg["model"].get("use_abacus", False) else None)
    rows = load_val_rows(
        cfg["train"]["data_path"], expected_representation)

    by_tier, ok_tier = collections.Counter(), collections.Counter()
    by_op, ok_op = collections.Counter(), collections.Counter()
    for r in rows:
        lhs = r["text"].split('=')[0]
        prompt = lhs + '='
        truth = true_answer(r["text"], reverse)
        pred = generate(model, prompt, stoi, itos, max_seq_len,
                        device, temp=0.0, reverse=reverse,
                        prompt_is_internal=True).split('=')[-1]
        hit = (pred == truth)
        by_tier[r["tier"]] += 1; ok_tier[r["tier"]] += hit
        op = next((o for o in V.OPERATORS if lhs.find(o, 1) != -1), None)
        if op:
            by_op[op] += 1; ok_op[op] += hit

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
    tot, okt = sum(by_tier.values()), sum(ok_tier.values())
    print(f"  overall    {okt:>4}/{tot:<4} = {100*okt/tot:5.1f}%")
    return {
        "tier": {t: ok_tier[t] / by_tier[t] for t in by_tier},
        "op": {op: ok_op[op] / by_op[op] for op in by_op},
        "overall": okt / tot,
    }


# prompts whose greedy answer is currently wrong (mult / division hard cases) --
# the exact cases where we want to know if the right answer is REACHABLE by
# sampling (i.e. whether RLVR/GRPO has any signal to sharpen).
PROBE_PROMPTS = [
    "999*999=", "111*999=", "888*999=", "777*777=", "997*998=", "123*456=",
    "10/3=", "2/3=", "100/7=", "1/7=", "999/999=", "500*200=",
]


def prompt_truth(prompt):
    """Ground-truth natural-order answer string for a prompt like '999*999='."""
    lhs = prompt.rstrip('=')
    op = next((o for o in V.OPERATORS if lhs.find(o, 1) != -1), None)
    i = lhs.find(op, 1)
    a, b = int(lhs[:i]), int(lhs[i+1:])
    r = compute(a, b, op)
    return None if r is None else str(r)


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
        with open(cfg["eval"]['data_path']) as f:
            prompts = yaml.safe_load(f)['data']
        temp = cfg["eval"]['temp']
        for prompt in prompts:
            print(generate(model, prompt, stoi, itos, max_seq_len, device, temp,
                           reverse=reverse))

    if args.mode in ('report', 'both'):
        tier_report(model, cfg, device, stoi, itos, reverse)

    if args.mode == 'probe':
        probe(model, cfg, device, stoi, itos, reverse, args.k, args.temp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/pretrain.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="override the checkpoint in the config (e.g. ckp/40.pt)")
    parser.add_argument("--mode", choices=['demo', 'report', 'both', 'probe'],
                        default='both',
                        help="demo=prompt list, report=per-tier acc, probe=pass@k")
    parser.add_argument("--k", type=int, default=200, help="samples per prompt (probe)")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temp (probe)")
    args = parser.parse_args()
    eval_model(args)


if '__main__' == __name__:
    main()
