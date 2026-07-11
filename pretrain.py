# type: ignore
import torch
import torch.nn as nn
import argparse
import json
import yaml
import math
from pathlib import Path
import vocab as V

from model import build_model
from torch.optim.lr_scheduler import LambdaLR

torch.manual_seed(42)


def load_examples(data_path, stoi, max_seq_len, split):
    pad_id = stoi[V.PAD]
    with open(data_path) as f:
        rows = (json.loads(line) for line in f)
        texts = [row["text"] for row in rows if row["split"] == split]
    if not texts:
        raise ValueError(f"no {split!r} examples found in {data_path}")

    encoded = []
    for text in texts:
        ids = [stoi[V.BOS]] + V.encode(text, stoi) + [stoi[V.EOS]]
        if len(ids) > max_seq_len:
            raise ValueError(
                f"{split} row {text!r} needs {len(ids)} tokens, but "
                f"max_seq_len={max_seq_len}; increase max_seq_len before training")
        ids += [pad_id] * (max_seq_len - len(ids))
        encoded.append(ids)

    all_ids = torch.tensor(encoded, dtype=torch.long)
    inp = all_ids[:, :-1]
    tgt = all_ids[:, 1:].clone()
    causal_mask = torch.tril(torch.ones(max_seq_len - 1, max_seq_len - 1)).unsqueeze(0)

    # We are training a calculator, not a model of randomly sampled input
    # expressions. Only score answer tokens (including EOS). In the shifted
    # tensors, the '=' position in inp predicts the first answer token.
    separators = inp.eq(stoi['='])
    missing_separator = ~separators.any(dim=1)
    if missing_separator.any():
        bad_index = missing_separator.nonzero(as_tuple=False)[0].item()
        raise ValueError(
            f"{split} row {texts[bad_index]!r} has no '=' token in the model "
            "input; check the vocabulary and max_seq_len")
    answer_positions = separators.cumsum(dim=1).gt(0)
    tgt.masked_fill_(~answer_positions, pad_id)
    return inp, tgt, causal_mask


def iter_batches(inputs, targets, causal_mask, batch_size, shuffle=False):
    """Create fresh batches; shuffled training examples are re-grouped each epoch."""
    indices = (torch.randperm(len(inputs)) if shuffle
               else torch.arange(len(inputs)))
    for batch_indices in indices.split(batch_size):
        yield inputs[batch_indices], targets[batch_indices], causal_mask


def train(args):

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    tc = cfg["train"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    stoi, itos = V.build_vocab()
    train_inputs, train_targets, train_mask = load_examples(
        tc["data_path"], stoi, mc["max_seq_len"], "train")
    val_inputs, val_targets, val_mask = load_examples(
        tc["data_path"], stoi, mc["max_seq_len"], "val")
    batch_size = int(tc["batch_size"])
    train_batch_count = math.ceil(len(train_inputs) / batch_size)
    val_batch_count = math.ceil(len(val_inputs) / batch_size)

    model = build_model(mc).to(device)

    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tc["lr"]),
        weight_decay=float(tc.get("weight_decay", 0.01)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi[V.PAD])

    total_steps = tc["epochs"] * train_batch_count
    warmup_steps = total_steps // 10  # 10% warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)                           # linear warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))               # cosine decay    
    scheduler = LambdaLR(optimizer, lr_lambda)
    exact_eval_every = int(tc.get("exact_eval_every", 0))
    target_accuracy = float(tc.get("target_accuracy", 1.01))
    checkpoint_path = Path(tc["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_min_accuracy = -1.0
    best_epoch = None

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for inp, tgt, mask in iter_batches(
                train_inputs, train_targets, train_mask, batch_size, shuffle=True):
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            logits = model(inp, mask)
            B, S, V_size = logits.shape
            loss = loss_fn(logits.reshape(B * S, V_size), tgt.reshape(B * S))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in iter_batches(
                    val_inputs, val_targets, val_mask, batch_size):
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                logits = model(inp, mask)
                B, S, V_size = logits.shape
                val_loss += loss_fn(logits.reshape(B * S, V_size), tgt.reshape(B * S)).item()
        save_every = int(tc.get("save_every", 0))
        if save_every and epoch % save_every == 0:
            epoch_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.epoch-{epoch}{checkpoint_path.suffix}")
            torch.save(model.state_dict(), epoch_path)
        print(f"epoch {epoch:>3}/{tc['epochs']}  train_loss={total_loss / train_batch_count:.4f}  val_loss={val_loss / val_batch_count:.4f}")

        if exact_eval_every and epoch % exact_eval_every == 0:
            # Exact greedy sequence accuracy is the actual objective. Validation
            # token loss can improve while a single wrong digit still makes the
            # whole arithmetic answer incorrect.
            from eval import tier_report
            metrics = tier_report(model, cfg, device, stoi, itos,
                                  reverse=cfg["eval"].get("reverse", True))
            category_accuracies = [*metrics["tier"].values(),
                                   *metrics["op"].values()]
            min_accuracy = min(category_accuracies)
            print(f"minimum category accuracy: {100 * min_accuracy:.1f}%")
            if min_accuracy > best_min_accuracy:
                best_min_accuracy = min_accuracy
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
                print(f"new best worst-category checkpoint -> {checkpoint_path}")
            if min_accuracy >= target_accuracy:
                print(f"target reached at epoch {epoch}")
                break

    if exact_eval_every:
        last_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.last{checkpoint_path.suffix}")
        torch.save(model.state_dict(), last_path)
        if best_epoch is None:
            torch.save(model.state_dict(), checkpoint_path)
            best_epoch = epoch
        print(f"best worst-category checkpoint: epoch {best_epoch} -> {checkpoint_path}")
        print(f"last checkpoint -> {last_path}")
    else:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved -> {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser()
    # Preserve `--mode train` for old commands, but evaluation now lives only in
    # eval.py so there cannot be two divergent decoding implementations.
    parser.add_argument("--mode", choices=['train'], default="train",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", default="config/pretrain.yaml")
    args = parser.parse_args()
    train(args)

if '__main__' == __name__:
    main()
