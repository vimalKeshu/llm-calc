# type: ignore
import torch
import torch.nn as nn
import argparse
import json
import yaml
import math
import vocab as V

from model import LLMCalcModel
from torch.optim.lr_scheduler import LambdaLR



def load_batches(data_path, stoi, max_seq_len, batch_size, split):
    pad_id = stoi[V.PAD]
    with open(data_path) as f:
        texts = [json.loads(line)["text"] for line in f if json.loads(line)["split"] == split]

    encoded = []
    for text in texts:
        ids = ([stoi[V.BOS]] + V.encode(text, stoi) + [stoi[V.EOS]])[:max_seq_len]
        ids += [pad_id] * (max_seq_len - len(ids))
        encoded.append(ids)

    all_ids = torch.tensor(encoded, dtype=torch.long)
    causal_mask = torch.tril(torch.ones(max_seq_len - 1, max_seq_len - 1)).unsqueeze(0)

    batches = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        batches.append((batch[:, :-1], batch[:, 1:], causal_mask))
    return batches


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

    stoi, _ = V.build_vocab()
    train_batches = load_batches(tc["data_path"], stoi, mc["max_seq_len"], tc["batch_size"], "train")
    val_batches   = load_batches(tc["data_path"], stoi, mc["max_seq_len"], tc["batch_size"], "val")

    model = LLMCalcModel(
        vocab_size=mc["vocab_size"],
        max_seq_len=mc["max_seq_len"],
        d_model=mc["d_model"],
        attention_heads=mc["attention_heads"],
        n_layers=mc["n_layers"],
        dropout=mc["dropout"],
    ).to(device)

    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]))
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi[V.PAD])

    total_steps = tc["epochs"] * len(train_batches)
    warmup_steps = total_steps // 10  # 10% warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)                           # linear warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))               # cosine decay    
    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for inp, tgt, mask in train_batches:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            logits = model(inp, mask)
            B, S, V_size = logits.shape
            loss = loss_fn(logits.reshape(B * S, V_size), tgt.reshape(B * S))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt, mask in val_batches:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                logits = model(inp, mask)
                B, S, V_size = logits.shape
                val_loss += loss_fn(logits.reshape(B * S, V_size), tgt.reshape(B * S)).item()
        torch.save(model.state_dict(), f'{epoch}.pt')
        print(f"epoch {epoch:>3}/{tc['epochs']}  train_loss={total_loss / len(train_batches):.4f}  val_loss={val_loss / len(val_batches):.4f}")

    torch.save(model.state_dict(), 'final.pt')
    print(f"saved -> {tc['checkpoint_path']}")


def eval(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)    

    mc = cfg["model"]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'device: {device}')

    model: nn.Module = LLMCalcModel(
                            vocab_size=mc["vocab_size"],
                            max_seq_len=mc["max_seq_len"],
                            d_model=mc["d_model"],
                            attention_heads=mc["attention_heads"],
                            n_layers=mc["n_layers"],
                            dropout=mc["dropout"],
                        ).to(device)
    model.load_state_dict(torch.load(cfg["train"]["checkpoint_path"], map_location=device))
    model.eval()
    stoi, itos = V.build_vocab()

    # tests = (["2+2=", 
    #           "2-2=", 
    #           "-2+2=", 
    #           "2*2=", 
    #           "100*2=", 
    #           "100/2=", 
    #           "9/2=", 
    #           "-5-2=", 
    #           "5-2=", 
    #           "-5+2=", 
    #           "333*333=", 
    #           "333*3=", 
    #           "333+333=", 
    #           "333/333="])

    tests = ([
      # --- carry propagation ---
      "999+1=",       # crosses digit boundary → 1000
      "999+999=",     # max addition → 1998
      "100-1=",       # borrow across zeros

      # --- large multiplication ---
      "999*999=",     # max → 998001
      "123*456=",     # irregular digits
      "500*200=",     # trailing zeros

      # --- subtraction producing negatives ---
      "1-999=",       # large negative result
      "100-999=",
      "5-9=",

      # --- negative operand all ops ---
      "-333+100=",
      "-333*3=",
      "-333/3=",
      "-999+999=",    # should be 0

      # --- decimal division (repeating) ---
      "1/3=",         # 0.333
      "2/3=",         # 0.667
      "10/3=",        # 3.333
      "100/7=",       # 14.286
      "1/7=",         # 0.143

      # --- zero edge cases ---
      "0*999=",
      "0/999=",
      "999*0=",
      "999-999=",     # should be 0

      # --- identity / trivial ---
      "999+0=",
      "999*1=",
      "999/1=",
      "999/999=",     # should be 1

      # --- cross-bucket hard ---
      "9*999=",       # 8991
      "999*9=",       # 8991 — symmetric, tests both orderings
      "99*99=",       # 9801
      "-99*99=",
    ])
    max_seq_len=int(mc["max_seq_len"])
    with torch.no_grad():
        for test in tests:
            tokens = [stoi[V.BOS]] + V.encode(test, stoi)

            while len(tokens) < max_seq_len:
                input_tensor = torch.tensor([tokens]).to(device)
                causal_mask = torch.tril(torch.ones(input_tensor.shape[1], input_tensor.shape[1])).unsqueeze(0).to(device)
                output_tensor = model(input_tensor, causal_mask)
                output_tensor = output_tensor[:, -1, :]   # (1, vocab_size)
                next_token = output_tensor.argmax(dim=-1) # the predicted token id
                # print(f'next_token: {next_token.item()} , end of sequence: {stoi[V.EOS]}')
                if next_token == stoi[V.EOS]:
                    break
                tokens.append(next_token.item())
            print(V.decode(tokens, itos))        
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'eval'], default="train")
    parser.add_argument("--config", default="pretrain.yaml")
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        eval(args)

if '__main__' == __name__:
    main()
