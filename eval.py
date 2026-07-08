import yaml
import torch
import argparse
import os 
import vocab as V 
import torch
import yaml
from generate_pretrain_samples import verify
from model import LLMCalcModel
 

@torch.no_grad()
def generate(model, prompt, stoi, itos, max_seq_len, device, temp=0.0) -> str:
    tokens = [stoi[V.BOS]] + V.encode(prompt, stoi)
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

    return V.decode(tokens, itos)

def probe(args):
    pass


def eval_model(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'device: {device}')
    mc = cfg["model"]
    eval_config = cfg["eval"]
    model: torch.nn.Module = LLMCalcModel(
                            vocab_size=mc["vocab_size"],
                            max_seq_len=mc["max_seq_len"],
                            d_model=mc["d_model"],
                            attention_heads=mc["attention_heads"],
                            n_layers=mc["n_layers"],
                            dropout=mc["dropout"],
                        ).to(device)
    model.load_state_dict(torch.load(eval_config["checkpoint_path"], map_location=device))
    model.eval()    

    with open(eval_config['data_path']) as f:
        eval_data = yaml.safe_load(f)  

    prompts = eval_data['data']
    temp = eval_config['temp']
    max_seq_len=int(mc["max_seq_len"])
    stoi, itos = V.build_vocab()

    for prompt in prompts:
        ans = generate(model, prompt, stoi, itos, max_seq_len, device, temp)
        print(ans)    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pretrain.yaml")
    args = parser.parse_args()
    eval_model(args) 


if '__main__' == __name__:
    main()