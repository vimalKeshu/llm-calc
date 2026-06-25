# type: ignore

# --- Special control tokens ---------------------------------------------
PAD = "<pad>" # padding so sequences in a batch have equal length
BOS = "<bos>" # beginning of sequence
EOS = "<eos>" # end of sequence -> teaches the model when to STOP
UNK = "<unk>" # fallback for any character not in the vocabulary

# Keep <pad> first so PAD == index 0.
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

# --- Content tokens ---------------------------------------------
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # 0-9 as character
OPERATORS = ['+', '-', '/', '*'] # math ops
SYMBOLS = ['=', '.'] # '=' separates question/answer; '.' for decimals


# --- Full ordered token list. ---------------------------------------------
TOKENS = SPECIAL_TOKENS + DIGITS + OPERATORS + SYMBOLS


def build_vocab() -> tuple[dict[str,int], dict[int,str]]:
    """Return (stoi, itos) lookup dicts."""
    stoi = {tok:i for i, tok in enumerate(TOKENS)}
    itos = {i:tok for i, tok in enumerate(TOKENS)}
    return (stoi, itos)

def encode(input: str, stoi: dict[str, int]) -> list[int]:
    """Turn a string like '12+34=46' into a list of token ids.
    Unknown characters map to <unk> instead of crashing.
    """
    ids: list[int] = [ stoi.get(ch, UNK) for ch in input ]
    return ids

def decode(ids: list[int], itos: dict[int, str]) -> str:
    """Turn a list of token ids back into a string.
    """
    chars: list[str] = []
    for id in ids:
        token: str = itos.get(id, UNK)
        if token == PAD or token == BOS or token == EOS or token==UNK:
            continue
        chars.append(token)
    return "".join(chars) 
        
# Convenience constants you'll reuse in later steps (dataset, model, training).
VOCAB_SIZE = len(TOKENS)

if '__main__' == __name__:
    stoi, itos = build_vocab()
    print(f"Vocab size: {VOCAB_SIZE}")
    print("Token -> id:")
    for tok, i in stoi.items():
        print(f"  {i:>2}  {tok!r}")

    sample = "128*64=8192"
    ids = encode(sample, stoi)
    print(f"\nEncode {sample!r} -> {ids}")
    print(f"Decode back        -> {decode(ids, itos)!r}")    


