# type: ignore

import torch
import torch.nn as nn 
import math 

class WordEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) :
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # multiply by sqrt(d_model) to scale the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)

class RMSLayerNormalization(nn.Module):
    """RMS Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        mean = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(mean + self.eps)

class RotaryEmbedding(nn.Module):

    def __init__(self, max_seq_len: int, h_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert h_dim % 2 == 0, "h_dim must be even (dims are rotated in pairs)."
        # --- Ingredient 1: the frequencies, one per pair ---
        # arange(0, head_dim, 2) gives 0, 2, 4, ... -> one entry per pair.
        inv_freq = 1.0 / (base ** (torch.arange(0, h_dim, 2).float() / h_dim)) # inv_freq shape: (head_dim/2,)

        # --- Ingredient 2: the positions ---
        t = torch.arange(max_seq_len).float()          # (max_seq_len,)

        # --- Combine: angle = position * frequency ---
        freqs = torch.outer(t, inv_freq)               # (max_seq_len, head_dim/2)

        # --- Cache cos and sin (the only things rotation actually needs) ---
        # register_buffer => moves with .to(device), but is NOT a trainable parameter.
        # persistent=False => kept out of the saved state_dict (it's recomputable).
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)        

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, heads, seq, head_dim)
        seq = x.shape[-2]

        # Slice out only the positions we need, and add broadcast dims for batch+heads.
        cos = self.cos[:seq].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim/2)
        sin = self.sin[:seq].unsqueeze(0).unsqueeze(0)

        # Split into the adjacent pairs: (0,1), (2,3), ...
        x1 = x[..., 0::2]   # even indices -> first element of each pair
        x2 = x[..., 1::2]   # odd indices  -> second element of each pair

        # The 2D rotation of each pair by its angle:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave the pairs back into the original (0,1,2,3,...) layout.
        return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(x)

class MultiHeadAttentionBlock(nn.Module):
    """Causal self attention."""

    def __init__(self, max_seq_len: int, d_model: int, h: int, dropout: float = 0.5) -> None:
        super().__init__()
        assert d_model % h == 0, "Model dimension should be divisible by no. of attention heads."
        self.d_model = d_model
        self.heads = h 
        self.d_head = d_model // h

        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.rope = RotaryEmbedding(max_seq_len, self.d_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        shape:    (batch,  heads,  seq,  head_dim)
        positive:    0       1      2       3
        negative:   -4      -3     -2      -1
        """
        b, s, d = x.shape
        q: torch.Tensor = self.w_q(x) 
        k: torch.Tensor = self.w_k(x)
        v: torch.Tensor = self.w_v(x)

        q_heads = q.view(b, s, self.heads, self.d_head).transpose(1, 2)
        k_heads = k.view(b, s, self.heads, self.d_head).transpose(1, 2)
        v_heads = v.view(b, s, self.heads, self.d_head).transpose(1, 2)

        q_heads = self.rope(q_heads)        # <-- rotate Q
        k_heads = self.rope(k_heads)        # <-- rotate K

        attn_score = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask==0, value=float('-inf'))

        attn = torch.nn.functional.softmax(attn_score, dim=-1)
        attn = self.dropout(attn)
        attn_output = attn @ v_heads
        output = attn_output.transpose(1, 2).contiguous().reshape(b, s, d)

        return self.w_o(output)

class MLP(nn.Module):
    """Feed forward network. Also called MLP."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)      # expand
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)      # project back
        self.act = nn.GELU()                     # nonlinearity

    def forward(self, x: torch.Tensor):
        """(batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)"""
        x_temp = self.fc1(x)
        x_temp = self.act(x_temp)

        x_temp = self.dropout(x_temp)
        x_final = self.fc2(x_temp)

        return x_final

class DecoderBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, 
                 max_seq_len: int, 
                 d_model: int, 
                 attention_heads: int, 
                 d_ff:int, 
                 eps: float = 1e-6, 
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.layer_norm_1 = RMSLayerNormalization(d_model=d_model, eps=eps)
        self.layer_norm_2 = RMSLayerNormalization(d_model=d_model, eps=eps)
        self.attn = MultiHeadAttentionBlock(max_seq_len, d_model, attention_heads, dropout)
        # Keep attention and MLP regularisation under the same config value.
        # Previously MLP silently used its 0.5 default even when the model was
        # configured with dropout=0.05.
        self.ffn = MLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x_norm_1 = self.layer_norm_1(x)
        x_attn = self.attn(x_norm_1, mask)
        x_dropout_1 = self.dropout(x_attn)
        x = x + x_dropout_1

        x_norm_2 = self.layer_norm_2(x)
        x_ffn = self.ffn(x_norm_2)
        x_dropout_2 = self.dropout(x_ffn)
        x = x + x_dropout_2

        return x

class LLMCalcModel(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 max_seq_len: int, 
                 d_model: int, 
                 attention_heads: int,
                 n_layers: int,  
                 eps: float = 1e-6, 
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.embedding: nn.Module = WordEmbeddings(d_model, vocab_size)
        layers=[]
        for _ in range(n_layers):
            layers.append(DecoderBlock(max_seq_len, d_model, attention_heads, 4 * d_model, eps, dropout))
        self.decoder_layers = nn.ModuleList(layers)
        self.layer_norm = RMSLayerNormalization(d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.embedding(x)

        for decoder in self.decoder_layers:
            x = decoder(x, mask)

        x = self.layer_norm(x)
        x = self.projection(x)

        return x 


class LoopLlmCalc(nn.Module):
    """Looped (recursive) variant of the calculator model.

    Reuses the building blocks from model.py, but instead of stacking `n_layers`
    distinct decoder blocks it applies a small stack `n_loops` times, sharing the
    same weights across iterations. Effective depth = n_layers * n_loops, while the
    parameter count stays that of just `n_layers` blocks -- cheap depth for the
    sequential carry/borrow chains that hard arithmetic needs. A per-iteration
    `step_emb` lets each pass know which loop it is (Universal-Transformer style).
    """
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int,
                 attention_heads: int,
                 n_layers: int,
                 n_loops: int = 1,
                 eps: float = 1e-6,
                 dropout: float = 0.5) -> None:
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.n_loops = n_loops
        self.embedding = WordEmbeddings(d_model, vocab_size)
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(max_seq_len, d_model, attention_heads, 4 * d_model, eps, dropout)
             for _ in range(n_layers)])
        self.step_emb = nn.Embedding(n_loops, d_model)      # per-iteration signal
        self.register_buffer("loop_ids", torch.arange(n_loops), persistent=False)
        self.layer_norm = RMSLayerNormalization(d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for t in range(self.n_loops):
            x = x + self.step_emb(self.loop_ids[t])
            for decoder in self.decoder_layers:
                x = decoder(x, mask)
        x = self.layer_norm(x)
        return self.projection(x)

def build_model(mc: dict) -> nn.Module:
    """Build the looped model when the config sets n_loops > 1, else the plain
    LLMCalcModel. Lets pretrain.py / eval.py stay arch-agnostic."""
    common = dict(vocab_size=mc["vocab_size"], max_seq_len=mc["max_seq_len"],
                  d_model=mc["d_model"], attention_heads=mc["attention_heads"],
                  n_layers=mc["n_layers"], dropout=mc["dropout"])
    n_loops = mc.get("n_loops", 1)
    if n_loops > 1:
        return LoopLlmCalc(n_loops=n_loops, **common)
    return LLMCalcModel(**common)


if '__main__' == __name__:
    calc: LLMCalcModel = LLMCalcModel(vocab_size=32, 
                                      max_seq_len=8, 
                                      d_model=8, 
                                      attention_heads=2, 
                                      n_layers=4)
    # EMBEDDING:  
    #     32 * 8 = 256 
    # Decoder block:
    #     MHA: 8 * 8  = 64 * 4                = 256  
    #     FFN: (8 * 32 + 32) +  (32 * 8 + 8)  = 552 
    #     RMS: 8 = 8 * 2                      = 016
    #     -----------------------------------------
    #     Total                               = 824
    #     4 Decoder block:
    #     824 * 4 = 3296
    # Others:
    #     RMS: 8             = 008
    #     PROJECTION: 8 * 32 = 256
    #     -------------------------
    #     Total              = 264
    # Model params:
    # 0256 + 3296 + 0264 = 3816

    total = 0
    for p in calc.parameters():
        total += p.numel()
    print("total params:", total)    


    # m = LoopLlmCalc(vocab_size=21, max_seq_len=18, d_model=64,
    #                 attention_heads=4, n_layers=2, n_loops=4)
    # print("params:", f"{sum(p.numel() for p in m.parameters()):,}",
    #       "| effective depth:", m.n_loops * len(m.decoder_layers))



                
            






    
