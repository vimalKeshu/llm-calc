# type: ignore

import torch
import torch.nn as nn 
import math 
import vocab as V

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


class RandomizedAbacusEmbedding(nn.Module):
    """Digit-place embeddings for the calculator's mixed number formats.

    Integer +, -, and * samples are expected to contain reversed magnitudes, so
    digit exponents increase from left to right inside every digit span:
    ``321 -> [units, tens, hundreds] -> [0, 1, 2]``.

    Division samples remain in natural order. Their digit exponents are derived
    from the decimal point (or an implicit decimal point after an integer). The
    answer is expected to use the fixed ``DDD.ddd`` format, which lets partial
    autoregressive answer prefixes receive stable place IDs before the decimal
    point has been generated.

    During training a single scalar beta is shared by the whole batch. Shifting
    every digit by the same beta trains higher embedding rows without breaking
    the equality between corresponding arithmetic columns. Evaluation always
    uses beta=0.
    """

    def __init__(self,
                 d_model: int,
                 digit_token_ids: list[int],
                 division_token_id: int,
                 equals_token_id: int,
                 decimal_token_id: int,
                 base: int = 4,
                 max_offset: int = 16,
                 min_decimal_exponent: int = -3,
                 max_integer_exponent: int = 5,
                 zero_offset_probability: float = 0.5) -> None:
        super().__init__()
        if base + min_decimal_exponent < 1:
            raise ValueError(
                "base must keep the smallest decimal place above reserved ID 0")
        if max_offset < 0:
            raise ValueError("max_offset must be >= 0")
        if not 0.0 <= zero_offset_probability <= 1.0:
            raise ValueError("zero_offset_probability must be in [0, 1]")

        self.d_model = d_model
        self.base = base
        self.max_offset = max_offset
        self.min_decimal_exponent = min_decimal_exponent
        self.max_integer_exponent = max_integer_exponent
        self.zero_offset_probability = zero_offset_probability
        self.division_token_id = division_token_id
        self.equals_token_id = equals_token_id
        self.decimal_token_id = decimal_token_id

        # ID 0 is reserved for tokens which are not digits. max_offset also
        # provides unused-at-beta=0 rows for later length-extrapolation tests.
        table_size = base + max_integer_exponent + max_offset + 1
        self.embedding = nn.Embedding(table_size, d_model, padding_idx=0)
        self.decimal_anchor = nn.Embedding(2, d_model, padding_idx=0)
        self.register_buffer(
            "digit_token_ids", torch.tensor(digit_token_ids), persistent=False)

    def _digit_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.isin(input_ids, self.digit_token_ids)

    @staticmethod
    def _previous(mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)

    @staticmethod
    def _next(mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)

    def _reversed_integer_exponents(
            self, digit_mask: torch.Tensor) -> torch.Tensor:
        """Return 0,1,2,... inside each reversed contiguous digit span."""
        _, seq_len = digit_mask.shape
        indices = torch.arange(
            seq_len, device=digit_mask.device).unsqueeze(0).expand_as(digit_mask)
        starts = digit_mask & ~self._previous(digit_mask)
        last_start = torch.where(
            starts, indices, torch.zeros_like(indices)).cummax(dim=1).values
        return indices - last_start

    def _natural_decimal_exponents(
            self,
            input_ids: torch.Tensor,
            digit_mask: torch.Tensor,
            division_rows: torch.Tensor) -> torch.Tensor:
        """Return decimal exponents for natural-order division numbers."""
        _, seq_len = input_ids.shape
        indices = torch.arange(
            seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        decimal_mask = input_ids.eq(self.decimal_token_id)
        numeric_mask = digit_mask | decimal_mask

        starts = numeric_mask & ~self._previous(numeric_mask)
        ends = numeric_mask & ~self._next(numeric_mask)
        start_indices = torch.where(
            starts, indices, torch.zeros_like(indices)).cummax(dim=1).values

        end_candidates = torch.where(
            ends, indices, torch.full_like(indices, seq_len))
        end_indices = torch.cummin(
            end_candidates.flip(1), dim=1).values.flip(1)

        previous_dot = torch.where(
            decimal_mask, indices, torch.full_like(indices, -1)
        ).cummax(dim=1).values
        next_dot_candidates = torch.where(
            decimal_mask, indices, torch.full_like(indices, seq_len))
        next_dot = torch.cummin(
            next_dot_candidates.flip(1), dim=1).values.flip(1)

        previous_dot_is_local = (
            (previous_dot >= start_indices) & (previous_dot <= end_indices))
        next_dot_is_local = (
            (next_dot >= start_indices) & (next_dot <= end_indices))
        decimal_indices = torch.where(
            previous_dot_is_local,
            previous_dot,
            torch.where(next_dot_is_local, next_dot, end_indices + 1),
        )

        # Digits to the left of the decimal have exponents 0,1,2,... when
        # counted from right to left. Digits to its right use -1,-2,-3,...
        exponents = torch.where(
            indices < decimal_indices,
            decimal_indices - indices - 1,
            decimal_indices - indices,
        )

        # During autoregressive division the decimal point is not visible when
        # the first answer digits are embedded. The agreed DDD.ddd format gives
        # them a known schedule: +2,+1,0,dot,-1,-2,-3.
        equals_seen = input_ids.eq(self.equals_token_id).cumsum(dim=1).gt(0)
        answer_numeric = numeric_mask & equals_seen & division_rows
        answer_ordinal = answer_numeric.cumsum(dim=1) - 1
        scheduled_answer_exponents = torch.where(
            answer_ordinal <= 2,
            2 - answer_ordinal,
            3 - answer_ordinal,
        )
        return torch.where(
            answer_numeric & digit_mask, scheduled_answer_exponents, exponents)

    def _sample_beta(self, device: torch.device) -> torch.Tensor:
        if not self.training or self.max_offset == 0:
            return torch.zeros((), dtype=torch.long, device=device)

        sampled = torch.randint(
            0, self.max_offset + 1, (), dtype=torch.long, device=device)
        force_zero = torch.rand((), device=device) < self.zero_offset_probability
        return torch.where(force_zero, torch.zeros_like(sampled), sampled)

    def position_ids(
            self,
            input_ids: torch.Tensor,
            beta: int | torch.Tensor | None = None) -> torch.Tensor:
        """Build the learned-embedding lookup IDs for a token batch.

        ``beta`` is exposed for deterministic tests and diagnostics. Normal
        training and inference should omit it.
        """
        digit_mask = self._digit_mask(input_ids)
        division_rows = input_ids.eq(
            self.division_token_id).any(dim=1, keepdim=True)

        reversed_exponents = self._reversed_integer_exponents(digit_mask)
        decimal_exponents = self._natural_decimal_exponents(
            input_ids, digit_mask, division_rows)
        exponents = torch.where(
            division_rows, decimal_exponents, reversed_exponents)

        if beta is None:
            beta_tensor = self._sample_beta(input_ids.device)
        else:
            beta_tensor = torch.as_tensor(
                beta, dtype=torch.long, device=input_ids.device)
            if beta_tensor.numel() != 1:
                raise ValueError("beta must be a scalar")

        digit_position_ids = self.base + beta_tensor + exponents
        position_ids = torch.where(
            digit_mask, digit_position_ids, torch.zeros_like(digit_position_ids))

        # Fail with a representation-specific error instead of an opaque
        # embedding lookup failure if data violates the configured place range.
        invalid = digit_mask & (
            (position_ids <= 0) | (position_ids >= self.embedding.num_embeddings))
        if torch.any(invalid):
            invalid_id = position_ids[invalid][0].item()
            raise ValueError(
                f"Abacus position ID {invalid_id} is outside [1, "
                f"{self.embedding.num_embeddings - 1}]; check number format "
                "or increase the configured Abacus range")
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = self.position_ids(input_ids)
        place_features = self.embedding(positions)
        decimal_features = self.decimal_anchor(
            input_ids.eq(self.decimal_token_id).long())
        return (place_features + decimal_features) * math.sqrt(self.d_model)

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
                 dropout: float = 0.5,
                 abacus_config: dict | None = None) -> None:
        super().__init__()
        self.embedding: nn.Module = WordEmbeddings(d_model, vocab_size)
        self.abacus = (
            RandomizedAbacusEmbedding(d_model=d_model, **abacus_config)
            if abacus_config is not None else None)
        layers=[]
        for _ in range(n_layers):
            layers.append(DecoderBlock(max_seq_len, d_model, attention_heads, 4 * d_model, eps, dropout))
        self.decoder_layers = nn.ModuleList(layers)
        self.layer_norm = RMSLayerNormalization(d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        token_ids = x
        x = self.embedding(token_ids)
        if self.abacus is not None:
            x = x + self.abacus(token_ids)

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
                 dropout: float = 0.5,
                 abacus_config: dict | None = None) -> None:
        super().__init__()
        if n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        self.n_loops = n_loops
        self.embedding = WordEmbeddings(d_model, vocab_size)
        self.abacus = (
            RandomizedAbacusEmbedding(d_model=d_model, **abacus_config)
            if abacus_config is not None else None)
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(max_seq_len, d_model, attention_heads, 4 * d_model, eps, dropout)
             for _ in range(n_layers)])
        self.step_emb = nn.Embedding(n_loops, d_model)      # per-iteration signal
        self.register_buffer("loop_ids", torch.arange(n_loops), persistent=False)
        self.layer_norm = RMSLayerNormalization(d_model)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        token_ids = x
        x = self.embedding(token_ids)
        if self.abacus is not None:
            x = x + self.abacus(token_ids)
        for t in range(self.n_loops):
            x = x + self.step_emb(self.loop_ids[t])
            for decoder in self.decoder_layers:
                x = decoder(x, mask)
        x = self.layer_norm(x)
        return self.projection(x)

def build_model(mc: dict) -> nn.Module:
    """Build the looped model when the config sets n_loops > 1, else the plain
    LLMCalcModel. Lets pretrain.py / eval.py stay arch-agnostic."""
    abacus_config = None
    if mc.get("use_abacus", False):
        stoi, _ = V.build_vocab()
        abacus_config = dict(
            digit_token_ids=[stoi[digit] for digit in V.DIGITS],
            division_token_id=stoi['/'],
            equals_token_id=stoi['='],
            decimal_token_id=stoi['.'],
            base=int(mc.get("abacus_base", 4)),
            max_offset=int(mc.get("max_abacus_offset", 16)),
            min_decimal_exponent=int(
                mc.get("min_abacus_decimal_exponent", -3)),
            max_integer_exponent=int(
                mc.get("max_abacus_integer_exponent", 5)),
            zero_offset_probability=float(
                mc.get("abacus_zero_offset_probability", 0.5)),
        )

    common = dict(vocab_size=mc["vocab_size"], max_seq_len=mc["max_seq_len"],
                  d_model=mc["d_model"], attention_heads=mc["attention_heads"],
                  n_layers=mc["n_layers"], dropout=mc["dropout"],
                  abacus_config=abacus_config)
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



                
            






    
