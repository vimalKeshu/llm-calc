## Research basis

- [Transformers Can Do Arithmetic with the Right Embeddings](https://arxiv.org/pdf/2405.17399v2), McLeish et al., NeurIPS 2024.

## Representation rules

### Addition, subtraction, and multiplication

Reverse the magnitude digits of every operand and answer. Keep a negative sign
in front of the reversed magnitude.

```text
Normal expression       Internal training expression
123+45=168              321+54=861
500-123=377             005-321=773
123*45=5535             321*54=5355
-123+45=-78             -321+54=-87
```

Zeros created by reversal are significant digits, not padding. For example,
`500` must become `005`; stripping those zeros would make reverse-decoding
incorrect.

Do not add zero padding to variable-length integers beyond zeros naturally
created by reversal.

### Division

Keep division operands and answers in natural, most-significant-digit-first
order. This matches the direction of long division and gives decimal digits in
their normal order.

Use a canonical internal answer with three integer slots and exactly three
fractional slots:

```text
User-facing expression  Internal training expression
1/3=0.333               1/3=000.333
6/3=2                   6/3=002.000
10/4=2.5                10/4=002.500
-3/2=-1.5               -3/2=-001.500
```

This fixed answer shape lets inference assign decimal-place embeddings before
the complete answer has been generated. After decoding, strip leading integer
zeros and insignificant trailing fractional zeros for user-facing output.

Division remains rounded to three fractional digits. The internal decimal
point is always present, including for exact integer quotients.

### Signs and operators

- A negative sign stays before the magnitude; its digits alone are reversed.
- The subtraction operator is not moved or reversed.
- The existing tokenizer may continue mapping a unary `-` to the internal `~`
  token so that unary signs and binary subtraction remain distinct.
- Signs, operators, `=`, BOS, EOS, and padding do not receive digit-place
  embeddings.

## Unified place coordinate

Use a semantic decimal exponent `e` for each digit:

```text
Place              e
10^5               5
10^4               4
thousands          3
hundreds           2
tens               1
units              0
tenths            -1
hundredths        -2
thousandths       -3
```

For reversed integer operations, exponents increase as the string is read from
left to right:

```text
Internal number:  3   2   1       # represents 123
Exponent e:       0   1   2
Meaning:        units tens hundreds
```

For natural-order division numbers, calculate exponents relative to the decimal
point, or relative to an implicit decimal point after the final digit:

```text
Number:       9    6    8    .    1     2     3
Exponent e:  +2   +1    0   dot  -1    -2    -3
```

This gives the same units, tens, and hundreds coordinates to digits regardless
of whether their textual representation is reversed or natural.

The decimal point does not consume an ordinary digit position. It keeps its
normal `.` token embedding and receives a dedicated decimal-anchor embedding.

## Randomized Abacus offset

The paper samples an offset `beta` from a range controlled by hyperparameter
`k`. The offset is shared by all numbers in a training batch so equal decimal
places remain aligned.

For this decimal-aware extension, map exponent `e` to a learned table index:

```text
abacus_id = BASE + beta + e
```

Choose `BASE >= 4` so exponent `-3` remains positive when `beta = 0`. Reserve
index zero for non-numeric tokens.

Recommended initial values:

```text
BASE = 4
max integer exponent = 5     # six-digit multiplication answers
min fractional exponent = -3
k = 16
```

Training behavior:

- Sample one `beta` for the batch.
- Use the same `beta` for operands and answers and across all four operations.
- Because this project has much less data than the paper, use `beta = 0` for
  roughly half of training batches and sample from `[0, k]` otherwise. This
  ensures that the exact table entries used at inference receive enough
  updates.

Inference behavior:

- Always use `beta = 0`.
- Never sample or change the offset during generation.

## Why randomization may help longer numbers

A purely learned place embedding cannot use an embedding row that never
received training. Random offsets expose higher embedding rows even when
training operands are short.

If training operands have at most three digits, their highest operand exponent
is `2`. Testing five-digit operands requires exponent `4`. An offset range of at
least two can expose the relevant higher rows during training:

```text
training exponent 2 + beta 2
    uses the row needed by
test exponent 4 + beta 0
```

This creates an opportunity for length extrapolation, not a guarantee. Evaluate
three-, four-, and five-digit operands separately using exact answer match.

## Encoding and decoding summary

### Before training

1. Compute the normal arithmetic result.
2. For `+`, `-`, and `*`, reverse every operand and answer magnitude.
3. For `/`, keep operands natural and format the answer as `DDD.ddd`.
4. Tokenize the internal expression.
5. Produce a parallel place-ID tensor using the operation-aware rules above.
6. Apply one shared randomized offset during training.

### During generation

1. Detect the operator from the prompt.
2. Use `beta = 0`.
3. Assign reversed integer coordinates for `+`, `-`, and `*`.
4. Assign the known `DDD.ddd` output-place schedule for `/`.
5. Generate until EOS.

### User-facing decoding

- For `+`, `-`, and `*`, reverse the generated answer magnitude back to
  natural order while preserving its sign.
- For `/`, keep the generated answer in natural order, remove leading integer
  zeros, and remove insignificant trailing fractional zeros.
- Validate the final string before displaying it.
