# type: ignore
"""
Curriculum-aware arithmetic data generator.

Design objective
----------------
Produce a training set that teaches a small model arithmetic *easy -> hard*, with:
  1. Enough coverage of every difficulty region that hard cases GENERALIZE
     (not memorize) -- especially the long carry/borrow chains that a
     left-to-right char model struggles with (999*999, 111*999, ...).
  2. A difficulty TAG on every row (tier 0..4 + a fine score) so the training
     loop can shuffle OR schedule a curriculum without regenerating.
  3. A held-out eval set that is provably DISJOINT from train, so "passing"
     a hard case measures generalization, not leakage.

Internal number format
----------------------
For addition, subtraction, and multiplication, every operand and answer
magnitude is reversed so the units digit comes first. Signs stay in front. For
example, 123+45=168 is stored as 321+54=861, and 500-123=377 is stored as
005-321=773. Zeros created by reversal are meaningful and must be preserved.

Division stays in natural order and uses a fixed DDD.ddd answer, e.g.
1/3=000.333 and 6/3=002.000. This makes decimal-place embeddings stable during
autoregressive generation. Evaluation converts all internal answers back to
normal user-facing strings.
"""
import argparse
import json
import random
from itertools import product

import vocab as V

# --- operand domain -----------------------------------------------------------
MAX = 999                                   # operands are at most 3 digits
DIGIT_RANGES = {1: (0, 9), 2: (10, 99), 3: (100, 999)}
TIER_NAMES = {0: "trivial", 1: "easy", 2: "medium", 3: "hard", 4: "hardest"}


# --- core arithmetic ----------------------------------------------------------
def compute(a, b, op):
    """Return the numeric result (int, or float rounded to 3dp for division),
    or None if the expression is undefined (division by zero)."""
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    # op == '/'
    if b == 0:
        return None
    r = round(a / b, 3)
    return int(r) if r == int(r) else r


def is_rounded_div(a, b, result):
    """True when a/b did not terminate cleanly and had to be rounded
    (i.e. a 'hard' repeating decimal like 1/3, 100/7)."""
    return abs(a / b - result) > 1e-9


def reverse_magnitude(value):
    """Reverse magnitude digits while preserving an optional leading sign."""
    s = str(value)
    neg = s.startswith('-')
    mag = s[1:] if neg else s
    return ('-' if neg else '') + mag[::-1]


def unreverse_magnitude(value):
    """Inverse of :func:`reverse_magnitude`."""
    neg = value.startswith('-')
    mag = value[1:] if neg else value
    return ('-' if neg else '') + mag[::-1]


def reverse_answer(result):
    """Backward-compatible name for reversing an integer answer magnitude."""
    return reverse_magnitude(result)


def unreverse_answer(ans):
    """Backward-compatible name for decoding a reversed integer answer."""
    return unreverse_magnitude(ans)


def should_reverse(op):
    """Integer column algorithms use the paper's units-first representation."""
    return op in ('+', '-', '*')


def format_division_answer(result):
    """Format a rounded quotient as the fixed internal DDD.ddd representation."""
    value = float(result)
    sign = '-' if value < 0 else ''
    magnitude = abs(value)
    if magnitude >= 1000:
        raise ValueError(
            f"division result {result!r} does not fit the DDD.ddd format")
    return sign + f"{magnitude:07.3f}"


def normalize_division_answer(answer):
    """Convert an internal DDD.ddd quotient to a user-facing decimal string."""
    try:
        value = float(answer)
    except ValueError:
        return answer
    if value == 0:
        return '0'
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip('0').rstrip('.')


def encode_prompt(prompt, internal_format=True):
    """Convert a natural prompt such as ``123+45=`` to the model's format."""
    if not internal_format:
        return prompt
    suffix = '=' if prompt.endswith('=') else ''
    lhs = prompt[:-1] if suffix else prompt
    op = next((candidate for candidate in V.OPERATORS
               if lhs.find(candidate, 1) != -1), None)
    if op is None:
        raise ValueError(f"could not find arithmetic operator in {prompt!r}")
    op_index = lhs.find(op, 1)
    left, right = lhs[:op_index], lhs[op_index + 1:]
    if should_reverse(op):
        left = reverse_magnitude(left)
        right = reverse_magnitude(right)
    return f"{left}{op}{right}{suffix}"


def decode_internal_answer(op, answer, internal_format=True):
    """Convert a generated/stored internal answer to normal user-facing form."""
    if not internal_format:
        return answer
    if should_reverse(op):
        return unreverse_magnitude(answer)
    if op == '/':
        return normalize_division_answer(answer)
    return answer


def make_text(a, b, op, result, reverse=True):
    """Create a natural expression or the agreed Abacus internal expression."""
    if not reverse:
        return f"{a}{op}{b}={result}"
    if should_reverse(op):
        return (f"{reverse_magnitude(a)}{op}{reverse_magnitude(b)}="
                f"{reverse_magnitude(result)}")
    return f"{a}{op}{b}={format_division_answer(result)}"


def verify(a, b, op, result):
    """Independent recomputation as a safety net against generator bugs."""
    expected = compute(a, b, op)
    if expected is None or result is None:
        return False
    return abs(float(expected) - float(result)) < 1e-6


# --- carry / borrow chain length (the real difficulty driver) -----------------
def _carries(x, y):
    """Number of carry positions when adding non-negative x + y."""
    c = carry = 0
    while x or y:
        s = x % 10 + y % 10 + carry
        carry = 1 if s >= 10 else 0
        c += carry
        x //= 10
        y //= 10
    return c


def _borrows(x, y):
    """Number of borrow positions when subtracting x - y, x >= y >= 0."""
    b = borrow = 0
    while y or borrow:
        d = x % 10 - y % 10 - borrow
        borrow = 1 if d < 0 else 0
        b += borrow
        x //= 10
        y //= 10
    return b


def additive_chain(a, b, op):
    """Carry/borrow chain length for + and -, accounting for operand signs."""
    u, v = a, (b if op == '+' else -b)
    if (u >= 0) == (v >= 0):                 # magnitudes add
        return _carries(abs(u), abs(v))
    return _borrows(max(abs(u), abs(v)), min(abs(u), abs(v)))  # magnitudes subtract


def ndigits(x):
    return len(str(abs(x)))


# --- difficulty classification ------------------------------------------------
def classify(a, b, op, result):
    """Return (tier:int 0..4, score:float). score = tier + a within-tier
    fraction so rows can be finely ordered for curriculum scheduling."""
    la, lb = ndigits(a), ndigits(b)
    neg = (result is not None) and (float(result) < 0)

    # ---- tier 0: trivial / identity / degenerate ----
    if op in '+-' and (a == 0 or b == 0):
        return 0, 0.0
    if op == '-' and a == b:                              # a - a = 0
        return 0, 0.0
    if op == '*' and (a in (0, 1) or b in (0, 1)):
        return 0, 0.0
    if op == '/' and (a == 0 or b == 1):
        return 0, 0.0

    if op in '+-':
        m = max(la, lb)
        ch = additive_chain(a, b, op)
        boundary = abs(result) >= 1000                   # crosses the 3-digit wall
        if m <= 1:
            tier = 1
        elif m == 2 or min(la, lb) == 1:
            tier = 2
        elif ch <= 2 and not boundary and not neg:
            tier = 3
        else:
            tier = 4
        sub = min(0.99, (ch + (1 if boundary else 0) + (1 if neg else 0)) / 6)
        return tier, tier + sub

    if op == '*':
        hi, lo = max(la, lb), min(la, lb)
        if hi == 1:
            tier = 1
        elif hi == 2 and lo == 1:
            tier = 2
        elif (hi == 2 and lo == 2) or (hi == 3 and lo == 1):
            tier = 3
        else:                                            # 3x2, 3x3
            tier = 4
        big = (abs(a) >= 800 and abs(b) >= 800)          # the hardest carry corner
        sub = min(0.99, (la * lb + (3 if big else 0) + (1 if neg else 0)) / 14)
        return tier, tier + sub

    # op == '/'
    hard_dec = is_rounded_div(a, b, result)
    if la <= 1 and not hard_dec:
        tier = 1
    elif la == 2 and not hard_dec:
        tier = 2
    elif (la == 3 and not hard_dec) or (hard_dec and la <= 2):
        tier = 3
    else:                                                # hard repeating, 3-digit
        tier = 4
    sub = min(0.99, ((3 if hard_dec else 0) + lb + (1 if neg else 0)) / 8)
    return tier, tier + sub


# --- candidate generation -----------------------------------------------------
def rand_by_digits(d):
    lo, hi = DIGIT_RANGES[d]
    return random.randint(lo, hi)


def draw_candidate(neg_ratio):
    """Region-weighted random (a, b, op) covering the whole space, tilted toward
    larger operands so the hard tiers fill. Sign applied to `a` only (the vocab
    only encodes a leading '-' as negative; `b` stays non-negative)."""
    op = random.choices(['+', '-', '*', '/'], weights=[3, 3, 4, 3])[0]
    da = random.choices([1, 2, 3], weights=[1, 3, 6])[0]
    db = random.choices([1, 2, 3], weights=[1, 3, 6])[0]
    a = rand_by_digits(da)
    b = rand_by_digits(db)
    if op == '/':
        b = max(1, b)
    if a != 0 and random.random() < neg_ratio:
        a = -a
    return a, b, op


def draw_balanced(neg_ratio):
    """Uniform draw across ops and digit-lengths -- used to build diverse pools
    (the final tier *frequencies* are set separately in compose())."""
    op = random.choice(['+', '-', '*', '/'])
    a = rand_by_digits(random.choice([1, 2, 3]))
    b = rand_by_digits(random.choice([1, 2, 3]))
    if op == '/':
        b = max(1, b)
    if a != 0 and random.random() < neg_ratio:
        a = -a
    return a, b, op


def easy_pool(neg_ratio):
    """Exhaustively enumerate single-digit operand pairs so the 'easy' tier has
    full unique coverage (its space is tiny and random draws rarely hit it)."""
    out = []
    for a in range(0, 10):
        for b in range(0, 10):
            for op in V.OPERATORS:
                out.append((a, b, op))
                if a != 0:
                    out.append((-a, b, op))
    return out


def trivial_pool(neg_ratio):
    """Enumerate identities, zeros, and self-subtraction across all digit lengths."""
    out = []
    reps = [1, 5, 9, 10, 50, 99, 100, 500, 999]
    for v in reps:
        for signed in ({v, -v} if v else {v}):
            out += [(signed, 0, '+'), (signed, 0, '-'),
                    (0, v, '+'), (signed, 1, '*'), (signed, 0, '*'),
                    (0, v, '*'), (signed, 1, '/'), (v, v, '-')]
    return out


def boundary_pool():
    """Additive expressions that cross the 3-digit wall (hardest carry/borrow)."""
    out = []
    for x in [1, 2, 5, 9, 10, 50, 99, 100, 500, 900, 990, 999]:
        out += [(1000 - x, x, '+') if 1000 - x <= 999 else (999, x, '+'),
                (999, x, '+'), (1000 - x if 1000 - x <= 999 else 999, x, '-')]
        out += [(x, 1000 - x, '-'), (100, x, '-'), (100 + x, x, '-')]
    return [(a, b, op) for (a, b, op) in out if 0 <= a <= 999 and 0 <= b <= 999]


def hard_corner_pool(count, neg_ratio):
    """Dense coverage of the >=800 x >=800 multiply corner where the longest
    carry chains live (999*999 etc.)."""
    out = []
    for _ in range(count):
        a = random.randint(800, 999)
        b = random.randint(800, 999)
        if random.random() < neg_ratio:
            a = -a
        out.append((a, b, '*'))
    return out


# --- held-out eval set (built FIRST, then excluded from train) ----------------
HARD_SUITE = [
    (999, 1, '+'), (999, 999, '+'), (100, 1, '-'), (999, 999, '*'),
    (123, 456, '*'), (500, 200, '*'), (1, 999, '-'), (100, 999, '-'),
    (5, 9, '-'), (-333, 100, '+'), (-333, 3, '*'), (-333, 3, '/'),
    (-999, 999, '+'), (1, 3, '/'), (2, 3, '/'), (10, 3, '/'), (100, 7, '/'),
    (1, 7, '/'), (0, 999, '*'), (0, 999, '/'), (999, 0, '*'), (999, 999, '-'),
    (999, 0, '+'), (999, 1, '*'), (999, 1, '/'), (999, 999, '/'), (9, 999, '*'),
    (999, 9, '*'), (99, 99, '*'), (-99, 99, '*'), (111, 999, '*'),
    (888, 999, '*'), (777, 777, '*'), (997, 998, '*'),
]


def build_eval(per_tier, neg_ratio):
    """The explicit hard suite plus a random per-tier held-out slice. Returns a
    list of (a,b,op) and the exclusion set of those triples."""
    eval_set = []
    seen = set()

    def add(a, b, op):
        r = compute(a, b, op)
        if r is None or not verify(a, b, op, r):
            return
        key = (a, b, op)
        if key in seen:
            return
        seen.add(key)
        eval_set.append(key)

    for a, b, op in HARD_SUITE:
        add(a, b, op)

    tier_counts = {t: 0 for t in range(5)}
    attempts = 0
    while min(tier_counts.values()) < per_tier and attempts < per_tier * 5000:
        attempts += 1
        a, b, op = draw_candidate(neg_ratio)
        r = compute(a, b, op)
        if r is None:
            continue
        t, _ = classify(a, b, op, r)
        if tier_counts[t] < per_tier and (a, b, op) not in seen:
            add(a, b, op)
            tier_counts[t] += 1
    return eval_set, seen


# --- assembling the train set -------------------------------------------------
# Optimise the worst category rather than average accuracy. The easy categories
# saturate early, so most examples are allocated to the two categories that
# require algorithmic generalisation. Sampling is also balanced by operation
# inside every tier; the old tier-only sampling left some tier/op cells with as
# few as 4.6k examples while others had 54k.
TIER_DIST = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.30, 4: 0.40}


def build_pools(n, exclude, neg_ratio, corner_frac=0.04):
    """Build a DIVERSE unique sample pool per tier (disjoint from `exclude`).
    Frequency is handled later in compose(); here we only maximise coverage."""
    pools = {t: {} for t in range(5)}        # tier -> {(a,b,op): result}

    def add(a, b, op):
        key = (a, b, op)
        if key in exclude:
            return
        r = compute(a, b, op)
        if r is None or not verify(a, b, op, r):
            return
        t, _ = classify(a, b, op, r)
        pools[t].setdefault(key, r)

    # structured seeds: guarantee full coverage of the small / rare regions
    for a, b, op in trivial_pool(neg_ratio):
        add(a, b, op)
    for a, b, op in easy_pool(neg_ratio):
        add(a, b, op)
    for a, b, op in boundary_pool():
        add(a, b, op)
    for a, b, op in hard_corner_pool(int(n * corner_frac), neg_ratio):
        add(a, b, op)

    # balanced random fill for pool variety in the larger tiers
    variety = max(30000, n // 8)
    attempts = 0
    while attempts < n * 8 and any(len(pools[t]) < variety for t in (2, 3, 4)):
        attempts += 1
        add(*draw_balanced(neg_ratio))
    return pools


def compose(pools, n):
    """Sample `n` rows, balanced by op within the hard-focused tier mix."""
    rows = []
    tier_targets = {t: int(n * TIER_DIST[t]) for t in range(5)}
    tier_targets[4] += n - sum(tier_targets.values())
    for t, target in tier_targets.items():
        per_op, remainder = divmod(target, len(V.OPERATORS))
        for op_index, op in enumerate(V.OPERATORS):
            items = [(key, result) for key, result in pools[t].items()
                     if key[2] == op]
            if not items:
                raise RuntimeError(f"empty data pool for tier={t}, op={op}")
            count = per_op + (op_index < remainder)
            for (a, b, sampled_op), result in random.choices(items, k=count):
                rows.append((a, b, sampled_op, result))
    random.shuffle(rows)
    return rows


# --- reporting ----------------------------------------------------------------
def report(rows):
    from collections import Counter
    tiers = Counter(r["tier"] for r in rows)
    ops = Counter(r["text"].split('=')[0][1:].lstrip('0123456789')[:1]
                  if False else None for r in rows)  # placeholder, computed below
    op_c = Counter()
    for r in rows:
        lhs = r["text"].split('=')[0]
        for op in V.OPERATORS:
            if lhs.find(op, 1) != -1:
                op_c[op] += 1
                break
    print("tier distribution:", {TIER_NAMES[t]: tiers[t] for t in range(5)})
    print("op   distribution:", dict(op_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400000, help="train samples")
    ap.add_argument("--eval-per-tier", type=int, default=150)
    ap.add_argument("--neg-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sample/pretrain.jsonl")
    ap.add_argument("--no-reverse", action="store_true",
                    help="disable the Abacus internal number format")
    args = ap.parse_args()
    random.seed(args.seed)
    reverse = not args.no_reverse

    eval_set, exclude = build_eval(args.eval_per_tier, args.neg_ratio)
    pools = build_pools(args.n, exclude, args.neg_ratio)
    composed = compose(pools, args.n)

    rows = []
    representation = "abacus-v1" if reverse else "natural-v1"
    for (a, b, op, r) in composed:
        t, score = classify(a, b, op, r)
        rows.append({"text": make_text(a, b, op, r, reverse),
                     "split": "train", "tier": t,
                     "difficulty": round(score, 4),
                     "representation": representation})

    for (a, b, op) in eval_set:
        r = compute(a, b, op)
        t, score = classify(a, b, op, r)
        rows.append({"text": make_text(a, b, op, r, reverse),
                     "split": "val", "tier": t,
                     "difficulty": round(score, 4),
                     "representation": representation})

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    reversed_ops = [op for op in V.OPERATORS if reverse and should_reverse(op)]
    rev_desc = ",".join(reversed_ops) if reversed_ops else "none (all natural)"
    print(f"wrote {n_train} train + {n_val} val -> {args.out} "
          f"(reversed ops: {rev_desc}; fixed decimal division: {reverse})")
    print("\ntrain:"); report([r for r in rows if r["split"] == "train"])
    print("\nval:  "); report([r for r in rows if r["split"] == "val"])


if __name__ == "__main__":
    main()
