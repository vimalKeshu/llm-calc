# type: ignore
import random
import json
import argparse
from itertools import product
import vocab as V

BUCKETS = [(0, 9), (10, 99), (100, 999)]


def make_expr(a, b, op):
    if op == '+':
        c = a + b
    elif op == '-':
        c = a - b
    elif op == '*':
        c = a * b
    else:  # '/'
        b = max(1, b)
        result = round(a / b, 3)
        c = int(result) if result == int(result) else result
    return f"{a}{op}{b}={c}"


def verify(expr):
    try:
        lhs, rhs = expr.split('=')
        for op in V.OPERATORS:
            idx = lhs.find(op, 1)
            if idx == -1:
                continue
            a, b = float(lhs[:idx]), float(lhs[idx+1:])
            if op == '+':   expected = round(a + b, 3)
            elif op == '-': expected = round(a - b, 3)
            elif op == '*': expected = round(a * b, 3)
            else:
                if b == 0: return False
                expected = round(a / b, 3)
            return abs(expected - round(float(rhs), 3)) < 1e-6
    except Exception:
        return False
    return False


def generate_train(n, clean_div_ratio=0.3, edge_ratio=0.05, neg_ratio=0.2):
    samples = set()
    cells = list(product(BUCKETS, BUCKETS, V.OPERATORS))
    per_cell = max(1, int(n * (1 - edge_ratio)) // len(cells))

    for (lo_a, hi_a), (lo_b, hi_b), op in cells:
        count = 0
        while count < per_cell:
            a = random.randint(lo_a, hi_a)
            b = random.randint(lo_b, hi_b)
            if op == '/':
                b = max(1, b)
                if random.random() < clean_div_ratio:
                    k = random.randint(1, max(1, 999 // b))
                    a = min(b * k, 999)
            if a != 0 and random.random() < neg_ratio:
                a = -a
            expr = make_expr(a, b, op)
            if verify(expr):
                samples.add(expr)
                count += 1

    edge_pool = []
    for op in V.OPERATORS:
        for v in range(0, 10):
            for a, b in [(0, v), (v, 1), (v, v), (-v, 1), (-v, v)]:
                expr = make_expr(a, b, op)
                if verify(expr):
                    edge_pool.append(expr)
    random.shuffle(edge_pool)
    for expr in edge_pool[:int(n * edge_ratio)]:
        samples.add(expr)

    while len(samples) < n:
        (lo_a, hi_a), (lo_b, hi_b), op = random.choice(cells)
        a = random.randint(lo_a, hi_a)
        b = max(1, random.randint(lo_b, hi_b)) if op == '/' else random.randint(lo_b, hi_b)
        if a != 0 and random.random() < neg_ratio:
            a = -a
        expr = make_expr(a, b, op)
        if verify(expr):
            samples.add(expr)

    return list(samples)[:n]


def bucket_values(lo, hi, x):
    """Return x evenly spaced integers across [lo, hi] inclusive."""
    if x == 1:
        return [lo]
    step = (hi - lo) / (x - 1)
    return list(dict.fromkeys(round(lo + i * step) for i in range(x)))


def generate_val(x=10):
    """Deterministic: x evenly spaced values per bucket per cell, including negative a."""
    samples = set()
    for (lo_a, hi_a), (lo_b, hi_b), op in product(BUCKETS, BUCKETS, V.OPERATORS):
        vals_a = bucket_values(lo_a, hi_a, x)
        vals_b = bucket_values(lo_b, hi_b, x)
        for a, b in product(vals_a, vals_b):
            b = max(1, b) if op == '/' else b
            for signed_a in ([a, -a] if a != 0 else [a]):
                expr = make_expr(signed_a, b, op)
                if verify(expr):
                    samples.add(expr)
    return list(samples)


def report(samples):
    op_counts = {op: 0 for op in V.OPERATORS}
    digit_counts = {}
    for s in samples:
        lhs = s.split('=')[0]
        for op in V.OPERATORS:
            idx = lhs.find(op, 1)
            if idx != -1:
                op_counts[op] += 1
                d = len(lhs[:idx])
                digit_counts[d] = digit_counts.get(d, 0) + 1
                break
    print("op  distribution:", op_counts)
    print("a digit-len dist:", dict(sorted(digit_counts.items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=100000, help="total train samples")
    parser.add_argument("--x",    type=int, default=10,    help="val samples per bucket cell")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    train_samples = generate_train(args.n)
    random.shuffle(train_samples)
    val_samples = generate_val(args.x)
    random.shuffle(val_samples)

    with open("pretrain.jsonl", "w") as f:
        for expr in train_samples:
            f.write(json.dumps({"text": expr, "split": "train"}) + "\n")
        for expr in val_samples:
            f.write(json.dumps({"text": expr, "split": "val"}) + "\n")

    print(f"wrote {len(train_samples)} train + {len(val_samples)} val -> pretrain.jsonl")
    print("\ntrain:"); report(train_samples)
    print("\nval:");   report(val_samples)


if '__main__' == __name__:
    main()
