# Three-digit calculator data distribution specification (v2)

This specification covers the supported calculator domain: two operands whose
absolute values are at most `999`, the operations `+`, `-`, `*`, and `/`, a
possibly negative first operand, negative-by-negative multiplication, division
rounded to three decimal places, and `NAN` for division by zero.
The objective is not to enumerate every possible expression. It is to cover
every arithmetic mechanism and every important interaction with explicit,
testable quotas.

The machine-readable source of truth is
[`config/data_distribution_v2.yaml`](../config/data_distribution_v2.yaml).

## Why the old distribution must be replaced

The previous generator balanced coarse difficulty tiers and deliberately added
an `800..999 * 800..999` hard-corner pool. The measured training distribution
therefore contained about 2,500 rows in each `8xx/9xx * 8xx/9xx` cell, versus
roughly 60--120 rows in an ordinary hundreds cell. The epoch-40 model reached
92% on fresh `9xx * 9xx` expressions but only 34% on `7xx * 7xx`.

Version 2 makes the distribution itself a contract. Generation fails if a
required quota or split invariant cannot be met.

## Global contract

- Training size: 400,000 rows.
- Validation size: 4,000 rows.
- Operations: exactly 25% each.
- Operand domain: `0..999` in magnitude; division includes a zero divisor and
  maps it to `NAN`.
- Signs: 20% negative first operands inside every operation and operand-length
  cell, using largest-remainder integer apportionment for very small cells.
- Multiplication sign patterns are exactly 80% `a*b`, 10% `-a*b`, and 10%
  `-a*-b`. Thus the overall negative-first ratio remains 20% while explicitly
  teaching that two negative factors produce a positive result. A negative
  second operand remains unsupported for the other three operations.
- Division by zero is defined as the string answer `NAN`, including `0/0` and
  negative-numerator cases. It is encoded as one atomic `<nan>` answer token,
  increasing the vocabulary from 21 to 22 tokens. Decoding displays `NAN`.
- Training repetitions are permitted for small or rare arithmetic facts, but a
  prompt may appear no more than 25 times.
- Validation prompts are unique and absent from training.
- Prompts in `config/pretrain_algorithmic_eval.yaml` are reserved as a test set
  and cannot appear in either generated split.
- For unsigned addition and multiplication validation expressions, the swapped
  operand order is also absent from training. This makes commutativity testing
  meaningful.
- Addition, subtraction, and multiplication use reversed magnitudes. Division
  remains natural order with a fixed `DDD.ddd` internal answer.
- Every encoded expression, including BOS and EOS, must fit 20 tokens.

## Operand-length distribution

The following matrix is applied independently to addition, subtraction, and
multiplication. Small facts remain represented, but most capacity is allocated
to full three-digit operands.

| First operand | Second: 1 digit | Second: 2 digits | Second: 3 digits |
|---|---:|---:|---:|
| **1 digit** | 0.1% | 0.9% | 4.0% |
| **2 digits** | 0.9% | 6.1% | 14.0% |
| **3 digits** | 4.0% | 14.0% | 56.0% |

This gives each of those operations 56,000 full `3x3` rows in a 100,000-row
operation budget.

Division uses a separate matrix because `/0`, `/1`, exact three-digit
quotients, and rounded three-digit quotients all require a one-digit divisor.
Using the shared matrix left only 5% one-digit divisors, which the identity
cases consumed completely.

| First operand | Divisor: 1 digit | Divisor: 2 digits | Divisor: 3 digits |
|---|---:|---:|---:|
| **1 digit** | 1% | 3% | 3% |
| **2 digits** | 8% | 8% | 5% |
| **3 digits** | 21% | 21% | 30% |

This reserves 30% of division for one-digit divisors while retaining 30,000
full `3x3` divisions. Quotas are based on magnitude length; zero is a one-digit
operand.

## Addition scenarios

Unsigned addition is divided by the actual carry mechanism. A cascading carry
means that an incoming carry changes a raw column sum of nine into another
carry. One-step propagation, such as `199+1`, is separated from two-step
propagation, such as `999+1` or `899+101`. Independent multi-carry cases have
multiple carried columns but no such causal dependency. Overflow is separate
only when it is not already a cascading case.

Signed addition is crossed with magnitude-subtraction complexity: negative and
positive outcomes each receive simple, ordinary multi-borrow, and
borrow-across-zero bands, while equality is its own zero-result band.

| Scenario | Share of addition |
|---|---:|
| Identity or zero operand | 4% |
| No carried column | 14% |
| Exactly one carried column | 16% |
| Multiple independent carried columns | 14% |
| One-step cascading carry | 10% |
| Two-step cascading carry | 6% |
| Non-cascading overflow to four digits | 16% |
| Signed negative result, simple magnitude subtraction | 3% |
| Signed negative result, multiple ordinary borrows | 3% |
| Signed negative result, borrow through zero | 2% |
| Signed result is zero | 2% |
| Signed positive result, simple magnitude subtraction | 4% |
| Signed positive result, multiple ordinary borrows | 4% |
| Signed positive result, borrow through zero | 2% |

Addition also has an independent digit-pattern marginal: 20% of rows contain
at least one zero digit and 80% contain no zero digits. It is allocated jointly
across all compatible carry/sign scenarios above. Identity rows necessarily
contain zero, while independent multi-carry rows necessarily contain no zero;
the remaining zero quota is apportioned across the other scenarios. Operand
length and sign remain separate exact marginals. Requiring 20% zeros
independently inside every length/sign cell would be impossible because all
addition identity cases contain zero.

All 81 ordered full-width regions from `1xx+1xx` through `9xx+9xx` receive
equal total coverage, differing by at most one row. The full `3x3` cell grid
and the 80/20 first-operand sign quota are both exact marginals; their joint is
left flexible because carry and overflow mechanisms are not feasible at equal
rates in every hundreds cell.

Every addition mechanism is also crossed with the feasible coarse operand
length families: both operands at most two digits, three-digit left only,
three-digit right only, and both operands three digits. Structural
impossibilities are explicit: a two-step cascade cannot occur when both
operands have at most two digits, and a non-cascading four-digit overflow
requires two three-digit operands. Signed rows exactly consume the configured
20% negative-first quota inside every length family.

## Subtraction scenarios

Unsigned non-negative results are divided by borrow-chain structure. Borrowing
through a zero digit, such as `100-1` or `500-499`, is isolated from other
multi-borrow cases. A negative result is separate because the model must select
a sign and subtract the magnitudes in the opposite order; that branch is split
again by simple versus complex magnitude subtraction. With a negative first
operand, subtraction becomes magnitude addition, so those rows are divided by
carry complexity.

Positive-first negative results receive 40% rather than the earlier 30%. This
matches the geometry of a balanced non-negative operand domain more closely
and is required to populate the positive-first side of every feasible `3x3`
hundreds cell while retaining the exact 20% negative-first marginal.

| Scenario | Share of subtraction |
|---|---:|
| Subtract zero | 3% |
| Equal operands (self-subtraction) | 2% |
| No borrowed column | 9% |
| Exactly one borrowed column | 11% |
| Multiple borrows without crossing zero | 11% |
| Positive result with borrow propagation through zero | 4% |
| Negative result, simple magnitude subtraction | 12% |
| Negative result, multiple ordinary magnitude borrows | 16% |
| Negative result, reversed-magnitude borrow through zero | 12% |
| Negative first operand, zero or one carry | 10% |
| Negative first operand, independent multiple carries | 5% |
| Negative first operand, cascading carry | 5% |

Subtraction uses a crossed 25% zero-digit / 75% no-zero marginal. Its larger
zero share prevents subtract-zero and the two borrow-across-zero branches from
consuming nearly the whole zero budget, leaving explicit zero-digit coverage
for other mechanisms. Subtracting zero is separated from equal nonzero operands because
the former is restricted to a one-digit right operand, whereas self-subtraction
uses equal operand widths.

All 81 ordered full-width regions from `1xx-1xx` through `9xx-9xx` receive
equal total coverage, and the 80/20 first-operand sign split is enforced inside
each cell. This specifically prevents signed rows or mixed-length
negative-result examples from consuming the entire sign-change budget:
validation must retain at least 100 full `3x3` cases with a non-negative first
operand and a negative result.

Subtraction mechanisms are likewise crossed with every feasible coarse length
family. Positive-result mechanisms cover three-digit minuends with shorter
subtrahends, negative-result mechanisms cover the opposite orientation, and
negative-first carry mechanisms cover small, both mixed orientations, and full
`3x3` operands. Signed rows exactly match the 20% negative-first quota in each
family. This prevents hard borrow or sign branches from being concentrated in
one operand-width shape.

## Multiplication scenarios

The global digit-pattern marginal prevents easy sparse products from dominating
the operation.

| Scenario | Share of multiplication |
|---|---:|
| Zero/one identity | 5% |
| Non-identity expression containing a zero digit | 20% |
| Non-zero repeated-digit operand such as `777` | 10% |
| Dense mixed digits | 65% |

The digit-pattern quotas are crossed with a separate sign-pattern marginal:

| Sign pattern | Share of multiplication |
|---|---:|
| `a*b` | 80% |
| `-a*b` | 10% |
| `-a*-b` | 10% |

The scenario marginal is also crossed with the feasible coarse operand-length
families. Identity/zero expressions cannot be full `3x3`, because at least one
operand is `0` or `1`; their rows are split across small and both mixed-width
families. The zero-digit, repeated-digit, and dense scenarios divide the
remaining capacity in the same small/left-mixed/right-mixed/full proportions.
This prevents full-width multiplication from collapsing to dense mixed-digit
products only.

Full `3x3` multiplication has two additional mandatory axes:

1. Every ordered hundreds region from `1xx*1xx` through `9xx*9xx` receives the
   same number of rows, differing by at most one due to integer rounding.
2. Inside each hundreds cell, candidates are divided into four equally sampled
   bands according to the schoolbook central-column total
   `a0*b2 + a1*b1 + a2*b0 + carry_in`. Bands are cell-relative quartiles, so
   every numeric region contributes both its easier and harder central carries.

Each feasible full-width scenario is separately balanced across those four
central bands. Thus zero-digit, repeated-digit, and dense products all exercise
the complete central-carry range. The global 80/10/10 sign marginal remains
exact within every fine operand-length cell; validation additionally requires
all three sign forms to occur in every multiplication scenario.

This directly targets the measured failure: units and tens were essentially
perfect, while the central three-part column was only 58% correct.

## Division scenarios

Division is divided by the mathematical form of the quotient rather than a
coarse difficulty tier. The earlier coarse version put 38% of its supposedly
nontrivial exact examples in `a/a=1`, contained no exact examples with divisors
`2..9`, and put 69% of its near-integer budget next to zero. The following
bands remove those concentrations.

| Scenario | Share of division |
|---|---:|
| Division by zero, answer `NAN` | 4% |
| Zero numerator with nonzero divisor | 2% |
| Divide by one | 3% |
| Unit-magnitude quotient, `abs(a)=b>1` | 3% |
| Exact quotient `2..9` | 10% |
| Exact quotient `10..99` | 9% |
| Exact quotient `100+` | 3% |
| Nonzero quotient no farther than 0.05 from zero | 3% |
| Non-integer quotient within 0.05 below an integer `1+` | 5% |
| Non-integer quotient within 0.05 above an integer `1+` | 5% |
| Terminating below one, one decimal place | 2% |
| Terminating below one, two decimal places | 2% |
| Terminating below one, three decimal places | 2% |
| Terminating at least one, one decimal place | 2% |
| Terminating at least one, two decimal places | 2% |
| Terminating at least one, three decimal places | 2% |
| Rounded quotient between 0.05 and 0.1 | 5% |
| Rounded quotient between 0.1 and 1 | 10% |
| Rounded quotient `1..9.999...` | 13% |
| Rounded quotient `10..99.999...` | 9% |
| Rounded quotient `100+` | 4% |

A decimal terminates within three places when the reduced denominator contains
only factors 2 and 5 and requires at most three fractional digits. All other
non-integer quotients exercise rounding. `NAN` is a categorical answer and does
not use the fixed `DDD.ddd` numeric shape.

Boundary bands are checked before terminating/repeating classification. Near
zero is separate from near integers `1+`, and the latter is exactly balanced
above and below the boundary. Each genuinely rounded band is also exactly 50%
round-down and 50% round-up. Division signs are allocated jointly with every
mechanism (80% positive and 20% negative globally); zero-numerator rows are the
necessary positive-only exception, with the small sign correction distributed
over the remaining mechanisms.

Full `3x3` division has an additional spatial invariant: every ordered
numerator/divisor hundreds region from `1xx/1xx` through `9xx/9xx` receives the
same number of rows, differing by at most one. This prevents the strong divisor
region skew found in the previous build.

## Required validation report

Generation must recompute and verify every answer, then report and assert:

- exact row totals by split and operation;
- exact operand-length and sign quotas;
- exact operation-specific scenario quotas;
- exact multiplication scenario-by-length and full-width
  scenario-by-central-band quotas;
- exact zero-digit marginals for addition and subtraction;
- exact division mechanism/sign quotas and 50/50 rounding directions;
- no prompt overlap between validation and training;
- no swapped validation leakage for unsigned `+` and `*`;
- multiplication `3x3` counts for all 81 hundreds cells;
- four central-complexity bands per multiplication hundreds cell;
- division `3x3` counts for all 81 numerator/divisor hundreds cells;
- addition and subtraction `3x3` counts for all 81 ordered hundreds cells;
- the exact 80/20 first-operand sign split inside every subtraction `3x3`
  hundreds cell;
- at least 20 validation rows in every operation scenario, at least six rows in
  every multiplication hundreds cell, at least one row in every cell/band
  intersection, at least 140 rows in each global central band, and every sign
  form in every multiplication scenario;
- at least three validation rows in every division hundreds cell and at least
  ten rows for each rounding direction inside every rounded division scenario;
- at least six validation rows in every addition/subtraction hundreds cell and
  at least 100 full-width negative-result subtraction probes;
- prompt repetition counts and the configured maximum;
- encoded sequence-length compliance;
- representation and answer round-trip correctness.

The held-out algorithmic suite and the multiplication behavior audit remain
test sets. They must not be incorporated into training quotas or used as
training prompts.

## Generator and outputs

`generate_data_v2.py` reads the YAML contract, builds bounded candidate pools,
solves the requested marginals as a flow-allocation problem, maximizes unique
prompts before allowing repetition, validates the completed dataset, and only
then writes it.

```bash
python generate_data_v2.py
```

Default outputs:

- `sample/pretrain_v2.jsonl`: 400,000 train rows and 4,000 validation rows.
- `sample/pretrain_v2_report.json`: the machine-readable quota and integrity
  report.

## Accepted seed-42 build: uniqueness and repetition

The tables below describe the generated training split, not additional quota
requirements. A problem is an ordered prompt, so `a+b` and `b+a` are distinct.
Each scenario/length cell uses the notation:

`training rows / unique problems / maximum copies of one problem`

A dash means that the generated build has no row in that intersection. The
number of repeated rows in a cell is `training rows - unique problems`.

### Operation totals

| Operation | Rows | Unique problems | Repeated rows | Average copies | Maximum copies |
|---|---:|---:|---:|---:|---:|
| Addition | 100,000 | 94,638 | 5,362 | 1.057 | 8 |
| Subtraction | 100,000 | 96,778 | 3,222 | 1.033 | 12 |
| Multiplication | 100,000 | 99,757 | 243 | 1.002 | 2 |
| Division | 100,000 | 77,153 | 22,847 | 1.296 | 25 |
| **Total** | **400,000** | **368,326** | **31,674** | **1.086** | **25** |

### Coarse operand-length bands

`small` means both operands have at most two digits. `left mixed` means only
the first operand has three digits, `right mixed` means only the second operand
has three digits, and `full` means `3x3`.

| Operation | Band | Rows | Unique | Repeated rows | Average copies | Maximum |
|---|---|---:|---:|---:|---:|---:|
| `+` | Small | 8,000 | 6,304 | 1,696 | 1.269 | 8 |
| `+` | Left mixed | 18,000 | 16,920 | 1,080 | 1.064 | 4 |
| `+` | Right mixed | 18,000 | 16,920 | 1,080 | 1.064 | 4 |
| `+` | Full | 56,000 | 54,494 | 1,506 | 1.028 | 6 |
| `-` | Small | 8,000 | 6,692 | 1,308 | 1.195 | 12 |
| `-` | Left mixed | 18,000 | 16,471 | 1,529 | 1.093 | 3 |
| `-` | Right mixed | 18,000 | 17,927 | 73 | 1.004 | 2 |
| `-` | Full | 56,000 | 55,688 | 312 | 1.006 | 5 |
| `*` | Small | 8,000 | 7,757 | 243 | 1.031 | 2 |
| `*` | Left mixed | 18,000 | 18,000 | 0 | 1.000 | 1 |
| `*` | Right mixed | 18,000 | 18,000 | 0 | 1.000 | 1 |
| `*` | Full | 56,000 | 56,000 | 0 | 1.000 | 1 |
| `/` | Small | 20,000 | 8,660 | 11,340 | 2.309 | 25 |
| `/` | Left mixed | 42,000 | 32,107 | 9,893 | 1.308 | 9 |
| `/` | Right mixed | 8,000 | 8,000 | 0 | 1.000 | 1 |
| `/` | Full | 30,000 | 28,386 | 1,614 | 1.057 | 12 |

### Addition scenario by length band

| Scenario | Small | Left mixed | Right mixed | Full `3x3` |
|---|---:|---:|---:|---:|
| Identity or zero | 1,000 / 179 / 6 | 1,500 / 870 / 2 | 1,500 / 870 / 2 | — |
| No carry | 1,400 / 1,400 / 1 | 3,780 / 3,780 / 1 | 3,780 / 3,780 / 1 | 5,040 / 5,040 / 1 |
| Single carry | 1,600 / 1,600 / 1 | 4,680 / 4,680 / 1 | 4,680 / 4,680 / 1 | 5,040 / 5,040 / 1 |
| Independent multi-carry | 1,400 / 1,400 / 1 | 2,030 / 2,030 / 1 | 2,030 / 2,030 / 1 | 8,540 / 8,540 / 1 |
| One-step cascading carry | 1,000 / 430 / 3 | 1,530 / 1,530 / 1 | 1,530 / 1,530 / 1 | 5,940 / 5,940 / 1 |
| Two-step cascading carry | — | 880 / 430 / 4 | 880 / 430 / 4 | 4,240 / 3,450 / 2 |
| Non-cascading overflow | — | — | — | 16,000 / 16,000 / 1 |
| Signed negative, simple | 700 / 700 / 1 | 1,400 / 1,400 / 1 | — | 900 / 900 / 1 |
| Signed negative, multi-borrow | — | 1,200 / 1,200 / 1 | — | 1,800 / 1,800 / 1 |
| Signed negative, borrow through zero | — | 1,000 / 1,000 / 1 | — | 1,000 / 1,000 / 1 |
| Signed result zero | 400 / 95 / 8 | — | — | 1,600 / 884 / 6 |
| Signed positive, simple | 500 / 500 / 1 | — | 2,000 / 2,000 / 1 | 1,500 / 1,500 / 1 |
| Signed positive, multi-borrow | — | — | 1,000 / 1,000 / 1 | 3,000 / 3,000 / 1 |
| Signed positive, borrow through zero | — | — | 600 / 600 / 1 | 1,400 / 1,400 / 1 |

Addition repetition is confined to identity, equality, and scarce cascading
carry intersections. Every ordinary no-carry, single-carry, independent-carry,
overflow, and non-equality signed intersection is unique.

### Subtraction scenario by length band

| Scenario | Small | Left mixed | Right mixed | Full `3x3` |
|---|---:|---:|---:|---:|
| Subtract zero | 600 / 92 / 7 | 2,400 / 871 / 3 | — | — |
| Equal operands | 888 / 88 / 12 | — | — | 1,112 / 800 / 5 |
| No borrow | 1,297 / 1,297 / 1 | 4,230 / 4,230 / 1 | — | 3,473 / 3,473 / 1 |
| Single borrow | 947 / 947 / 1 | 4,030 / 4,030 / 1 | — | 6,023 / 6,023 / 1 |
| Multi-borrow | — | 2,470 / 2,470 / 1 | — | 8,530 / 8,530 / 1 |
| Borrow through zero | — | 1,270 / 1,270 / 1 | — | 2,730 / 2,730 / 1 |
| Negative result, simple | 2,668 / 2,668 / 1 | — | 4,520 / 4,520 / 1 | 4,812 / 4,812 / 1 |
| Negative result, multi-borrow | — | — | 5,800 / 5,800 / 1 | 10,200 / 10,200 / 1 |
| Negative result, borrow through zero | — | — | 4,080 / 4,007 / 2 | 7,920 / 7,920 / 1 |
| Signed low-carry | 800 / 800 / 1 | 1,800 / 1,800 / 1 | 1,800 / 1,800 / 1 | 5,600 / 5,600 / 1 |
| Signed independent multi-carry | 400 / 400 / 1 | 900 / 900 / 1 | 900 / 900 / 1 | 2,800 / 2,800 / 1 |
| Signed cascading carry | 400 / 400 / 1 | 900 / 900 / 1 | 900 / 900 / 1 | 2,800 / 2,800 / 1 |

Subtraction repetition is almost entirely caused by the finite `a-0` and
`a-a` identity populations. The only non-identity repetition is 73 additional
rows in the right-mixed negative-result borrow-through-zero intersection.

### Multiplication scenario by length band

| Scenario | Small | Left mixed | Right mixed | Full `3x3` |
|---|---:|---:|---:|---:|
| Identity or zero | 1,000 / 757 / 2 | 2,000 / 2,000 / 1 | 2,000 / 2,000 / 1 | — |
| Contains a zero digit | 1,474 / 1,474 / 1 | 3,369 / 3,369 / 1 | 3,368 / 3,368 / 1 | 11,789 / 11,789 / 1 |
| Repeated-digit operand | 737 / 737 / 1 | 1,684 / 1,684 / 1 | 1,684 / 1,684 / 1 | 5,895 / 5,895 / 1 |
| Dense mixed digits | 4,789 / 4,789 / 1 | 10,947 / 10,947 / 1 | 10,948 / 10,948 / 1 | 38,316 / 38,316 / 1 |

Multiplication has 99,757 unique prompts. Its only repetition is 243 additional
copies in the finite small identity/zero population, and no prompt occurs more
than twice. Every non-identity scenario/length intersection is fully unique.
The sign-pattern marginals remain exactly 80,000 `a*b`, 10,000 `-a*b`, and
10,000 `-a*-b` rows.

The full-width central-column bands are exactly balanced and unique:

| Cell-relative central band | Rows | Unique | Maximum copies |
|---:|---:|---:|---:|
| 0, lowest quartile | 14,000 | 14,000 | 1 |
| 1 | 14,000 | 14,000 | 1 |
| 2 | 14,000 | 14,000 | 1 |
| 3, highest quartile | 14,000 | 14,000 | 1 |

Within the 56,000 full-width rows, the zero-digit, repeated-digit, and dense
scenarios contribute 11,789, 5,895, and 38,316 examples respectively. Each of
those scenario totals is divided almost equally among the four central bands;
the global band totals are exactly 14,000 each.

### Division scenario by length band

| Scenario | Small | Left mixed | Right mixed | Full `3x3` |
|---|---:|---:|---:|---:|
| Division by zero | 2,283 / 154 / 25 | 1,717 / 1,055 / 2 | — | — |
| Zero numerator | 1,103 / 79 / 25 | — | 897 / 897 / 1 | — |
| Divide by one | 1,584 / 167 / 25 | 1,416 / 1,416 / 1 | — | — |
| Unit quotient | 1,384 / 166 / 25 | — | — | 1,616 / 1,417 / 4 |
| Exact quotient, one digit | 3,723 / 292 / 25 | 2,860 / 1,208 / 4 | — | 3,417 / 2,002 / 12 |
| Exact quotient, two digits | 1,624 / 212 / 15 | 7,376 / 3,341 / 9 | — | — |
| Exact quotient, three digits | — | 3,000 / 1,607 / 3 | — | — |
| Near zero, nonzero | 389 / 389 / 1 | — | 2,611 / 2,611 / 1 | — |
| Near integer below | 526 / 526 / 1 | 462 / 462 / 1 | — | 4,012 / 4,012 / 1 |
| Near integer above | 414 / 414 / 1 | 1,694 / 1,694 / 1 | — | 2,892 / 2,892 / 1 |
| Terminating below one, 1 dp | 321 / 267 / 4 | — | 285 / 285 / 1 | 1,394 / 1,394 / 1 |
| Terminating below one, 2 dp | 249 / 249 / 1 | — | 70 / 70 / 1 | 1,681 / 1,681 / 1 |
| Terminating below one, 3 dp | 146 / 146 / 1 | — | 262 / 262 / 1 | 1,592 / 1,592 / 1 |
| Terminating at least one, 1 dp | 584 / 584 / 1 | 1,416 / 1,416 / 1 | — | — |
| Terminating at least one, 2 dp | 460 / 460 / 1 | 964 / 964 / 1 | — | 576 / 576 / 1 |
| Terminating at least one, 3 dp | 177 / 177 / 1 | 1,202 / 1,202 / 1 | — | 621 / 621 / 1 |
| Rounded quotient `0.05..0.1` | 1,125 / 470 / 6 | — | 3,875 / 3,875 / 1 | — |
| Rounded quotient `0.1..1` | 2,240 / 2,240 / 1 | — | — | 7,760 / 7,760 / 1 |
| Rounded quotient, one integer digit | 1,477 / 1,477 / 1 | 7,084 / 7,084 / 1 | — | 4,439 / 4,439 / 1 |
| Rounded quotient, two integer digits | 191 / 191 / 1 | 8,809 / 8,809 / 1 | — | — |
| Rounded quotient, three integer digits | — | 4,000 / 1,849 / 4 | — | — |

Division repetition is concentrated in finite identity and exact-quotient
populations: `/0`, `0/b`, `/1`, `a/a`, and small exact quotients. Rounded,
near-boundary, and terminating bands are predominantly or completely unique.
The maximum of 25 copies is the configured global repetition cap.
