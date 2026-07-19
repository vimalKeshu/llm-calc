# V3 compact division dataset

This dataset keeps three decimal places for division while removing redundant
integer padding:

- `000.640` becomes `0.640`
- `-004.014` becomes `-4.014`
- `294.000` remains `294.000`
- division by zero remains `NAN`

The v2 generator behavior remains the default. V3 opts into the compact format
through `domain.division_answer_format: compact_fixed_precision` and uses the
separate `abacus-v2-compact-division` representation label.

## Distribution

The generated training split has 500,000 rows:

- addition: 100,000
- subtraction: 100,000
- multiplication: 100,000
- division: 200,000

The validation split has 5,000 rows in the same 20/20/20/40 proportion. This
preserves the absolute 100,000-row exposure for each previously trained
operation, so division is increased without reducing addition or subtraction.

Division emphasizes the failure regimes observed during checkpoint evaluation:

- exact two-digit quotients
- values immediately below and above integer boundaries
- terminating decimals greater than or equal to one
- rounded quotients below one
- rounded quotients with a one-digit integer part

These groups account for 142,000 of the 200,000 division training rows. There
are 80,000 3x3 division rows and 60,000 signed division rows. Every focus row is
tagged with `division_capability_focus`, and all division rows have a
`division_capability_group` for later sampling and evaluation.

## Artifacts and validation

- Spec: `config/data_distribution_v3_compact_division.yaml`
- Dataset: `sample/pretrain_v3_compact_division.jsonl`
- Report: `sample/pretrain_v3_compact_division_report.json`

The generated report verifies all answers and sequence lengths, with zero
train/validation overlap, zero reserved-test overlap, and zero swapped
commutative validation leakage. The standalone unit suite also verifies that
the old fixed-width format remains the default.
