# type: ignore
"""Quota-driven generator for the three-digit calculator distribution v2.

The distribution contract lives in ``config/data_distribution_v2.yaml``.
Unlike the older tier sampler, this generator allocates exact row budgets to
arithmetic mechanisms and fails if the requested marginals are infeasible.
"""

import argparse
import bisect
import collections
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

import vocab as V
from generate_data import (
    _borrows,
    classify,
    compute,
    decode_internal_answer,
    make_text,
    ndigits,
    parse_expression,
    should_reverse,
    unreverse_magnitude,
    verify,
)


COMMUTATIVE_OPERATIONS = {"+", "*"}
SIGNS = ("positive", "negative")


def _deep_merge(base, override):
    """Recursively merge a derived distribution specification."""
    result = dict(base)
    for key, value in override.items():
        if (isinstance(value, dict)
                and isinstance(result.get(key), dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_spec(path, _seen=None):
    """Load a YAML spec, supporting a relative ``extends`` base spec."""
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"cyclic distribution spec inheritance at {path}")
    seen.add(path)
    with path.open() as f:
        spec = yaml.safe_load(f)
    parent = spec.pop("extends", None)
    if parent is None:
        return spec
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(load_spec(parent_path, seen), spec)


def multiplication_sign_pattern(a, b):
    if a >= 0 and b >= 0:
        return "positive_positive"
    if a < 0 and b >= 0:
        return "negative_positive"
    if a < 0 and b < 0:
        return "negative_negative"
    raise ValueError("positive*negative is outside the configured grammar")


def supports_commutative_swap(a, b, op):
    if op == "+":
        return a >= 0 and b >= 0
    if op == "*":
        return ((a >= 0 and b >= 0) or (a < 0 and b < 0))
    return False


def apportion(total, weights):
    """Largest-remainder integer apportionment preserving mapping order."""
    items = list(weights.items())
    weight_sum = sum(float(weight) for _, weight in items)
    if total < 0 or weight_sum <= 0:
        raise ValueError("apportion requires a non-negative total and positive weights")
    ideals = [total * float(weight) / weight_sum for _, weight in items]
    counts = [math.floor(value) for value in ideals]
    remainder = total - sum(counts)
    order = sorted(
        range(len(items)), key=lambda i: (-(ideals[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    return {items[i][0]: counts[i] for i in range(len(items))}


def digit_values(length, negative=False):
    if length == 1:
        start, end = (1 if negative else 0), 9
    elif length == 2:
        start, end = 10, 99
    elif length == 3:
        start, end = 100, 999
    else:
        raise ValueError(f"unsupported operand length: {length}")
    return range(start, end + 1)


def parse_length_key(key):
    left, right = key.split('x', 1)
    return int(left), int(right)


def length_key(a, b):
    return f"{ndigits(a)}x{ndigits(b)}"


def operand_digit_pattern(a, b):
    return ("contains_zero_digit"
            if '0' in str(abs(a)) or '0' in str(abs(b))
            else "no_zero_digits")


def operand_length_band(a, b):
    da, db = ndigits(a), ndigits(b)
    if da == 3 and db == 3:
        return "full"
    if da == 3:
        return "left_mixed"
    if db == 3:
        return "right_mixed"
    return "small"


def operation_targets(total, spec):
    return apportion(
        total,
        {op: config["weight"] for op, config in spec["operations"].items()})


def length_targets(op, total, spec):
    weights = spec["operations"][op].get(
        "operand_length_weights", spec["operand_length_weights"])
    return apportion(total, weights)


def scenario_targets(op, total, spec):
    return apportion(total, spec["operations"][op]["scenario_weights"])


def division_sign(a):
    return "negative" if a < 0 else "positive"


def division_rounding_direction(a, b, scenario=None):
    """Return whether three-decimal rounding increases the truncated value."""
    if scenario is None:
        scenario = scenario_division(a, b)
    if not scenario.startswith("rounded_"):
        return "not_applicable"
    quotient = abs(a) / b
    truncated = math.floor(quotient * 1000) / 1000
    return "up" if round(quotient, 3) > truncated else "down"


def two_by_two_allocation(row_counts, column_counts):
    """Construct an exact 2x2 table close to independent marginals."""
    rows = list(row_counts)
    columns = list(column_counts)
    if len(rows) != 2 or len(columns) != 2:
        raise ValueError("two_by_two_allocation requires two rows and columns")
    total = sum(row_counts.values())
    if total != sum(column_counts.values()):
        raise ValueError("2x2 row and column totals differ")
    first_row, second_row = rows
    first_column, second_column = columns
    lower = max(
        0,
        row_counts[first_row] - column_counts[second_column],
        column_counts[first_column] - row_counts[second_row],
    )
    upper = min(row_counts[first_row], column_counts[first_column])
    ideal = (row_counts[first_row] * column_counts[first_column] / total
             if total else 0)
    top_left = min(upper, max(lower, int(round(ideal))))
    return {
        (first_row, first_column): top_left,
        (first_row, second_column): row_counts[first_row] - top_left,
        (second_row, first_column): column_counts[first_column] - top_left,
        (second_row, second_column): (
            row_counts[second_row]
            - (column_counts[first_column] - top_left)),
    }


def division_allocation_targets(total, spec):
    """Cross every division mechanism with sign and rounded-result direction."""
    mechanisms = scenario_targets("/", total, spec)
    sign_weights = spec["operations"]["/"]["sign_pattern_weights"]
    global_signs = apportion(total, sign_weights)
    forced_positive = {"zero_numerator"}
    forced_positive_count = sum(mechanisms[name] for name in forced_positive)
    if forced_positive_count > global_signs["positive"]:
        raise ValueError("zero-numerator quota exceeds positive division quota")
    flexible = {
        scenario: count for scenario, count in mechanisms.items()
        if scenario not in forced_positive}
    negative_by_scenario = apportion(global_signs["negative"], flexible)
    direction_weights = spec["operations"]["/"][
        "rounding_direction_weights"]

    targets = collections.OrderedDict()
    for scenario, count in mechanisms.items():
        negative = (0 if scenario in forced_positive
                    else negative_by_scenario[scenario])
        signs = {"positive": count - negative, "negative": negative}
        if scenario.startswith("rounded_"):
            directions = apportion(count, direction_weights)
            joint = two_by_two_allocation(signs, directions)
            for sign in signs:
                for direction in directions:
                    joint_count = joint[(sign, direction)]
                    if joint_count:
                        targets[(scenario, sign, direction)] = joint_count
        else:
            for sign, sign_count in signs.items():
                if sign_count:
                    targets[(scenario, sign, "not_applicable")] = sign_count
    return targets


def length_band_targets(op, total, spec):
    targets = collections.Counter()
    for key, count in length_targets(op, total, spec).items():
        da, db = parse_length_key(key)
        band = ("full" if da == db == 3 else
                "left_mixed" if da == 3 else
                "right_mixed" if db == 3 else "small")
        targets[band] += count
    return {band: targets[band] for band in (
        "small", "left_mixed", "right_mixed", "full")}


def scenario_length_allocations(op, total, spec, mechanisms):
    """Split scenarios across coarse length bands and repair exact marginals.

    Signed scenarios are repaired separately from unsigned scenarios.  This
    makes their band totals agree with the configured negative-first quota,
    rather than allowing a global band repair to move signed rows into length
    families that do not have enough negative primary slots.
    """
    weights_by_scenario = spec["operations"][op][
        "scenario_length_band_weights"]
    if set(weights_by_scenario) != set(mechanisms):
        raise ValueError(f"{op} scenario length-band configuration is incomplete")
    allocations = {
        scenario: apportion(count, weights_by_scenario[scenario])
        for scenario, count in mechanisms.items()}
    global_targets = collections.Counter(length_band_targets(op, total, spec))

    negative_ratio = float(spec["domain"]["negative_first_operand_ratio"])
    signed_targets = collections.Counter()
    for key, count in length_targets(op, total, spec).items():
        da, db = parse_length_key(key)
        band = ("full" if da == db == 3 else
                "left_mixed" if da == 3 else
                "right_mixed" if db == 3 else "small")
        signs = apportion(
            count, {"positive": 1.0 - negative_ratio,
                    "negative": negative_ratio})
        signed_targets[band] += signs["negative"]
    unsigned_targets = collections.Counter({
        band: global_targets[band] - signed_targets[band]
        for band in global_targets})

    def repair(scenarios, targets):
        current = collections.Counter()
        for scenario in scenarios:
            current.update(allocations[scenario])
        if sum(current.values()) != sum(targets.values()):
            raise RuntimeError(
                f"{op} signed scenario total does not match sign quota")
        while current != targets:
            deficits = [band for band in targets
                        if current[band] < targets[band]]
            surpluses = [band for band in targets
                        if current[band] > targets[band]]
            candidates = []
            for scenario in scenarios:
                allocation = allocations[scenario]
                scenario_weights = weights_by_scenario[scenario]
                scenario_total = mechanisms[scenario]
                for surplus in surpluses:
                    if allocation.get(surplus, 0) <= 0:
                        continue
                    for deficit in deficits:
                        if deficit not in scenario_weights:
                            continue
                        surplus_error = (
                            allocation.get(surplus, 0)
                            - scenario_total * scenario_weights.get(surplus, 0))
                        deficit_error = (
                            allocation.get(deficit, 0)
                            - scenario_total * scenario_weights[deficit])
                        before = surplus_error ** 2 + deficit_error ** 2
                        after = ((surplus_error - 1) ** 2
                                 + (deficit_error + 1) ** 2)
                        candidates.append(
                            (after - before, scenario, surplus, deficit))
            if not candidates:
                raise RuntimeError(
                    f"cannot repair {op} scenario length-band allocation: "
                    f"{dict(current)} != {dict(targets)}")
            _, scenario, surplus, deficit = min(candidates)
            allocations[scenario][surplus] -= 1
            allocations[scenario][deficit] = (
                allocations[scenario].get(deficit, 0) + 1)
            current[surplus] -= 1
            current[deficit] += 1

    signed_scenarios = [
        scenario for scenario in mechanisms if scenario.startswith("signed_")]
    unsigned_scenarios = [
        scenario for scenario in mechanisms if scenario not in signed_scenarios]
    repair(signed_scenarios, signed_targets)
    repair(unsigned_scenarios, unsigned_targets)
    return allocations


def pattern_length_combination_possible(op, scenario, pattern, band):
    if (op == "+" and scenario == "cascading_carry_one_step" and
            band == "small" and pattern == "contains_zero_digit"):
        return False
    if (op == "-" and scenario == "signed_cascading_carry" and
            band == "small" and pattern == "contains_zero_digit"):
        return False
    if (op == "-" and scenario == "signed_independent_multi_carry" and
            band in ("small", "left_mixed", "right_mixed") and
            pattern == "contains_zero_digit"):
        return False
    return True


def cross_pattern_and_length(op, scenario, pattern_counts, band_counts):
    """Create an exact 2x3 table while respecting structural impossibility."""
    zero = "contains_zero_digit"
    no_zero = "no_zero_digits"
    zero_target = pattern_counts.get(zero, 0)
    bounds = {}
    for band, count in band_counts.items():
        zero_allowed = pattern_length_combination_possible(
            op, scenario, zero, band)
        no_zero_allowed = pattern_length_combination_possible(
            op, scenario, no_zero, band)
        if not zero_allowed and not no_zero_allowed:
            raise RuntimeError(f"no pattern is possible for {scenario} {band}")
        minimum = count if not no_zero_allowed else 0
        maximum = count if zero_allowed else 0
        bounds[band] = [minimum, maximum]
    minimum_total = sum(value[0] for value in bounds.values())
    maximum_total = sum(value[1] for value in bounds.values())
    if not minimum_total <= zero_target <= maximum_total:
        raise RuntimeError(
            f"{op} {scenario} zero target {zero_target} is incompatible with "
            f"length bands {band_counts}")
    zeros = {band: value[0] for band, value in bounds.items()}
    remaining = zero_target - minimum_total
    while remaining:
        slack = {band: bounds[band][1] - zeros[band]
                 for band in band_counts if zeros[band] < bounds[band][1]}
        if not slack:
            raise AssertionError("zero-pattern capacity disappeared")
        proposal = apportion(remaining, slack)
        moved = 0
        for band, amount in proposal.items():
            amount = min(amount, slack[band])
            zeros[band] += amount
            remaining -= amount
            moved += amount
        if not moved:
            band = next(iter(slack))
            zeros[band] += 1
            remaining -= 1
    targets = collections.OrderedDict()
    for band, count in band_counts.items():
        if zeros[band]:
            targets[(scenario, zero, band)] = zeros[band]
        if count - zeros[band]:
            targets[(scenario, no_zero, band)] = count - zeros[band]
    return targets


def multiplication_allocation_targets(total, spec):
    """Cross multiplication scenario, length, central band, and sign.

    Identity expressions cannot have two three-digit operands.  The remaining
    scenarios divide the residual capacity in the same proportions across all
    four length families. Full-width rows are then balanced across the four
    cell-relative central-column bands. The primary allocation independently
    preserves the exact 80/10/10 sign marginal inside every operand-length
    cell. Every scenario must contain all three sign forms, but sign is not
    forced as a fourth exact joint axis because that is integer-infeasible in
    the small validation micro-cells.
    """
    operation = spec["operations"]["*"]
    mechanisms = scenario_targets("*", total, spec)
    weights_by_scenario = operation["scenario_length_band_weights"]
    if set(weights_by_scenario) != set(mechanisms):
        raise ValueError(
            "multiplication scenario length-band configuration is incomplete")
    allocations = {
        scenario: apportion(count, weights_by_scenario[scenario])
        for scenario, count in mechanisms.items()}
    global_targets = collections.Counter(length_band_targets("*", total, spec))
    current = collections.Counter()
    for allocation in allocations.values():
        current.update(allocation)

    while current != global_targets:
        deficits = [band for band in global_targets
                    if current[band] < global_targets[band]]
        surpluses = [band for band in global_targets
                    if current[band] > global_targets[band]]
        candidates = []
        for scenario, allocation in allocations.items():
            scenario_weights = weights_by_scenario[scenario]
            scenario_total = mechanisms[scenario]
            for surplus in surpluses:
                if allocation.get(surplus, 0) <= 0:
                    continue
                for deficit in deficits:
                    if deficit not in scenario_weights:
                        continue
                    surplus_error = (
                        allocation.get(surplus, 0)
                        - scenario_total * scenario_weights.get(surplus, 0))
                    deficit_error = (
                        allocation.get(deficit, 0)
                        - scenario_total * scenario_weights[deficit])
                    before = surplus_error ** 2 + deficit_error ** 2
                    after = ((surplus_error - 1) ** 2
                             + (deficit_error + 1) ** 2)
                    candidates.append(
                        (after - before, scenario, surplus, deficit))
        if not candidates:
            raise RuntimeError(
                "cannot repair multiplication scenario length bands: "
                f"{dict(current)} != {dict(global_targets)}")
        _, scenario, surplus, deficit = min(candidates)
        allocations[scenario][surplus] -= 1
        allocations[scenario][deficit] = (
            allocations[scenario].get(deficit, 0) + 1)
        current[surplus] -= 1
        current[deficit] += 1

    central_weights = operation["three_by_three"][
        "scenario_central_band_weights"]
    if len(central_weights) != 4:
        raise ValueError(
            "multiplication scenario central-band weights require four values")
    central_weight_map = collections.OrderedDict(
        (band, weight) for band, weight in enumerate(central_weights))
    full_scenario_totals = collections.OrderedDict(
        (scenario, allocations[scenario].get("full", 0))
        for scenario in mechanisms
        if allocations[scenario].get("full", 0))
    full_central_allocations = split_micro_categories(
        full_scenario_totals, central_weight_map)
    targets = collections.OrderedDict()
    for scenario in mechanisms:
        for band, count in allocations[scenario].items():
            if not count:
                continue
            if band == "full":
                central_counts = full_central_allocations[scenario]
                for central, central_count in central_counts.items():
                    if central_count:
                        targets[(scenario, band, central)] = central_count
            else:
                targets[(scenario, band, "not_applicable")] = count
    return targets


def allocation_targets(op, total, spec):
    """Return exact joint flow-column targets for operation-specific axes."""
    mechanisms = scenario_targets(op, total, spec)
    if op == "/":
        return division_allocation_targets(total, spec)
    if op == "*":
        return multiplication_allocation_targets(total, spec)
    if op not in ("+", "-"):
        return mechanisms
    pattern_targets = apportion(
        total, spec["operations"][op]["digit_pattern_weights"])
    zero_target = pattern_targets["contains_zero_digit"]
    # These are consequences of the scenario definitions, not sampling
    # preferences. Addition identity contains operand zero; an independent
    # multi-carry cannot contain a zero digit. Borrow-across-zero necessarily
    # contains a zero in the minuend.
    forced_zero = (
        {"identity_or_zero",
         "signed_negative_borrow_across_zero",
         "signed_positive_borrow_across_zero"}
        if op == "+" else
        {"subtract_zero", "borrow_across_zero",
         "negative_result_borrow_across_zero"})
    forced_no_zero = ({"independent_multi_carry"} if op == "+" else set())
    forced_zero_count = sum(mechanisms[name] for name in forced_zero)
    remaining_zero_target = zero_target - forced_zero_count
    compatible = {
        scenario: count for scenario, count in mechanisms.items()
        if scenario not in forced_zero | forced_no_zero}
    if remaining_zero_target < 0 or remaining_zero_target > sum(
            compatible.values()):
        raise ValueError(
            f"{op} zero-digit target is incompatible with scenario quotas")
    additional_zeros = apportion(remaining_zero_target, compatible)
    targets = collections.OrderedDict()
    for scenario, count in mechanisms.items():
        if scenario in forced_zero:
            zero_count = count
        elif scenario in forced_no_zero:
            zero_count = 0
        else:
            zero_count = additional_zeros[scenario]
        if zero_count:
            targets[(scenario, "contains_zero_digit")] = zero_count
        if count - zero_count:
            targets[(scenario, "no_zero_digits")] = count - zero_count
    length_allocations = scenario_length_allocations(
        op, total, spec, mechanisms)
    pattern_counts = collections.defaultdict(collections.Counter)
    for (scenario, pattern), count in targets.items():
        pattern_counts[scenario][pattern] += count
    crossed = collections.OrderedDict()
    for scenario in mechanisms:
        crossed.update(cross_pattern_and_length(
            op, scenario, pattern_counts[scenario],
            length_allocations[scenario]))
    return crossed


def configured_allocation_categories(op, train_total, val_total, spec):
    """Return every category required by either split in stable order."""
    categories = collections.OrderedDict()
    for total in (train_total, val_total):
        for category in allocation_targets(op, total, spec):
            categories.setdefault(category, None)
    return categories


def split_micro_categories(micro_totals, weights):
    """Apportion categories per micro-cell and repair global rounding drift."""
    categories = list(weights)
    total = sum(micro_totals.values())
    global_targets = apportion(total, weights)
    allocations = {
        key: apportion(count, weights) for key, count in micro_totals.items()}
    current = collections.Counter()
    for allocation in allocations.values():
        current.update(allocation)

    while current != collections.Counter(global_targets):
        deficits = [
            category for category in categories
            if current[category] < global_targets[category]]
        surpluses = [
            category for category in categories
            if current[category] > global_targets[category]]
        if not deficits or not surpluses:
            raise AssertionError("could not repair category apportionment")
        deficit = max(
            deficits, key=lambda category: global_targets[category] - current[category])
        surplus = max(
            surpluses, key=lambda category: current[category] - global_targets[category])
        best_key = min(
            (key for key in micro_totals if allocations[key][surplus] > 0),
            key=lambda key: (
                abs((allocations[key][surplus] - 1)
                    - micro_totals[key] * weights[surplus])
                + abs((allocations[key][deficit] + 1)
                      - micro_totals[key] * weights[deficit])
                - abs(allocations[key][surplus]
                      - micro_totals[key] * weights[surplus])
                - abs(allocations[key][deficit]
                      - micro_totals[key] * weights[deficit]),
                str(key),
            ))
        allocations[best_key][surplus] -= 1
        allocations[best_key][deficit] += 1
        current[surplus] -= 1
        current[deficit] += 1
    return allocations


def multiplication_central_total(a, b):
    """Schoolbook k=2 total, including carries produced by k=0 and k=1."""
    left = [int(digit) for digit in reversed(f"{abs(a):03d}")]
    right = [int(digit) for digit in reversed(f"{abs(b):03d}")]
    carry = 0
    for column in range(3):
        partial = sum(
            left[i] * right[column - i]
            for i in range(3)
            if 0 <= column - i < 3)
        total = partial + carry
        if column == 2:
            return total
        carry = total // 10
    raise AssertionError("unreachable")


def build_central_thresholds():
    """Approximate central-total quartile boundaries inside all 81 cells."""
    thresholds = {}
    for left_hundreds in range(1, 10):
        for right_hundreds in range(1, 10):
            totals = [
                multiplication_central_total(a, b)
                for a in range(left_hundreds * 100, left_hundreds * 100 + 100)
                for b in range(right_hundreds * 100, right_hundreds * 100 + 100)
            ]
            totals.sort()
            thresholds[(left_hundreds, right_hundreds)] = tuple(
                totals[len(totals) * numerator // 4]
                for numerator in (1, 2, 3))
    return thresholds


def central_band(a, b, thresholds):
    cell = (abs(a) // 100, abs(b) // 100)
    return bisect.bisect_right(
        thresholds[cell], multiplication_central_total(a, b))


def build_primary_targets(op, total, spec):
    """Create exact operand-length/sign targets and operation micro-cells."""
    negative_ratio = float(spec["domain"]["negative_first_operand_ratio"])
    lengths = length_targets(op, total, spec)
    multiplication_sign_weights = spec["operations"]["*"].get(
        "sign_pattern_weights", {})
    division_sign_weights = spec["operations"]["/"].get(
        "sign_pattern_weights", {})
    primary = collections.OrderedDict()
    for key, count in lengths.items():
        da, db = parse_length_key(key)
        if op == "*" and da == 3 and db == 3:
            cells = [(left, right) for left in range(1, 10)
                     for right in range(1, 10)]
            cell_totals = apportion(count, {cell: 1 for cell in cells})
            band_weights = spec["operations"]["*"]["three_by_three"][
                "central_total_band_weights"]
            micro_totals = collections.OrderedDict()
            for cell_index, (cell, cell_count) in enumerate(cell_totals.items()):
                # Rotate tied remainders between cells. This preserves the
                # within-cell max difference of one and makes the four global
                # central-band totals equal whenever the 3x3 total permits it.
                band_order = [
                    (cell_index + offset) % 4 for offset in range(4)]
                bands = apportion(
                    cell_count,
                    {band: band_weights[band] for band in band_order})
                for band, band_count in bands.items():
                    micro_totals[(cell[0], cell[1], band)] = band_count
            sign_allocations = split_micro_categories(
                micro_totals, multiplication_sign_weights)
            for (left, right, band), micro_count in micro_totals.items():
                for sign_pattern in multiplication_sign_weights:
                    primary[("mul3", left, right, band, sign_pattern)] = (
                        sign_allocations[(left, right, band)][sign_pattern])
        elif op == "*":
            sign_patterns = apportion(count, multiplication_sign_weights)
            for sign_pattern, pattern_count in sign_patterns.items():
                primary[("length", da, db, sign_pattern)] = pattern_count
        elif op == "/" and da == 3 and db == 3:
            cells = [(numerator, divisor) for numerator in range(1, 10)
                     for divisor in range(1, 10)]
            cell_totals = collections.OrderedDict(
                apportion(count, {cell: 1 for cell in cells}))
            sign_allocations = split_micro_categories(
                cell_totals, division_sign_weights)
            for (numerator, divisor), cell_count in cell_totals.items():
                for sign in division_sign_weights:
                    primary[("div3", numerator, divisor, sign)] = (
                        sign_allocations[(numerator, divisor)][sign])
        elif op in ("+", "-") and da == 3 and db == 3:
            cells = [(left, right) for left in range(1, 10)
                     for right in range(1, 10)]
            cell_totals = collections.OrderedDict(
                apportion(count, {cell: 1 for cell in cells}))
            prefix = "add3" if op == "+" else "sub3"
            if op == "-":
                sign_weights = {
                    "positive": 1.0 - negative_ratio,
                    "negative": negative_ratio,
                }
                sign_allocations = split_micro_categories(
                    cell_totals, sign_weights)
                for (left, right), cell_count in cell_totals.items():
                    for sign in SIGNS:
                        primary[(prefix, left, right, sign)] = (
                            sign_allocations[(left, right)][sign])
            else:
                for (left, right), cell_count in cell_totals.items():
                    primary[(prefix, left, right)] = cell_count
        else:
            sign_weights = (division_sign_weights if op == "/" else
                            {"positive": 1.0 - negative_ratio,
                             "negative": negative_ratio})
            signs = apportion(count, sign_weights)
            for sign in SIGNS:
                primary[("length", da, db, sign)] = signs[sign]
    if sum(primary.values()) != total:
        raise AssertionError("primary target apportionment changed the row total")
    return primary


def addition_carry_profile(a, b):
    """Return (carry count, whether an incoming carry triggers another)."""
    carry = 0
    carry_count = 0
    cascading = False
    x, y = a, b
    while x or y or carry:
        raw_sum = x % 10 + y % 10
        next_carry = 1 if raw_sum + carry >= 10 else 0
        cascading |= bool(carry and raw_sum == 9 and next_carry)
        carry_count += next_carry
        carry = next_carry
        x //= 10
        y //= 10
    return carry_count, cascading


def addition_cascade_depth(a, b):
    """Longest run in which incoming carries causally trigger later carries."""
    carry = 0
    current_depth = 0
    maximum_depth = 0
    x, y = a, b
    while x or y or carry:
        raw_sum = x % 10 + y % 10
        next_carry = 1 if raw_sum + carry >= 10 else 0
        if carry and raw_sum == 9 and next_carry:
            current_depth += 1
            maximum_depth = max(maximum_depth, current_depth)
        else:
            current_depth = 0
        carry = next_carry
        x //= 10
        y //= 10
    return maximum_depth


def borrow_profile(minuend, subtrahend):
    """Return (borrow count, whether a borrow propagates through a zero)."""
    borrow = 0
    borrow_count = 0
    across_zero = False
    x, y = minuend, subtrahend
    while x or y or borrow:
        x_digit = x % 10
        y_digit = y % 10
        across_zero |= bool(borrow and x_digit == 0)
        next_borrow = 1 if x_digit - y_digit - borrow < 0 else 0
        borrow_count += next_borrow
        borrow = next_borrow
        x //= 10
        y //= 10
    return borrow_count, across_zero


def magnitude_subtraction_band(left, right):
    if left == right:
        return "simple"
    borrows, across_zero = borrow_profile(max(left, right), min(left, right))
    if across_zero:
        return "borrow_across_zero"
    if borrows >= 2:
        return "multi_borrow"
    return "simple"


def scenario_addition(a, b):
    result = a + b
    if a < 0:
        band = magnitude_subtraction_band(abs(a), b)
        if result < 0:
            return f"signed_negative_{band}"
        if result == 0:
            return "signed_result_zero"
        return f"signed_positive_{band}"
    if a == 0 or b == 0:
        return "identity_or_zero"
    carries, _ = addition_carry_profile(a, b)
    cascade_depth = addition_cascade_depth(a, b)
    if cascade_depth >= 2:
        return "cascading_carry_two_step"
    if cascade_depth == 1:
        return "cascading_carry_one_step"
    if result >= 1000:
        return "overflow_non_cascading"
    if carries == 0:
        return "no_carry"
    if carries == 1:
        return "single_carry"
    return "independent_multi_carry"


def scenario_subtraction(a, b):
    if a < 0:
        carries, _ = addition_carry_profile(abs(a), b)
        if addition_cascade_depth(abs(a), b):
            return "signed_cascading_carry"
        if carries >= 2:
            return "signed_independent_multi_carry"
        return "signed_low_carry"
    if b == 0:
        return "subtract_zero"
    if a == b:
        return "equal_operands"
    if a < b:
        band = magnitude_subtraction_band(a, b)
        return f"negative_result_{band}"
    borrows, across_zero = borrow_profile(a, b)
    if across_zero:
        return "borrow_across_zero"
    if borrows == 0:
        return "no_borrow"
    if borrows == 1:
        return "single_borrow"
    return "multi_borrow"


def repeated_digits(value):
    magnitude = str(abs(value))
    return len(magnitude) >= 2 and len(set(magnitude)) == 1


def scenario_multiplication(a, b):
    if abs(a) in (0, 1) or abs(b) in (0, 1):
        return "identity_or_zero"
    if '0' in str(abs(a)) or '0' in str(abs(b)):
        return "contains_zero_digit"
    if repeated_digits(a) or repeated_digits(b):
        return "repeated_digit_operand"
    return "dense_mixed_digits"


def terminating_decimal_places(numerator, denominator):
    divisor = math.gcd(numerator, denominator)
    reduced = denominator // divisor
    twos = fives = 0
    while reduced % 2 == 0:
        reduced //= 2
        twos += 1
    while reduced % 5 == 0:
        reduced //= 5
        fives += 1
    return max(twos, fives) if reduced == 1 else None


def scenario_division(a, b):
    numerator = abs(a)
    if b == 0:
        return "division_by_zero"
    if numerator == 0:
        return "zero_numerator"
    if b == 1:
        return "divide_by_one"
    if numerator == b:
        return "unit_quotient"
    if numerator % b == 0:
        quotient = numerator // b
        if quotient < 10:
            return "exact_q_1digit"
        if quotient < 100:
            return "exact_q_2digit"
        return "exact_q_3digit"
    quotient = numerator / b
    if quotient <= 0.05 + 1e-12:
        return "near_zero_nonzero"
    nearest_integer = round(quotient)
    if (nearest_integer >= 1 and
            abs(quotient - nearest_integer) <= 0.05 + 1e-12):
        return ("near_integer_below" if quotient < nearest_integer
                else "near_integer_above")
    places = terminating_decimal_places(numerator, b)
    if places is not None and places <= 3:
        magnitude = "below_one" if quotient < 1 else "at_least_one"
        return f"terminating_{magnitude}_{places}dp"
    if quotient < 0.1:
        return "rounded_below_small"
    if quotient < 1:
        return "rounded_below_medium"
    if quotient < 10:
        return "rounded_ge1_q1digit"
    if quotient < 100:
        return "rounded_ge1_q2digit"
    return "rounded_ge1_q3digit"


def division_capability_group(scenario):
    """Map division scenarios to the capability regimes used in v3 audits."""
    if scenario == "exact_q_2digit":
        return "exact_two_digit_quotient"
    if scenario in ("near_integer_below", "near_integer_above"):
        return "near_integer_boundary"
    if scenario == "rounded_below_medium":
        return "rounded_below_one"
    if scenario == "rounded_ge1_q1digit":
        return "rounded_one_digit_integer"
    if scenario in (
            "terminating_at_least_one_1dp",
            "terminating_at_least_one_2dp",
            "terminating_at_least_one_3dp"):
        return "terminating_at_least_one"
    return "general"


SCENARIO_FUNCTIONS = {
    "+": scenario_addition,
    "-": scenario_subtraction,
    "*": scenario_multiplication,
    "/": scenario_division,
}


def primary_for_candidate(op, a, b, thresholds):
    if op == "*":
        sign = multiplication_sign_pattern(a, b)
    else:
        sign = "negative" if a < 0 else "positive"
    da, db = ndigits(a), ndigits(b)
    if op == "*" and da == 3 and db == 3:
        return ("mul3", abs(a) // 100, abs(b) // 100,
                central_band(a, b, thresholds), sign)
    if op == "/" and da == 3 and db == 3:
        return ("div3", abs(a) // 100, b // 100, sign)
    if op in ("+", "-") and da == 3 and db == 3:
        prefix = "add3" if op == "+" else "sub3"
        if op == "-":
            return (prefix, abs(a) // 100, b // 100, sign)
        return (prefix, abs(a) // 100, b // 100)
    return ("length", da, db, sign)


def reservoir_add(pools, seen, edge, candidate, cap, rng):
    seen[edge] += 1
    bucket = pools.setdefault(edge, [])
    if len(bucket) < cap:
        bucket.append(candidate)
        return
    replacement = rng.randrange(seen[edge])
    if replacement < cap:
        bucket[replacement] = candidate


def build_candidate_pools(op, primary_limits, allocation_categories, spec,
                          thresholds, rng):
    """Enumerate the supported domain into bounded, diverse edge reservoirs."""
    cap_limit = int(spec["dataset"]["candidate_pool_cap"])
    cap_multiplier = int(spec["dataset"]["candidate_pool_multiplier"])
    caps = {
        primary: max(64, min(cap_limit, limit * cap_multiplier))
        for primary, limit in primary_limits.items()
    }
    pools = {}
    seen = collections.Counter()
    allowed_categories = set(allocation_categories)
    if op == "*":
        sign_configurations = [
            ("positive_positive", False, False),
            ("negative_positive", True, False),
            ("negative_negative", True, True),
        ]
    else:
        sign_configurations = [
            ("positive", False, False),
            ("negative", True, False),
        ]
    for _, negative_a, negative_b in sign_configurations:
        for da in range(1, 4):
            for magnitude_a in digit_values(da, negative=negative_a):
                a = -magnitude_a if negative_a else magnitude_a
                for db in range(1, 4):
                    for magnitude_b in digit_values(db, negative=negative_b):
                        b = -magnitude_b if negative_b else magnitude_b
                        primary = primary_for_candidate(
                            op, a, b, thresholds)
                        if primary not in primary_limits:
                            continue
                        scenario = SCENARIO_FUNCTIONS[op](a, b)
                        configured_scenarios = set(
                            spec["operations"][op]["scenario_weights"])
                        if scenario not in configured_scenarios:
                            raise ValueError(
                                f"unconfigured {op} scenario {scenario!r}")
                        if op in ("+", "-"):
                            category = (
                                scenario,
                                operand_digit_pattern(a, b),
                                operand_length_band(a, b),
                            )
                        elif op == "*":
                            length_band = operand_length_band(a, b)
                            central = (
                                central_band(a, b, thresholds)
                                if length_band == "full"
                                else "not_applicable")
                            category = (
                                scenario,
                                length_band,
                                central,
                            )
                        elif op == "/":
                            category = (
                                scenario,
                                division_sign(a),
                                division_rounding_direction(a, b, scenario),
                            )
                        else:
                            category = scenario
                        if category not in allowed_categories:
                            # A structurally valid combination may intentionally
                            # receive a zero quota after global length-band repair.
                            continue
                        reservoir_add(
                            pools, seen, (primary, category), (a, b),
                            caps[primary], rng)
    return {edge: tuple(candidates) for edge, candidates in pools.items()}, seen


@dataclass
class FlowEdge:
    to: int
    reverse: int
    capacity: int
    original: int


class Dinic:
    def __init__(self, node_count):
        self.graph = [[] for _ in range(node_count)]

    def add_edge(self, source, target, capacity):
        forward_index = len(self.graph[source])
        reverse_index = len(self.graph[target])
        self.graph[source].append(
            FlowEdge(target, reverse_index, capacity, capacity))
        self.graph[target].append(FlowEdge(source, forward_index, 0, 0))
        return forward_index

    def maximum_flow(self, source, sink):
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = collections.deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity and levels[edge.to] < 0:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if levels[sink] < 0:
                return total
            positions = [0] * len(self.graph)

            def send(node, amount):
                if node == sink:
                    return amount
                while positions[node] < len(self.graph[node]):
                    edge = self.graph[node][positions[node]]
                    if edge.capacity and levels[edge.to] == levels[node] + 1:
                        sent = send(edge.to, min(amount, edge.capacity))
                        if sent:
                            edge.capacity -= sent
                            self.graph[edge.to][edge.reverse].capacity += sent
                            return sent
                    positions[node] += 1
                return 0

            while True:
                sent = send(source, 10 ** 18)
                if not sent:
                    break
                total += sent


def allocate_rows(primary_targets, category_counts, pools, repeat_limit,
                  unique=False):
    """Solve exact primary and scenario marginals as a bipartite max flow."""
    primaries = list(primary_targets)
    scenarios = list(category_counts)
    source = 0
    primary_offset = 1
    scenario_offset = primary_offset + len(primaries)
    sink = scenario_offset + len(scenarios)
    flow = Dinic(sink + 1)
    source_edges = {}
    for index, primary in enumerate(primaries):
        source_edges[primary] = flow.add_edge(
            source, primary_offset + index, primary_targets[primary])
    sink_edges = {}
    for index, scenario in enumerate(scenarios):
        sink_edges[scenario] = flow.add_edge(
            scenario_offset + index, sink, category_counts[scenario])

    tracked = collections.defaultdict(list)
    for primary_index, primary in enumerate(primaries):
        for scenario_index, scenario in enumerate(scenarios):
            candidates = pools.get((primary, scenario), ())
            if not candidates:
                continue
            # First expose only one slot per retained prompt. Running this flow
            # before adding repetition capacity maximizes prompt diversity.
            capacity = min(len(candidates), primary_targets[primary])
            node = primary_offset + primary_index
            edge_index = flow.add_edge(
                node, scenario_offset + scenario_index, capacity)
            tracked[(primary, scenario)].append((node, edge_index))

    expected = sum(primary_targets.values())
    achieved = flow.maximum_flow(source, sink)
    if not unique and achieved < expected and repeat_limit > 1:
        for primary_index, primary in enumerate(primaries):
            for scenario_index, scenario in enumerate(scenarios):
                candidates = pools.get((primary, scenario), ())
                if not candidates:
                    continue
                capacity = min(
                    len(candidates) * (repeat_limit - 1),
                    primary_targets[primary])
                if not capacity:
                    continue
                node = primary_offset + primary_index
                edge_index = flow.add_edge(
                    node, scenario_offset + scenario_index, capacity)
                tracked[(primary, scenario)].append((node, edge_index))
        achieved += flow.maximum_flow(source, sink)
    if achieved != expected:
        missing_scenarios = {
            scenario: flow.graph[scenario_offset + index][
                sink_edges[scenario]].capacity
            for index, scenario in enumerate(scenarios)
            if flow.graph[scenario_offset + index][sink_edges[scenario]].capacity
        }
        missing_primaries = {
            primary: flow.graph[source][source_edges[primary]].capacity
            for primary in primaries
            if flow.graph[source][source_edges[primary]].capacity
        }
        raise RuntimeError(
            f"distribution is infeasible: allocated {achieved}/{expected}; "
            f"unfilled primary capacities={missing_primaries}; "
            f"unfilled scenario capacities={missing_scenarios}")

    allocation = {}
    for edge, graph_locations in tracked.items():
        used = sum(
            flow.graph[node][edge_index].original
            - flow.graph[node][edge_index].capacity
            for node, edge_index in graph_locations)
        if used:
            allocation[edge] = used
    return allocation


def filtered_pools(pools, op, excluded):
    return {
        edge: tuple(
            (a, b) for a, b in candidates if (a, b, op) not in excluded)
        for edge, candidates in pools.items()
    }


def sample_allocation(op, allocation, pools, rng, unique=False):
    samples = []
    used = set()
    for (primary, category), count in allocation.items():
        candidates = list(pools[(primary, category)])
        rng.shuffle(candidates)
        if unique and count > len(candidates):
            raise RuntimeError(
                f"not enough unique candidates for {op} {primary} {category}")
        for index in range(count):
            a, b = candidates[index if unique else index % len(candidates)]
            key = (a, b, op)
            if unique and key in used:
                raise AssertionError("candidate appeared in two allocation edges")
            used.add(key)
            scenario = category[0] if isinstance(category, tuple) else category
            samples.append({
                "a": a,
                "b": b,
                "op": op,
                "scenario": scenario,
                "primary": primary,
            })
            if op in ("+", "-"):
                samples[-1]["digit_pattern"] = category[1]
    rng.shuffle(samples)
    return samples


def parse_natural_prompt(prompt):
    left, right, op, _ = parse_expression(prompt)
    return int(left), int(right), op


def load_reserved_prompts(spec):
    reserved = set()
    for suite_path in spec["dataset"].get("reserved_test_suites", []):
        with open(suite_path) as f:
            suite = yaml.safe_load(f)
        groups = suite.get("groups", [{"data": suite.get("data", [])}])
        for group in groups:
            for prompt in group.get("data", []):
                key = parse_natural_prompt(prompt)
                reserved.add(key)
                a, b, op = key
                if supports_commutative_swap(a, b, op):
                    reserved.add((b, a, op))
    return reserved


def load_excluded_jsonl(paths, internal_format=True):
    """Load prompts from existing datasets so a fresh split cannot reuse them."""
    excluded = set()
    for path in paths:
        with open(path) as f:
            for line_number, line in enumerate(f, 1):
                row = json.loads(line)
                lhs = row["text"].split("=", 1)[0]
                left, right, op, _ = parse_expression(lhs)
                if internal_format and should_reverse(op):
                    left = unreverse_magnitude(left)
                    right = unreverse_magnitude(right)
                try:
                    key = (int(left), int(right), op)
                except ValueError as exc:
                    raise ValueError(
                        f"cannot parse prompt at {path}:{line_number}: {lhs!r}"
                    ) from exc
                excluded.add(key)
                a, b, operation = key
                if supports_commutative_swap(a, b, operation):
                    excluded.add((b, a, operation))
    return excluded


def build_rows(samples, split, reverse, max_seq_len,
               division_answer_format="fixed_width", representation=None,
               division_focus_scenarios=()):
    rows = []
    stoi, _ = V.build_vocab()
    if representation is None:
        representation = "abacus-v1" if reverse else "natural-v1"
    focus_scenarios = set(division_focus_scenarios)
    for sample in samples:
        a, b, op = sample["a"], sample["b"], sample["op"]
        result = compute(a, b, op)
        if result is None or not verify(a, b, op, result):
            raise AssertionError(f"arithmetic verification failed for {(a, b, op)}")
        text = make_text(
            a, b, op, result, reverse,
            division_answer_format=division_answer_format)
        token_count = 1 + len(V.encode(text, stoi)) + 1
        if token_count > max_seq_len:
            raise ValueError(
                f"{text!r} requires {token_count} tokens; limit is {max_seq_len}")
        stored_answer = text.split('=', 1)[1]
        decoded_answer = decode_internal_answer(
            op, stored_answer, internal_format=reverse)
        if decoded_answer != str(result):
            raise AssertionError(
                f"representation round trip failed: {text!r} -> {decoded_answer!r} "
                f"instead of {str(result)!r}")
        tier, difficulty = classify(a, b, op, result)
        row = {
            "text": text,
            "split": split,
            "tier": tier,
            "difficulty": round(difficulty, 4),
            "representation": representation,
            "operation": op,
            "scenario": sample["scenario"],
            "operand_digits": length_key(a, b),
            "first_operand_negative": a < 0,
            "second_operand_negative": b < 0,
        }
        if op == "*":
            row["multiplication_sign_pattern"] = multiplication_sign_pattern(a, b)
            row["operand_length_band"] = operand_length_band(a, b)
        if op in ("+", "-"):
            row["operand_digit_pattern"] = operand_digit_pattern(a, b)
            row["operand_length_band"] = operand_length_band(a, b)
        if op == "/":
            row["division_sign"] = division_sign(a)
            row["rounding_direction"] = division_rounding_direction(
                a, b, sample["scenario"])
            row["division_answer_format"] = division_answer_format
            row["division_capability_group"] = division_capability_group(
                sample["scenario"])
            row["division_capability_focus"] = (
                sample["scenario"] in focus_scenarios)
        primary = sample["primary"]
        if primary[0] == "mul3":
            row["hundreds_cell"] = f"{primary[1]}xx*{primary[2]}xx"
            row["central_total_band"] = primary[3]
            row["central_total"] = multiplication_central_total(a, b)
        elif primary[0] == "div3":
            row["hundreds_cell"] = f"{primary[1]}xx/{primary[2]}xx"
        elif primary[0] == "add3":
            row["hundreds_cell"] = f"{primary[1]}xx+{primary[2]}xx"
        elif primary[0] == "sub3":
            row["hundreds_cell"] = f"{primary[1]}xx-{primary[2]}xx"
        rows.append(row)
    return rows


def expected_sign_counts(cell_total, negative_ratio):
    return apportion(
        cell_total,
        {"positive": 1.0 - negative_ratio, "negative": negative_ratio})


def validate_distribution(train_samples, val_samples, train_rows, val_rows,
                          spec, reserved):
    """Assert the specification and return a JSON-serializable audit report."""
    expected_split_sizes = {
        "train": int(spec["dataset"]["train_rows"]),
        "val": int(spec["dataset"]["validation_rows"]),
    }
    sample_splits = {"train": train_samples, "val": val_samples}
    row_splits = {"train": train_rows, "val": val_rows}
    report = {
        "spec": spec["name"],
        "version": spec["version"],
        "division_answer_format": spec["domain"].get(
            "division_answer_format", "fixed_width"),
        "splits": {},
    }

    for split, samples in sample_splits.items():
        if len(samples) != expected_split_sizes[split]:
            raise AssertionError(
                f"{split} has {len(samples)}, expected {expected_split_sizes[split]}")
        op_expected = operation_targets(len(samples), spec)
        op_actual = collections.Counter(sample["op"] for sample in samples)
        if dict(op_actual) != dict(op_expected):
            raise AssertionError(f"{split} operation quota mismatch")

        split_report = {
            "rows": len(samples),
            "operations": dict(op_actual),
            "by_operation": {},
        }
        for op, op_total in op_expected.items():
            op_samples = [sample for sample in samples if sample["op"] == op]
            expected_lengths = length_targets(op, op_total, spec)
            actual_lengths = collections.Counter(
                length_key(sample["a"], sample["b"]) for sample in op_samples)
            if dict(actual_lengths) != dict(expected_lengths):
                raise AssertionError(f"{split} {op} operand-length quota mismatch")

            expected_scenarios = scenario_targets(op, op_total, spec)
            actual_scenarios = collections.Counter(
                sample["scenario"] for sample in op_samples)
            if dict(actual_scenarios) != dict(expected_scenarios):
                raise AssertionError(f"{split} {op} scenario quota mismatch")
            if split == "val":
                minimum = int(spec["validation"][
                    "min_validation_rows_per_scenario"])
                below_minimum = {
                    scenario: count for scenario, count in actual_scenarios.items()
                    if count < minimum}
                if below_minimum:
                    raise AssertionError(
                        f"{split} {op} scenarios below minimum {minimum}: "
                        f"{below_minimum}")

            negative_ratio = float(
                spec["domain"]["negative_first_operand_ratio"])
            for cell, cell_total in expected_lengths.items():
                if op == "/":
                    expected_signs = apportion(
                        cell_total,
                        spec["operations"]["/"]["sign_pattern_weights"])
                else:
                    expected_signs = expected_sign_counts(
                        cell_total, negative_ratio)
                in_cell = [sample for sample in op_samples
                           if length_key(sample["a"], sample["b"]) == cell]
                actual_signs = collections.Counter(
                    "negative" if sample["a"] < 0 else "positive"
                    for sample in in_cell)
                normalized_signs = {
                    sign: actual_signs[sign] for sign in expected_signs}
                if normalized_signs != dict(expected_signs):
                    raise AssertionError(
                        f"{split} {op} {cell} sign quota mismatch: "
                        f"{normalized_signs} != {dict(expected_signs)}")

                if op == "*":
                    expected_patterns = apportion(
                        cell_total,
                        spec["operations"]["*"]["sign_pattern_weights"])
                    actual_patterns = collections.Counter(
                        multiplication_sign_pattern(sample["a"], sample["b"])
                        for sample in in_cell)
                    normalized_patterns = {
                        pattern: actual_patterns[pattern]
                        for pattern in expected_patterns}
                    if normalized_patterns != dict(expected_patterns):
                        raise AssertionError(
                            f"{split} multiplication {cell} sign-pattern quota "
                            f"mismatch: {normalized_patterns} != "
                            f"{dict(expected_patterns)}")

            split_report["by_operation"][op] = {
                "operand_lengths": dict(actual_lengths),
                "scenarios": dict(actual_scenarios),
            }
            if op in ("+", "-"):
                actual_allocation = collections.Counter(
                    (sample["scenario"],
                     operand_digit_pattern(sample["a"], sample["b"]),
                     operand_length_band(sample["a"], sample["b"]))
                    for sample in op_samples)
                expected_allocation = allocation_targets(op, op_total, spec)
                normalized_allocation = {
                    category: actual_allocation[category]
                    for category in expected_allocation}
                if normalized_allocation != dict(expected_allocation):
                    raise AssertionError(
                        f"{split} {op} scenario/digit-pattern/length-band "
                        f"quota mismatch: {normalized_allocation} != "
                        f"{dict(expected_allocation)}")
                actual_joint = collections.Counter()
                actual_scenario_lengths = collections.Counter()
                for (scenario, pattern, band), count in actual_allocation.items():
                    actual_joint[(scenario, pattern)] += count
                    actual_scenario_lengths[(scenario, band)] += count
                expected_patterns = apportion(
                    op_total,
                    spec["operations"][op]["digit_pattern_weights"])
                actual_patterns = collections.Counter(
                    operand_digit_pattern(sample["a"], sample["b"])
                    for sample in op_samples)
                normalized_patterns = {
                    pattern: actual_patterns[pattern]
                    for pattern in expected_patterns}
                if normalized_patterns != dict(expected_patterns):
                    raise AssertionError(
                        f"{split} {op} digit-pattern quota mismatch: "
                        f"{normalized_patterns} != {dict(expected_patterns)}")
                split_report["by_operation"][op]["digit_patterns"] = dict(
                    actual_patterns)
                split_report["by_operation"][op][
                    "scenario_digit_patterns"] = {
                        f"{scenario}|{pattern}": count
                        for (scenario, pattern), count in actual_joint.items()}
                split_report["by_operation"][op][
                    "scenario_length_bands"] = {
                        f"{scenario}|{band}": count
                        for (scenario, band), count
                        in actual_scenario_lengths.items()}
            if op == "*":
                actual_allocation = collections.Counter()
                for sample in op_samples:
                    band = operand_length_band(sample["a"], sample["b"])
                    central = (sample["primary"][3]
                               if band == "full" else "not_applicable")
                    actual_allocation[(
                        sample["scenario"], band, central)] += 1
                expected_allocation = allocation_targets("*", op_total, spec)
                normalized_allocation = {
                    category: actual_allocation[category]
                    for category in expected_allocation}
                if normalized_allocation != dict(expected_allocation):
                    raise AssertionError(
                        f"{split} multiplication scenario/length/central "
                        "quota mismatch")
                scenario_lengths = collections.Counter()
                scenario_central = collections.Counter()
                scenario_signs = collections.Counter()
                for (scenario, band, central), count in actual_allocation.items():
                    scenario_lengths[(scenario, band)] += count
                    if central != "not_applicable":
                        scenario_central[(scenario, central)] += count
                for sample in op_samples:
                    scenario_signs[(
                        sample["scenario"],
                        multiplication_sign_pattern(
                            sample["a"], sample["b"]),
                    )] += 1
                if split == "val":
                    sign_minimum = int(spec["validation"][
                        "min_validation_rows_per_multiplication_scenario_sign"])
                    for scenario in expected_scenarios:
                        counts = {
                            sign: scenario_signs[(scenario, sign)]
                            for sign in spec["operations"]["*"][
                                "sign_pattern_weights"]}
                        if min(counts.values()) < sign_minimum:
                            raise AssertionError(
                                f"validation multiplication {scenario} lacks "
                                f"a sign form: {counts}")
                split_report["by_operation"][op]["sign_patterns"] = dict(
                    collections.Counter(
                        multiplication_sign_pattern(sample["a"], sample["b"])
                        for sample in op_samples))
                split_report["by_operation"][op][
                    "scenario_length_bands"] = {
                        f"{scenario}|{band}": count
                        for (scenario, band), count
                        in scenario_lengths.items()}
                split_report["by_operation"][op][
                    "scenario_central_bands"] = {
                        f"{scenario}|{central}": count
                        for (scenario, central), count
                        in scenario_central.items()}
                split_report["by_operation"][op]["scenario_signs"] = {
                    f"{scenario}|{sign}": count
                    for (scenario, sign), count in scenario_signs.items()}
            if op == "/":
                focus_scenarios = set(spec["operations"]["/"].get(
                    "capability_focus_scenarios", []))
                capability_groups = collections.Counter(
                    division_capability_group(sample["scenario"])
                    for sample in op_samples)
                focus_counts = collections.Counter(
                    sample["scenario"] for sample in op_samples
                    if sample["scenario"] in focus_scenarios)
                split_report["by_operation"][op][
                    "capability_groups"] = dict(capability_groups)
                split_report["by_operation"][op][
                    "capability_focus_scenarios"] = dict(focus_counts)
                split_report["by_operation"][op][
                    "capability_focus_rows"] = sum(focus_counts.values())
                actual_joint = collections.Counter(
                    (sample["scenario"],
                     division_sign(sample["a"]),
                     division_rounding_direction(
                         sample["a"], sample["b"], sample["scenario"]))
                    for sample in op_samples)
                expected_joint = allocation_targets(op, op_total, spec)
                normalized_joint = {
                    category: actual_joint[category]
                    for category in expected_joint}
                if normalized_joint != dict(expected_joint):
                    raise AssertionError(
                        f"{split} division scenario/sign/direction quota "
                        f"mismatch: {normalized_joint} != "
                        f"{dict(expected_joint)}")
                direction_minimum = int(spec["validation"][
                    "min_validation_rows_per_division_rounding_direction"])
                direction_by_scenario = collections.defaultdict(
                    collections.Counter)
                sign_by_scenario = collections.defaultdict(collections.Counter)
                for sample in op_samples:
                    scenario = sample["scenario"]
                    sign_by_scenario[scenario][
                        division_sign(sample["a"])] += 1
                    direction = division_rounding_direction(
                        sample["a"], sample["b"], scenario)
                    if direction != "not_applicable":
                        direction_by_scenario[scenario][direction] += 1
                if split == "val":
                    for scenario, counts in direction_by_scenario.items():
                        if set(counts) != {"down", "up"}:
                            raise AssertionError(
                                f"validation {scenario} lacks a rounding direction")
                        if min(counts.values()) < direction_minimum:
                            raise AssertionError(
                                f"validation {scenario} rounding direction "
                                f"below minimum {direction_minimum}")
                split_report["by_operation"][op]["scenario_signs"] = {
                    scenario: dict(counts)
                    for scenario, counts in sign_by_scenario.items()}
                split_report["by_operation"][op]["rounding_directions"] = {
                    scenario: dict(counts)
                    for scenario, counts in direction_by_scenario.items()}

        multiplication = [sample for sample in samples
                          if sample["op"] == "*" and
                          length_key(sample["a"], sample["b"]) == "3x3"]
        cell_counts = collections.Counter(
            f"{abs(sample['a']) // 100}xx*{abs(sample['b']) // 100}xx"
            for sample in multiplication)
        if len(cell_counts) != 81 or max(cell_counts.values()) - min(
                cell_counts.values()) > 1:
            raise AssertionError(f"{split} multiplication hundreds cells are uneven")
        if split == "val" and min(cell_counts.values()) < int(
                spec["validation"][
                    "min_validation_rows_per_multiplication_hundreds_cell"]):
            raise AssertionError("validation multiplication cell minimum not met")
        band_counts = collections.defaultdict(collections.Counter)
        global_band_counts = collections.Counter()
        for sample in multiplication:
            primary = sample["primary"]
            cell = f"{primary[1]}xx*{primary[2]}xx"
            band_counts[cell][primary[3]] += 1
            global_band_counts[primary[3]] += 1
        for cell, counts in band_counts.items():
            if set(counts) != {0, 1, 2, 3}:
                raise AssertionError(f"{split} {cell} is missing a central band")
            if max(counts.values()) - min(counts.values()) > 1:
                raise AssertionError(f"{split} {cell} central bands are uneven")
            if split == "val" and min(counts.values()) < int(
                    spec["validation"][
                        "min_validation_rows_per_multiplication_cell_band"]):
                raise AssertionError(
                    f"validation {cell} central-band minimum not met")
        band_weights = spec["operations"]["*"]["three_by_three"][
            "central_total_band_weights"]
        expected_global_bands = apportion(
            len(multiplication),
            {band: band_weights[band] for band in range(4)})
        if dict(global_band_counts) != dict(expected_global_bands):
            raise AssertionError(
                f"{split} global multiplication central-band quota mismatch: "
                f"{dict(global_band_counts)} != {dict(expected_global_bands)}")
        if split == "val" and min(global_band_counts.values()) < int(
                spec["validation"][
                    "min_validation_rows_per_multiplication_central_band"]):
            raise AssertionError("validation global central-band minimum not met")
        split_report["multiplication_3x3"] = {
            "hundreds_cells": dict(sorted(cell_counts.items())),
            "cell_min": min(cell_counts.values()),
            "cell_max": max(cell_counts.values()),
            "central_bands": dict(sorted(global_band_counts.items())),
            "central_bands_balanced": True,
            "sign_patterns": dict(collections.Counter(
                multiplication_sign_pattern(sample["a"], sample["b"])
                for sample in multiplication)),
        }

        division = [sample for sample in samples
                    if sample["op"] == "/" and
                    length_key(sample["a"], sample["b"]) == "3x3"]
        division_cells = collections.Counter(
            f"{abs(sample['a']) // 100}xx/{sample['b'] // 100}xx"
            for sample in division)
        if len(division_cells) != 81 or max(division_cells.values()) - min(
                division_cells.values()) > 1:
            raise AssertionError(
                f"{split} division hundreds cells are uneven")
        if split == "val" and min(division_cells.values()) < int(
                spec["validation"][
                    "min_validation_rows_per_division_hundreds_cell"]):
            raise AssertionError("validation division cell minimum not met")
        split_report["division_3x3"] = {
            "hundreds_cells": dict(sorted(division_cells.items())),
            "cell_min": min(division_cells.values()),
            "cell_max": max(division_cells.values()),
            "signs": dict(collections.Counter(
                division_sign(sample["a"]) for sample in division)),
        }

        for arithmetic_op, report_key, symbol in (
                ("+", "addition_3x3", "+"),
                ("-", "subtraction_3x3", "-")):
            arithmetic = [sample for sample in samples
                          if sample["op"] == arithmetic_op and
                          length_key(sample["a"], sample["b"]) == "3x3"]
            arithmetic_cells = collections.Counter(
                f"{abs(sample['a']) // 100}xx{symbol}"
                f"{sample['b'] // 100}xx"
                for sample in arithmetic)
            if len(arithmetic_cells) != 81 or max(
                    arithmetic_cells.values()) - min(
                        arithmetic_cells.values()) > 1:
                raise AssertionError(
                    f"{split} {arithmetic_op} hundreds cells are uneven")
            if split == "val" and min(arithmetic_cells.values()) < int(
                    spec["validation"][
                        "min_validation_rows_per_add_sub_hundreds_cell"]):
                raise AssertionError(
                    f"validation {arithmetic_op} cell minimum not met")
            signs_balanced_within_every_cell = False
            if arithmetic_op == "-":
                cells = [(left, right) for left in range(1, 10)
                         for right in range(1, 10)]
                cell_totals = collections.OrderedDict(
                    apportion(len(arithmetic), {cell: 1 for cell in cells}))
                sign_weights = {
                    "positive": 1.0 - float(
                        spec["domain"]["negative_first_operand_ratio"]),
                    "negative": float(
                        spec["domain"]["negative_first_operand_ratio"]),
                }
                expected_cell_signs = split_micro_categories(
                    cell_totals, sign_weights)
                actual_cell_signs = collections.Counter(
                    (abs(sample["a"]) // 100, sample["b"] // 100,
                     "negative" if sample["a"] < 0 else "positive")
                    for sample in arithmetic)
                normalized_cell_signs = {
                    (left, right, sign): actual_cell_signs[(left, right, sign)]
                    for left, right in cells for sign in SIGNS}
                expected_flat_cell_signs = {
                    (left, right, sign): expected_cell_signs[
                        (left, right)][sign]
                    for left, right in cells for sign in SIGNS}
                if normalized_cell_signs != expected_flat_cell_signs:
                    raise AssertionError(
                        f"{split} {arithmetic_op} hundreds-cell sign quotas "
                        "do not match")
                signs_balanced_within_every_cell = True
            operation_report = {
                "hundreds_cells": dict(sorted(arithmetic_cells.items())),
                "cell_min": min(arithmetic_cells.values()),
                "cell_max": max(arithmetic_cells.values()),
                "signs_balanced_within_every_cell": (
                    signs_balanced_within_every_cell),
                "signs": dict(collections.Counter(
                    "negative" if sample["a"] < 0 else "positive"
                    for sample in arithmetic)),
            }
            if arithmetic_op == "-":
                negative_results = sum(
                    sample["scenario"].startswith("negative_result_")
                    for sample in arithmetic)
                required = int(spec["validation"][
                    "min_validation_rows_three_by_three_negative_result"])
                if split == "val" and negative_results < required:
                    raise AssertionError(
                        "validation 3x3 negative-result subtraction minimum "
                        f"not met: {negative_results} < {required}")
                operation_report["negative_result_rows"] = negative_results
            split_report[report_key] = operation_report
        report["splits"][split] = split_report

    train_keys = [(sample["a"], sample["b"], sample["op"])
                  for sample in train_samples]
    val_keys = [(sample["a"], sample["b"], sample["op"])
                for sample in val_samples]
    train_set, val_set = set(train_keys), set(val_keys)
    if train_set & val_set:
        raise AssertionError("train/validation prompt overlap")
    if len(val_set) != len(val_keys):
        raise AssertionError("validation prompts are not unique")
    if train_set & reserved or val_set & reserved:
        raise AssertionError("reserved algorithmic test prompt leaked into data")
    for a, b, op in val_set:
        if supports_commutative_swap(a, b, op) and (b, a, op) in train_set:
            raise AssertionError(f"swapped validation leakage for {(a, b, op)}")

    repetitions = collections.Counter(train_keys)
    max_repetitions = max(repetitions.values())
    repetition_limit = int(spec["dataset"]["max_train_repetitions_per_prompt"])
    if max_repetitions > repetition_limit:
        raise AssertionError(
            f"maximum prompt repetition {max_repetitions} exceeds {repetition_limit}")
    report["integrity"] = {
        "train_unique_prompts": len(train_set),
        "validation_unique_prompts": len(val_set),
        "maximum_train_prompt_repetition": max_repetitions,
        "train_validation_overlap": 0,
        "reserved_test_overlap": 0,
        "swapped_commutative_validation_leakage": 0,
        "all_answers_verified": True,
        "all_sequences_within_limit": True,
    }
    if len(train_rows) != len(train_samples) or len(val_rows) != len(val_samples):
        raise AssertionError("row conversion changed split sizes")
    return report


def print_report(report):
    for split, split_report in report["splits"].items():
        print(f"\n{split}: {split_report['rows']:,} rows")
        print("  operations:", split_report["operations"])
        for op, details in split_report["by_operation"].items():
            print(f"  {op} scenarios: {details['scenarios']}")
            if op in ("+", "-"):
                print(f"  {op} digit patterns: {details['digit_patterns']}")
            if op == "*":
                print(f"  {op} sign patterns: {details['sign_patterns']}")
            if op == "/":
                print(f"  {op} rounding directions: "
                      f"{details['rounding_directions']}")
        multiplication = split_report["multiplication_3x3"]
        print(
            "  multiplication 3x3 hundreds-cell rows: "
            f"min={multiplication['cell_min']}, max={multiplication['cell_max']}; "
            "four central bands balanced in every cell")
        division = split_report["division_3x3"]
        print(
            "  division 3x3 hundreds-cell rows: "
            f"min={division['cell_min']}, max={division['cell_max']}")
        for label, key in (("addition", "addition_3x3"),
                           ("subtraction", "subtraction_3x3")):
            operation = split_report[key]
            suffix = (f"; negative results={operation['negative_result_rows']}"
                      if key == "subtraction_3x3" else "")
            print(
                f"  {label} 3x3 hundreds-cell rows: "
                f"min={operation['cell_min']}, max={operation['cell_max']}"
                f"{suffix}")
    print("\nintegrity:", report["integrity"])


def generate(spec, reverse=True, excluded_prompts=()):
    seed = int(spec["dataset"]["seed"])
    rng = random.Random(seed)
    thresholds = build_central_thresholds()
    reserved = load_reserved_prompts(spec)
    reserved.update(excluded_prompts)
    train_operation_counts = operation_targets(
        int(spec["dataset"]["train_rows"]), spec)
    val_operation_counts = operation_targets(
        int(spec["dataset"]["validation_rows"]), spec)
    max_repetitions = int(
        spec["dataset"]["max_train_repetitions_per_prompt"])

    all_train = []
    all_val = []
    for op in spec["domain"]["operations"]:
        print(f"building candidate pools for {op!r}...")
        train_primary = build_primary_targets(
            op, train_operation_counts[op], spec)
        val_primary = build_primary_targets(op, val_operation_counts[op], spec)
        combined_primary = collections.OrderedDict({
            primary: train_primary.get(primary, 0) + val_primary.get(primary, 0)
            for primary in list(train_primary) + [
                key for key in val_primary if key not in train_primary]
        })
        train_categories = allocation_targets(
            op, train_operation_counts[op], spec)
        val_categories = allocation_targets(op, val_operation_counts[op], spec)
        categories = configured_allocation_categories(
            op, train_operation_counts[op], val_operation_counts[op], spec)
        pools, population = build_candidate_pools(
            op, combined_primary, categories, spec, thresholds, rng)
        print(
            f"  retained {sum(len(pool) for pool in pools.values()):,} "
            f"candidates from {sum(population.values()):,} classified expressions")

        val_pools = filtered_pools(pools, op, reserved)
        val_allocation = allocate_rows(
            val_primary,
            val_categories,
            val_pools,
            repeat_limit=1,
            unique=True)
        val_samples = sample_allocation(
            op, val_allocation, val_pools, rng, unique=True)

        excluded_train = set(reserved)
        for sample in val_samples:
            a, b = sample["a"], sample["b"]
            excluded_train.add((a, b, op))
            if supports_commutative_swap(a, b, op):
                excluded_train.add((b, a, op))
        train_pools = filtered_pools(pools, op, excluded_train)
        train_allocation = allocate_rows(
            train_primary,
            train_categories,
            train_pools,
            repeat_limit=max_repetitions,
            unique=False)
        train_samples = sample_allocation(
            op, train_allocation, train_pools, rng, unique=False)
        all_train.extend(train_samples)
        all_val.extend(val_samples)

    rng.shuffle(all_train)
    rng.shuffle(all_val)
    max_seq_len = int(spec["validation"]["require_supported_sequence_length"])
    division_answer_format = spec["domain"].get(
        "division_answer_format", "fixed_width")
    representation = spec["domain"].get(
        "representation", "abacus-v1" if reverse else "natural-v1")
    focus_scenarios = spec["operations"]["/"].get(
        "capability_focus_scenarios", [])
    train_rows = build_rows(
        all_train, "train", reverse, max_seq_len,
        division_answer_format=division_answer_format,
        representation=representation,
        division_focus_scenarios=focus_scenarios)
    val_rows = build_rows(
        all_val, "val", reverse, max_seq_len,
        division_answer_format=division_answer_format,
        representation=representation,
        division_focus_scenarios=focus_scenarios)
    report = validate_distribution(
        all_train, all_val, train_rows, val_rows, spec, reserved)
    return train_rows, val_rows, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="config/data_distribution_v2.yaml")
    parser.add_argument("--out", default=None,
                        help="override dataset.output_path")
    parser.add_argument("--report-out", default=None,
                        help="override dataset.report_path")
    parser.add_argument("--train-rows", type=int, default=None)
    parser.add_argument("--validation-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument(
        "--exclude-jsonl", action="append", default=[],
        help=("exclude every prompt in an existing internal-format JSONL "
              "dataset; may be supplied more than once"))
    args = parser.parse_args()

    spec = load_spec(args.spec)
    if args.train_rows is not None:
        spec["dataset"]["train_rows"] = args.train_rows
    if args.validation_rows is not None:
        spec["dataset"]["validation_rows"] = args.validation_rows
    if args.seed is not None:
        spec["dataset"]["seed"] = args.seed

    reverse = not args.no_reverse
    excluded_prompts = load_excluded_jsonl(
        args.exclude_jsonl, internal_format=reverse)
    train_rows, val_rows, report = generate(
        spec, reverse=reverse, excluded_prompts=excluded_prompts)
    output_path = Path(args.out or spec["dataset"]["output_path"])
    report_path = Path(args.report_out or spec["dataset"]["report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
        for row in val_rows:
            f.write(json.dumps(row) + "\n")
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print_report(report)
    print(f"\nwrote {len(train_rows):,} train + {len(val_rows):,} val -> "
          f"{output_path}")
    print(f"wrote validation report -> {report_path}")


if __name__ == "__main__":
    main()
