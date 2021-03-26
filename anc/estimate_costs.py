import datetime
import itertools

from typing import Tuple, NamedTuple, Iterable, Iterator, Optional

import math
import matplotlib.pyplot as plt

import os.path as pathutils

Parameters = NamedTuple(
    'Parameters',
    [
        # Physical gate error rate.
        ('gate_err', float),
        # Time it takes to trigger a logical measurement, error correct it,
        # and decide which measurement to do next.
        ('reaction_time', datetime.timedelta),
        # Time it takes to measure the surface code's local stabilizers.
        ('cycle_time', datetime.timedelta),
        # Window size over exponent bits. (g0 in paper)
        ('exp_window', int),
        # Window size over multiplication bits. (g1 in paper)
        ('mul_window', int),
        # Bits between runways used during parallel additions. (g2 in paper)
        ('runway_sep', int),
        # Level 2 code distance.
        ('code_distance', int),
        # Level 1 code distance.
        ('l1_distance', int),
        # Error budget.
        ('max_total_err', float),
        # Problem size.
        ('n', int),
        # Number of controlled group operations required.
        ('n_e', int),
        # Whether or not to use two levels of 15-to-1 distillation.
        ('use_t_t_distillation', bool),
        # Number of bits of padding to use for runways and modular coset.
        ('deviation_padding', int),
    ]
)


def parameters_to_attempt(n: int,
                          n_e: int,
                          gate_error_rate: float,
                          ) -> Iterator[Parameters]:
    l1_distances = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    l2_distances = range(9, 51, 2)
    exp_windows = [4, 5, 6]
    mul_windows = [4, 5, 6]
    runway_seps = [512, 768, 1024, 1536, 2048]
    dev_offs = range(2, 10)

    for d1, d2, exp_window, mul_window, runway_sep, dev_off in itertools.product(
            l1_distances,
            l2_distances,
            exp_windows,
            mul_windows,
            runway_seps,
            dev_offs):
        if mul_window > exp_window or n % runway_sep != 0:
            continue
        distill_types = [False]
        if d1 == 15 and d2 >= 31:
            distill_types.append(True)
        for b in distill_types:
            yield Parameters(
                gate_err=gate_error_rate,
                reaction_time=datetime.timedelta(microseconds=10),
                cycle_time=datetime.timedelta(microseconds=1),
                exp_window=exp_window,
                mul_window=mul_window,
                runway_sep=runway_sep,
                l1_distance=d1,
                code_distance=d2,
                max_total_err=0.8,
                n=n,
                n_e=n_e,
                use_t_t_distillation=b,
                deviation_padding=int(math.ceil(math.log2(n*n*n_e)) + dev_off))


def topological_error_per_unit_cell(
        code_distance: int,
        gate_err: float) -> float:
    return 0.1 * (100 * gate_err) ** ((code_distance + 1) / 2)


def total_topological_error(code_distance: int,
                            gate_err: float,
                            unit_cells: int) -> float:
    """
    Args:
        code_distance: Diameter of logical qubits.
        gate_err: Physical gate error rate.
        unit_cells: Number of d*d*1 spacetime cells at risk.
    """
    return unit_cells * topological_error_per_unit_cell(
        code_distance,
        gate_err)


def compute_distillation_error(tof_count: int,
                               params: Parameters) -> float:
    """Estimate the total chance of CCZ magic state distillation failing.

    Args:
        tof_count: Number of CCZ states to distill.
        params: Algorithm construction parameters.

    References:
        Based on spreadsheet "calculator-CCZ-2T-resources.ods" from
        https://arxiv.org/abs/1812.01238 by Gidney and Fowler
    """
    if params.use_t_t_distillation:
        return 4*9*10**-17  # From FIG 1 of https://arxiv.org/abs/1812.01238

    l1_distance = params.l1_distance
    l2_distance = params.code_distance

    # Level 0
    L0_distance = l1_distance // 2
    L0_distillation_error = params.gate_err
    L0_topological_error = total_topological_error(
        unit_cells=100,  # Estimated 100 for T injection.
        code_distance=L0_distance,
        gate_err=params.gate_err)
    L0_total_T_error = L0_distillation_error + L0_topological_error

    # Level 1
    L1_topological_error = total_topological_error(
        unit_cells=1100,  # Estimated 1000 for factory, 100 for T injection.
        code_distance=l1_distance,
        gate_err=params.gate_err)
    L1_distillation_error = 35 * L0_total_T_error**3
    L1_total_T_error = L1_distillation_error + L1_topological_error

    # Level 2
    L2_topological_error = total_topological_error(
        unit_cells=1000,  # Estimated 1000 for factory.
        code_distance=l2_distance,
        gate_err=params.gate_err)
    L2_distillation_error = 28 * L1_total_T_error**2
    L2_total_CCZ_or_2T_error = L2_topological_error + L2_distillation_error

    return tof_count * L2_total_CCZ_or_2T_error


DeviationProperties = NamedTuple(
    'DeviationProperties',
    [
        ('piece_count', int),
        ('piece_len', int),
        ('reg_len', int),
        ('inner_loop_count', int),
        ('deviation_error', float),
    ]
)


def compute_deviation_properties(params: Parameters) -> DeviationProperties:
    piece_count = int(math.ceil(params.n / params.runway_sep))
    piece_len = params.runway_sep + params.deviation_padding
    reg_len = params.n + params.deviation_padding * piece_count

    # Temporarily adding carry runways into main register avoids need to
    # iterate over their bits when multiplying with that register as input.
    mul_in_bits = params.n + params.deviation_padding + 2
    inner_loop_count = int(math.ceil(params.n_e * 2 * mul_in_bits / (
            params.exp_window * params.mul_window)))

    classical_deviation_error = inner_loop_count * piece_count / 2**params.deviation_padding
    quantum_deviation_error = 4*math.sqrt(classical_deviation_error)
    return DeviationProperties(
        piece_count=piece_count,
        piece_len=piece_len,
        reg_len=reg_len,
        inner_loop_count=inner_loop_count,
        deviation_error=quantum_deviation_error,
    )


def probability_union(*ps: float) -> float:
    t = 1
    for p in ps:
        if p >= 1:
            # This happens when e.g. using the union bound to upper bound
            # a probability by using a frequency. The frequency estimate can
            # exceed 1 error per run.
            return 1
        t *= 1 - p
    return 1 - t


def logical_factory_dimensions(params: Parameters
                               ) -> Tuple[int, int, float]:
    """Determine the width, height, depth of the magic state factory."""
    if params.use_t_t_distillation:
        return 12*2, 8*2, 6  # Four T2 factories

    l1_distance = params.l1_distance
    l2_distance = params.code_distance

    t1_height = 4 * l1_distance / l2_distance
    t1_width = 8 * l1_distance / l2_distance
    t1_depth = 5.75 * l1_distance / l2_distance

    ccz_depth = 5
    ccz_height = 6
    ccz_width = 3
    storage_width = 2 * l1_distance / l2_distance

    ccz_rate = 1 / ccz_depth
    t1_rate = 1 / t1_depth
    t1_factories = int(math.ceil((ccz_rate * 8) / t1_rate))
    t1_factory_column_height = t1_height * math.ceil(t1_factories / 2)

    width = int(math.ceil(t1_width * 2 + ccz_width + storage_width))
    height = int(math.ceil(max(ccz_height, t1_factory_column_height)))
    depth = max(ccz_depth, t1_depth)

    return width, height, depth


def board_logical_dimensions(params: Parameters,
                             register_len: int) -> Tuple[int, int, int]:
    """Computes the dimensions of the surface code board in logical qubits.

    Assumes a single-threaded execution. For parallel execution, pass in
    parameters for an individual adder piece.

    Returns:
        width, height, distillation_area
    """

    factory_width, factory_height, factory_depth = (
        logical_factory_dimensions(params))
    ccz_time = factory_depth * params.cycle_time * params.code_distance
    factory_pair_count = int(math.ceil(ccz_time / params.reaction_time / 2))
    total_width = (factory_width + 1) * factory_pair_count + 1

    # FIG. 15 Lattice surgery implementation of the CZ fixups
    cz_fixups_box_height = 3

    # FIG. 23. Implementation of the MAJ operation in lattice surgery.
    adder_height = 3

    # FIG. 31. Data layout during a parallel addition.
    routing_height = 6
    reg_height = int(math.ceil(register_len / (total_width - 2)))
    total_height = sum([
        factory_height * 2,
        cz_fixups_box_height * 2,
        adder_height,
        routing_height,
        reg_height * 3,
    ])
    distillation_area = factory_height * factory_width * factory_pair_count * 2

    return total_width, total_height, distillation_area


CostEstimate = NamedTuple(
    'CostEstimate',
    [
        ('params', Parameters),
        ('toffoli_count', int),
        ('total_error', float),
        ('distillation_error', float),
        ('topological_data_error', float),
        ('total_hours', float),
        ('total_megaqubits', int),
        ('total_volume_megaqubitdays', float)
    ]
)


def physical_qubits_per_logical_qubit(code_distance: int) -> int:
    return (code_distance + 1)**2 * 2


def estimate_algorithm_cost(params: Parameters) -> Optional[CostEstimate]:
    """Determine algorithm single-shot layout and costs for given parameters."""

    post_process_error = 1e-2  # assumed to be below 1%
    dev = compute_deviation_properties(params)

    # Derive values for understanding inner loop.
    adder_depth = dev.piece_len * 2 - 1
    lookup_depth = 2 ** (params.exp_window + params.mul_window) - 1
    unlookup_depth = 2 * math.sqrt(lookup_depth)

    # Derive values for understanding overall algorithm.
    piece_width, piece_height, piece_distillation = board_logical_dimensions(
        params, dev.piece_len)
    logical_qubits = piece_width * piece_height * dev.piece_count
    distillation_area = piece_distillation * dev.piece_count

    tof_count = (adder_depth * dev.piece_count
                 + lookup_depth
                 + unlookup_depth) * dev.inner_loop_count

    # Code distance lets us compute time taken.
    inner_loop_time = (
            adder_depth * params.reaction_time +
            # Double speed lookup.
            lookup_depth * params.code_distance * params.cycle_time / 2 +
            unlookup_depth * params.code_distance * params.cycle_time / 2)
    total_time = inner_loop_time * dev.inner_loop_count

    # Upper-bound the topological error:
    surface_code_cycles = total_time / params.cycle_time
    topological_error = total_topological_error(
        unit_cells=(logical_qubits  - distillation_area) * surface_code_cycles,
        code_distance=params.code_distance,
        gate_err=params.gate_err)

    # Account for the distillation error:
    distillation_error = compute_distillation_error(
        tof_count=tof_count,
        params=params)

    # Check the total error.
    total_error = probability_union(
        topological_error,
        distillation_error,
        dev.deviation_error,
        post_process_error,
    )
    if total_error >= params.max_total_err:
        return None

    # Great!
    total_qubits = logical_qubits * physical_qubits_per_logical_qubit(
        params.code_distance)
    total_time = total_time.total_seconds()

    # Format.
    total_hours = total_time / 60 ** 2
    total_megaqubits = total_qubits / 10 ** 6
    total_volume_megaqubitdays = (total_hours / 24) * total_megaqubits
    total_error = math.ceil(100 * total_error) / 100

    return CostEstimate(
        params=params,
        toffoli_count=tof_count,
        distillation_error=distillation_error,
        topological_data_error=topological_error,
        total_error=total_error,
        total_hours=total_hours,
        total_megaqubits=total_megaqubits,
        total_volume_megaqubitdays=total_volume_megaqubitdays)


def rank_estimate(costs: CostEstimate) -> float:
    # Slight preference for decreasing space over decreasing time.
    skewed_volume = costs.total_megaqubits**1.2 * costs.total_hours
    return skewed_volume / (1 - costs.total_error)


def estimate_best_problem_cost(n: int, n_e: int, gate_error_rate: float) -> Optional[CostEstimate]:
    estimates = [estimate_algorithm_cost(params)
                 for params in parameters_to_attempt(n, n_e, gate_error_rate)]
    surviving_estimates = [e for e in estimates if e is not None]
    return min(surviving_estimates, key=rank_estimate, default=None)


# ------------------------------------------------------------------------------

def reduce_significant(q: float) -> float:
    """Return only the n most significant digits."""
    if q == 0:
        return 0
    n = math.floor(math.log(q, 10))
    result = math.ceil(q / 10**(n-1)) * 10**(n-1)

    # Handle poor precision in float type.
    if result < 0.1:
        return round(result * 100) / 100
    elif result < 10:
        return round(result * 10) / 10
    else:
        return round(result)


def fips_strength_level(n):
    # From FIPS 140-2 IG CMVP, page 110.
    #
    # This is extrapolated from the asymptotic complexity of the sieving
    # step in the general number field sieve (GNFS).
    ln = math.log
    return (1.923 * (n * ln(2))**(1/3) * ln(n * ln(2))**(2/3) - 4.69) / ln(2)


def fips_strength_level_rounded(n): # NIST-style rounding
    return 8 * round(fips_strength_level(n) / 8)


TABLE_HEADER = [
    'n',
    'n_e',
    'phys_err',
    'd1',
    'd2',
    'dev_off',
    'g_mul',
    'g_exp',
    'g_sep',
    '%',
    'volume',
    'E:volume',
    'Mqb',
    'hours',
    'E:hours',
    'tt_distill',
    'B tofs',
]


def tabulate_cost_estimate(costs: CostEstimate):
    assert costs.params.gate_err in [1e-3, 1e-4]
    gate_error_desc = r"0.1\%" if costs.params.gate_err == 1e-3 else r"0.01\%"
    row = [
        costs.params.n,
        costs.params.n_e,
        gate_error_desc,
        costs.params.l1_distance,
        costs.params.code_distance,
        costs.params.deviation_padding - int(math.ceil(math.log2(costs.params.n**2*costs.params.n_e))),
        costs.params.mul_window,
        costs.params.exp_window,
        costs.params.runway_sep,
        str(math.ceil(100 * costs.total_error)) + r"\%",
        reduce_significant(costs.total_volume_megaqubitdays),
        reduce_significant(costs.total_volume_megaqubitdays / (1 - costs.total_error)),
        reduce_significant(costs.total_megaqubits),
        reduce_significant(costs.total_hours),
        reduce_significant(costs.total_hours / (1 - costs.total_error)),
        costs.params.use_t_t_distillation,
        reduce_significant(costs.toffoli_count / 10**9),
    ]
    print('&'.join('${}$'.format(e).ljust(10) for e in row) + '\\\\')

# ------------------------------------------------------------------------------


# RSA
def eh_rsa(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    delta = 20 # Required to respect assumptions in the analysis.
    m = math.ceil(n / 2) - 1
    l = m - delta
    n_e = m + 2 * l
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_rsa_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # With maximum tradeoffs.
    return estimate_best_problem_cost(n, math.ceil(n / 2), gate_error_rate)


# General DLP
def shor_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    m = n - 1 + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    m = n - 1
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_general_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    m = n - 1
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# Schnorr DLP
def shor_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]:
    delta = 5 # Required to reach 99% success probability.
    z = fips_strength_level_rounded(n)
    m = 2 * z + delta
    n_e = 2 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr(n, gate_error_rate) -> Optional[CostEstimate]: # Single run.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_schnorr_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


# Short DLP
def eh_dlp_short(n, gate_error_rate) -> Optional[CostEstimate]:
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = 3 * m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)


def eh_dlp_short_max_tradeoffs(n, gate_error_rate) -> Optional[CostEstimate]: # Multiple runs with maximal tradeoff.
    z = fips_strength_level_rounded(n)
    m = 2 * z
    n_e = m
    return estimate_best_problem_cost(n, n_e, gate_error_rate)

# ------------------------------------------------------------------------------


def tabulate():
    gate_error_rates = [1e-3, 1e-4]
    moduli = [1024, 2048, 3072, 4096, 8192, 12288, 16384]

    datasets = [
        ("RSA, via Ekera-Håstad with s = 1 in a single run:", eh_rsa),
        ("Discrete logarithms, Schnorr group, via Shor:", shor_dlp_schnorr),
        ("Discrete logarithms, Schnorr group, via Ekera-Håstad with s = 1 in a single run:", eh_dlp_schnorr),
        ("Discrete logarithms, short exponent, via Ekerå-Håstad with s = 1 in a single run:", eh_dlp_short),
        ("Discrete logarithms, general, via Shor:", shor_dlp_general),
        ("Discrete logarithms, general, via Ekerå with s = 1 in a single run:", eh_dlp_general),
    ]

    for name, func in datasets:
        print()
        print(name)
        print('&'.join(str(e).ljust(10) for e in TABLE_HEADER) + '\\\\')
        print('\hline')
        for e in gate_error_rates:
            for n in moduli:
                tabulate_cost_estimate(func(n, e))


def significant_bits(n: int) -> int:
    assert n >= 0
    high = n.bit_length()
    low = (n ^ (n - 1)).bit_length()
    return high - low + 1


def plot():
    # Choose bit sizes to plot.
    max_steps = 64
    bits = [1024 * s for s in range(1, max_steps + 1)]
    bits = [e for e in bits if significant_bits(e) <= 3]
    max_y = 1024 * max_steps

    datasets = [
        ('C0', 'RSA via Ekerå-Håstad', eh_rsa, 1e-3, 'o'),
        ('C5', 'RSA via Ekerå-Håstad - 0.01% gate error instead of 0.1%', eh_rsa, 1e-4, '*'),
        ('C1', 'Short DLP or Schnorr DLP via EH', eh_dlp_short, 1e-3, 's'),
        ('C3', 'Schnorr DLP via Shor', shor_dlp_schnorr, 1e-3, 'd'),
        ('C2', 'General DLP via EH', eh_dlp_general, 1e-3, 'P'),
        ('C4', 'General DLP via Shor', shor_dlp_general, 1e-3, 'X'),
    ]

    plt.subplots(figsize=(16, 9)) # force 16 x 9 inches layout for the PDF

    for color, name, func, gate_error_rate, marker in datasets:
        valid_ns = []
        hours = []
        megaqubits = []

        for n in bits:
            cost = func(n, gate_error_rate)
            if cost is not None:
                expected_hours = cost.total_hours / (1 - cost.total_error)
                hours.append(expected_hours)
                megaqubits.append(cost.total_megaqubits)
                valid_ns.append(n)

        plt.plot(valid_ns, hours, color=color, label=name + ', hours', marker=marker)
        plt.plot(valid_ns, megaqubits, color=color, label=name + ', megaqubits', linestyle='--', marker=marker)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1024, max_y)
    plt.xticks(bits, [str(e) for e in bits], rotation=90)
    yticks = [(5 if e else 1)*10**k
              for k in range(6)
              for e in range(2)][:-1]
    plt.yticks(yticks, [str(e) for e in yticks])
    plt.minorticks_off()
    plt.grid(True)
    plt.xlabel('modulus length n (bits)')
    plt.ylabel('expected time (hours) and physical qubit count (megaqubits)')
    plt.gcf().subplots_adjust(bottom=0.16)

    plt.legend(loc='upper left', shadow=False)

    plt.tight_layout() # truncate margins

    # Export the figure to a PDF file.
    path = pathutils.dirname(pathutils.realpath(__file__))
    path = pathutils.normpath(path + "/../assets/rsa-dlps-extras.pdf")
    plt.savefig(path)


if __name__ == '__main__':
    tabulate()

    plot()

    plt.show()
