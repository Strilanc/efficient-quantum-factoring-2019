import math
from typing import Any, List, Dict

import numpy as np


def estimate_abstract_to_physical(tof_count: float,
                                  abstract_qubits: float,
                                  measurement_depth: float,
                                  prefers_parallel: bool = False,
                                  prefers_serial: bool = False
                                  ) -> Any:
    for code_distance in range(25, 101):
        logical_qubit_area = code_distance**2 * 2
        ccz_factory_duration = 5*code_distance
        ccz_factory_footprint = 15*8*logical_qubit_area
        ccz_factory_volume = ccz_factory_footprint * ccz_factory_duration

        reaction_time = 10
        assert not (prefers_parallel and prefers_serial)

        if prefers_parallel:
            # Use time optimal computation.
            runtime = reaction_time * measurement_depth
            routing_overhead_factor = 1.5
        elif prefers_serial:
            # Use serial distillation.
            runtime = tof_count * ccz_factory_duration
            routing_overhead_factor = 1.01
        else:
            # Do something intermediate.
            runtime = ccz_factory_duration * measurement_depth
            routing_overhead_factor = 1.25

        data_footprint = abstract_qubits * logical_qubit_area
        data_volume = data_footprint * runtime
        unit_cells = abstract_qubits * runtime
        error_per_unit_cell = 10**-math.ceil(code_distance / 2 + 1)
        data_error = unit_cells * error_per_unit_cell
        if data_error > 0.25:
            continue

        distill_volume = tof_count * ccz_factory_volume
        factory_count = int(math.ceil(distill_volume / runtime / ccz_factory_footprint))
        distill_footprint = factory_count * ccz_factory_footprint

        total_volume = data_volume * routing_overhead_factor + distill_volume
        total_footprint = (data_footprint + distill_footprint) * routing_overhead_factor

        microseconds_per_day = 10**6 * 60 * 60 * 24
        qubit_microseconds_per_megaqubit_day = 10**6 * microseconds_per_day

        return (
            code_distance,
            factory_count,
            tof_count,
            runtime / microseconds_per_day,
            total_footprint / 10.0**6,
            total_volume / qubit_microseconds_per_megaqubit_day,
        )

    raise NotImplementedError()


def sig_figs(v, r):
    if v < 10**9:
        return str(np.round(v, 1)).rjust(r)
    return "{:0.2g}".format(v).rjust(r)


def output(construction, result):
    cd, fact, tof, time, foot, vol = result
    print(construction[0].rjust(20),
          str(cd).rjust(20),
          str(fact).rjust(20),
          sig_figs(tof / 10**9, 20),
          sig_figs(vol, 20),
          sig_figs(foot, 20),
          sig_figs(time, 20))


def make_constructions(n: int, ecc_n: int):
    constructions = []

    def include(name, tof_count, abstract_qubits, measurement_depth):
        constructions.append(
            (name, tof_count, abstract_qubits, measurement_depth))

    lgn = math.log2(n)
    include("Vedral",
            tof_count=80*n**3,
            abstract_qubits=7*n + 1,
            measurement_depth=80*n**3)
    include("Zalka (basic)",
            tof_count=12*n**3,
            abstract_qubits=3*n,
            measurement_depth=12*n**3)
    include("Zalka (log add)",
            tof_count=52*n**3,
            abstract_qubits=5*n,
            measurement_depth=600*n**2)
    include("Zalka (fft mult)",
            tof_count=2**17 * n**2,
            abstract_qubits=96*n,
            measurement_depth=2**17 * n**1.2)
    include("Beauregard",
            tof_count=576 * n**3 * lgn**2,
            abstract_qubits=2*n+3,
            measurement_depth=144*n**3*lgn)
    include("Fowler",
            tof_count=40 * n**3,
            abstract_qubits=3 * n,
            measurement_depth=40 * n**3)
    # from https://arxiv.org/pdf/1706.06752.pdf Table 2
    counts = {
        512: 6.41 * 10**10,
        1024: 5.81 * 10**11,
        2048: 5.20 * 10**12,
        3072: 1.86 * 10**13,
        7680: 3.30 * 10**14,
        15360: 2.87 * 10**15,
    }
    include("Haner",
            tof_count=counts[n],
            abstract_qubits=2 * n + 2,
            measurement_depth=52 * n**3)
    L = np.exp(n**(1/3) * np.log2(n)**(2/3))
    q = 1.387
    include("Bernstein",
            tof_count=L**q,
            abstract_qubits=n**(2/3),
            measurement_depth=L**q)
    include("(ours)",
            tof_count=0.0009*n**3*lgn + 0.3*n**3,
            abstract_qubits=3*n + 6*lgn,
            measurement_depth=0.9*n**2*lgn + 400*n**2)

    n = ecc_n
    lgn = math.log2(n)
    # from https://arxiv.org/pdf/1706.06752.pdf Table 2
    counts = {
        110: 9.44 * 10**9,
        160: 2.97 * 10**10,
        192: 5.30 * 10**10,
        224: 8.43 * 10**10,
        256: 1.26 * 10**11,
        384: 4.52 * 10**11,
        521: 1.14 * 10**12,
    }
    include("ECC224 Roetteler",
            tof_count=counts[n],
            abstract_qubits=9*n,
            measurement_depth=448*n**3*lgn)

    return constructions


def print_construction_costs(n: int,
                             ecc_n: int,
                             parameters: List[Dict[str, Any]]):
    print("n={} ecc_n={}".format(n, ecc_n))
    print("construction".rjust(20),
          "code distance".rjust(20),
          "factories".rjust(20),
          "billion Tof+T/2".rjust(20),
          "megaqubitdays".rjust(20),
          "megaqubits".rjust(20),
          "days".rjust(20))
    for c in make_constructions(n, ecc_n):
        name, tof_count, abstract_qubits, measurement_depth = c
        costs = []
        for p in parameters:
            costs.append(
                estimate_abstract_to_physical(tof_count=tof_count,
                                              abstract_qubits=abstract_qubits,
                                              measurement_depth=measurement_depth,
                                              **p))
        best = tuple([min(e) for e in zip(*costs)])
        output(c, best)


def main():
    show_cases = False
    for n, ecc_n in [(1024, 160),
                     (2048, 224),
                     (3072, 256)]:
        if show_cases:
            print()
            print("SERIAL")
            print_construction_costs(n, ecc_n, [{'prefers_serial': True}])

            print()
            print("PARALLEL")
            print_construction_costs(n, ecc_n, [{'prefers_parallel': True}])

            print()
            print("INTERMEDIATE")
            print_construction_costs(n, ecc_n, [{}])

            print()
            print("BEST PER")
        print_construction_costs(n, ecc_n, [{}, {'prefers_serial': True}, {'prefers_parallel': True}])


if __name__ == '__main__':
    main()
