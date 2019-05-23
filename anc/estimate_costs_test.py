import datetime

from . import estimate_costs


def test_topological_error_per_unit_cell():
    f = estimate_costs.topological_error_per_unit_cell
    assert abs(f(7, 1e-3) - 1e-5) < 1e-20
    assert abs(f(15, 1e-3) == 1e-9) < 1e-20
    assert abs(f(31, 1e-3) == 1e-17) < 1e-20


def test_logical_board_dimensions():
    def f(n):
        return estimate_costs.board_logical_dimensions(
            estimate_costs.Parameters(
                gate_err=None,
                reaction_time=datetime.timedelta(microseconds=10),
                cycle_time=datetime.timedelta(microseconds=1),
                exp_window=None,
                mul_window=None,
                runway_sep=None,
                code_distance=27,
                l1_distance=17,
                max_total_err=None,
                n=None,
                n_e=None,
                use_t_t_distillation=None,
                deviation_padding=None), n)
    assert f(1024)[:2] == (113, 61)
    assert f(1100)[:2] == (113, 61)


def test_logical_factory_dimensions():
    def f(d1, d2):
        return estimate_costs.logical_factory_dimensions(
            estimate_costs.Parameters(
                gate_err=None,
                reaction_time=None,
                cycle_time=None,
                exp_window=None,
                mul_window=None,
                runway_sep=None,
                code_distance=d2,
                l1_distance=d1,
                max_total_err=None,
                n=None,
                n_e=None,
                use_t_t_distillation=None,
                deviation_padding=None))
    assert f(15, 25) == (14, 8, 5)
    assert f(15, 27) == (13, 7, 5)
    assert f(17, 27) == (15, 8, 5)
    assert f(17, 29) == (14, 8, 5)
    assert f(17, 31) == (13, 7, 5)


def test_distillation_error():
    def f(d1, d2):
        params = estimate_costs.Parameters(
            gate_err=1e-3,
            reaction_time=None,
            cycle_time=None,
            exp_window=None,
            mul_window=None,
            runway_sep=None,
            code_distance=d2,
            l1_distance=d1,
            max_total_err=None,
            n=None,
            n_e=None,
            use_t_t_distillation=False,
            deviation_padding=None)
        return 1 / estimate_costs.compute_distillation_error(1, params)

    # Compare with values taken from spreadsheet.
    assert 1e10 <= f(15, 25) <= 2e10
    assert 1e10 <= f(15, 31) <= 2e10
    assert 8e10 <= f(17, 25) <= 10e10
    assert 4e11 <= f(17, 27) <= 6e11
    assert 8e11 <= f(17, 29) <= 10e11
    assert 9e11 <= f(17, 31) <= 11e11
    assert 9e12 <= f(19, 31) <= 11e12


def test_significant_bits():
    f = estimate_costs.significant_bits
    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 2
    assert f(4) == 1
    assert f(5) == 3
    assert f(6) == 2
    assert f(7) == 3
    assert f(8) == 1
    assert f(9) == 4
    assert f(10) == 3
    assert f(11) == 4
    assert f((1 << 100) + 1) == 101


def test_compute_deviation_properties():
    params = estimate_costs.Parameters(
        gate_err=None,
        reaction_time=None,
        cycle_time=None,
        exp_window=5,
        mul_window=4,
        runway_sep=1024,
        code_distance=None,
        l1_distance=None,
        max_total_err=None,
        n=2048,
        n_e=2048*1.5,
        use_t_t_distillation=False,
        deviation_padding=45)
    f = estimate_costs.compute_deviation_properties
    a = f(params)
    print(a)
    assert a.deviation_error <= 0.001
    assert a.piece_count == 2
    assert a.piece_len == 1024 + 45
    assert a.reg_len == 2048 + 90
    assert 650000 <= a.inner_loop_count <= 660000


def test_estimate_algorithms_cost():
    f = estimate_costs.estimate_algorithm_cost
    p27 = estimate_costs.Parameters(
        gate_err=1e-3,
        reaction_time=datetime.timedelta(microseconds=10),
        cycle_time=datetime.timedelta(microseconds=1),
        exp_window=5,
        mul_window=4,
        runway_sep=1024,
        l1_distance=17,
        code_distance=27,
        max_total_err=1.0,
        n=2048,
        n_e=2048*1.5,
        use_t_t_distillation=False,
        deviation_padding=43)
    p29 = estimate_costs.Parameters(
        gate_err=1e-3,
        reaction_time=datetime.timedelta(microseconds=10),
        cycle_time=datetime.timedelta(microseconds=1),
        exp_window=5,
        mul_window=4,
        runway_sep=1024,
        l1_distance=17,
        code_distance=29,
        max_total_err=1.0,
        n=2048,
        n_e=2048*1.5,
        use_t_t_distillation=False,
        deviation_padding=43)

    c27 = f(p27)
    c29 = f(p29)

    assert c29.total_error < c27.total_error
    assert c29.total_megaqubits > c27.total_megaqubits
