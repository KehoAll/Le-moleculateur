import numpy as np

from core import parse_formula, solve_stoichiometry


def test_parse_formula_supports_hydrates_and_charges():
    assert parse_formula("CuSO4·5H2O") == {"Cu": 1.0, "S": 1.0, "O": 9.0, "H": 10.0}
    assert parse_formula("NH4+") == {"N": 1.0, "H": 4.0}
    assert parse_formula("SO4^2-") == {"S": 1.0, "O": 4.0}


def test_parse_formula_parentheses_and_spacing():
    assert parse_formula("Fe2(SO4)3") == {"Fe": 2.0, "S": 3.0, "O": 12.0}
    assert parse_formula(" Na Cl ") == {"Na": 1.0, "Cl": 1.0}


def test_solve_stoichiometry_nonneg():
    final_dict = {"A": 1.0}
    precursor_dicts = [{"A": 1.0}, {"A": -1.0}]
    x, residuals, rank, singular_values = solve_stoichiometry(
        final_dict, precursor_dicts, nonneg=True
    )
    assert np.all(x >= 0.0)
    assert rank >= 1
    assert len(singular_values) == len(precursor_dicts)
