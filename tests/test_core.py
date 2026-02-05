import numpy as np

from core import (
    get_molar_mass_and_uncert,
    parse_formula,
    remove_spectators,
    suggest_spectators,
    solve_stoichiometry,
)


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
    assert len(singular_values) == min(len(final_dict), len(precursor_dicts))


def test_remove_spectators_simple():
    prec_dict = parse_formula("CaCl2")
    spectators = [("Cl", parse_formula("Cl"))]
    active, removed = remove_spectators(prec_dict, spectators)
    assert active == {"Ca": 1.0}
    assert removed == [("Cl", 2)]


def test_remove_spectators_polyatomic():
    prec_dict = parse_formula("Fe(NO3)3")
    spectators = [("NO3", parse_formula("NO3"))]
    active, removed = remove_spectators(prec_dict, spectators)
    assert active == {"Fe": 1.0}
    assert removed == [("NO3", 3)]


def test_solution_simple_case():
    solute_dict = parse_formula("Ca")
    prec_dict = parse_formula("CaCl2")
    spectators = [("Cl", parse_formula("Cl"))]
    active_dict, _ = remove_spectators(prec_dict, spectators)
    x, _, _, _ = solve_stoichiometry(solute_dict, [active_dict], nonneg=True)

    concentration = 0.1
    volume_ml = 100.0
    n_target = concentration * (volume_ml / 1000.0)
    n_prec = x[0] * n_target

    dict_masses = {"Ca": (40.0, 0.0), "Cl": (35.0, 0.0)}
    M_prec, _ = get_molar_mass_and_uncert(prec_dict, dict_masses)
    mass = M_prec * n_prec
    assert np.isclose(mass, 0.01 * 110.0)


def test_suggest_spectators_auto():
    solute_dict = parse_formula("KNO3")
    precursor_dicts = [parse_formula("AgNO3"), parse_formula("K2CO3")]
    suggested = suggest_spectators(solute_dict, precursor_dicts)
    names = {name for name, _ in suggested}
    assert "Ag" in names
    assert "CO3" in names
    assert "NO3" not in names
