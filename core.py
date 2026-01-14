import os
import re
import ast

import numpy as np
import pandas as pd


def extract_molar_masses_and_uncertainties(path, sheet_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier Excel introuvable: {path}")
    df = pd.read_excel(path, sheet_name=sheet_name)
    required_columns = {"Symbol", "Standard atomic weight", "Unnamed: 4"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier Excel: "
            + ", ".join(sorted(missing_columns))
        )

    symbols = df["Symbol"][1:]
    masses = df["Standard atomic weight"][1:]
    errs = df["Unnamed: 4"][1:]

    def parse_maybe_interval(value):
        if isinstance(value, str):
            try:
                pair = ast.literal_eval(value)  # ex: "[10.8, 11.0]"
                return float(np.mean(pair))
            except (ValueError, SyntaxError):
                pass
        return float(value)

    result = {}
    for s, m, e in zip(symbols, masses, errs):
        s = str(s).strip()
        if not s:
            continue

        m_val = parse_maybe_interval(m)

        if pd.isna(e):
            e_val = 0.0
        else:
            try:
                pair_err = ast.literal_eval(str(e))
                e_val = abs(pair_err[1] - pair_err[0]) / 2.0
            except (ValueError, SyntaxError, TypeError):
                e_val = float(e)

        result[s] = (m_val, e_val)

    return result


def parse_formula(formula):
    formula = re.sub(r"\s+", "", formula)
    if not formula:
        raise ValueError("Formule brute vide ou invalide.")

    def strip_charge(fragment):
        fragment = re.sub(r"\^[0-9]+[+-]$", "", fragment)
        fragment = re.sub(r"(?:(?:\d+)?[+-])$", "", fragment)
        return fragment

    def parse_fragment(fragment):
        token_pattern = r"([A-Z][a-z]?|\d+(?:\.\d+)?|\(|\))"
        tokens = re.findall(token_pattern, fragment)
        if not tokens:
            raise ValueError("Formule brute vide ou invalide.")

        def is_number(token):
            return re.fullmatch(r"\d+(?:\.\d+)?", token) is not None

        def is_element(token):
            return re.fullmatch(r"[A-Z][a-z]?", token) is not None

        stack = [{}]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "(":
                stack.append({})
                i += 1
                continue
            if tok == ")":
                if len(stack) == 1:
                    raise ValueError("Parenthese fermante sans ouvrante.")
                group = stack.pop()
                multiplier = 1.0
                if i + 1 < len(tokens) and is_number(tokens[i + 1]):
                    multiplier = float(tokens[i + 1])
                    i += 1
                for el, count in group.items():
                    stack[-1][el] = stack[-1].get(el, 0.0) + count * multiplier
                i += 1
                continue
            if is_element(tok):
                multiplier = 1.0
                if i + 1 < len(tokens) and is_number(tokens[i + 1]):
                    multiplier = float(tokens[i + 1])
                    i += 1
                stack[-1][tok] = stack[-1].get(tok, 0.0) + multiplier
                i += 1
                continue
            if is_number(tok):
                raise ValueError(f"Nombre inattendu dans la formule: '{tok}'.")
            raise ValueError(f"Jeton inconnu dans la formule: '{tok}'.")

        if len(stack) != 1:
            raise ValueError("Parenthese ouvrante sans fermante.")

        return stack[0]

    parts = re.split(r"[·.]", formula)
    total = {}
    for raw in parts:
        fragment = strip_charge(raw)
        if not fragment:
            continue
        coeff_match = re.match(r"^(\d+(?:\.\d+)?)(.*)$", fragment)
        if coeff_match:
            part_coeff = float(coeff_match.group(1))
            fragment = coeff_match.group(2)
        else:
            part_coeff = 1.0
        if not fragment:
            raise ValueError("Formule brute invalide apres coefficient.")
        fragment_dict = parse_fragment(fragment)
        for el, count in fragment_dict.items():
            total[el] = total.get(el, 0.0) + count * part_coeff
    return total


def validate_elements(formula_dict, dict_masses, label):
    missing = [el for el in formula_dict.keys() if el not in dict_masses]
    if missing:
        raise ValueError(
            f"Elements inconnus dans {label}: {', '.join(sorted(missing))}"
        )


def get_molar_mass_and_uncert(comp_dict, dict_masses):
    M = 0.0
    dM2 = 0.0
    for el, count in comp_dict.items():
        if el not in dict_masses:
            raise ValueError(f"Element '{el}' introuvable dans le tableau de masses atomiques.")
        M_el, dM_el = dict_masses[el]
        M += count * M_el
        dM2 += (count * dM_el) ** 2
    return M, np.sqrt(dM2)


def nnls_solve(P, f, tol=1e-12, max_iter=None):
    if max_iter is None:
        max_iter = 3 * P.shape[1]

    x = np.zeros(P.shape[1])
    passive = np.zeros(P.shape[1], dtype=bool)
    w = P.T @ (f - P @ x)
    iteration = 0

    while np.any(w > tol) and iteration < max_iter:
        t = int(np.argmax(w))
        passive[t] = True
        z = np.zeros_like(x)
        while True:
            if not np.any(passive):
                break
            P_passive = P[:, passive]
            z_passive, _, _, _ = np.linalg.lstsq(P_passive, f, rcond=None)
            z[passive] = z_passive
            z[~passive] = 0.0
            if np.all(z[passive] > tol):
                break
            negative = (z <= tol) & passive
            alpha = np.min(x[negative] / (x[negative] - z[negative]))
            x = x + alpha * (z - x)
            newly_inactive = (x <= tol) & passive
            passive[newly_inactive] = False
            z[newly_inactive] = 0.0
        x = z
        w = P.T @ (f - P @ x)
        iteration += 1

    return x


def solve_stoichiometry(final_dict, precursor_dicts, nonneg=False):
    elements = list(final_dict.keys())
    nb_el = len(elements)
    nb_prec = len(precursor_dicts)

    P = np.zeros((nb_el, nb_prec))
    f = np.zeros(nb_el)

    for i, el in enumerate(elements):
        f[i] = final_dict[el]
        for j, pdict in enumerate(precursor_dicts):
            P[i, j] = pdict.get(el, 0.0)

    if nonneg:
        x = nnls_solve(P, f)
        residual = np.sum((P @ x - f) ** 2)
        residuals = np.array([residual])
    else:
        x, residuals, _, _ = np.linalg.lstsq(P, f, rcond=None)

    rank = np.linalg.matrix_rank(P)
    singular_values = np.linalg.svd(P, compute_uv=False)
    return x, residuals, rank, singular_values
