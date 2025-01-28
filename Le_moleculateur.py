import sys
import os
import re
import ast
import numpy as np
import pandas as pd
from datetime import datetime

from gooey import Gooey, GooeyParser

###############################################################################
# 1) Lecture des masses atomiques ET de leurs incertitudes
###############################################################################
def extract_molar_masses_and_uncertainties(path):
    """
    Lit un fichier Excel conforme a https://doi.org/10.1515/pac-2019-0603,
    c.-a-d. des colonnes 'Symbol', 'Standard atomic weight' et 'Unnamed: 4'
    ou 'Standard atomic weight' peut etre un float ou un intervalle,
    et 'Unnamed: 4' l'incertitude standard (ou NaN si non specifiee).

    Retourne un dictionnaire {Symbole : (Moyenne, Incertitude)}.
    """
    df = pd.read_excel(path)
    
    symbols = df['Symbol'][1:]
    masses  = df['Standard atomic weight'][1:]
    errs    = df['Unnamed: 4'][1:]
    
    def parse_maybe_interval(value):
        if isinstance(value, str):
            try:
                pair = ast.literal_eval(value)  # ex: "[10.8, 11.0]"
                return float(np.mean(pair))
            except:
                pass
        return float(value)
    
    result = {}
    for s, m, e in zip(symbols, masses, errs):
        s = str(s).strip()
        if not s:
            continue
        
        # Masse moyenne
        m_val = parse_maybe_interval(m)

        # Incertitude
        if pd.isna(e):
            e_val = 0.0
        else:
            try:
                pair_err = ast.literal_eval(str(e))
                e_val = abs(pair_err[1] - pair_err[0]) / 2.0
            except:
                e_val = float(e)
        
        result[s] = (m_val, e_val)
    
    return result


###############################################################################
# 2) Parsing d'une formule brute => dict { El : stoich }
###############################################################################
def parse_formula(formula):
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    parts = re.findall(pattern, formula)
    
    comp_dict = {}
    for (el, num_str) in parts:
        if not num_str:
            coeff = 1.0
        else:
            coeff = float(num_str)
        comp_dict[el] = comp_dict.get(el, 0.0) + coeff
    return comp_dict


###############################################################################
# 3) Masse molaire et incertitude d'un compose
###############################################################################
def get_molar_mass_and_uncert(comp_dict, dict_masses):
    M = 0.0
    dM2 = 0.0
    for el, count in comp_dict.items():
        if el not in dict_masses:
            raise ValueError(f"Element '{el}' introuvable dans le tableau de masses atomiques.")
        M_el, dM_el = dict_masses[el]
        M     += count * M_el
        dM2   += (count * dM_el) ** 2
    return M, np.sqrt(dM2)


###############################################################################
# 4) Resolution stoech => x
###############################################################################
def solve_stoichiometry(final_dict, precursor_dicts):
    elements = list(final_dict.keys())
    nb_el = len(elements)
    nb_prec = len(precursor_dicts)
    
    P = np.zeros((nb_el, nb_prec))
    f = np.zeros(nb_el)
    
    for i, el in enumerate(elements):
        f[i] = final_dict[el]
        for j, pdict in enumerate(precursor_dicts):
            P[i, j] = pdict.get(el, 0.0)
    
    x, residuals, rank, s = np.linalg.lstsq(P, f, rcond=None)
    return x


###############################################################################
# 5) Conversion d'un dict stoech => formule brute (sans re-normalisation stricte)
###############################################################################
def compose_formula_fixed_stoich(stoich_dict, decimals=2):
    filtered = {el: val for el, val in stoich_dict.items() if val > 1e-14}
    if not filtered:
        return ""
    
    parts = []
    for el in sorted(filtered.keys()):
        val = round(filtered[el], decimals)
        if abs(val - round(val)) < 10**(-decimals):
            int_val = int(round(val))
            if int_val == 1:
                parts.append(el)
            else:
                parts.append(f"{el}{int_val}")
        else:
            parts.append(f"{el}{val}")
    return "".join(parts)


###############################################################################
# 6) Pour doubler la sortie : console + fichier (Tee)
###############################################################################
class Tee(object):
    """Redirige ce qui est ecrit dans sys.stdout vers plusieurs flux a la fois."""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()


###############################################################################
# 7) Programme principal (avec Gooey)
###############################################################################
@Gooey(
    program_name="Le Moleculateur",
    default_size=(800, 720),
    optional_args_label="Options"
)
def main():
    parser = GooeyParser(
        description="Calcule les masses de precurseurs (avec incertitudes), "
                    "et permet d'ajuster la decomposition en CO2 / O2 / H2O."
    )

    parser.add_argument(
        "--Prod",
        metavar="Formule finale (obligatoire)",
        help="Formule brute du produit final (ex: Ca5Ga6O14).",
        required=True,
        widget="TextField"
    )

    parser.add_argument(
        "--m",
        metavar="Masse en grammes (defaut: 1.0g)",
        help="Masse en grammes (defaut: 1.0g).",
        default=1.0,
        type=float,
        widget="DecimalField"
    )

    parser.add_argument(
        "--Precs",
        metavar="Precurseurs (obligatoire)",
        help="Liste des precurseurs separes par des espaces (ex: CaO Ga2O3).",
        required=True,
        widget="TextField"
    )

    parser.add_argument(
        "--Purete",
        metavar="Purete (%) (optionnel)",
        help="Pourcentages de purete (ex: 99 99.99), meme ordre que les precurseurs.",
        default="",
        widget="TextField"
    )

    parser.add_argument(
        "--prec",
        metavar="Precision (decimales) (optionnel)",
        help="Nombre de decimales pour l'affichage (defaut: 4).",
        default=4,
        type=int,
        widget="IntegerField"
    )

    # Fichier Excel obligatoire, AVEC une valeur par defaut
    parser.add_argument(
        "--path",
        metavar="Fichier Excel (obligatoire)",
        help="Chemin du fichier Excel (masses atomiques).",
        required=True,
        default=os.path.join(os.getcwd(), 'Atomic weights.xlsx'),
        widget="FileChooser"
    )

    # Groupe "Degagement de gaz" + 3 checkboxes en francais
    gas_group = parser.add_argument_group(
        "Degagement de gaz",
        "Cochez pour autoriser la production des gaz suivants :"
    )

    gas_group.add_argument(
        "--releaseCO2",
        action="store_true",
        help="",  # on vide 'help', on va utiliser gooey_options pour le label
        gooey_options={"label": "Relacher CO2"}
    )
    gas_group.add_argument(
        "--releaseO2",
        action="store_true",
        help="",
        gooey_options={"label": "Relacher O2"}
    )
    gas_group.add_argument(
        "--releaseH2O",
        action="store_true",
        help="",
        gooey_options={"label": "Relacher H2O"}
    )

    # Recupere args Gooey
    args = parser.parse_args()

    # --- Conversion des champs ---
    final_formula = args.Prod.strip()
    mass_target   = float(args.m)
    decimals      = args.prec
    excel_path    = args.path

    # Precs => split
    precs_str = args.Precs.strip()
    if not precs_str:
        print("ERREUR : Aucun precurseur fourni.")
        sys.exit(1)
    precursors = precs_str.split()

    # Purete => split
    purete_str = args.Purete.strip()
    if purete_str:
        raw_purete = purete_str.split()
        purete_list = [float(x) for x in raw_purete]
    else:
        purete_list = None

    # Lecture du fichier Excel
    dict_masses = extract_molar_masses_and_uncertainties(excel_path)

    # Parse la formule finale
    final_dict = parse_formula(final_formula)
    M_final, dM_final = get_molar_mass_and_uncert(final_dict, dict_masses)

    # Parse des precurseurs
    precursor_dicts = [parse_formula(p) for p in precursors]

    # Ajout des pseudo-precurseurs si coche
    byproducts = []

    if args.releaseCO2:
        byproducts.append(("CO2", {"C":1,"O":2}))
        if "C" not in final_dict:
            final_dict["C"] = 0.0

    if args.releaseO2:
        byproducts.append(("O2", {"O":2}))

    if args.releaseH2O:
        byproducts.append(("H2O", {"H":2,"O":1}))
        if "H" not in final_dict:
            final_dict["H"] = 0.0

    # On les ajoute a la liste precurseurs
    for (byp_name, byp_dict) in byproducts:
        precursors.append(byp_name)
        precursor_dicts.append(byp_dict)

    # Nombre initial de precurseurs reels
    nb_init_prec = len(precs_str.split())

    # Gestion purete
    if (purete_list is None) or (len(purete_list) == 0):
        purete_list = [100.0] * nb_init_prec
    else:
        if len(purete_list) < nb_init_prec:
            print("Avertissement : nombre de puretes specifie < nb precurseurs.")
            while len(purete_list) < nb_init_prec:
                purete_list.append(100.0)
        elif len(purete_list) > nb_init_prec:
            print("Avertissement : trop de valeurs de purete, surplus ignore.")
        purete_list = purete_list[:nb_init_prec]

    # Resolution
    x_stoich = solve_stoichiometry(final_dict, precursor_dicts)
    
    n_final = mass_target / M_final

    # Distinction masses positives / negatives
    results_consumed = []
    results_byproducts = []

    for i, prec_name in enumerate(precursors):
        comp_dict_i = precursor_dicts[i]
        M_prec_i, dM_prec_i = get_molar_mass_and_uncert(comp_dict_i, dict_masses)

        x_i = x_stoich[i]
        n_prec_i = x_i * n_final
        mass_prec_i = M_prec_i * n_prec_i
        dm_prec_i   = dM_prec_i * abs(n_prec_i)

        if i < nb_init_prec:
            # Appliquer purete
            pcent = purete_list[i]
            alpha = pcent / 100.0
            mass_corr = mass_prec_i / alpha
            dmass_corr = dm_prec_i / alpha

            if mass_prec_i >= 0:
                results_consumed.append((prec_name, mass_corr, dmass_corr, pcent, mass_corr - mass_prec_i))
            else:
                results_byproducts.append((prec_name, -mass_corr, -dmass_corr))
        else:
            # Pseudo-precurseur
            if mass_prec_i >= 0:
                results_consumed.append((prec_name, mass_prec_i, dm_prec_i, 100.0, 0.0))
            else:
                results_byproducts.append((prec_name, -mass_prec_i, -dm_prec_i))

    # Stoechiometrie reconstituee
    total_elements = {}
    partial_stoech = []
    for i, p_dict in enumerate(precursor_dicts):
        xi = x_stoich[i]
        sto_dict_i = {}
        for el, val_el in p_dict.items():
            contrib = xi * val_el
            total_elements[el] = total_elements.get(el, 0.0) + contrib
            sto_dict_i[el] = contrib
        partial_stoech.append((precursors[i], sto_dict_i))

    # Normalisation
    sorted_by_stoich = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    ref_el, ref_val = sorted_by_stoich[0]
    if ref_el in total_elements and total_elements[ref_el] > 1e-14:
        scale = ref_val / total_elements[ref_el]
    else:
        scale = 1.0
    scaled_elements = {el: val*scale for el,val in total_elements.items()}

    # Verif ecart +/-5%
    tol = 0.05
    errors_found = False
    for e in final_dict:
        target = final_dict[e]
        found = scaled_elements.get(e, 0.0)
        if target != 0:
            diff_rel = abs(found - target) / target
            if diff_rel > tol:
                print(f"\nATTENTION : Ecart trop grand pour l'element {e} :")
                print(f"  Souhaite = {target:.3f},   Obtenu = {found:.3f}")
                print(f"  Ecart relatif = {diff_rel*100:.2f}% > {tol*100:.1f}%\n")
                errors_found = True

    recon_formula = compose_formula_fixed_stoich(scaled_elements, decimals=3)

    # Sauvegarde de la sortie
    folder_name = final_formula
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    filename_base = f"{final_formula}_with_{'_'.join(precs_str.split())}"
    filename_base = f"{timestamp}_{filename_base}"
    
    i = 1
    while True:
        candidate_filename = f"{filename_base}_{i}.txt"
        candidate_path = os.path.join(folder_name, candidate_filename)
        if not os.path.exists(candidate_path):
            break
        i += 1
    
    f_out = open(candidate_path, 'w', encoding='utf-8')
    backup_stdout = sys.stdout
    tee = Tee(backup_stdout, f_out)
    sys.stdout = tee

    # Affichage final
    print(f"Resultats sauvegardes dans : {candidate_path}\n")
    print(f"Produit final vise : {final_formula}")
    print(f"Masse visee       : {mass_target:.3f} g")
    print(f"Masse molaire du produit final : {M_final:.3f} +/- {dM_final:.3f} g/mol\n")

    print("===== Precurseurs (a peser) =====\n")
    for (pname, mass_corr, dmass_corr, pcent, delta_m) in results_consumed:
        print(f" * {pname:<8s} : {mass_corr:.{decimals}f} +/- {dmass_corr:.{decimals}f} g  "
              f"(purete={pcent:.1f}%, delta={delta_m:.4f} g)")

    print("\n===== Sous-produits rejetes (masse > 0 => degage) =====\n")
    if len(results_byproducts) == 0:
        print("Aucun sous-produit calcule (mass negative)")
    else:
        for (pname, mass_rel, dmass_rel) in results_byproducts:
            print(f" - {pname:<8s} : {mass_rel:.{decimals}f} +/- {dmass_rel:.{decimals}f} g liberes")

    print("\n===== Stoechiometrie totale reconstituee (pour 1 mole du produit) =====")
    for el in sorted(total_elements.keys()):
        print(f"  Element {el} : {total_elements[el]:.3f}")

    print("\n===== Stoechiometrie apportee par chaque precurseur (pour 1 mole du produit) =====")
    for i, (prc_name, sto_dict) in enumerate(partial_stoech):
        print(f"  Precurseur : {prc_name} (x={x_stoich[i]:.3f})")
        for el in sorted(sto_dict.keys()):
            print(f"    {el} : {sto_dict[el]:.3f}")

    print(f"\nFormule brute reconstituee (normalisee sur {ref_el}) : {recon_formula}")

    sys.stdout = backup_stdout
    f_out.close()
    print(f"\nLe fichier de resultats a ete cree ici : {candidate_path}")


if __name__ == "__main__":
    main()
