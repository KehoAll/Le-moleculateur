import sys
import os
import re
import json
import csv
import math
from datetime import datetime

from gooey import Gooey, GooeyParser

from core import (
    extract_molar_masses_and_uncertainties,
    parse_formula,
    validate_elements,
    get_molar_mass_and_uncert,
    solve_stoichiometry,
)

APP_VERSION = "1.4.0"
CONFIG_FILENAME = ".le_moleculateur_config.json"


def load_last_config(config_path):
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def save_last_config(config_path, payload):
    try:
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass


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
    default_size=(800, 780),
    optional_args_label="Options"
)
def main():
    config_path = os.path.join(os.getcwd(), CONFIG_FILENAME)
    last_config = load_last_config(config_path)
    parser = GooeyParser(
        description="Calcule les masses de precurseurs (avec incertitudes), "
                    "et permet d'ajuster la decomposition en CO2 / O2 / H2O / NO2 + autres gaz libres."
    )

    parser.add_argument(
        "--Prod",
        metavar="Formule finale (obligatoire)",
        help="Formule brute du produit final (ex: Ca5Ga6O14).",
        required=True,
        default=last_config.get("Prod", ""),
        widget="TextField"
    )

    parser.add_argument(
        "--m",
        metavar="Masse en grammes (defaut: 1.0g)",
        help="Masse en grammes (defaut: 1.0g).",
        default=last_config.get("m", 1.0),
        type=float,
        widget="DecimalField"
    )

    parser.add_argument(
        "--Precs",
        metavar="Precurseurs (obligatoire)",
        help="Liste des precurseurs separes par des espaces (ex: CaO Ga2O3).",
        required=True,
        default=last_config.get("Precs", ""),
        widget="TextField"
    )

    parser.add_argument(
        "--Purete",
        metavar="Purete (%) (optionnel)",
        help="Pourcentages de purete (ex: 99 99.99), meme ordre que les precurseurs.",
        default=last_config.get("Purete", ""),
        widget="TextField"
    )

    # NEW: Elément de normalisation optionnel
    parser.add_argument(
        "--PivotEl",
        metavar="Element de normalisation (optionnel)",
        help="Symbole d'un element (ex: O) pour normaliser la formule reconstituee sur cet element.",
        default=last_config.get("PivotEl", ""),
        widget="TextField"
    )

    parser.add_argument(
        "--prec",
        metavar="Precision (decimales) (optionnel)",
        help="Nombre de decimales pour l'affichage (defaut: 4).",
        default=last_config.get("prec", 4),
        type=int,
        widget="IntegerField"
    )

    parser.add_argument(
        "--tol",
        metavar="Tolerance relative (optionnel)",
        help="Tolerance relative pour la verification (defaut: 0.001 = 0.1%).",
        default=last_config.get("tol", 0.001),
        type=float,
        widget="DecimalField"
    )
    parser.add_argument(
        "--sheet",
        metavar="Feuille Excel (optionnel)",
        help="Nom ou index de feuille Excel (defaut: 0).",
        default=last_config.get("sheet", "0"),
        widget="TextField"
    )

    # Fichier Excel obligatoire, AVEC une valeur par defaut
    parser.add_argument(
        "--path",
        metavar="Fichier Excel (obligatoire)",
        help="Chemin du fichier Excel (masses atomiques).",
        required=True,
        default=last_config.get(
            "path", os.path.join(os.getcwd(), 'Atomic weights.xlsx')
        ),
        widget="FileChooser"
    )

    # Groupe "Degagement de gaz"
    gas_group = parser.add_argument_group(
        "Degagement de gaz",
        "Cochez pour autoriser la production des gaz suivants :"
    )

    gas_group.add_argument(
        "--releaseCO2",
        action="store_true",
        help="",
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
    # NEW: case NO2
    gas_group.add_argument(
        "--releaseNO2",
        action="store_true",
        help="",
        gooey_options={"label": "Relacher NO2"}
    )

    # NEW: champ libre pour d'autres gaz
    parser.add_argument(
        "--OtherGases",
        metavar="Autres gaz a relacher (optionnel)",
        help="Liste d'especes gazeuses supplementaires separees par des espaces ou des virgules (ex: CO NO N2 H2S).",
        default=last_config.get("OtherGases", ""),
        widget="TextField"
    )
    parser.add_argument(
        "--exportCSV",
        action="store_true",
        default=last_config.get("exportCSV", False),
        help="",
        gooey_options={"label": "Exporter CSV (resultats)"}
    )
    parser.add_argument(
        "--nonneg",
        action="store_true",
        default=last_config.get("nonneg", False),
        help="",
        gooey_options={"label": "Forcer les quantites negatives a zero"}
    )

    args = parser.parse_args()

    final_formula = args.Prod.strip()
    mass_target   = float(args.m)
    decimals      = args.prec
    tol           = float(args.tol)
    excel_path    = args.path
    sheet_raw     = str(args.sheet).strip()
    pivot_el_user = args.PivotEl.strip()
    run_timestamp = datetime.now()

    # Liste des precurseurs
    precs_str = args.Precs.strip()
    if not precs_str:
        print("ERREUR : Aucun precurseur fourni.")
        sys.exit(1)
    precursors = precs_str.split()

    # Purete
    purete_str = args.Purete.strip()
    if purete_str:
        raw_purete = purete_str.split()
        purete_list = [float(x) for x in raw_purete]
    else:
        purete_list = None

    # Lecture excel + parse formule finale
    if sheet_raw == "":
        sheet_name = 0
    else:
        sheet_name = int(sheet_raw) if sheet_raw.isdigit() else sheet_raw
    dict_masses = extract_molar_masses_and_uncertainties(excel_path, sheet_name=sheet_name)
    final_dict = parse_formula(final_formula)
    validate_elements(final_dict, dict_masses, "formule finale")
    M_final, dM_final = get_molar_mass_and_uncert(final_dict, dict_masses)

    # Parse des precurseurs
    precursor_dicts = [parse_formula(p) for p in precursors]
    for prec_name, prec_dict in zip(precursors, precursor_dicts):
        validate_elements(prec_dict, dict_masses, f"precurseur {prec_name}")

    # === Gestion des sous-produits "gaz" ===
    byproducts = []

    def ensure_elements_in_final(byp_dict):
        """Pour chaque element d'un gaz byproduct, s'il n'est pas dans la formule finale, l'ajouter avec 0."""
        for el in byp_dict.keys():
            if el not in final_dict:
                final_dict[el] = 0.0

    if args.releaseCO2:
        byproducts.append(("CO2", {"C":1, "O":2}))
        if "C" not in final_dict:
            final_dict["C"] = 0.0

    if args.releaseO2:
        byproducts.append(("O2", {"O":2}))
        # O peut etre deja dans le solide, pas necessaire de forcer O=0

    if args.releaseH2O:
        byproducts.append(("H2O", {"H":2, "O":1}))
        if "H" not in final_dict:
            final_dict["H"] = 0.0

    # NEW: NO2
    if args.releaseNO2:
        byproducts.append(("NO2", {"N":1, "O":2}))
        if "N" not in final_dict:
            final_dict["N"] = 0.0

    # NEW: autres gaz libres (ex: CO, NO, N2, H2S, ...)
    other_gases_str = args.OtherGases.strip()
    if other_gases_str:
        tokens = re.split(r'[,\s]+', other_gases_str)
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            try:
                d_tok = parse_formula(tok)
                if len(d_tok) == 0:
                    print(f"Avertissement : gaz '{tok}' non reconnu, ignore.")
                    continue
                byproducts.append((tok, d_tok))
                ensure_elements_in_final(d_tok)
            except Exception as ex:
                print(f"Avertissement : gaz '{tok}' non interpretable ({ex}), ignore.")

    # De-duplication eventuelle des byproducts par nom
    if byproducts:
        dedup = []
        seen = set()
        for name, dct in byproducts:
            if name not in seen:
                dedup.append((name, dct))
                seen.add(name)
        byproducts = dedup

    # On les ajoute dans la liste
    for (byp_name, byp_dict) in byproducts:
        precursors.append(byp_name)
        precursor_dicts.append(byp_dict)

    # Nombre de precurseurs reels (saisis)
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

    eps = 1e-14

    # Resolution stoechiometrie
    x_stoich, residuals, rank, singular_values = solve_stoichiometry(
        final_dict, precursor_dicts, nonneg=args.nonneg
    )
    n_final = mass_target / M_final

    # On stocke separément:
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
            # Precurseur reel
            pcent = purete_list[i]
            alpha = pcent / 100.0
            mass_corr = mass_prec_i / alpha
            dmass_corr = dm_prec_i / alpha
            if mass_prec_i >= 0:
                results_consumed.append((prec_name, mass_corr, dmass_corr, pcent, mass_corr - mass_prec_i, M_prec_i))
            else:
                results_byproducts.append((prec_name, abs(mass_corr), abs(dmass_corr)))
        else:
            # Pseudo-precurseur (gaz) => sous-produit degage
            mass_rel = abs(mass_prec_i)
            dmass_rel = abs(dm_prec_i)
            results_byproducts.append((prec_name, mass_rel, dmass_rel))

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

    # === Normalisation : par l'element choisi si fourni, sinon comme avant ===
    # Choix par defaut = element au coeff le plus grand dans la formule cible
    sorted_by_stoich = sorted(final_dict.items(), key=lambda x: x[1], reverse=True)
    default_ref_el, default_ref_val = sorted_by_stoich[0]

    use_ref_el = default_ref_el
    use_ref_val = default_ref_val

    if pivot_el_user:
        if pivot_el_user not in final_dict or final_dict[pivot_el_user] <= 0.0:
            print(f"Avertissement : element de normalisation '{pivot_el_user}' absent ou nul dans la formule finale. "
                  f"Normalisation par defaut sur {default_ref_el}.")
        else:
            use_ref_el = pivot_el_user
            use_ref_val = final_dict[pivot_el_user]

    denom = total_elements.get(use_ref_el, 0.0)
    if denom > eps:
        scale = use_ref_val / denom
    else:
        print(f"Avertissement : impossible de normaliser sur {use_ref_el} (denominateur nul). "
              f"Facteur d'echelle fixe a 1.0.")
        scale = 1.0

    scaled_elements = {el: val*scale for el,val in total_elements.items()}

    # Verif ecarts
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

    # Sauvegarde
    folder_name = final_formula
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    timestamp_str = run_timestamp.strftime("%Y_%m_%d_%H_%M")
    filename_base = f"{final_formula}_with_{'_'.join(precs_str.split())}"
    if byproducts:
        filename_base += f"_gas_{'_'.join([name for name,_ in byproducts])}"
    filename_base = f"{timestamp_str}_{filename_base}"
    
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

    # === Affichage final ===
    print(f"Resultats sauvegardes dans : {candidate_path}\n")
    print("===== Parametres d'entree =====")
    print(f"Version            : {APP_VERSION}")
    print(f"Date execution     : {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fichier Excel      : {excel_path}")
    print(f"Feuille Excel      : {sheet_name}")
    print(f"Precurseurs        : {', '.join(precs_str.split())}")
    if purete_list:
        print(f"Purete (%)         : {', '.join(f'{p:.2f}' for p in purete_list)}")
    if byproducts:
        print(f"Gaz autorises      : {', '.join(name for name, _ in byproducts)}")
    else:
        print("Gaz autorises      : aucun")
    print(f"Tol. verification  : {tol}")
    print(f"Export CSV         : {'oui' if args.exportCSV else 'non'}")
    print(f"Contraintes >= 0   : {'oui' if args.nonneg else 'non'}\n")
    print(f"Produit final vise : {final_formula}")
    print(f"Masse visee       : {mass_target:.3f} g")
    print(f"Masse molaire du produit final : {M_final:.3f} +/- {dM_final:.3f} g/mol\n")

    if pivot_el_user:
        print(f"Normalisation de la formule reconstituee sur : {use_ref_el}\n")

    print("===== Precurseurs (a peser) =====\n")
    for (pname, mass_corr, dmass_corr, pcent, delta_m, M_prec_i) in results_consumed:
        purete_str_print = ""
        if abs(pcent - 100.0) > 1e-9:
            purete_str_print = f"(purete={pcent:.1f}%, delta={delta_m:.4f} g)"
        print(f" * {pname:<8s} : {mass_corr:.{decimals}f} +/- {dmass_corr:.{decimals}f} g  {purete_str_print}")

    print("\n===== Details sur les precurseurs consommes =====\n")

    total_mass = 0.0
    total_moles = 0.0
    total_mass_uncert_sq = 0.0
    consumed_data = []
    for (pname, mass_corr, dmass_corr, pcent, delta_m, M_prec_i) in results_consumed:
        if M_prec_i != 0:
            n_prec = mass_corr / M_prec_i
        else:
            n_prec = 0.0
        consumed_data.append((pname, mass_corr, n_prec))
        total_mass += max(mass_corr, 0)
        total_moles += max(n_prec, 0)
        total_mass_uncert_sq += dmass_corr ** 2

    if total_mass < 1e-12 or total_moles < 1e-12:
        print("Aucun precurseur reel consomme !")
    else:
        total_mass_uncert = math.sqrt(total_mass_uncert_sq)
        print(f"Total masse precurseurs : {total_mass:.{decimals}f} +/- {total_mass_uncert:.{decimals}f} g")
        for (pname, m_corr, n_corr) in consumed_data:
            frac_mass = m_corr / total_mass
            frac_mole = n_corr / total_moles
            print(f" - {pname:<8s} : masse = {m_corr:.{decimals}f} g,  fraction massique = {frac_mass*100:.2f}%, "
                  f"n = {n_corr:.{decimals}f} mol, fraction molaire = {frac_mole*100:.2f}%")

    print("\n===== Sous-produits rejetes (masse > 0 => degage) =====\n")
    if len(results_byproducts) == 0:
        print("Aucun sous-produit calcule (mass negative)")
    else:
        total_byproduct_uncert_sq = 0.0
        for (pname, mass_rel, dmass_rel) in results_byproducts:
            print(f" - {pname:<8s} : {mass_rel:.{decimals}f} +/- {dmass_rel:.{decimals}f} g liberes")
            total_byproduct_uncert_sq += dmass_rel ** 2
        total_byproduct_uncert = math.sqrt(total_byproduct_uncert_sq)
        total_byproduct_mass = sum(mass for _, mass, _ in results_byproducts)
        print(f"\nTotal sous-produits : {total_byproduct_mass:.{decimals}f} +/- {total_byproduct_uncert:.{decimals}f} g")

    print("\n===== Stoechiometrie totale reconstituee (pour 1 mole du produit) =====")
    for el in sorted(total_elements.keys()):
        print(f"  Element {el} : {total_elements[el]:.3f}")

    print("\n===== Diagnostics de resolution =====")
    if residuals is not None and len(residuals) > 0:
        print(f"  Residus (somme) : {residuals[0]:.6g}")
    else:
        print("  Residus (somme) : non disponibles")
    print(f"  Rang matrice    : {rank}")
    if singular_values is not None and len(singular_values) > 0:
        print(f"  Valeurs sing.   : {', '.join(f'{val:.6g}' for val in singular_values)}")

    print("\n===== Stoechiometrie apportee par chaque precurseur (pour 1 mole du produit) =====")
    for i, (prc_name, sto_dict) in enumerate(partial_stoech):
        print(f"  Precurseur : {prc_name} (x={x_stoich[i]:.3f})")
        for el in sorted(sto_dict.keys()):
            print(f"    {el} : {sto_dict[el]:.3f}")

    print(f"\nFormule brute reconstituee (normalisee sur {use_ref_el}) : {recon_formula}")

    sys.stdout = backup_stdout
    f_out.close()
    print(f"\nLe fichier de resultats a ete cree ici : {candidate_path}")

    if args.exportCSV:
        csv_path = os.path.splitext(candidate_path)[0] + ".csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([
                "type", "nom", "masse_g", "incertitude_g", "purete_pct",
                "delta_masse_g", "masse_molaire_g_mol"
            ])
            for (pname, mass_corr, dmass_corr, pcent, delta_m, M_prec_i) in results_consumed:
                writer.writerow([
                    "precurseur", pname, f"{mass_corr:.{decimals}f}",
                    f"{dmass_corr:.{decimals}f}", f"{pcent:.2f}",
                    f"{delta_m:.{decimals}f}", f"{M_prec_i:.6f}"
                ])
            for (pname, mass_rel, dmass_rel) in results_byproducts:
                writer.writerow([
                    "sous-produit", pname, f"{mass_rel:.{decimals}f}",
                    f"{dmass_rel:.{decimals}f}", "", "", ""
                ])
        print(f"Fichier CSV cree : {csv_path}")

    save_last_config(
        config_path,
        {
            "Prod": final_formula,
            "m": mass_target,
            "Precs": precs_str,
            "Purete": args.Purete.strip(),
            "PivotEl": pivot_el_user,
            "prec": decimals,
            "tol": tol,
            "path": excel_path,
            "sheet": str(sheet_name),
            "OtherGases": args.OtherGases.strip(),
            "exportCSV": args.exportCSV,
            "nonneg": args.nonneg,
        }
    )


if __name__ == "__main__":
    main()
