Le Moleculateur

Le Moleculateur est un outil Python (avec interface graphique grâce à Gooey) permettant de calculer les quantités de précurseurs nécessaires pour synthétiser un composé chimique donné, en tenant compte :

- Des masses molaires (avec incertitudes) lues depuis un fichier Excel
- De la pureté des précurseurs
- D’une éventuelle décomposition en libérant certains gaz (CO₂, O₂, H₂O)

Le programme effectue une résolution stœchiométrique par moindres carrés, puis génère un rapport avec les quantités à peser et les sous-produits émis (sous forme de fichier texte).

Le programme s'exécute en lançant "Le_moleculateur.exe".

En cas de modification du fichier .py, la compilation de ce dernier en .exe se fait dans la console au même dossier que le .py : pyinstaller --onefile Le_moleculateur.py 
Après la compilation le fichier .exe ainsi généé se trouve dans le dossier "dist".

Conseils pour reduire la taille de l'executable :
- Preferer "--onedir" (moins compact mais plus legible) si un dossier de sortie est acceptable.
- Utiliser UPX pour compresser l'executable (option "--upx-dir" de PyInstaller).
- Exclure les modules inutiles via "--exclude-module" (ex: matplotlib, scipy) si non utilises.
- Nettoyer les datas embarquees si elles ne sont pas necessaires au runtime.
- Utiliser le fichier "Le_moleculateur.optimized.spec" (mode onedir + exclusions) : pyinstaller Le_moleculateur.optimized.spec

Nouveautes (version 1.4.0) :
- Moteur de calcul isole dans "core.py" pour faciliter les tests.
- Resolution stoechiometrique avec option non-negatif plus robuste (NNLS interne).
- Diagnostic et synthese des incertitudes globales (totaux precurseurs/sous-produits).
- Tests automatises (pytest) pour le parsing et la resolution.
- Build optimise via "Le_moleculateur.optimized.spec".

Nouveautes (version 1.3.0) :
- Support des formules avec parenthèses, hydrates (ex: CuSO4·5H2O) et charges finales (+/-).
- Parametre "--sheet" pour selectionner la feuille Excel.
- Option "Exporter CSV" pour obtenir un fichier .csv a cote du rapport texte.
- Option "Forcer les quantites negatives a zero" pour bloquer les coefficients negatifs.
- Memorisation automatique des derniers parametres (fichier .le_moleculateur_config.json).
