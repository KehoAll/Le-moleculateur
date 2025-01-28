Le Moleculateur

Le Moleculateur est un outil Python (avec interface graphique grâce à Gooey) permettant de calculer les quantités de précurseurs nécessaires pour synthétiser un composé chimique donné, en tenant compte :

- Des masses molaires (avec incertitudes) lues depuis un fichier Excel
- De la pureté des précurseurs
- D’une éventuelle décomposition en libérant certains gaz (CO₂, O₂, H₂O)

Le programme effectue une résolution stœchiométrique par moindres carrés, puis génère un rapport avec les quantités à peser et les sous-produits émis (sous forme de fichier texte).

Le programme s'exécute en lançant "Le_moleculateur.exe".

En cas de modification du fichier .py, la compilation de ce dernier en .exe se fait dans la console au même dossier que le .py : pyinsaller --onefile Le_moleculateur.py 
Après la compilation le fichier .exe ainsi généé se trouve dans le dossier "dist".
