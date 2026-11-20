# Projet IA moteur

Ce prototype Python classe les vibrations d’un roulement dans quatre catégories : sain, défaut de bague intérieure, défaut de bille ou défaut de bague extérieure. L’application Streamlit V2 compare les prédictions de Random Forest et de SVM. Chaque modèle vote sur des fenêtres de 2 048 échantillons.

## Lancer l’application sur Windows

Le projet a été testé avec Python 3.14. Depuis PowerShell dans le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app_v2.py
```

Les modèles préentraînés sont fournis. Dans l’application, sélectionne la fréquence d’origine et importe un fichier MATLAB contenant une variable dont le nom se termine par `_DE_time`. La V1 se lance avec `app.py` et permet de choisir un seul modèle.

## Essayer avec deux fichiers

Les mesures brutes viennent du Bearing Data Center de la Case Western Reserve University. Télécharge les fichiers auprès de la source, puis place-les dans le dossier `data/` du projet avec les noms suivants.

| Source | Nom dans le projet | Fréquence | État connu |
| --- | --- | --- | --- |
| [Normal_0 · Normal Baseline Data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data) | 97.mat | 48 000 Hz | Sain |
| [IR007_0 · 12k Drive End Bearing Fault Data](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data) | 105.mat | 12 000 Hz | Bague intérieure |

Le pourcentage affiché correspond à la part des fenêtres attribuées à la classe la plus fréquente. Le projet utilise des essais de laboratoire.

## Régénérer les modèles V2

Pour réentraîner les modèles, place les 28 fichiers suivants dans `data/`. Dans chaque ligne, les fichiers sont listés dans l’ordre des charges de 0 à 3 chevaux. Pour la bague extérieure, choisis la série centrée à 6 h sur la page CWRU.

| Série CWRU | Numéros des fichiers .mat |
| --- | --- |
| Normal_0 à Normal_3 | 97, 98, 99, 100 |
| IR007_0 à IR007_3 | 105, 106, 107, 108 |
| B007_0 à B007_3 | 118, 119, 120, 121 |
| Bague extérieure 0,007 pouce à 6 h | 130, 131, 132, 133 |
| IR014_0 à IR014_3 | 169, 170, 171, 172 |
| B014_0 à B014_3 | 185, 186, 187, 188 |
| Bague extérieure 0,014 pouce à 6 h | 197, 198, 199, 200 |

```powershell
.\.venv\Scripts\python.exe features_v2.py
.\.venv\Scripts\python.exe train_v2.py
```

Ces commandes régénèrent `features_v2.csv`, les deux modèles V2 et leurs matrices de confusion. La V1 utilise `features.py` et `train.py` avec les quatre premières séries du tableau. Pour le test supplémentaire `test_021.py`, ajoute aussi 212.mat (IR021_3), 225.mat (B021_3) et 237.mat (bague extérieure à 6 h, 0,021 pouce, 3 chevaux).

## Guides en français

- [Comprendre le projet IA moteur](docs/01%20-%20Comprendre%20le%20projet%20IA%20moteur.pdf)
- [Comprendre le code pas à pas](docs/02%20-%20Comprendre%20le%20code%20pas%20a%20pas.pdf)
