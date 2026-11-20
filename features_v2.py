"""Génère le tableau des caractéristiques utilisé par les modèles V2."""

from pathlib import Path

import pandas as pd

from analyse_vibratoire import (
    FS_CIBLE,
    calculer_caracteristiques,
    harmoniser_frequence,
    lire_signal_mat,
)


DOSSIER_PROJET = Path(__file__).resolve().parent

# Pour chaque taille de défaut, les fichiers suivent les charges de 0 à 3 HP.
FICHIERS_PAR_CLASSE = {
    "Sain": {"normal": [97, 98, 99, 100]},
    "Bague intérieure": {
        "007": [105, 106, 107, 108],
        "014": [169, 170, 171, 172],
    },
    "Bille": {
        "007": [118, 119, 120, 121],
        "014": [185, 186, 187, 188],
    },
    "Bague extérieure": {
        "007": [130, 131, 132, 133],
        "014": [197, 198, 199, 200],
    },
}


def creer_dataset(dossier_donnees):
    tableaux = []
    for classe, fichiers_par_taille in FICHIERS_PAR_CLASSE.items():
        frequence_origine = 48_000 if classe == "Sain" else FS_CIBLE
        for taille_defaut, numeros in fichiers_par_taille.items():
            for charge, numero in enumerate(numeros):
                fichier = f"{numero}.mat"
                signal = lire_signal_mat(dossier_donnees / fichier)
                signal = harmoniser_frequence(signal, frequence_origine)

                mesures = calculer_caracteristiques(signal)
                mesures.insert(0, "Fichier", fichier)
                mesures.insert(1, "Charge_HP", charge)
                mesures.insert(2, "Taille", taille_defaut)
                mesures["Classe"] = classe
                tableaux.append(mesures)

    return pd.concat(tableaux, ignore_index=True)


def main():
    dataset = creer_dataset(DOSSIER_PROJET / "data")
    dataset.to_csv(DOSSIER_PROJET / "features_v2.csv", index=False)
    print("Dataset V2 créé !")
    print("Dimensions :", dataset.shape)
    print("\nNombre de fichiers :", dataset["Fichier"].nunique())
    print("\nFenêtres par classe :")
    print(dataset["Classe"].value_counts())


if __name__ == "__main__":
    main()
