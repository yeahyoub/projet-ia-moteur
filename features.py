"""Génère le tableau des caractéristiques utilisé par les modèles V1."""

from pathlib import Path

import pandas as pd

from analyse_vibratoire import (
    FS_CIBLE,
    calculer_caracteristiques,
    harmoniser_frequence,
    lire_signal_mat,
)


DOSSIER_PROJET = Path(__file__).resolve().parent

# Dans chaque série, les fichiers suivent les charges de 0 à 3 HP.
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


def creer_dataset(dossier_donnees, *, version=1):
    tableaux = []
    for classe, fichiers_par_taille in FICHIERS_PAR_CLASSE.items():
        frequence_origine = 48_000 if classe == "Sain" else FS_CIBLE
        for taille_defaut, numeros in fichiers_par_taille.items():
            if version == 1 and taille_defaut not in ("normal", "007"):
                continue
            for charge, numero in enumerate(numeros):
                fichier = f"{numero}.mat"
                signal = lire_signal_mat(dossier_donnees / fichier)
                signal = harmoniser_frequence(signal, frequence_origine)

                mesures = calculer_caracteristiques(signal, version=version)
                mesures.insert(0, "Fichier", fichier)
                mesures.insert(1, "Charge_HP", charge)
                if version == 2:
                    mesures.insert(2, "Taille", taille_defaut)
                mesures["Classe"] = classe
                tableaux.append(mesures)

    return pd.concat(tableaux, ignore_index=True)


def main():
    dataset = creer_dataset(DOSSIER_PROJET / "data")
    dataset.to_csv(DOSSIER_PROJET / "features.csv", index=False)
    print("Dataset créé avec succès !")
    print("Dimensions :", dataset.shape)
    print(dataset.groupby(["Classe", "Charge_HP"]).size())


if __name__ == "__main__":
    main()
