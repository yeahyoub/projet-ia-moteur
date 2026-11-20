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

# Fichiers ordonnés par charge de 0 à 3 HP pour chaque état du roulement.
FICHIERS_PAR_CLASSE = {
    "Sain": [97, 98, 99, 100],
    "Bague intérieure": [105, 106, 107, 108],
    "Bille": [118, 119, 120, 121],
    "Bague extérieure": [130, 131, 132, 133],
}


def creer_dataset(dossier_donnees):
    tableaux = []
    for classe, numeros in FICHIERS_PAR_CLASSE.items():
        frequence_origine = 48_000 if classe == "Sain" else FS_CIBLE
        for charge, numero in enumerate(numeros):
            fichier = f"{numero}.mat"
            signal = lire_signal_mat(dossier_donnees / fichier)
            signal = harmoniser_frequence(signal, frequence_origine)

            mesures = calculer_caracteristiques(signal, version=1)
            mesures.insert(0, "Fichier", fichier)
            mesures.insert(1, "Charge_HP", charge)
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
