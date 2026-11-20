"""Évalue les modèles V2 sur trois fichiers de défaut de taille 0,021 pouce."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from analyse_vibratoire import calculer_caracteristiques, lire_signal_mat


DOSSIER_PROJET = Path(__file__).resolve().parent
CLASSES_REELLES = {
    "212.mat": "Bague intérieure",
    "225.mat": "Bille",
    "237.mat": "Bague extérieure",
}


def afficher_diagnostic(nom, predictions, classe_reelle):
    effectifs = pd.Series(predictions).value_counts()
    diagnostic = effectifs.index[0]
    part_majoritaire = effectifs.iloc[0] / len(predictions) * 100
    part_correcte = np.mean(predictions == classe_reelle) * 100

    print(f"\n{nom}")
    print("Diagnostic :", diagnostic)
    print(f"Vote majoritaire : {part_majoritaire:.1f}%")
    print(f"Fenêtres correctes : {part_correcte:.1f}%")
    print("Répartition :")
    print(effectifs.to_string())


def main():
    modeles = {
        "Random Forest": joblib.load(DOSSIER_PROJET / "random_forest_v2.joblib"),
        "SVM": joblib.load(DOSSIER_PROJET / "svm_v2.joblib"),
    }
    for fichier, classe_reelle in CLASSES_REELLES.items():
        print(f"\n===== {fichier} =====")
        print("Défaut réel :", classe_reelle)

        signal = lire_signal_mat(DOSSIER_PROJET / "data" / fichier)
        mesures = calculer_caracteristiques(signal)
        for nom, modele in modeles.items():
            predictions = modele.predict(mesures)
            afficher_diagnostic(nom, predictions, classe_reelle)


if __name__ == "__main__":
    main()
