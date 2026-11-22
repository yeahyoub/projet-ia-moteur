from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from analyse_vibratoire import CARACTERISTIQUES_V2
from entrainement import CHARGE_TEST_HP, chemins_modeles, creer_modeles


DOSSIER_PROJET = Path(__file__).resolve().parent


def bilan_par_fichier(donnees_test, predictions):
    details = donnees_test[["Fichier", "Classe"]].assign(Prediction=predictions)
    lignes = []
    for fichier, fenetres in details.groupby("Fichier"):
        classe_reelle = fenetres["Classe"].iloc[0]
        classe_predite = fenetres["Prediction"].mode().iloc[0]
        lignes.append(
            {
                "Fichier": fichier,
                "Réel": classe_reelle,
                "Prédit": classe_predite,
                "Correct": classe_reelle == classe_predite,
            }
        )
    return pd.DataFrame(lignes)


def main():
    donnees = pd.read_csv(DOSSIER_PROJET / "features_v2.csv")
    donnees_entrainement = donnees.loc[donnees["Charge_HP"] < CHARGE_TEST_HP]
    donnees_test = donnees.loc[donnees["Charge_HP"] == CHARGE_TEST_HP]

    fichiers_entrainement = set(donnees_entrainement["Fichier"])
    fichiers_test = set(donnees_test["Fichier"])
    if not fichiers_entrainement.isdisjoint(fichiers_test):
        raise ValueError("Des fichiers figurent dans les jeux d’entraînement et de test.")

    caracteristiques = list(CARACTERISTIQUES_V2)
    caracteristiques_entrainement = donnees_entrainement[caracteristiques]
    caracteristiques_test = donnees_test[caracteristiques]
    classes_entrainement = donnees_entrainement["Classe"]
    classes_test = donnees_test["Classe"]

    print("Fichiers entraînement :", len(fichiers_entrainement))
    print("Fichiers test :", len(fichiers_test))

    chemins = chemins_modeles(DOSSIER_PROJET, version=2)
    for nom, modele in creer_modeles().items():
        print(f"\n=========================\n{nom}\n=========================")
        modele.fit(caracteristiques_entrainement, classes_entrainement)
        predictions = modele.predict(caracteristiques_test)

        accuracy = accuracy_score(classes_test, predictions)
        score_f1 = f1_score(classes_test, predictions, average="macro")
        print("Accuracy par fenêtre :", f"{accuracy:.2%}")
        print("F1-score macro :", f"{score_f1:.3f}")
        print(classification_report(classes_test, predictions, zero_division=0))

        bilan = bilan_par_fichier(donnees_test, predictions)
        print("\nRésultats par fichier :")
        print(bilan.to_string(index=False))
        print("Accuracy par fichier :", f"{bilan['Correct'].mean():.2%}")

        chemin_modele = chemins[nom]
        affichage = ConfusionMatrixDisplay.from_predictions(
            classes_test, predictions, xticks_rotation=45, cmap="Blues"
        )
        figure = affichage.figure_
        figure.tight_layout()
        figure.savefig(DOSSIER_PROJET / f"confusion_{chemin_modele.stem}.png", dpi=160)
        plt.close(figure)
        joblib.dump(modele, chemin_modele)

    print("\nEntraînement V2 terminé !")


if __name__ == "__main__":
    main()
