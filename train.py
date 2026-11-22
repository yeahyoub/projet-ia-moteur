from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from analyse_vibratoire import CARACTERISTIQUES_V1
from entrainement import CHARGE_TEST_HP, chemins_modeles, creer_modeles


DOSSIER_PROJET = Path(__file__).resolve().parent


def main():
    donnees = pd.read_csv(DOSSIER_PROJET / "features.csv")
    masque_test = donnees["Charge_HP"] == CHARGE_TEST_HP
    donnees_entrainement = donnees.loc[~masque_test]
    donnees_test = donnees.loc[masque_test]
    caracteristiques = list(CARACTERISTIQUES_V1)
    caracteristiques_entrainement = donnees_entrainement[caracteristiques]
    caracteristiques_test = donnees_test[caracteristiques]
    classes_entrainement = donnees_entrainement["Classe"]
    classes_test = donnees_test["Classe"]

    print("Échantillons entraînement :", len(caracteristiques_entrainement))
    print("Échantillons test :", len(caracteristiques_test))

    modeles = creer_modeles()
    for nom, modele in modeles.items():
        print(f"\n===== {nom} =====")
        modele.fit(caracteristiques_entrainement, classes_entrainement)
        predictions = modele.predict(caracteristiques_test)

        accuracy = accuracy_score(classes_test, predictions)
        print(f"Accuracy : {accuracy:.2%}")
        print(classification_report(classes_test, predictions, zero_division=0))

        affichage = ConfusionMatrixDisplay.from_predictions(
            classes_test, predictions, xticks_rotation=45, cmap="Blues"
        )
        affichage.ax_.set_title(f"Matrice de confusion - {nom}")
        affichage.figure_.tight_layout()
        plt.show()

    for nom, chemin in chemins_modeles(DOSSIER_PROJET).items():
        joblib.dump(modeles[nom], chemin)
    print("Modèles enregistrés !")


if __name__ == "__main__":
    main()
