"""Génère le tableau des caractéristiques utilisé par les modèles V2."""

from features import DOSSIER_PROJET, creer_dataset as creer_dataset_commun


def creer_dataset(dossier_donnees):
    return creer_dataset_commun(dossier_donnees, version=2)


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
