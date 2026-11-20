"""Affiche la forme du signal et son RMS pour chaque état du moteur."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyse_vibratoire import SignalIntrouvable, lire_signal_mat


DOSSIER_DONNEES = Path(__file__).resolve().parent / "data"
ENREGISTREMENTS = (
    ("Moteur sain", "97.mat"),
    ("Bague intérieure", "105.mat"),
    ("Bille", "118.mat"),
    ("Bague extérieure", "130.mat"),
)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for axe, (etat, fichier) in zip(axes.flat, ENREGISTREMENTS):
        try:
            signal = lire_signal_mat(DOSSIER_DONNEES / fichier)
        except SignalIntrouvable as erreur:
            raise ValueError(f"Signal DE introuvable dans {fichier}") from erreur

        rms = np.sqrt(np.mean(signal**2))
        print(f"{etat} : RMS = {rms:.5f}")

        axe.plot(signal[:2000], linewidth=0.8)
        axe.set(title=etat, xlabel="Échantillons", ylabel="Amplitude")
        axe.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
