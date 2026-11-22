"""Compare les spectres vibratoires des quatre états du moteur."""

from pathlib import Path

import matplotlib.pyplot as plt

from analyse_vibratoire import FS_CIBLE, calculer_spectre, harmoniser_frequence, lire_signal_mat


DOSSIER_DONNEES = Path(__file__).resolve().parent / "data"
ENREGISTREMENTS = (
    ("Moteur sain", "97.mat", 48_000),
    ("Bague intérieure", "105.mat", FS_CIBLE),
    ("Bille", "118.mat", FS_CIBLE),
    ("Bague extérieure", "130.mat", FS_CIBLE),
)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    for axe, (etat, fichier, frequence_origine) in zip(axes.flat, ENREGISTREMENTS):
        signal = lire_signal_mat(DOSSIER_DONNEES / fichier)
        extrait = harmoniser_frequence(signal, frequence_origine)[:4096]
        frequences, spectre = calculer_spectre(extrait)

        axe.plot(frequences, spectre)
        axe.set(
            title=etat,
            xlabel="Fréquence (Hz)",
            ylabel="Amplitude spectrale",
            xlim=(0, 3000),
        )
        axe.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
