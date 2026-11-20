"""Préparation et analyse des signaux vibratoires au format MATLAB."""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from scipy.stats import kurtosis


FS_CIBLE = 12_000
TAILLE_FENETRE = 2_048
FENETRES_PAR_LOT = 128
EPSILON = 1e-12

CARACTERISTIQUES_V1 = (
    "RMS",
    "Ecart_type",
    "Kurtosis",
    "Frequence_dominante",
)
CARACTERISTIQUES_V2 = (
    "RMS",
    "Ecart_type",
    "Kurtosis",
    "Peak_to_Peak",
    "Crest_Factor",
    "Frequence_dominante",
    "Spectral_Centroid",
    "Energie_Bande",
)

FENETRE_HANN = np.hanning(TAILLE_FENETRE)
FREQUENCES = np.fft.rfftfreq(TAILLE_FENETRE, d=1 / FS_CIBLE)
BANDE_1_A_3_KHZ = slice(
    FREQUENCES.searchsorted(1_000),
    FREQUENCES.searchsorted(3_000, side="right"),
)


class SignalIntrouvable(ValueError):
    """Le fichier MATLAB ne contient aucun signal côté entraînement."""


def lire_signal_mat(source, *, dtype=None):
    """Lit le premier signal vibratoire côté entraînement."""
    contenu = loadmat(source)
    cle_signal = next((cle for cle in contenu if cle.endswith("_DE_time")), None)
    if cle_signal is None:
        raise SignalIntrouvable(
            "Aucune variable '_DE_time' trouvée dans ce fichier MATLAB."
        )
    return np.asarray(contenu[cle_signal], dtype=dtype).ravel()


def harmoniser_frequence(signal, frequence_origine):
    if frequence_origine == 48_000:
        return resample_poly(signal, up=1, down=4)
    return signal


def calculer_caracteristiques(signal, *, version=2):
    """Extrait les caractéristiques de chaque fenêtre complète à 12 kHz."""
    if version not in (1, 2):
        raise ValueError(f"Version de caractéristiques inconnue : {version}")

    colonnes = CARACTERISTIQUES_V1 if version == 1 else CARACTERISTIQUES_V2
    nombre_fenetres = len(signal) // TAILLE_FENETRE
    if nombre_fenetres == 0:
        return pd.DataFrame(columns=colonnes)

    fenetres = signal[: nombre_fenetres * TAILLE_FENETRE].reshape(-1, TAILLE_FENETRE)

    # Traiter les longues mesures par lots pour limiter la mémoire utilisée.
    lots = []
    for debut in range(0, nombre_fenetres, FENETRES_PAR_LOT):
        lot = fenetres[debut : debut + FENETRES_PAR_LOT]
        lots.append(_caracteristiques_du_lot(lot, colonnes, version))
    return pd.concat(lots, ignore_index=True)


def _caracteristiques_du_lot(fenetres, colonnes, version):
    rms = np.sqrt(np.mean(fenetres**2, axis=1))
    fenetres_centrees = fenetres - np.mean(fenetres, axis=1, keepdims=True)
    spectres = np.abs(np.fft.rfft(fenetres_centrees * FENETRE_HANN, axis=1))

    # Ignorer la composante continue pour chercher le pic.
    indices_pics = np.argmax(spectres[:, 1:], axis=1) + 1
    mesures = {
        "RMS": rms,
        "Ecart_type": np.std(fenetres, axis=1),
        "Kurtosis": kurtosis(fenetres, axis=1, fisher=False),
        "Frequence_dominante": FREQUENCES[indices_pics],
    }
    if version == 2:
        puissances = spectres**2
        mesures.update(
            {
                "Peak_to_Peak": np.ptp(fenetres, axis=1),
                "Crest_Factor": np.max(np.abs(fenetres), axis=1) / (rms + EPSILON),
                "Spectral_Centroid": np.sum(FREQUENCES * spectres, axis=1)
                / (np.sum(spectres, axis=1) + EPSILON),
                "Energie_Bande": np.sum(puissances[:, BANDE_1_A_3_KHZ], axis=1)
                / (np.sum(puissances[:, 1:], axis=1) + EPSILON),
            }
        )
    return pd.DataFrame(mesures, columns=colonnes)
