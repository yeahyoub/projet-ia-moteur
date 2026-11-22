"""Diagnostic Streamlit des roulements avec comparaison des deux modèles."""

from collections import Counter
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analyse_vibratoire import (
    FS_CIBLE,
    TAILLE_FENETRE,
    calculer_caracteristiques,
    calculer_spectre,
    harmoniser_frequence,
    lire_signal_mat,
)
from entrainement import chemins_modeles


CLASSES = ["Sain", "Bague intérieure", "Bille", "Bague extérieure"]


@st.cache_resource
def charger_modeles():
    racine = Path(__file__).resolve().parent
    chemins = chemins_modeles(racine, version=2)
    manquants = [chemin.name for chemin in chemins.values() if not chemin.is_file()]
    if manquants:
        raise FileNotFoundError("Fichiers de modèles manquants : " + ", ".join(manquants))
    return {nom: joblib.load(chemin) for nom, chemin in chemins.items()}


def charger_signal(source, frequence_origine):
    signal = lire_signal_mat(source, dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("Le signal contient des valeurs non finies.")

    signal = harmoniser_frequence(signal, frequence_origine)
    if signal.size < TAILLE_FENETRE:
        raise ValueError(f"Signal trop court pour former une fenêtre de {TAILLE_FENETRE} points.")
    return signal


@st.cache_data(max_entries=8, show_spinner=False)
def analyser_signal(contenu, frequence_origine):
    signal = charger_signal(BytesIO(contenu), frequence_origine)
    caracteristiques = calculer_caracteristiques(signal)
    if caracteristiques.isna().to_numpy().any():
        raise ValueError("Des caractéristiques non numériques ont été calculées.")
    return signal[:4096].copy(), caracteristiques


def figure_signal_et_spectre(signal):
    segment = signal[:4096]
    taille = len(segment)
    temps = np.arange(taille) / FS_CIBLE
    frequences, amplitude = calculer_spectre(segment)

    figure, (axe_signal, axe_spectre) = plt.subplots(
        2, 1, figsize=(10, 6), constrained_layout=True
    )
    axe_signal.plot(temps, segment, color="#3889c5", linewidth=0.8)
    axe_signal.set(title="Signal vibratoire", xlabel="Temps (s)", ylabel="Amplitude")
    axe_signal.grid(alpha=0.25)

    axe_spectre.plot(frequences, amplitude, color="#f59b42", linewidth=0.8)
    axe_spectre.set(
        title="Spectre fréquentiel (FFT)",
        xlabel="Fréquence (Hz)",
        ylabel="Amplitude spectrale",
        xlim=(0, FS_CIBLE / 2),
    )
    axe_spectre.grid(alpha=0.25)
    return figure


def diagnostiquer(modele, tableau):
    predictions = modele.predict(tableau)
    effectifs = Counter(predictions)
    classe_majoritaire, nombre = effectifs.most_common(1)[0]
    pourcentage = 100 * nombre / len(predictions)
    return classe_majoritaire, pourcentage, effectifs


def afficher_diagnostics(diagnostics):
    st.subheader("Comparaison des diagnostics")
    for colonne, (nom, diagnostic) in zip(st.columns(len(diagnostics)), diagnostics.items()):
        classe, pourcentage, effectifs = diagnostic
        repartition = pd.DataFrame(
            {"Fenêtres": [effectifs.get(classe, 0) for classe in CLASSES]},
            index=pd.Index(CLASSES, name="Classe"),
        )
        with colonne:
            st.markdown(f"**{nom}**")
            st.metric("Classe prédite", classe)
            st.metric("Part des fenêtres dans cette classe", f"{pourcentage:.1f} %")
            st.bar_chart(repartition)

    classes_predites = {diagnostic[0] for diagnostic in diagnostics.values()}
    if len(classes_predites) == 1:
        st.success("Les deux modèles indiquent la même classe.")
    else:
        st.warning(
            "Les deux modèles ne sont pas d'accord. Le résultat est incertain ; "
            "une vérification complémentaire est recommandée."
        )


def afficher_analyse(fichier, frequence):
    try:
        modeles = charger_modeles()
        signal, caracteristiques = analyser_signal(fichier.getvalue(), frequence)
        diagnostics = {
            nom: diagnostiquer(modele, caracteristiques)
            for nom, modele in modeles.items()
        }
    except Exception as erreur:
        st.error(f"Impossible d'analyser le fichier : {erreur}")
        st.stop()

    st.success(
        f"{fichier.name} analysé : {len(caracteristiques)} fenêtres "
        f"de {TAILLE_FENETRE} points, à {FS_CIBLE:,} Hz.".replace(",", " ")
    )

    figure = figure_signal_et_spectre(signal)
    st.pyplot(figure)
    plt.close(figure)

    afficher_diagnostics(diagnostics)
    with st.expander("Voir les huit caractéristiques extraites"):
        st.dataframe(caracteristiques.head(10), width="stretch")


def main():
    st.set_page_config(page_title="IA moteur - diagnostic V2", page_icon="⚙️", layout="wide")
    st.title("⚙️ Diagnostic intelligent des moteurs électriques")
    st.write("Détection de défauts de roulements par analyse vibratoire et intelligence artificielle.")
    st.caption("Version 2 : Random Forest et SVM, huit caractéristiques, fenêtres de 2048 échantillons.")

    frequence = st.selectbox(
        "Fréquence d'échantillonnage du fichier d'origine",
        options=[FS_CIBLE, 48_000],
        format_func=lambda valeur: f"{valeur:,} Hz".replace(",", " "),
    )
    st.info(
        "Pour les fichiers de défaut CWRU utilisés ici : 12 000 Hz. "
        "Pour les fichiers sains 97.mat à 100.mat de la base à 48 kHz : 48 000 Hz. "
        "Vérifie toujours la fréquence réelle du fichier."
    )
    fichier = st.file_uploader("Importer un signal vibratoire (.mat)", type=["mat"])
    if fichier is not None:
        afficher_analyse(fichier, frequence)

    st.divider()
    st.caption(
        "Prototype pédagogique entraîné sur des mesures de laboratoire CWRU. "
        "La part des fenêtres classées dans une catégorie correspond à un vote "
        "majoritaire, pas à une probabilité de fiabilité. "
        "Ce logiciel ne remplace pas un diagnostic industriel."
    )


if __name__ == "__main__":
    main()
