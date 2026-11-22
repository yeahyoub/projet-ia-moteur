"""Diagnostic Streamlit des moteurs avec le modèle choisi par l’utilisateur."""

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from analyse_vibratoire import (
    FS_CIBLE,
    TAILLE_FENETRE,
    SignalIntrouvable,
    calculer_caracteristiques,
    harmoniser_frequence,
    lire_signal_mat,
)
from entrainement import chemins_modeles


@st.cache_resource
def charger_modeles():
    racine = Path(__file__).resolve().parent
    return {nom: joblib.load(chemin) for nom, chemin in chemins_modeles(racine).items()}


@st.cache_data(max_entries=8, show_spinner=False)
def analyser_signal(contenu, frequence_origine):
    signal = lire_signal_mat(BytesIO(contenu))
    signal = harmoniser_frequence(signal, frequence_origine)
    apercu = signal[:2000].copy()
    if signal.size < TAILLE_FENETRE:
        return apercu, None

    return apercu, calculer_caracteristiques(signal, version=1)


def afficher_diagnostic(modele, caracteristiques):
    predictions = modele.predict(caracteristiques)
    effectifs = pd.Series(predictions).value_counts()
    classe = effectifs.idxmax()
    pourcentage = effectifs.max() / len(predictions) * 100

    st.subheader("Résultat du diagnostic")
    colonne_classe, colonne_part = st.columns(2)
    colonne_classe.metric("État prédit", classe)
    colonne_part.metric("Fenêtres classées dans cet état", f"{pourcentage:.1f}%")

    st.subheader("Répartition des prédictions")
    st.bar_chart(effectifs)
    st.info(
        "Prototype académique : ce résultat ne constitue "
        "pas un diagnostic industriel certifié."
    )


def main():
    st.set_page_config(page_title="Diagnostic IA des moteurs", page_icon="⚙️", layout="wide")
    st.title("⚙️ Diagnostic intelligent des moteurs électriques")
    st.write("Détection des défauts de roulements par intelligence artificielle")

    try:
        modeles = charger_modeles()
    except FileNotFoundError as erreur:
        st.error(f"Fichier de modèle introuvable ({Path(erreur.filename).name}).")
        return

    algorithme = st.selectbox("Choisir l'algorithme", list(modeles))
    frequence = st.selectbox(
        "Fréquence d'échantillonnage du fichier", [FS_CIBLE, 48_000]
    )
    fichier = st.file_uploader("Importer un signal vibratoire", type=["mat"])
    if fichier is None:
        return

    try:
        apercu, caracteristiques = analyser_signal(fichier.getvalue(), frequence)
    except SignalIntrouvable:
        st.error("Aucun signal DE_time trouvé.")
        return
    except (OSError, ValueError) as erreur:
        st.error(str(erreur))
        return

    st.subheader("Signal vibratoire")
    st.line_chart(apercu)
    if caracteristiques is None:
        st.error("Signal trop court pour l'analyse.")
        return

    afficher_diagnostic(modeles[algorithme], caracteristiques)


if __name__ == "__main__":
    main()
