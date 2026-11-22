from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CHARGE_TEST_HP = 3


def chemins_modeles(dossier_projet, *, version=1):
    suffixe = "_v2" if version == 2 else ""
    return {
        "Random Forest": dossier_projet / f"random_forest{suffixe}.joblib",
        "SVM": dossier_projet / f"svm{suffixe}.joblib",
    }


def creer_modeles():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, class_weight="balanced"),
        ),
    }
