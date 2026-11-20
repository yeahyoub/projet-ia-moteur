from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
