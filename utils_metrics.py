import numpy as np
from sklearn.metrics import roc_auc_score

def safe_auc(y_true, y_proba):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        if y_proba is None:
            return None

        y_proba = np.asarray(y_proba)
        if y_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_proba))
        elif y_proba.ndim == 2:
            if y_proba.shape[1] == 2:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                return float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted'))
        return None
    except Exception:
        return None