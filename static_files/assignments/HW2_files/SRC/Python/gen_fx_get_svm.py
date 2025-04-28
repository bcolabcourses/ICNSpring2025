
import numpy as np
from sklearn.metrics import confusion_matrix
from gen_fx_get_equal_part import gen_fx_get_equal_part
from gen_fx_MC_SVM import gen_fx_MC_SVM

def gen_fx_get_svm(grp, I, rate, rep):
    """
    Trains and evaluates an SVM classifier over multiple repetitions.

    Parameters:
        grp (np.ndarray): Class labels (N,)
        I (np.ndarray): Feature matrix (N, D)
        rate (float): Proportion of training data per class
        rep (int): Number of repetitions

    Returns:
        out (dict): Contains accuracy, confusion matrices, per-class accuracy, and last model
        mdl: Last trained model
    """
    grp = np.array(grp)
    Pt = []  # Accuracy scores
    Tu = []  # Per-class accuracy
    Ct = []  # Confusion matrices

    for _ in range(rep):
        test, train = gen_fx_get_equal_part(grp, rate)

        cls, mdl = gen_fx_MC_SVM(I[test], I[train], grp[train])

        accuracy = np.sum(cls == grp[test]) / np.sum(test)
        Pt.append(accuracy)

        conf = confusion_matrix(grp[test], cls)
        Ct.append(conf)

        per_class_acc = np.diag(conf) / np.sum(conf, axis=1)
        Tu.append(per_class_acc)

    out = {
        "pt": np.array(Pt),
        "C": np.array(Ct),
        "tu": np.array(Tu),
        "model": mdl
    }

    return out, mdl
