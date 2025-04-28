
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from scipy.stats import mode

def gen_fx_MC_SVM(sample, train, grp):
    """
    Multi-class SVM classification using one-vs-one strategy with voting.

    Parameters:
        sample (np.ndarray): Feature matrix of samples to classify (MxD)
        train (np.ndarray): Feature matrix of training samples (NxD)
        grp (np.ndarray): Class labels for the training samples (Nx1 or 1D)

    Returns:
        class_pred (np.ndarray): Predicted class labels for the input samples (Mx1)
        model (OneVsOneClassifier): Trained One-vs-One multi-class SVM model
    """
    grp = np.array(grp).flatten()
    classNo = np.unique(grp)
    cls_nu = len(classNo)

    # Store predictions from all pairwise classifiers
    vote_matrix = np.zeros((sample.shape[0], cls_nu * (cls_nu - 1) // 2))
    
    idx = 0
    for i in range(cls_nu):
        for j in range(i + 1, cls_nu):
            class_i = classNo[i]
            class_j = classNo[j]

            # Select samples from the two classes
            ix = np.where((grp == class_i) | (grp == class_j))[0]
            X_pair = train[ix]
            y_pair = grp[ix]

            # Train binary SVM
            clf = SVC(kernel='linear')
            clf.fit(X_pair, y_pair)

            # Predict test samples
            vote_matrix[:, idx] = clf.predict(sample)
            idx += 1

    # Majority voting
    class_pred, _ = mode(vote_matrix, axis=1)
    class_pred = class_pred.flatten()

    # Train ECOC-style model (sklearn uses OvO under the hood)
    model = OneVsOneClassifier(SVC(kernel='linear'))
    model.fit(train, grp)

    return class_pred, model
