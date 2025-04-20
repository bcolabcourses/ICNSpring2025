
import numpy as np

def gen_fx_get_equal_part(grp, rate):
    """
    Splits data into balanced training and testing sets for each class.

    Parameters:
        grp (np.ndarray): A vector of class labels for each sample (N,)
        rate (float): Proportion of samples per class to assign to training (e.g., 0.7)

    Returns:
        test (np.ndarray): Boolean array indicating test samples
        train (np.ndarray): Boolean array indicating train samples
    """
    grp = np.array(grp)
    catNo = np.unique(grp)
    asiz = []

    for cat in catNo:
        asiz.append(np.sum(grp == cat))

    minL = min(asiz)
    TrSiz = int(np.floor(rate * minL))
    TeSiz = minL - TrSiz

    train = np.zeros(grp.shape, dtype=bool)
    test = np.zeros(grp.shape, dtype=bool)

    for cat in catNo:
        ix = np.where(grp == cat)[0]
        np.random.shuffle(ix)
        train[ix[:TrSiz]] = True
        test[ix[-TeSiz:]] = True

    return test, train
