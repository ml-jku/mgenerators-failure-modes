import uuid
from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime

import numpy as np
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score


def timestamp(adduuid=False):
    s = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    if adduuid:
        s = s + '_' + uuid.uuid4().hex
    return s


def can_list(smiles):
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.MolToSmiles(m) for m in ms if m is not None]


def one_ecfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=1024))
        return fp
    except:
        return None


def ecfp(smiles, radius=2, n_jobs=12):
    with Pool(n_jobs) as pool:
        X = pool.map(partial(one_ecfp, radius=radius), smiles)
    return X


def calc_auc(clf, X_test, y_test):
    preds = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)


def score(smiles_list, clf):
    """Makes predictions for a list of smiles. Returns none if smiles is invalid"""
    X = ecfp(smiles_list)
    X_valid = [x for x in X if x is not None]
    if len(X_valid) == 0:
        return X

    preds_valid = clf.predict_proba(np.array(X_valid))[:, 1]
    preds = []
    i = 0
    for li, x in enumerate(X):
        if x is None:
            # print(smiles_list[li], Chem.MolFromSmiles(smiles_list[li]), x)
            preds.append(None)
        else:
            preds.append(preds_valid[i])
            assert preds_valid[i] is not None
            i += 1
    return preds


class TPScoringFunction(BatchScoringFunction):
    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def raw_score_list(self, smiles_list):
        return score(smiles_list, self.clf)
