"""This script fits classifiers with different random seeds using
different splits of data. This can then be used to get a better estimate
of which optimization/control scores combinations are likely for training
and test data
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils import ecfp


def get_predictions(chid, n_estimators=100, n_jobs=8, n_runs=10):
    # read data and calculate ecfp fingerprints
    assay_file = f'./assays/processed/{chid}.csv'
    print(f'Reading data from: {assay_file}')
    df = pd.read_csv(assay_file)
    X = np.array(ecfp(df.smiles))
    y = np.array(df.label)

    # train classifiers and store them in dictionary
    p11s = []
    p21s = []
    p12s = []
    p22s = []
    y1s = []
    y2s = []
    for i in range(n_runs):
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, stratify=y)
        y1s.append(y1)
        y2s.append(y2)
        clfs = {}
        clfs['Split1'] = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs)
        clfs['Split1'].fit(X1, y1)

        clfs['Split2'] = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=n_jobs)
        clfs['Split2'].fit(X2, y2)

        p11s.append(clfs['Split1'].predict_proba(X1)[:, 1])
        p21s.append(clfs['Split2'].predict_proba(X1)[:, 1])
        p12s.append(clfs['Split1'].predict_proba(X2)[:, 1])
        p22s.append(clfs['Split2'].predict_proba(X2)[:, 1])

    predictions = [p11s, p21s, p12s, p22s, y1s, y2s]
    predictions = [np.concatenate(x) for x in predictions]
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_results",
        type=str,
        default='./results/goal_directed_paper/',
        help="Directory pointing to goal-directed generation results.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="How many runs to perform. For small datasets more runs will give better plots in the end.")
    args = parser.parse_args()

    chids = os.listdir(os.path.join(args.dir_results, 'graph_ga'))

    trainset_predictions = {}
    for chid in chids:
        trainset_predictions[chid] = get_predictions(chid, n_runs=args.n_runs)

    fn_results = os.path.join(args.dir_results, 'predictions.p')
    with open(fn_results, 'wb') as f:
        pickle.dump(trainset_predictions, f)

    print(f'Wrote data to: {fn_results}')
