import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from time import gmtime, strftime, time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from generators import GB_GA_Generator
from utils import TPScoringFunction, calc_auc, ecfp, score

def timestamp():
    return strftime("%Y-%m-%d_%H:%M:%S", gmtime())

def fit_clfs(chid, n_estimators, n_jobs):
    """
    Args:
        chid: which assay to use:
        external_file:
    """
    # read data and calculate ecfp fingerprints
    assay_file = f'./assays/processed/{chid}.csv'
    print(f'Reading data from: {assay_file}')
    df = pd.read_csv(assay_file)
    X = np.array(ecfp(df.smiles))
    y = np.array(df.label)

    # split in equally sized sets. Stratify to get same label distributions
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, stratify=y)

    balance = (np.mean(y1), np.mean(y2))

    # train classifiers and store them in dictionary
    clfs = {}
    clfs['Split1'] = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=n_jobs)
    clfs['Split1'].fit(X1, y1)

    clfs['Split1_alt'] = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=n_jobs)
    clfs['Split1_alt'].fit(X1, y1)

    clfs['Split2'] = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=n_jobs)
    clfs['Split2'].fit(X2, y2)

    # calculate AUCs for the clfs
    aucs = {}
    aucs['Split1'] = calc_auc(clfs['Split1'], X2, y2)
    aucs['Split1_alt'] = calc_auc(clfs['Split1_alt'], X2, y2)
    aucs['Split2'] = calc_auc(clfs['Split2'], X1, y1)
    print("AUCs:")
    for k, v in aucs.items():
        print(f'{k}: {v}')

    return clfs, aucs, balance


def optimize(chid,
             n_estimators,
             n_jobs,
             external_file,
             n_external,
             seed,
             optimizer_args):

    config = locals()
    np.random.seed(seed)

    #set up logging
    results_dir = os.path.join('./results', 'graph_ga', chid, timestamp())
    os.makedirs(results_dir)

    config_file = os.path.join(results_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)



    clfs, aucs, balance = fit_clfs(chid, n_estimators, n_jobs)
    results = {}
    results['AUC'] = aucs
    results['balance'] = balance

    clf_file = os.path.join(results_dir, 'classifiers.p')
    with open(clf_file, 'wb') as f:
        pickle.dump(clfs, f)

    # Create guacamol scoring function with clf trained on split 1
    scoring_function = TPScoringFunction(clfs['Split1'])

    # run optimization
    t0 = time()
    optimizer = GB_GA_Generator(**optimizer_args)
    smiles_history = optimizer.generate_optimized_molecules(
        scoring_function, 100, get_history=True)

    t1 = time()
    opt_time = t1 - t0

    # make a list of dictionaries for every time step
    statistics = []
    for optimized_smiles in smiles_history:
        row = {}
        row['smiles'] = optimized_smiles
        row['preds'] = {}
        row['ratio_active'] = {}
        row['mean_pred'] = {}
        for k, clf in clfs.items():
            preds = score(optimized_smiles, clf)
            row['preds'][k] = preds
        statistics.append(row)

    results['statistics'] = statistics

    stat_time = time() - t1
    # add predictions on external set
    # load external smiles for evaluation
    with open(external_file) as f:
        external_smiles = f.read().split()
    external_smiles = np.random.choice(external_smiles, n_external)
    results['predictions_external'] = {k: score(external_smiles, clf) for k, clf in clfs.items()}

    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)

    print(f'Storing results in {results_dir}')
    print(f'Optimization time {opt_time:.2f}')
    print(f'Statistics time {stat_time:.2f}')


if __name__ == '__main__':
    config = dict(
        chid='CHEMBL3888429',
        n_estimators=100,
        n_jobs=8,
        external_file='./data/guacamol_v1_test.smiles.can',
        n_external=3000,
        seed=101,
        optimizer_args=dict(smi_file='./data/guacamol_v1_valid.smiles.can',
                            population_size=100,
                            offspring_size=200,
                            generations=5,
                            mutation_rate=0.01,
                            n_jobs=-1,
                            random_start=True,
                            patience=150,
                            canonicalize=False))

    optimize(**config)
