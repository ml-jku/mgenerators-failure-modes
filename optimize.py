import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from time import gmtime, strftime, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from guacamol_baselines.graph_ga.goal_directed_generation import \
    GB_GA_Generator
from utils import TPScoringFunction, calc_auc, ecfp, score


def optimize(chid,
             n_estimators,
             n_jobs,
             external_file,
             n_external,
             seed,
             optimizer_args):

    config = locals()
    np.random.seed(seed)

    # create dictionary to store results in
    results = {}

    # read data and calculate ecfp fingerprints
    assay_file = f'./assays/processed/{chid}.csv'
    print(f'Reading data from: {assay_file}')
    df = pd.read_csv(assay_file)
    X = np.array(ecfp(df.smiles))
    y = np.array(df.label)

    # load external smiles for evaluation
    with open(external_file) as f:
        external_smiles = f.read().split()
    external_smiles = np.random.choice(external_smiles, n_external)

    # split in equally sized sets. Stratify to get same label distributions
    X1, X2, y1, y2 = train_test_split(
        X, y, test_size=0.5, stratify=y)
    results['balance'] = (np.mean(y1), np.mean(y2))

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

    clfs['all'] = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=n_jobs)
    clfs['all'].fit(X, y)

    # calculate AUCs for the clfs
    results['AUC'] = {}
    results['AUC']['Split1'] = calc_auc(clfs['Split1'], X2, y2)
    results['AUC']['Split1_alt'] = calc_auc(clfs['Split1_alt'], X2, y2)
    results['AUC']['Split2'] = calc_auc(clfs['Split2'], X1, y1)
    print("AUCs:")
    for k, v in results['AUC'].items():
        print(f'{k}: {v}')

    # Create guacamol scoring function with clf trained on split 1
    scoring_function = TPScoringFunction(clfs['Split1'])

    # run optimization
    opt_time = time()
    optimizer = GB_GA_Generator(**optimizer_args)
    smiles_history = optimizer.generate_optimized_molecules(
        scoring_function, 100, get_history=True)

    opt_time = time() - opt_time
    # make a list of dictionaries for every time step

    stat_time = time()

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
            row['ratio_active'][k] = (np.array(preds) > 0.5).mean()
            row['mean_pred'][k] = np.array(preds).mean()
        statistics.append(row)

    stat_time = time() - stat_time
    results['statistics'] = statistics

    # add predictions on external set
    results['predictions_external'] = {
        k: score(external_smiles, clf) for k, clf in clfs.items()}

    def timestamp():
        return strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    results_dir = os.path.join('./results', 'graph_ga', chid, timestamp())
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    print(f'Optimization time {opt_time:.2f}')
    print(f'Statistics time {stat_time:.2f}')
    print(f'Storing results in {results_dir}')
    results_file = os.path.join(results_dir, 'results.json')

    with open(results_file, 'w') as f:
        json.dump(results, f)

    config_file = os.path.join(results_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)


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
