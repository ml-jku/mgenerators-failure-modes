import json
import os
from collections import defaultdict
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def flatten(dicts):
    "Takes a list of dictionaries with same keys and joins them up into lists."
    flat = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            flat[k].append(v)

    return flat


def get_splits(run_dir):
    df1 = pd.read_csv(run_dir / 'split1.csv', index_col=False)
    df1['Split'] = 0
    df2 = pd.read_csv(run_dir / 'split2.csv', index_col=False)
    df2['Split'] = 1
    df = pd.concat([df1, df2])
    df.index = np.arange(len(df))
    return df


@lru_cache(maxsize=64)
def load_chid(chid_dir, order, **kwargs):
    runs = [run for run in os.listdir(chid_dir) if os.path.isfile(chid_dir / run / 'results.json')]
    accumulate = []
    for run in runs:
        run_dir = chid_dir / run
        split_info = get_splits(run_dir)
        with open(run_dir / 'results.json', 'r') as f:
            results = json.load(f)

        # some runs aborted earlier. Solve this by adding the last entry a few times!
        # TODO: solve this in a better way.
        n_gen = len(results['statistics'])

        if n_gen != 151:
            results['statistics'] += [results['statistics'][-1]] * \
                (151 - n_gen)

        # create a dictionary containing arrays of shape [n_iter, n_molecules]
        preds_internal = flatten([row['preds']
                                  for row in results['statistics']])
        smiles = [row['smiles'] for row in results['statistics']]
        # array for each clf and split
        preds_external = results['predictions_external']
        accumulate.append((preds_internal, preds_external,
                           results['AUC'], smiles, split_info))

    # preds_internal, preds_external, aucs = [flatten(x) for x in list(zip(*accumulate))]
    preds_internal, preds_external, aucs, smiles, split_info = list(
        zip(*accumulate))
    preds_internal = flatten(preds_internal)
    preds_external = flatten(preds_external)
    aucs = flatten(aucs)
    # legacy compatibility
    for d in [preds_internal, preds_external, aucs]:
        if 'all' in d:
            del d['all']
    preds_internal, preds_external, aucs = [
        {k: d[k] for k in order} for d in [preds_internal, preds_external, aucs]]
    return preds_internal, preds_external, aucs, smiles, split_info


def median_score_single(pred, color=None, label=None, **kwargs):
    medians = np.array([[np.median(y) for y in x] for x in pred])
    q25 = np.array([[np.percentile(y, 25) for y in x] for x in pred])
    q75 = np.array([[np.percentile(y, 75) for y in x] for x in pred])

    n_runs = medians.shape[0]
    for i in range(n_runs):
        plt.plot(medians[i], c=color, label=label, **kwargs)
        label = None  # avoid multiple legend entries
        plt.fill_between(
            np.arange(medians[i].shape[0]), q25[i], q75[i], alpha=.1, color=color)


def median_score_compound(pred, color=None, label=None, shade=False, **kwargs):
    # pred is triply nested list [n_runs, n_epochs, n_mol(variable_size) ]

    # get mean scores over runs
    pred_mean = np.array([[np.mean(y) for y in x] for x in pred]).T

    median = np.median(pred_mean, 1)
    q25 = np.percentile(pred_mean, 25, axis=1)
    q75 = np.percentile(pred_mean, 75, axis=1)

    plt.plot(median, c=color, label=label, **kwargs)

    plt.fill_between(
        np.arange(median.shape[0]), q25, q75, alpha=.1, color=color)


def plot_wrapper(preds_internal, primitive, name, col_dict, legend_dict, ls_dict, skip=[], xlabel=None, ylabel=None, ax=None, legend=True, leg_lw=3, **kwargs):
    if ax is not None:
        plt.sca(ax)
    for k, pred in preds_internal.items():
        if k in skip:
            continue
        primitive(
            pred, color=col_dict[k], label=legend_dict[k], ls=ls_dict[k], **kwargs)

    if legend:
        leg = plt.legend(loc='upper center', bbox_to_anchor=(
            0.5, 1.14), ncol=3, frameon=False)
        for line in leg.get_lines():
            line.set_linewidth(leg_lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# RF gives predictions that are on the grid of np.linspace(0,1,n_trees+1). Shake this up a little
def jitter(a, scale=1):
    a = np.array(a)
    return a + np.random.normal(loc=0, scale=scale, size=a.shape)


def countour(p11s, p21s, y1s, ax, levels=3, scatter=False):
    idx = np.array(y1s, bool)
    x = p11s[idx]
    y = p21s[idx]
    xx, yy = np.mgrid[0:1:100j, 0:1:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.contour(xx, yy, f, colors='black', levels=levels, alpha=0.5)
    if scatter:
        ax.scatter(x, y, s=1)
