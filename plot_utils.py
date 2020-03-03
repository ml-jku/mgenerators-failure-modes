import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def npflatten(dicts):
    "Takes a list of dictionaries with same keys and joins them up into numpy array. Similar to pandas but also works with higher dim arrays"
    flat = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            flat[k].append(v)
    return {k: np.array(v) for k, v in flat.items()}


def load_chid(chid_dir, order):
    runs = [x for x in os.listdir(chid_dir) if x != 'summary']

    accumulate = []
    for run in runs:
        run_dir = chid_dir/run
        with open(run_dir/'results.json', 'r') as f:
            results = json.load(f)

        # some runs aborted earlier. Hack this away by adding the last entry a few times!
        n_gen = len(results['statistics'])
        if n_gen != 151:
            results['statistics'] += [results['statistics'][-1]] * \
                (151 - n_gen)

        # create a dictionary containing arrays of shape [n_iter, n_molecules]
        preds_internal = npflatten([row['preds']
                                    for row in results['statistics']])
        # array for each clf and split
        preds_external = {k: np.array(
            v) for k, v in results['predictions_external'].items()}

        accumulate.append((preds_internal, preds_external, results['AUC']))

    preds_internal, preds_external, aucs = [
        npflatten(x) for x in list(zip(*accumulate))]
    for d in [preds_internal, preds_external, aucs]:
        if 'all' in d:
            del d['all']
    preds_internal, preds_external, aucs = [
        {k: d[k] for k in order} for d in [preds_internal, preds_external, aucs]]
    return preds_internal, preds_external, aucs


def median_score_single(pred, color=None, label=None, **kwargs):
    n_runs = pred.shape[0]

    medians = np.median(pred, 2)
    q25 = np.percentile(pred, 25, axis=2)
    q75 = np.percentile(pred, 75, axis=2)

    for i in range(n_runs):
        plt.plot(medians[i], c=color, label=label, **kwargs)
        label = None  # avoid multiple legend entries
        plt.fill_between(
            np.arange(medians[i].shape[0]), q25[i], q75[i], alpha=.1, color=color)


def ratio_active_single(pred, color=None, label=None, **kwargs):
    n_runs = pred.shape[0]
    means = np.mean(pred > 0.5, 2)
    stds = np.ones_like(means) * 0.02
    for i in range(n_runs):
        plt.plot(means[i], c=color, label=label, **kwargs)
        label = None
        plt.fill_between(np.arange(
            means[i].shape[0]), means[i]-stds[i], means[i]+stds[i], alpha=.1, color=color)


def median_score_compound(pred, color=None, label=None, **kwargs):
    # pred  has shape [n_runs, n_epochs, n_mol]
    # reshape to [n_epochs, n_runs, n_mol]
    pred = pred.transpose(1, 0, 2)

    # should we plot mean and standard deviations of means or all values?
    #pred = pred.reshape(pred.shape[1], -1)
    pred = pred.mean(2)
    median = np.median(pred, 1)
    q25 = np.percentile(pred, 25, axis=1)
    q75 = np.percentile(pred, 75, axis=1)

    plt.plot(median, c=color, label=label, **kwargs)
    plt.fill_between(
        np.arange(median.shape[0]), q25, q75, alpha=.1, color=color)


def ratio_active_compound(pred, color=None, label=None, **kwargs):
    """Computes ratio active for each run. Then plots median and quartiles"""
    pred = pred.transpose(1, 0, 2)
    pred = pred > 0.5

    # should we plot mean and standard deviations of means or all values?
    #pred = pred.reshape(pred.shape[1], -1)

    pred = pred.mean(2)
    median = np.median(pred, 1)
    q25 = np.percentile(pred, 25, axis=1)
    q75 = np.percentile(pred, 75, axis=1)

    plt.plot(median, c=color, label=label, **kwargs)
    plt.fill_between(
        np.arange(median.shape[0]), q25, q75, alpha=.1, color=color)


def plot_wrapper(preds_internal, primitive, name, xlabel, ylabel, col_dict, skip=False, ax=None, legend=True, leg_lw=3, **kwargs):
    if ax is not None:
        plt.sca(ax)

    for k, pred in preds_internal.items():
        if skip and k == 'Split1_alt':
            continue
        primitive(pred, color=col_dict[k], label=legend_dict[k], **kwargs)

    if legend:
        leg = plt.legend(loc='upper center', bbox_to_anchor=(
            0.5, 1.14), ncol=3, frameon=False)
        for line in leg.get_lines():
            line.set_linewidth(leg_lw)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
