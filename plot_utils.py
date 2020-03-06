import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def npflatten(dicts):
    "Takes a list of dictionaries with same keys and joins them up into numpy array. Similar to pandas but also works with higher dim arrays"
    flat = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            flat[k].append(v)
    # try:
    #     ret = {k: np.array(v) for k, v in flat.items()}
    # except:
    #     from itertools import chain
    #     hidu = flat
    #     for k, v in flat.items():
    #         print(len(v))
    #         print([len(x) for x in v]) #aoeu
    #         print(k, all(len(x)==336 for x in chain(*v)))
    #     raise
    return {k: np.array(v) for k, v in flat.items()}

def flatten(dicts):
    "Takes a list of dictionaries with same keys and joins them up into numpy array. Similar to pandas but also works with higher dim arrays"
    flat = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            flat[k].append(v)

    return flat

def load_chid(chid_dir, order, hack=True):
    runs = [run for run in os.listdir(chid_dir) if (run != 'summary') and os.path.isfile(chid_dir/run/'results.json')]
    accumulate = []
    for run in runs:
        run_dir = chid_dir/run
        with open(run_dir/'results.json', 'r') as f:
            results = json.load(f)

        # some runs aborted earlier. Hack this away by adding the last entry a few times!
        n_gen = len(results['statistics'])

        # TODO: solve this in a better way. It feels so wrong ;)
        if hack:
            if n_gen != 151:
                results['statistics'] += [results['statistics'][-1]] * \
                    (151 - n_gen)

        # create a dictionary containing arrays of shape [n_iter, n_molecules]
        preds_internal = flatten([row['preds'] for row in results['statistics']])
        # array for each clf and split
        preds_external = results['predictions_external']

        accumulate.append((preds_internal, preds_external, results['AUC']))

    # preds_internal, preds_external, aucs = [flatten(x) for x in list(zip(*accumulate))]
    preds_internal, preds_external, aucs = list(zip(*accumulate))
    preds_internal = flatten(preds_internal)
    preds_external = flatten(preds_external)
    aucs = flatten(aucs)
    # legacy compatibility
    for d in [preds_internal, preds_external, aucs]:
        if 'all' in d:
            del d['all']
    preds_internal, preds_external, aucs = [{k: d[k] for k in order} for d in [preds_internal, preds_external, aucs]]
    return preds_internal, preds_external, aucs


def median_score_single(pred, color=None, label=None, **kwargs):
    medians = np.array([[np.median(y) for y in x] for x in pred])
    q25 = np.array([[np.percentile(y, 25) for y in x] for x in pred])
    q75 = np.array([[np.percentile(y, 75) for y in x] for x in pred])

    n_runs = medians.shape[0]
    for i in range(n_runs):
        plt.plot(medians[i], c=color, label=label, **kwargs)
        label = None  # avoid multiple legend entries
        plt.fill_between(np.arange(medians[i].shape[0]), q25[i], q75[i], alpha=.1, color=color)


def ratio_active_single(pred, color=None, label=None, **kwargs):
    means = np.array([[np.mean(np.array(y) > 0.5) for y in x] for x in pred])

    n_runs = means.shape[0]

    stds = np.ones_like(means) * 0.02
    for i in range(n_runs):
        plt.plot(means[i], c=color, label=label, **kwargs)
        label = None
        plt.fill_between(np.arange(
            means[i].shape[0]), means[i]-stds[i], means[i]+stds[i], alpha=.1, color=color)


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





def ratio_active_compound(pred, color=None, label=None, **kwargs):
    """Computes ratio active for each run. Then plots median and quartiles"""

    pred_mean = np.array([[(np.array(y) > 0.5).mean() for y in x] for x in pred]).T
    median = np.median(pred_mean, 1)
    q25 = np.percentile(pred_mean, 25, axis=1)
    q75 = np.percentile(pred_mean, 75, axis=1)

    plt.plot(median, c=color, label=label, **kwargs)
    plt.fill_between(
        np.arange(median.shape[0]), q25, q75, alpha=.1, color=color)


def plot_wrapper(preds_internal, primitive, name, xlabel, ylabel, col_dict, legend_dict, skip=False, ax=None, legend=True, leg_lw=3, **kwargs):
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


