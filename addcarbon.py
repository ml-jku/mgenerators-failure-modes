import argparse
import json
import os
from pathlib import Path

import numpy as np
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import \
    DistributionMatchingGenerator
from Levenshtein import distance
from rdkit import Chem

# save some sampled molecules
from utils import timestamp

# from guacamol_baselines.random_smiles_sampler.distribution_learning import \
    # RandomSmilesSampler


class AddCarbonSampler(DistributionMatchingGenerator):
    """"""

    def __init__(self, smi_file):
        self.smi_file = smi_file
        with open(smi_file) as f_smiles:
            self.train_list = np.array(f_smiles.read().split())

        self.train_set = set(self.train_list)
        self.train_size = self.train_list.shape[0]

    def generate(self, number_samples):
        generated_smiles = list()

        total = 0

        # loop until we have enough samples
        while len(generated_smiles) < number_samples:
            # get a molecule and canonicalize it
            idx = np.random.choice(self.train_size)
            mol = Chem.MolFromSmiles(self.train_list[idx])
            orig_can = Chem.MolToSmiles(mol)

            # set up variables for finding minimum later
            min_d = 1e9
            min_smiles = None

            # loop over random positions
            for i in np.random.permutation(len(orig_can)):
                # Insert C add random spot and check if valid
                mut = orig_can[:i] + 'C' + orig_can[i:]
                mut_mol = Chem.MolFromSmiles(mut)
                if mut_mol is None:
                    continue

                # If it is valid compute canonical smiles and compare it to the original smiles
                mut_can = Chem.MolToSmiles(mut_mol)
                d = distance(orig_can, mut_can)

                # find smiles with minimal levenshtein distance to original.
                if (d < min_d) and (mut_can not in self.train_set):
                    min_d = d
                    min_smiles = mut_can
                    if min_d == 1:
                        break

            # If valid smiles has been found append it to results
            if min_smiles is not None:
                generated_smiles.append(min_smiles)

            total += 1
            if total % 100 == 0:
                print(len(generated_smiles) / total, total, ' ' * 50, end='\r')

        return list(generated_smiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smi_train",
        type=str,
        default='./data/guacamol_v1_train.smiles',
        help="File containing a SMILES on each line.")
    parser.add_argument(
        "--results_base",
        type=str,
        default='./results/addcarbon',
        help="Directory where to put results")
    args = parser.parse_args()

    # setup results directory
    dir_results = Path(f'{args.results_base}/{timestamp()}')
    os.makedirs(dir_results)

    add_carbon = AddCarbonSampler(args.smi_train)

    sampled_smiles = add_carbon.generate(10000)
    fn_sampled = str(dir_results / 'addcarbon_smiles.txt')
    with open(fn_sampled, 'w') as f:
        f.write('\n'.join(sampled_smiles))

    fn_guacamol_results = str(dir_results / 'guacamol_results.json')
    assess_distribution_learning(
        add_carbon, args.smi_train, json_output_file=fn_guacamol_results)

    with open(fn_guacamol_results) as f:
        results = json.load(f)

    for b in results['results']:
        print(f"{b['benchmark_name']}: {b['score']}")

    print(f'Saved results in: {dir_results}')
