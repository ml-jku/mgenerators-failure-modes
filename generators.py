# maybe replace this by explicit impots
from pathlib import Path
from typing import List, Optional

import joblib
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize, canonicalize_list
from joblib import delayed

import torch
from guacamol_baselines.graph_ga.goal_directed_generation import *
from guacamol_baselines.smiles_lstm_hc.rnn_generator import \
    SmilesRnnMoleculeGenerator
from guacamol_baselines.smiles_lstm_hc.rnn_utils import load_rnn_model


def mols2smiles(mols):
    return [Chem.MolToSmiles(m) for m in mols]


def load_smiles_from_file(self, smi_file):
    with open(smi_file) as f:
        if self.canonicalize:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)
        else:
            return f.read().split()


def top_k(self, smiles, scoring_function, k):
    joblist = (delayed(scoring_function.score)(s) for s in smiles)
    scores = self.pool(joblist)
    scored_smiles = list(zip(scores, smiles))
    scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
    return [smile for score, smile in scored_smiles][:k]


class GB_GA_Generator(GoalDirectedGenerator, Mixin):

    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, n_jobs=-1, random_start=False, patience=5, canonicalize=True):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience
        self.canonicalize = canonicalize
        self.load_smiles_from_file = load_smiles_from_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)

    # change top_k, these functions be mixed in ?

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self,
                                     scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None,
                                     get_history=False) -> List[str]:

        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(
                f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = np.random.choice(
                    self.all_smiles, self.population_size)
            else:
                starting_population = self.top_k(
                    self.all_smiles, scoring_function, self.population_size)

        # select initial population
        # this is also slow
        # population_smiles = heapq.nlargest(self.population_size, starting_population, key=scoring_function.score)
        starting_scores = scoring_function.score_list(starting_population)
        population_smiles = [x for _, x in sorted(zip(
            starting_scores, starting_population), key=lambda pair: pair[0], reverse=True)]

        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]

        # this is slow. Don't know exactly why. maybe pickling classifiers is not too nice
        # population_scores_old = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
        population_scores = scoring_function.score_list(
            mols2smiles(population_mol))

        # evolution: go go go!!
        t0 = time()

        patience = 0

        population_history = []
        population_history.append([Chem.MolToSmiles(m)
                                   for m in population_mol])

        for generation in range(self.generations):
            # new_population
            mating_pool = make_mating_pool(
                population_mol, population_scores, self.offspring_size)
            offspring_mol = self.pool(delayed(reproduce)(
                mating_pool, self.mutation_rate) for _ in range(self.population_size))

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            # population_scores = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)
            population_scores = scoring_function.score_list(
                [Chem.MolToSmiles(m) for m in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[
                :self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            res_time = time() - t0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec | '
                  f'{res_time:.2f} rest ')

            population_history.append([Chem.MolToSmiles(m)
                                       for m in population_mol])

        # finally
        if get_history:
            return population_history
        else:
            return [Chem.MolToSmiles(m) for m in population_mol][:number_molecules]


class SmilesRnnDirectedGenerator(GoalDirectedGenerator, Mixin):
    def __init__(self, pretrained_model_path: str, n_epochs=4, mols_to_sample=1028, keep_top=512,
                 optimize_n_epochs=2, max_len=100, optimize_batch_size=64, number_final_samples=1028,
                 sample_final_model_only=False, random_start=False, smi_file=None, n_jobs=-1, canonicalize=True) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_batch_size = optimize_batch_size
        self.optimize_n_epochs = optimize_n_epochs
        self.pretrain_n_epochs = 0
        self.max_len = max_len
        self.number_final_samples = number_final_samples
        self.sample_final_model_only = sample_final_model_only
        self.random_start = random_start
        self.smi_file = smi_file
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.canonicalize = canonicalize
        self.load_smiles_from_file = load_smiles_from_file

    def generate_optimized_molecules(self,
                                     scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None,
                                     get_history=False) -> List[str]:

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = []
            else:
                all_smiles = self.load_smiles_from_file(self.smi_file)
                starting_population = self.top_k(
                    all_smiles, scoring_function, self.mols_to_sample)

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix('.json')

        model = load_rnn_model(
            model_def, self.pretrained_model_path, device, copy_to_cpu=True)

        generator = SmilesRnnMoleculeGenerator(model=model,
                                               max_len=self.max_len,
                                               device=device)

        molecules = generator.optimise(objective=scoring_function,
                                       start_population=starting_population,
                                       n_epochs=self.n_epochs,
                                       mols_to_sample=self.mols_to_sample,
                                       keep_top=self.keep_top,
                                       optimize_batch_size=self.optimize_batch_size,
                                       optimize_n_epochs=self.optimize_n_epochs,
                                       pretrain_n_epochs=self.pretrain_n_epochs)

        # take the molecules seen during the hill-climbing, and also sample from the final model
        samples = [m.smiles for m in molecules]
        if self.sample_final_model_only:
            samples.clear()
        samples += generator.sample(max(number_molecules,
                                        self.number_final_samples))

        # calculate the scores and return the best ones
        samples = canonicalize_list(samples)
        scores = scoring_function.score_list(samples)

        scored_molecules = zip(samples, scores)
        sorted_scored_molecules = sorted(
            scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True)

        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]
