# On failure modes of molecule generators and optimizers
Philipp Renz <sup>a</sup>,
Dries Van Rompaey  <sup>b</sup>,
Jörg Kurt Wegner  <sup>b</sup>,
Sepp Hochreiter  <sup>a</sup>,
Günter Klambauer  <sup>a</sup>,

<sup>a</sup> LIT AI Lab & Institute for Machine Learning, Johannes Kepler University Linz,Altenberger Strasse 69, A-4040 Linz, Austria
<sup>b</sup> High Dimensional Biology and Discovery Data Sciences, Janssen Research & Development, Janssen Pharmaceutica N.V., Turnhoutseweg 30, Beerse B-2340, Belgium

The paper can be found here:
https://chemrxiv.org/articles/On_Failure_Modes_of_Molecule_Generators_and_Optimizers/12213542

Feel free to send questions to renz@ml.jku.at.

## Code
Steps to reproduce the paper:

### Download Guacamol data splits
The compounds are used for distribution learning and for starting populations for the graph-based genetic algorithm.
```
mkdir data
wget -O data/guacamol_v1_all.smiles https://ndownloader.figshare.com/files/13612745
wget -O data/guacamol_v1_test.smiles https://ndownloader.figshare.com/files/13612757
wget -O data/guacamol_v1_valid.smiles https://ndownloader.figshare.com/files/13612766
wget -O data/guacamol_v1_train.smiles https://ndownloader.figshare.com/files/13612760
```
### Bioactivity data
The csv-files downloaded from ChEMBL are located in `assays/raw`.
Running the `preprocess.py` script will transform the data into binary classification tasks and store them in `assays/processed`.

## Experiments
For the distribution-learning experiment (AddCarbon model) is suffices to run `addcarbon.py`

For the goal-directed generation benchmarks more steps have to be taken.
1. `preprocess.py`: Preprocess the data to obtain binary classification tasks.
1. `run_goal_directed.py`: This runs all the molecular optimization experiments.
1. `predictions.py`: This fits  a classifier multiple times with different random seeds, mainly to estimate the optimization/control score combinations of split 1 actives. The results are used to get the contours in the scatter plots (Fig. 2, S1)
1. `plots.ipynb`: Notebook to create most of the plots in the paper
1. `nearest_neighbours.ipynb`: Notebook to calculate nearest neighbour distances and to create Fig. S4 (histograms over Tanimoto similarities)

## Special thanks
Special thanks goes out to the authors of [Guacamol](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) ([Github](https://github.com/BenevolentAI/guacamol)). Their code was very helpful in implementing our experiments.
