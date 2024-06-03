# A Bayesian Approach to Online Planning
The code for the paper "A Bayesian Approach to Online Planning" to be published in ICML 2024.

#### TODO add link to paper

## Abstract
The combination of Monte Carlo tree search and neural networks has revolutionized online planning. 
As neural network approximations are often imperfect, we ask whether uncertainty estimates about the network outputs could be used to improve planning. 
We develop a Bayesian planning approach that facilitates such uncertainty quantification, inspired by classical ideas from the meta-reasoning literature. 
We propose a Thompson sampling based algorithm for searching the tree of possible actions, for which we prove the first (to our knowledge) finite time Bayesian regret bound, and propose an efficient implementation for a restricted family of posterior distributions. 
In addition we propose a variant of the Bayes-UCB method applied to trees. 
Empirically, we demonstrate that on the ProcGen Maze and Leaper environments, when the uncertainty estimates are accurate but the neural network output is inaccurate, our Bayesian approach searches the tree much more effectively. 
In addition, we investigate whether popular uncertainty estimation methods are accurate enough to yield significant gains in planning.

## Installation
The code was tested on Python 3.8. Later versions may also work.

Install `baselines` from source:
```bash
git clone https://github.com/openai/baselines && cd baselines
python setup.py install
```

Install requirements:
```bash
git clone https://github.com/nirgreshler/bayesian-online-planning.git
cd bayesian-online-planning
pip install -r requirements.txt
```


## Simulating Procgen Environment
We provide the code to reproduce the results presented in the paper, on both Maze and Leaper environments from the Procgen benchmark (https://openai.com/index/procgen-benchmark/).
To run planning on a single environment instance, run the `simulate_procgen.py` script with the following arguments:
- `--env`: the name of the environment (either `maze` or `leaper`)
- `--planner`: the name of the planning algorithm (either `nmcts`, `bts`, `tsts` or `tsts_det`)
- `--model-path` (optional): the path to the neural network model (under neural_network/models). If not provided a default model will be used.
- `seed` (optional): the seed for the environment (default: 0)
- `--time-steps` (optional): the number of time steps to run the simulation (default: 100 for maze and 25 for leaper)
- `--search-budget` (optional): the number of search iterations for the planning algorithm (default: 100)
- `--results-folder` (optional): a path to save images of the environment state after each time step.

### Example
`python simulate_procgen.py -e maze -p nmcts`

## Changing Configuration Parameters
Some parameters used by the planner can be modified in the dataclass `Config` in `config\config.py`.
The usage of each parameter is documented there.