# Approximation-of-dissipative-systems-by-elastic-chains-numerical-evidence

This repository contains the code used in the paper *Approximation of dissipative systems by elastic chains: numerical evidence* by Bersani, Alberto Maria; Caressa, Paolo; Dell’Isola, Francesco. In: MATHEMATICS AND MECHANICS OF SOLIDS. - ISSN 1081-2865. - 28:2(2023), pp. 501-520. [10.1177/10812865221081851]

The code is contained in a single script [paper_code.py](paper_code.py).

To perform numerical computations we used the `scipy` Python package under the Python 3.8 compiler.
We used the simulated annealing optimization function which implements a generalized annealing
by using its default parameters both for annealing temperature, number of iterations etc.
(see Xiang, Y. and Sun, D.Y. and Fan, W. and Gong, X.G.
"Generalized simulated annealing algorithm and its application to the Thomson model",
Physics Letters A, 233:3 (1997) 216-220.

The code can be executed in a Python 3 environment which provides standard libraries `numpy`, `scipy` and `matplotlib`,
easily installed via the `pip` tool.

The script optimizes parameters of a coupled system to approximate a damped one:
results are either written on png files or plotted interactively;
further information is printed.

The functions defined in this script may be easily engineered to run several classes of simulations and collect results.
