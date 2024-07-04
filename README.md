## Description
[![Conference](https://img.shields.io/badge/L4DC2024-Accepted-success)](https://l4dc.web.ox.ac.uk/home)


Code for the paper [Learning true objectives: Linear algebraic characterizations of identifiability in inverse reinforcement learning](https://proceedings.mlr.press/v242/shehab24a.html), published in [L4DC 2024](https://l4dc.web.ox.ac.uk/home). The extended version of the paper can be found [here](https://dx.doi.org/10.7302/23151).

## Usage
To clone the repository and install the requirements, run the following:
```Console
git clone https://github.com/mlshehab/learning_true_objectives.git
cd ./learning_true_objectives
pip install -r requirements.txt
```

The optional arguments for  `main.py` are given by:

```console
$ python main.py --help

usage: main.py [-h] [-d {fig1a,fig1b,fig1c}] [--with_features {dense,sparse,None}]

Code for L4DC 2024 Paper: Learning True Objectives

optional arguments:
  -h, --help            show this help message and exit
  -d {fig1a,fig1b,fig1c}
                        Specifying the dynamics
  --with_features {dense,sparse,None}
                        Specifying the features
```
The default for the argument `--with_features` is `None` and should not be specified if you don't want the feature based implemenation.

In order to reproduce the results of Figure 1 (a,b or c), run the command:

```console 
python main.py -d fig1a  # for results of figure 1 (a)
python main.py -d fig1b  # for results of figure 1 (b)
python main.py -d fig1c  # for results of figure 1 (c)
```

In order to reproduce the results of Figure 2 using **Dense Features**, run the command:

```console 
python main.py -d fig1c --with_features dense
```

In order to reproduce the results of Figure 2 using **Sparse Features**, run the command:

```console 
python main.py -d fig1c --with_features sparse
```


## Authors

[Mohamad Louai Shehab](https://scholar.google.com/citations?user=DJvw8dUAAAAJ&hl=en&oi=ao) 

[Antoine Aspeel](https://aaspeel.github.io/) 

[Nikos Arechiga](https://scholar.google.com/citations?user=NFWFaGAAAAAJ&hl=en)

[Andrew Best](https://scholar.google.com/citations?user=eIQlFbcAAAAJ&hl=en)

[Necmiye Ozay](https://web.eecs.umich.edu/~necmiye/)
