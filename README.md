## Description

Code for the paper "Learning True Objectives"

## Usage
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
python main.py -d fig1a 
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

Mohamad Louai Shehab  
[Antoine Aspeel](https://aaspeel.github.io/) 
[Necmiye Ozay](https://web.eecs.umich.edu/~necmiye/)
