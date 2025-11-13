# Ising model and graphons

This repository contains the Julia code for:
* Analyze bifurcations of the self-consistency equations with help of `BifurcationKit`
* Perform Monte-Carlo simulations

## Generation of bifurcation diagrams

To obtain datasets for bifurcation diagrams, one should run:
```
julia SW_graph_bifurcation_L_inf.jl N J p r beta_min beta_max
```
where:
* `N` is discretization parameter (we discretize graphon by the matrix of `N*N` sise, it is `Int64`)
* `J` is spin coupling in Ising model (`Float64`)
* `p` and `r` are parameters of small-world graphon (`Float64`)
* `beta_min` and `beta_max` are min and max inverse temperatures (`Float64`)

During the evaluation of the code, the line
```
Diagram was generated
```
should appear: it means that `BifurcationKit` have succeed and bifurcation diagram was generated without errors.

This diagram is parsed and finally the line
```
Data was extracted
```
appears, which means that `*.csv` datasets are created successfully. The structure of datasets is as follows:
* For each `i`-th branch, the `*.csv*` file with name `..._br_i_...csv` is created. This file contains `(beta, solution_norm)` array
* All bifurcation points are collected to file with name `BP_....csv`
* Explicit solutions that appear after crossing the bifurcation points are collected in files with names `_bp_...csv`
All these datasets can be proccessed in any library, which can read `*.csv`, like `pandas`

## Monte-Carlo simulations

* Details of simulations are described in `*.pdf` file in this repo
* All the simulations can be done with help of Jupyter notebook (`*.ipynb` file)
