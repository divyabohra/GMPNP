# 1D MPNP

This repository contains the implementation of **GMPNP** in 1D for a CO2 electrocatalysis system using the FEniCS finite element package in Python. 
The parameter and mesh files can be found in the utilities folder. 
The script for the Stern layer solves the Poisson equation in the Stern region using the results obtained from the GMPNP model. 
These scripts were used to derive the results published in [https://pubs.rsc.org/en/content/articlelanding/2019/ee/c9ee02485a#!divAbstract](url).

The environment.yml file can be used with conda to create the python environment needed to run the script. See [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#](url) for help regarding managing enviroments using conda. 