# 1D MPNP

This repository contains the implementation of **GMPNP** in 1D for a CO2 electrocatalysis system using the FEniCS finite element package in Python. 
The parameter and mesh files can be found in the utilities folder. 
The script for the Stern layer solves the Poisson equation in the Stern region using the results obtained from the GMPNP model. 
These scripts were used to derive the results published in [https://pubs.rsc.org/en/content/articlelanding/2019/ee/c9ee02485a#!divAbstract](url).

The environment.yml file can be used with conda to create the python environment needed to run the script using the command:

`conda env create -f environment.yml`

The environment name is the first line in the yaml file and is currently set as **FEniCS**.

Once the environment is created, activate it using:

`conda activate FEniCS`

To see the packages installed in the new environment, use:

`conda env list`

See [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#](url) for further help regarding managing enviroments using conda. 

In order to run the MPNP_CO2ER_EDL.py script, you will first need to modify the ***basepath*** and ***basepath_utilities*** variables according to the location of the python file and the utilities folder on your local machine. 

Activate the FEniCS virtual environment and run the script from the directory using:

`python MPNP_CO2ER_EDL.py`

As you will see in the script, you can vary all relevant parameters using command line arguements like in the example below. This makes it simpler to submit multiple jobs on a computational cluster.

`python MPNP_CO2ER_EDL.py --voltage_multiplier=-10.0 --cation='Cs'`

And you are good to go!