# IVC2425

# CONDA
## Create conda environment
conda create -n IVC2425Env310 -python=3.10

or

conda create -p PATH python=3.10


## Export conda environment
conda env export > environment.yml

## Import conda environment
conda env create -n Project_Environment_Name --file environment.yml

# PIP

## Import pip environment
pip install -r requirements.txt

## Export pip environment
pip freeze > requirements.txt

## Unofficial Windows Binaries for Python Extension Packages
<https://www.lfd.uci.edu/~gohlke/pythonlibs/>