# Sports as Sequences

Senior comprehensive project in statistics, Carleton College

## Reproducing results

This requires both a `conda` and `R` installation.
One can forgo `conda` if they want to manually install python dependencies.

```bash
# clone git repp
git clone http://github.com/alipatti/stat-comps
cd stat-comps

# create conda virtual environment and install dependencies
conda env create -f environment.yaml

cd code
python nba.py # fetch and clean the NBA data
python training.py # train the models (this takes a while)
python shapley.py # compute the shapley values for our model
R -f plots_and_tables.r # make the plots and tables for our paper

cd ../paper
latexmk paper.tex
```
