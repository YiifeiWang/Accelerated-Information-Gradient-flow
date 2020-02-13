# Bayesian neural network

We follows the implementation of Bayesian neural network in https://github.com/dilinwang820/matrix_svgd. 



## Requirements

python==2.7

tensorflow

numpy

pandas

xlrd>=1.0.0



## Datasets

Most of tested data can be obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). We elaborate on the placement of data as follows.

- __Boston__. This dataset can be downloaded from https://github.com/dilinwang820/Stein-Variational-Gradient-Descent. Place the file under the folder `./data/boston/`.

- __Combined__. This dataset can be downloaded from [this link](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant). Place all files under the folder `./data/combined/`.

- __Combined__. This dataset can be downloaded from [this link](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength). Place all files under the folder `./data/concrete/`.

- __Kin8nm__. This dataset can be downloaded from https://www.openml.org/d/189. Place the file under the folder `./data/kin8nm/`.

- __Wine__. This dataset can be downloaded from [this link](https://archive.ics.uci.edu/ml/datasets/wine+quality). Place all files under the folder `./data/wine/`.

- __Year__. This daraset can be downloaded from [this link](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD). Place the file under the folder `./data/year/`.




## Reproduction

To reproduce the results, simply run `bash run_all.sh` in command line. The results will be saved under `results`. 

To compute the averaged test RMSE and log-likelihood, please open `Evaluate.ipynb` using jupyter notebook. 