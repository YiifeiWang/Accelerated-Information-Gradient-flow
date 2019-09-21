## Accelerated Information Gradient flow

We present the codes for the numerical experiments.

### Instructions

- For the toy example: Direct run `toy_example.m`. Figures will be saved under `./result/toy/`

- For the Gaussian examples: for the part in the ODE level, run `Gaussian_example_flow.m`; for the part in the particle levle, run `Gaussian_example.m`. Figures will be saved under `./result/Gauss/`

- For the Bayesian logistic regression, the dataset can be downloaded from https://github.com/DartML/Stein-Variational-Gradient-Descent. The dataset file `covertype.mat'`shall be placed under the folder `./data/` . 

  There are two ways to select the kernel bandwidth, BM and MED. Run `Bayesian_BM.m` and `Bayesian_MED.m` to output datafiles. The results will be stored under `./result/`.  Run `Bayesian_plot.m` to plot the figures. You can use our existing datafiles to plot the figures. Figures will be saved under `./result/covtype/`.