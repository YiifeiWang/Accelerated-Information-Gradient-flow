# Accelerated Information Gradient flow

We present the codes for the numerical experiments in the paper [Accelerated Information Gradient flow](https://arxiv.org/pdf/1909.02102).



##  Abstract

We present a systematic framework for the Nesterov's accelerated gradient flows in the spaces of probabilities embedded with information metrics. Here two metrics are considered, including both the Fisher-Rao metric and the Wasserstein-$2$ metric. For the Wasserstein-$2$ metric case, we prove the convergence properties of the accelerated gradient flows, and introduce their formulations in Gaussian families. Furthermore, we propose a practical discrete-time algorithm in particle implementations with an adaptive restart technique.  We formulate a novel bandwidth selection method, which learns the Wasserstein-$2$ gradient direction from Brownian-motion samples. Experimental results including Bayesian inference show the strength of the current method compared with the state-of-the-art.



## Reproduction

### Toy example

Direct run `toy_example.m`. Figures will be saved under `./result/toy/`

### Gaussian examples

- For the part in the ODE level, run `Gaussian_example_flow.m`

- For the part in the particle levle, run `Gaussian_example.m`. Figures will be saved under `./result/Gauss/`

### Bayesian logistic regression

The Covertype dataset can be downloaded from https://github.com/DartML/Stein-Variational-Gradient-Descent. The dataset file `covertype.mat'`shall be placed under the folder `./data/` . 

There are two ways to select the kernel bandwidth, BM and MED. Run `Bayesian_BM.m` and `Bayesian_MED.m` to output datafiles. The results will be stored under `./result/`.  Run `Bayesian_plot.m` to plot the figures. You can use our existing datafiles to plot the figures. Figures will be saved under `./result/covtype/`.

### Bayesian neural network

 Detailed descriptions are placed under `./BNN/`. 



## Feedback

If you have any questions or comments, feel free to shoot me an [email](zackwang24@pku.edu.cn). 