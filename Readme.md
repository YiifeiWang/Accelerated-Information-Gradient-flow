# Accelerated Information Gradient flow

We present the codes for the numerical experiments in the paper [Accelerated Information Gradient flow](https://arxiv.org/pdf/1909.02102).



##  Abstract

We present a framework for Nesterov's accelerated gradient flows in probability space. Here four examples of information metrics are considered, including Fisher-Rao metric, Wasserstein-2 metric, Kalman-Wasserstein metric and Stein metric. For both Fisher-Rao and Wasserstein-2 metrics, we prove convergence properties of accelerated gradient flows. In implementations, we propose a sampling-efficient discrete-time algorithm for Wasserstein-2, Kalman-Wasserstein and Stein accelerated gradient flows with a restart technique. We also formulate a kernel bandwidth selection method, which learns the gradient of logarithm of density from Brownian-motion samples. Numerical experiments, including Bayesian logistic regression and Bayesian neural network, show the strength of the proposed methods compared with state-of-the-art algorithms.



## Reproduction

### Toy example

Direct run `toy_example.m`. Figures will be saved under `./result/toy/`

### Toy example of selecting bandwidth

Direct run `toy_example_bandwidth.m`. Figures will be saved under `./result/toy_bandwidth/`

### Bayesian logistic regression

The Covertype dataset can be downloaded from https://github.com/DartML/Stein-Variational-Gradient-Descent. The dataset file `covertype.mat'`shall be placed under the folder `./data/` . 

There are two ways to select the kernel bandwidth, BM and MED. Run `Bayesian_BM.m` and `Bayesian_MED.m` to output datafiles. The results will be stored under `./result/`.  Run `Bayesian_plot.m` to plot the figures. You can use our existing datafiles to plot the figures. Figures will be saved under `./result/covtype/`.

### Bayesian neural network

 Detailed descriptions are placed under `./BNN/`. 



## Feedback

If you have any questions or comments, feel free to shoot me an [email](zackwang24@pku.edu.cn). 