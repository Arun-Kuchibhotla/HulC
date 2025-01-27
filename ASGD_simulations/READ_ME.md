# HulC using Averaged Stochastic Gradient Descent (ASGD)

In the code provided, we run simulations corresponding to the paper "HulC using Averaged Sotchastic Gradient Descent (ASGD)" (to be published on ArXiv). 

We compare confidence intervals for linear and logistic regression using the following four inference techniques:
- robust sandwich confidence interval (as a baseline);
- [Chen et al's](https://arxiv.org/abs/1610.08637) ASGD plug-in confidence interval;
- [HulC](https://arxiv.org/abs/2105.14577) using ASGD;
- t-statistic on batch of ASGD estimators.


---

# Simulation details


In a simulation study, we assess the utility of HulC by comparing confidence regions for $\theta_{\infty} \in \mathbb{R}^d$ on two simple cases: linear regression and logistic regression. In both cases, we generate $n$ iid samples $X_i \in \mathbb{R}^d$, $X_i \sim N(\mu, \Sigma)$, where $\mu^\top = [1,0, \dots,0]$ and $\Sigma$ is either the identity, Toeplitz, or Equicorrelation. 

Mimicking simulations by [Chen et al (2016)](https://arxiv.org/abs/1610.08637), we consider dimension sizes $d=5, 20, 100$ and the three types of covariance schemes. For linear regression, the noise parameter is $\varepsilon \sim N(0, 1)$, that is, $Y_i = \theta_{\infty}^\top X_i + \varepsilon_i$ for $i = 1, \dots, n$.  For logistic regression, $Y_i \sim^{iid} \text{Bernoulli}(p_i)$, where $p_i := \frac{1}{1+\exp\{-\theta_{\infty}^\top X_i\}}$. In both cases, the parameter $\theta_{\infty}$ consists of coordinates that are linearly spaced between 0 and 1. For example, if $d=5$, then $\theta_{\infty} = [0, 0.25, 0.5, 0.75, 1]^\top$.

We compare coverage of four inference techniques:
- robust sandwich confidence interval (as a baseline);
- [Chen et al's](https://arxiv.org/abs/1610.08637) ASGD plug-in confidence interval;
- HulC using ASGD;
- t-statistic on ASGD estimators.

We aim to achieve the theoretical 95% coverage rate as we vary the sample size $n = 10^3, 10^4, 5\cdot 10^4, 10^5$, the dimension size $d$, the type of covariance matrix, and the hyperparameter $c$ in the ASGD step size $\eta_t = ct^{-0.505}$. In each run of $200$ independent experiments, we first generate the data. Given the data, we fix $c$ from a grid of values and record the coverage and width ratios for each inference technique; specifically, we check whether the $k$-th coordinate of the parameter, $\theta_{\infty (k)}$, falls within the corresponding confidence interval $CI_{(k)}$, assigning a value of $1$ if $\theta_{\infty (k)} \in \text{CI}_{(k)}$ and $0$ otherwise. The estimated coverage is then calculated as the proportion of the $200$ experiments in which the parameter was covered, with a target of approximately 95% (equivalent to $190$ out of $200$ independent instances of coverage). 

---

# Setup

## Running for the first time

To make sure we have the same capabilities (e.g., same versions of packages), it's best to use an anaconda environment. If you
are not familiar with conda you can read more about it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).


1. Clone the [HulC repository](https://github.com/Arun-Kuchibhotla/HulC). For cloning instructions, see [Git Hub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). 

2. On your PC, open your command prompt (e.g., Git Bash).

3. Change your working directory to the path of the Hulc repo.
`
cd C:/Users/yourusername/Hulc/
`

4. In the command prompt, create a conda environment (make sure you have already [installed Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html): 
`
conda create -n Hulc 
`

5. Activate the environment (from command prompt)
`
conda activate Hulc 
`

6. From the command prompt, install the required packages using pip: 

`
pip install -r requirements.txt
`

5. Open your preferred editor. I prefer [Spyder](https://www.spyder-ide.org/).
`
spyder
`

Or for VS Code, run:
`
code
`

## Running for the second+ time

1. On your PC, open your command prompt (e.g., Git Bash).

2. Change your working directory to the path of the Hulc repo.
`
cd C:/Users/yourusername/Hulc/
`

3. Activate the conda environment (from command prompt)
`
conda activate Hulc 
`

4. Open your preferred editor. I prefer [Spyder](https://www.spyder-ide.org/).
`
spyder
`

Or for VS Code, run:
`
code
`

Note that you can also open Spyder or VS code directly and use point-and-click menu items to open the correct Anaconda environment.


# Description of files

- **run_sims** directory:
    - **regression_simulations.py**: This is the main file used to generate S=200 simulations. 
    - **gen_data.py**: Generates data for linear and logistic regression.
	- **ASGD_Chen_functions**: Contains gradient functions to perform linear and logistic regression via stochastic gradient descent, as well as a function to produce ASGD plug-in confidence intervals according to [Chen et al 2016](https://arxiv.org/abs/1610.08637)
	- **ASGD_HulC_manual.py**: Contains functions that produce confidence intervals for parameter $\boldsymbol{\theta}_{\infty} \in \mathbb{R}^d$ according to the four techniques being studied (sandwich estimator, ASGD plug-in, HulC, and t-stat).
    - **hulc_batches.py**: Contains helper functions for producing the [HulC confidence intervals](https://arxiv.org/abs/2105.14577)
    - **HulC_example_figures.py**: This file generates linear & logistic regression data in multiple dimensions and performs ASGD for example figures. The goal is to show that ASGD accuracy is sensitive to hyperparameter c.


# How to run the simulations.



1. Adjust the working directory in the preambles of **ASGD_HulC_manual.py** and **regression_simulations.py**.

2. Open **regression_simulations.py** and follow the instructions in the preamble.