# Additional SGD Methods

In the code provided, we run simulations corresponding to the paper  ["Statistical Inference for Online Algorithms" (Carter and Kuchibhotla)](https://arxiv.org/abs/2105.14577). 

We compare confidence intervals for linear and logistic regression using the following four inference techniques:
- Wald-based robust sandwich confidence interval (as a baseline)
- [HulC](https://arxiv.org/abs/2105.14577) (based on one of the particular online algorithms listed below)
- $t$-statistic (also based on same algorithm used by HulC)
 
We run the following algorithms to estimate the parameter $\theta_{\infty} \in \mathbb{R}^d$:
- Average-iterate-implicit-SGD
- Last-iterate-implicit-SGD
- ROOT-SGD
- Truncated-SGD
- Noisy-truncated-SGD

For Averaged-SGD (ASGD), see summary [here](https://github.com/Arun-Kuchibhotla/HulC/tree/main/Online_Algorithms_HulC_simulations/ASGD_sims)


---

# Simulation details


In a simulation study, we assess the utility of HulC by comparing confidence regions for $\theta_{\infty} \in \mathbb{R}^d$ on two simple cases: linear regression and logistic regression. In both cases, we generate $n$ iid samples $X_i \in \mathbb{R}^d$, $X_i \sim N(\mu, \Sigma)$, where $\mu^\top = [1,0, \dots,0]$ and $\Sigma$ is either the identity, Toeplitz ($\Sigma_{ij}=0.5^{|i-j|}$), or Equicorrelation ($\Sigma_{ii}=1$, $\Sigma_{ij}=0.2$ if $i\ne j$, ). 



Mimicking simulations by [Chen et al (2016)](https://arxiv.org/abs/1610.08637), we consider dimension sizes $d=5, 20, 100$ and the three types of covariance schemes. For linear regression, the noise parameter is $\varepsilon \sim N(0, 1)$, that is, $Y_i = \theta_{\infty}^\top X_i + \varepsilon_i$ for $i = 1, \dots, n$.  For logistic regression, $Y_i \sim^{iid} \text{Bernoulli}(p_i)$, where $p_i := \frac{1}{1+\exp\{-\theta_{\infty}^\top X_i\}}$. In both cases, the parameter $\theta_{\infty}$ consists of coordinates that are linearly spaced between 0 and 1. For example, if $d=5$, then $\theta_{\infty} = [0, 0.25, 0.5, 0.75, 1]^\top$.

We run the following algorithms to estimate the parameter $\theta_{\infty} \in \mathbb{R}^d$:
- Averaged-SGD (ASGD) - see summary [here](https://github.com/Arun-Kuchibhotla/HulC/tree/main/Online_Algorithms_HulC_simulations/ASGD_sims)
- Average-iterate-implicit-SGD
- Last-iterate-implicit-SGD
- ROOT-SGD
- Truncated-SGD
- Noisy-truncated-SGD


We compare confidence intervals for linear and logistic regression using the following four inference techniques:
- Wald-based robust sandwich confidence interval (as a baseline)
- [HulC](https://arxiv.org/abs/2105.14577) (based on one of the particular online algorithms listed below)
- t-statistic (also based on same algorithm used by HulC)

We aim to achieve the theoretical 95% coverage rate as we vary the sample size $n = 10^3, 10^4, 5\cdot 10^4, 10^5$, the dimension size $d$, the type of covariance matrix, and the hyperparameter $c$ in the SGD step size $\eta_t = ct^{-0.505}$. In each run of $200$ independent experiments, we first generate the data. Given the data, we fix $c$ from a grid of values and record the coverage and width ratios for each inference technique; specifically, we check whether the $k$-th coordinate of the parameter, $\theta_{\infty (k)}$, falls within the corresponding confidence interval $CI_{(k)}$, assigning a value of $1$ if $\theta_{\infty (k)} \in \text{CI}_{(k)}$ and $0$ otherwise. The estimated coverage is then calculated as the proportion of the $200$ experiments in which the parameter was covered, with a target of approximately 95% (equivalent to $190$ out of $200$ independent instances of coverage). 

**The results of these simulations are available in a [dynamic Tableau graph](https://public.tableau.com/app/profile/selina.carter6629/viz/OnlineinferencesimulationsOLSandlogisticregression/Coverageandwidthratio_paper).**

The following plots display coverage and width ratios as a function of step size hyperparameter $c$ for linear and logistic regression (Toeplitz covariance) and dimension sizes $d=5$ and $d=20$.


### Average-iterate-implicit-SGD

<table>
  <tr>
    <th>Linear regression (diminsion = 5)</th>
    <th>Linear regression (diminsion = 20)0</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D5_Toeplitz_cov_wr_AISGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D20_Toeplitz_cov_wr_AISGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <th>Logistic regression (dimension = 5)</th>
    <th>Logistic regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D5_Toeplitz_cov_wr_AISGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D20_Toeplitz_cov_wr_AISGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

---

### Last-iterate-implicit-SGD

<table>
  <tr>
    <th>Linear regression (dimension = 5)</th>
    <th>Linear regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D5_Toeplitz_cov_wr_ISGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D20_Toeplitz_cov_wr_ISGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <th>Logistic regression (dimension = 5)</th>
    <th>Logistic regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D5_Toeplitz_cov_wr_ISGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D20_Toeplitz_cov_wr_ISGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

---

### ROOT-SGD

<table>
  <tr>
    <th>Linear regression (dimension = 5)</th>
    <th>Linear regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D5_Toeplitz_cov_wr_rootSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D20_Toeplitz_cov_wr_rootSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <th>Logistic regression (dimension = 5)</th>
    <th>Logistic regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D5_Toeplitz_cov_wr_rootSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D20_Toeplitz_cov_wr_rootSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

---

### Truncated-SGD

<table>
  <tr>
    <th>Linear regression (dimension = 5)</th>
    <th>Linear regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D5_Toeplitz_cov_wr_truncatedSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D20_Toeplitz_cov_wr_truncatedSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <th>Logistic regression (dimension = 5)</th>
    <th>Logistic regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D5_Toeplitz_cov_wr_truncatedSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D20_Toeplitz_cov_wr_truncatedSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>


### Noisy-truncated-SGD

<table>
  <tr>
    <th>Linear regression (dimension = 5)</th>
    <th>Linear regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D5_Toeplitz_cov_wr_noisytruncatedSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/linear_D20_Toeplitz_cov_wr_noisytruncatedSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <th>Logistic regression (dimension = 5)</th>
    <th>Logistic regression (dimension = 20)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D5_Toeplitz_cov_wr_noisytruncatedSGD_initTRUE.png" width="100%"></td>
    <td><img src="https://github.com/Arun-Kuchibhotla/HulC/blob/main/Online_Algorithms_HulC_simulations/additional_simulations/assets/logistic_D20_Toeplitz_cov_wr_noisytruncatedSGD_initTRUE.png" width="100%"></td>
  </tr>
</table>


## Main observations:

- There is no single value of step size hyperparameter $c$ that works uniformly for all algorithms.
- Especially when the dimension size is large, most algorithms are quite sensitive to $c$.
- The HulC and the $t$-stat methods generally produce correct coverage for appropriately chosen $c$.
- HulC width ratios are only slightly larger than the t-stat width ratios across all settings -- around $10\%$ wider on average across all settings for which coverage is achieved.
- As the sample size increases, the width ratios generally decrease, with the exception of last-iterate-implicit-SGD.
- For linear regression, the most reliable algorithm in terms of hyperparameter $c$ is average-iterate-implicit-SGD. (In this case, across all values of $c \in [0.0005, 2]$, both the HulC and t-stat methods produce correct coverage and width ratios are less than $1.8$).
- For logistic regression, achieving correct coverage is typically more challenging, but sometimes width ratios are smaller than for linear regression. (For example, using root SGD and truncated SGD, logistic regression produces smaller width ratios than linear regression).



---


# Setup

## Running the simulations for the first time.

Unlike the [ASGD simulations](https://github.com/Arun-Kuchibhotla/HulC/tree/main/Online_Algorithms_HulC_simulations/ASGD_sims), which are in python, these simulations are in R because we rely on the `sgd` package for most of the SGD functions (to our knowledge, average-iterate-implicit-SGD and last-iterate-implicit-SGD packages are not avaialable in python).

You don't need an Anaconda environment because the packages to install are contained in each R file.


1. Clone the [HulC repository](https://github.com/Arun-Kuchibhotla/HulC). For cloning instructions, see [Git Hub Docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). 

2. On your PC, open the folder `HulC/additional_simulations/run_sims`.

3. In R Studio, open additional_simsloop.R, and follow the instructions in the file.


## Running for the second+ time

Follow steps 2-3 above.


# Description of files

- **run_sims** directory:
    - **additional_sims.R**: This is the main file used to generate S=200 simulations for one particular setting.
    - **additional_sims_loop.R**: This is the main file used to generate S=200 simulations for a loop over several settings.
    - **gen_data.R**: Generates data for linear and logistic regression.
	- **HulC_sgd_functions.R**: Contains functions that produce confidence intervals for parameter $\boldsymbol{\theta}_{\infty} \in \mathbb{R}^d$ the HulC algorithm.


# How to run the simulations.



1. Adjust the working directory in the preambles of **ASGD_HulC_manual.py** and **regression_simulations.py**.

2. Open **regression_simulations.py** and follow the instructions in the preamble.


