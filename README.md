# HulC: Hull based confidence regions
<p align="center">
  <img src="Hulk.png?raw=true" alt="HulC"/>
</p>

We provide an introduction to the general purpose method of construction confidence intervals called HulC (Hull based confidence regions). HulC is developed by [Kuchibhotla, Balakrishnan and Wasserman (2021)](https://arxiv.org/abs/2105.14577). On this page, we provide both R and python implementations of the HulC method as well as reproducible code to obtain the plots and tables in the paper. This page also contains more illustrations of the method beyond those given in the paper. 

The "R" folder contains the R code for all the HulC procedures (in HulC.R) and the jupyter notebooks for illustrations using the functions in HulC.R. The python code to the same is in the "python" folder.
### Related papers:
[1] The "monotonic_regression_code" contains all the R code related to [Mallick, Sarkar, Kuchibhotla (2023)](https://arxiv.org/abs/2310.20058), where new asymptotic limit theory for the LSE monotonic regression estimator has been developed, with an improved understanding of its rate of convergence, adaptivity properties, and pointwise asymptotic distribution. HulC is shown to provide better inference in this new asymptotic regime compared to prior methods.
