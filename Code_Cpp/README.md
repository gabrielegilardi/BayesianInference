# Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC) Sampling and the Metropolis-Hastings Algorithm

## Features

- The code has been written in plain vanilla C++ and tested using g++ 8.1.0 (MinGW-W64).

- Likelihood (pdf) can be defined as an arbitrary function with any number of independent parameters.

- Prior functions are defined using an array of structures, and can be any pdf from function "prior_dist" in file "Metropolis.cpp" (other priors can be easily added).

- Jumps in the Metropolis-Hastings algorithm are proposed using a normal   distribution of the parameters.

- Function *random_number* in file *Metropolis.cpp* can be used to generate random numbers from any arbitrary pdf.

- Results are passed using a structure.

- Usage: *test.exe example*.

## Main Parameters

`example` Name of the example to run.

`likelihood` Name of the likelihood function.

`par` Array with the parameters of the likelihood function.

`n_data` Number of data to be sampled from the likelihood function.

`data` Array with the data sampled from the likelihood function.

`a0`, `b0` Support interval for the likelihood function.

`priors` Array of structures with the priors. Each prior is assigned to one of the likelihood parameter following the same order as in `par`.

`samples` Number of jumps to perform in the Metropolis-Hastings algorithm.

`par_init` Initial value for the parameters in the Metropolis-Hastings  algorithm.

`width_prop` Standard deviation of the normal distribution used to search the neighboroud of a parameter. A good value should give about 50% of accepted jumps.

`i0` Index specifying the burn-in/warm-up amount.

`posterior` Array containing the jumps (accepted or rejected) of all parameters.

`jumps` Number of jumps actually accepted.

`res` Structure with the results returned by the Metropolis-Hastings algorithm.

`save_res` Name of the file where to save the results. If empty, results are not saved.

## Examples

There are three examples: **Random**, **Coin**, and **Normal** (see *test.cpp* for the specific parameters and results). A brief description is given below. Plots similar to the ones in the Python version (see the [Readme](../Code_Python/README.md)) can be generated using the results saved in the file specified by `save_res`.


**Random:**

Generation of random numbers from a generic pdf.

**Coin:**

One parameter (theta), Bernoulli distribution as likelihood, beta distribution as prior, admit an analytical solution.

**Normal:**

Two parameters (mean and standard deviation), normal distribution as likelihood, normal distribution as prior for the mean, gamma distribution as prior for the standard deviation.

## References

- Wikipedia, "[Metropolis-Hastings Algorithm](https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm)".

- Wikipedia, "[Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)".

- "[Bayesian Statistics](https://en.wikipedia.org/wiki/Bayesian_statistics)", Chapter 2 in "[Advanced Algorithmic Trading](https://www.quantstart.com/advanced-algorithmic-trading-ebook/)", by M. Halls-Moore.
