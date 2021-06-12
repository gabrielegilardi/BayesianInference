# Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC) Sampling and the Metropolis-Hastings Algorithm

There are two (similar) versions, one in Python and one in C++:

- [Readme](./Code_Python/README.md) for the Python version.

- [Readme](./Code_Cpp/README.md) for the C++ version.

## References

- Wikipedia, "[Metropolis-Hastings Algorithm](https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm)".

- Wikipedia, "[Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)".

- "[Bayesian Statistics](https://en.wikipedia.org/wiki/Bayesian_statistics)", Chapter 2 in "[Advanced Algorithmic Trading](https://www.quantstart.com/advanced-algorithmic-trading-ebook/)", by M. Halls-Moore.

- Probabilistic programming in Python using [pymc3](https://docs.pymc.io/).

## Notes

- The Python version has one example more than the C++ version (*Coin_upd*).

- The differences in the results between the two versions are mostly due to the randomly generated initial data (array `data`). To compare solutions, simply generate and save the data using one of the versions and then load them in the other version.

- The remaining (very) small differences are due to the randomness implicit in the Metropolis-Hastings algorithm.
