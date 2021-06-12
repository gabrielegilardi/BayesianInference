"""
Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC)
Sampling and the Metropolis-Hastings Algorithm.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- Code has been written and tested in Python 3.8.5.
- Likelihood (pdf) can be defined as an arbitrary function with any number
  of independent parameters.
- Prior functions are defined using a list of list, and can be any pdf from
  function "prior_dist" in file "Metropolis.py" (other priors can be easily
  added).
- Jumps in the Metropolis-Hastings algorithm are proposed using a normal
  distribution of the parameters.
- Function "random_number" in file "Metropolis.py" can be used to generate
  random numbers from any arbitrary pdf.
- Results can be verified using the pymc3 library.
- Usage: python test.py <example>.

Main Parameters
---------------
example = Random, Coin, Normal, Coin_upd
    Name of the example to run.
likelihood
    Name of the likelihood function.
par
    Array with the parameters of the likelihood function.
n_data >=1
    Number of data to be sampled from the likelihood function.
data
    Array with the data sampled from the likelihood function.
a0, b0
    Support interval for the likelihood function.
priors
    List with the priors. Each prior is assigned to one of the likelihood
    parameter following the same order as in <par>.
samples > 0
    Number of jumps to perform in the Metropolis-Hastings algorithm.
par_init
    Initial value for the parameters in the Metropolis-Hastings algorithm.
width_prop > 0
    Standard deviation of the normal distribution used to search the neighboroud
    of a parameter. A good value should give about 50% of accepted jumps.
i0 >= 0
    Index specifying the burn-in/warm-up amount.
posterior
    Array containing the jumps (accepted or rejected) of all parameters.
jumps
    Number of jumps actually accepted.

Examples
--------
There are four examples: Random, Coin, Normal, and Coin_upd (see the code for
parameters and results).

- Random: generation of random numbers from a generic pdf.

- Coin: one parameter (theta), Bernoulli distribution as likelihood, beta
        distribution as prior, admit an analytical solution.

- Normal: two parameters (mean and standard deviation), normal distribution
          as likelihood, normal distribution as prior for the mean, gamma
          distribution as prior for the standard deviation, solution also
          checked with pymc3.

- Coin_upd: one parameter (theta), Bernoulli distribution as likelihood,
            uniform distribution as initial prior, previous posterior as
            successive prior.

References
----------
- Metropolis-Hastings algorithm @
  https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
- Markov chain Monte Carlo @
  https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
- Halls-Moore, 2016, "Bayesian statistics", chapter II in "Advanced Algorithmic
  Trading", @ https://www.quantstart.com/advanced-algorithmic-trading-ebook/
- Probabilistic programming in Python using pymc3 @ https://docs.pymc.io/
"""
if __name__ == '__main__':

    import sys
    import warnings

    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    from Metropolis import metropolis, random_number, prior_dist

    # To avoid the warning about log(0) when one of the probability is zero
    warnings.filterwarnings("ignore")

    # Read example to run
    if len(sys.argv) != 2:
        print("Usage: python test.py <example>")
        sys.exit(1)
    example = sys.argv[1]

    # Seed the random generator
    np.random.seed(123)

    # Generation of random numbers given the pdf
    if (example == 'Random'):

        def pdf_random(x, par):
            """
            Piece-wise pdf.
            """
            pdf = np.where((x > 0.0) * (x <= 1.0), 0.3 * x, 0.0)
            pdf += np.where((x > 1.0) * (x <= 2.0), -0.2 * x + 0.5, 0.0)
            pdf += np.where((x > 2.0) * (x <= 3.0), 0.1, 0.0)
            pdf += np.where((x > 3.0) * (x <= 4.0), 0.1 * x - 0.2, 0.0)
            pdf += np.where((x > 4.0) * (x <= 5.0), 0.2, 0.0)
            pdf += np.where((x > 5.0) * (x <= 7.0), -0.1 * x + 0.7, 0.0)

            return pdf

        # Parameters
        par = []
        a0, b0 = -1.0, +8.0

        # Randomly approximated pdf
        n_data = 50000
        data = random_number(pdf_random, par, a0, b0, size=n_data, n=1000)

        # Real pdf
        xx = np.linspace(a0, b0, 1000)
        yy = pdf_random(xx, par)

        # Plot
        plt.plot(xx, yy, label='Real')
        plt.hist(data, 100, histtype="step", density=True, label='Random')
        plt.xlabel('x')
        plt.xticks(np.arange(-1, 9, step=1))
        plt.xlim(-1, 8)
        plt.ylabel('pdf')
        plt.yticks(np.arange(0, 0.4, step=0.05))
        plt.ylim(0, 0.35)
        plt.grid(b=True)
        plt.legend()
        plt.show()

    # Coin flip example:
    # - one parameter (the probability tail comes up)
    # - Bernoulli distribution as likelihood
    # - beta distribution as prior
    # - admit an analytical solution
    elif (example == 'Coin'):

        def pdf_coin(x, par):
            """
            Bernoulli distribution.
            """
            theta = par[0]
            pdf = np.where(x, theta, 1.0-theta)

            return pdf

        # Generate data (0 = tail, 1 = head)
        likelihood = pdf_coin
        n_data = 50                 # Number of coin flip
        par = [0.63]                # Probability tail comes up
        data = (np.random.uniform(0, 1, size=n_data) > par[0])

        # Priors (coefficients are just arbitrary values used as example)
        alpha, beta = 4.0, 12.0
        priors = []
        priors.append(['beta', alpha, beta])

        # Solve
        samples = 10000
        par_init = [0.5]
        width_prop = 0.1
        i0 = int(np.floor(0.2 * samples))       # Burn-in period
        posterior, jumps = metropolis(data, likelihood, priors, samples=samples,
                                      par_init=par_init, width_prop=width_prop)

        # Results:
        # - accepted jumps = 52.9%
        # - <theta> mean and std = 0.289, 0.057
        print("\nAccepted jumps = {0:.1f}%".format(100 * jumps / samples))
        print("<theta> mean and std = {0:.3f}, {1:.3f}"
              .format(posterior[i0:, 0].mean(), posterior[i0:, 0].std()))

        # Analytical solution
        # - Bernoulli likelihood with beta prior results in a beta posterior
        # - posterior (alpha, beta) are converted to equivalent (mu, sigma)
        #   for comparison purpose
        # - Results: <theta> mean and std (anal.) = 0.288, 0.055
        # - Ref.: https://en.wikipedia.org/wiki/Conjugate_prior
        n_heads = data.sum()
        alpha_post = alpha + n_heads
        beta_post = beta + n_data - n_heads
        a = alpha_post + beta_post
        mu_post = alpha_post / a
        sigma_post = np.sqrt(alpha_post * beta_post / (a + 1.0)) / a
        print("\n<theta> mean and std (anal.) = {0:.3f}, {1:.3f}"
              .format(mu_post, sigma_post))

        # Plot posteriors (numerical, analytical, and histogram)
        plt.subplot(121)
        sns.kdeplot(posterior[i0:, 0], label='num.', c='b')
        plt.hist(posterior[i0:, 0], 30, histtype="step", density=True,
                 color='r', label='hist.')
        xx = np.linspace(0.0, 0.6, 1000)
        yy = stats.beta(alpha_post, beta_post).pdf(xx)
        plt.plot(xx, yy, label='anal.', c='g')
        plt.xlabel('x')
        plt.xticks(np.linspace(0.0, 0.6, num=7))
        plt.xlim(0.0, 0.6)
        plt.ylabel('$\Theta$')
        plt.yticks(np.linspace(0, 8, num=9))
        plt.ylim(0, 8)
        plt.grid(b=True)
        plt.legend()

        # Plot all <theta> values (both accepted and rejected)
        plt.subplot(122)
        plt.plot(posterior[:, 0], c='b')
        plt.xlabel('sample')
        plt.xticks(np.linspace(0, samples, num=5))
        plt.xlim(0, samples)
        plt.ylabel('$\Theta$')
        plt.yticks(np.linspace(0.1, 0.5, num=5))
        plt.ylim(0.1, 0.5)
        plt.axhline(posterior[i0:, 0].mean(), color='r')
        plt.grid(b=True)

        plt.show()

    # Example with normally distributed likelihood:
    # - two parameters (mean and standard deviation)
    # - Normal distribution as likelihood
    # - Normal distribution as prior for the mean <mu >and gamma distribution
    #   as prior for the standard deviation <sigma>
    # - solution checked with pymc3
    elif (example == 'Normal'):

        def pdf_normal(x, par):
            """
            Normal distribution.
            """
            mu = par[0]
            sigma = par[1]
            y = (x - mu) / sigma
            pdf = np.exp(-y * y / 2.0) / (sigma * np.sqrt(2.0 * np.pi))
            return pdf

        # Generate data
        likelihood = pdf_normal
        n_data = 20
        par = [-1.3, 1.0]               # Mean and standard deviation
        a0, b0 = -10.0, +10.0           # Support
        data = random_number(likelihood, par, a0, b0, size=n_data)

        # Priors (coefficients are just arbitrary values used as example)
        priors = []
        priors.append(['norm', 2.0, 1.0])               # Mean
        priors.append(['gamma', 6.0, 1.0])              # Standard deviation

        # Solve
        samples = 20000
        par_init = [2.0, 5.0]
        width_prop = 0.20
        i0 = int(np.floor(0.2 * samples))       # Burn-in period
        posterior, jumps = metropolis(data, likelihood, priors, samples=samples,
                                      par_init=par_init, width_prop=width_prop)

        # Results:
        # - accepted jumps = 52.8%
        # - <mu> mean and std = -1.182, 0.230
        # - <sigma> mean and std = 0.968, 0.198
        print("\nAccepted jumps = {0:.1f}%".format(100 * jumps / samples))
        print("<mu> mean and std = {0:.3f}, {1:.3f}"
              .format(posterior[i0:, 0].mean(), posterior[i0:, 0].std()))
        print("<sigma> mean and std = {0:.3f}, {1:.3f}"
              .format(posterior[i0:, 1].mean(), posterior[i0:, 1].std()))

        # Solve using pymc3 (set to false if pymc3 not installed)
        use_pymc3 = True
        if (use_pymc3):

            print("\n===== Solving using pymc3 =====")

            import pymc3 as pm

            with pm.Model():

                # Priors
                mu = pm.Normal('mu', 2.0, 1.0)
                sigma = pm.Gamma('sigma', 6.0, 1.0)

                # Best starting point
                start = pm.find_MAP()

                # Likelihood
                returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data)

                # Algorithm
                step = pm.Metropolis()

                # Solve
                trace = pm.sample(samples, step, return_inferencedata=False)

            # Results:
            # - start point =  {'mu': array(2.0), 'sigma': array(5.0)}
            # - <mu> mean and std = -1.188, 0.224
            # - <sigma> mean and std = 0.972, 0.202
            print("Start point = ", start)
            print("<mu> mean and std = {0:.3f}, {1:.3f}"
                  .format(trace[i0:]['mu'].mean(), trace[i0:]['mu'].std()))
            print("<sigma> mean and std = {0:.3f}, {1:.3f}"
                  .format(trace[i0:]['sigma'].mean(), trace[i0:]['sigma'].std()))

        # Plot <mu> posteriors (numerical, analytical, and histogram)
        plt.subplot(221)
        sns.kdeplot(posterior[i0:, 0], label='num.', c='b')
        plt.hist(posterior[i0:, 0], 50, histtype="step", density=True,
                 color='r', label='hist.')
        if (use_pymc3):
            sns.kdeplot(trace[i0:]['mu'], label='pymc3', c='g')
        plt.xlabel('x')
        plt.xticks(np.linspace(-2.50, 0, num=6))
        plt.xlim(-2.50, 0)
        plt.ylabel('$\mu$')
        plt.yticks(np.linspace(0, 2.5, num=6))
        plt.ylim(0, 2.5)
        plt.grid(b=True)
        plt.legend()

        # Plot <sigma> posteriors (numerical, analytical, and histogram)
        plt.subplot(222)
        sns.kdeplot(posterior[i0:, 1], label='num.', c='b')
        plt.hist(posterior[i0:, 1], 50, histtype="step", density=True,
                 color='r', label='hist.')
        if (use_pymc3):
            sns.kdeplot(trace[i0:]['sigma'], label='pymc3', c='g')
        plt.xlabel('x')
        plt.xticks(np.linspace(0.0, 2, num=5))
        plt.xlim(0.0, 2)
        plt.ylabel('$\sigma$')
        plt.yticks(np.linspace(0, 2.5, num=6))
        plt.ylim(0, 2.5)
        plt.grid(b=True)
        plt.legend()

        # Plot all <mu> values (both accepted and rejected)
        plt.subplot(223)
        plt.plot(posterior[:, 0], c='b')
        plt.xlabel('sample')
        plt.xticks(np.linspace(0, samples, num=5))
        plt.xlim(0, samples)
        plt.ylabel('$\mu$')
        plt.yticks(np.linspace(-2, 0, num=5))
        plt.ylim(-2, 0)
        plt.axhline(posterior[i0:, 0].mean(), color='r')
        plt.grid(b=True)

        # Plot all <sigma> values (both accepted and rejected)
        plt.subplot(224)
        plt.plot(posterior[:, 1], c='b')
        plt.xlabel('sample')
        plt.xticks(np.linspace(0, samples, num=5))
        plt.xlim(0, samples)
        plt.ylabel('$\sigma$')
        plt.yticks(np.linspace(0.0, 2.50, num=6))
        plt.ylim(0.0, 2.50)
        plt.axhline(posterior[i0:, 1].mean(), color='r')
        plt.grid(b=True)

        plt.show()

    # Coin flip example with updates:
    # - one parameter (the probability tail comes up)
    # - Bernoulli distribution as likelihood
    # - uniform distribution as initial prior
    # - previous posterior as successive prior
    #
    # The mean of the posterior should tend to the probability head comes up,
    # i.e. (1-par[0]), while its standard deviation should become smaller and
    # smaller.
    elif (example == 'Coin_upd'):

        def pdf_coin_upd(x, par):
            """
            Bernoulli distribution.
            """
            theta = par[0]
            pdf = np.where(x, theta, 1.0-theta)

            return  pdf

        # Parameters
        likelihood = pdf_coin_upd
        n_data = 50                     # Number of coin flip
        par = [0.63]                    # Probability tail comes up
        samples = 5000
        i0 = int(np.floor(0.2 * samples))       # Burn-in period
        mu = 0.5
        width_prop = 0.1

        # Initial prior is uniform
        XX = np.linspace(0.0, 1.0, 100)
        YY = np.ones(len(XX))
        plt.plot(XX, YY, label='init')

        # Perform n_steps
        tot_heads = 0
        tot_data = 0
        n_steps = 15
        for step in range(n_steps):

            # Generate data (0 = tail, 1 = head)
            data = (np.random.uniform(0, 1, size=n_data) > par[0])
            tot_heads += data.sum()
            tot_data += len(data)

            # Prior is expressed as generic array
            priors = []
            priors.append(['generic', XX, YY])

            # Solve using the previous mean as initial value
            par_init = [mu]
            posterior, jumps = metropolis(data, likelihood, priors,
                                          samples=samples, par_init=par_init,
                                          width_prop=width_prop)
            mu = posterior[i0:, 0].mean()

            # Print results for each step
            print("\nStep", step+1)
            print("- head freq. = {0:.1f}%".format(100 * tot_heads / tot_data))
            print("- accepted jumps = {0:.1f}%".format(100 * jumps / samples))
            print("- <theta> mean and std = {0:.3f}, {1:.3f}"
                  .format(mu, posterior[i0:, 0].std()))

            # Use the posterior as new prior (adding 10% tails on the side)
            x_min, x_max = np.min(posterior[:, 0]), np.max(posterior[:, 0])
            xx = np.linspace(x_min, x_max, 100)
            yy = stats.gaussian_kde(posterior[:, 0])(xx)
            d = x_max - x_min
            XX = np.concatenate([[xx[0] - 0.1 * d], xx, [xx[-1] + 0.1 * d]])
            YY = np.concatenate([[0], yy, [0]])
            if (((step+1) % 3) == 0):
                plt.plot(XX, YY, label='step ' + str(step+1))

        # Plot interpolated posteriors
        plt.xlabel('x')
        plt.xticks(np.linspace(0.2, 0.5, num=7))
        plt.xlim(0.2, 0.5)
        plt.ylabel('$\Theta$')
        plt.yticks(np.linspace(0, 24, num=7))
        plt.ylim(0, 24)
        plt.axvline(1.0 - par[0], c='k', ls='--')
        plt.grid(b=True)
        plt.legend()
        plt.show()

    else:
        print("\n", example)
        print("--> Example not found.\n")
        sys.exit(1)
