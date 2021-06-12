"""
Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC)
Sampling and the Metropolis-Hastings Algorithm.

Copyright (c) 2021 Gabriele Gilardi
"""

import sys
import numpy as np
from scipy import stats
import scipy.special as sp


def integ(func, par, a, b, n=50):
    """
    Returns the integral of function <func> in the interval [a, b] using
    Simpson's rule with <n> steps.

    Ref.: https://en.wikipedia.org/wiki/Newton-Cotes_formulas
    """
    # Step size
    x = np.linspace(a, b, 2*n+1)
    h = (x[1] - x[0])

    # Evaluate the function simultaneously in all points
    y = func(x, par)

    # Add contributions using Simpson's rule
    f = (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum() + y[-1]) * h / 3

    return f


def random_number(pdf, par, a0, b0, size=1, maxIt=1000, n=50, tolF=1.e-7,
                  tolX=1.e-7):
    """
    Returns random numbers from the generic probability distribution <pdf>,
    defined by parameters <par> and with support in [a0, b0].

    Procedure:
    - generate a random number <Y> in [0-1] using a uniform distribution.
    - assume the random number is the cdf(X) of the <pdf>.
    - use the bisection method to find <X> solving the equation Y-cdf(X)=0.
    - return X as random number for the probability distribution <pdf>.

    Refs.: https://en.wikipedia.org/wiki/Inverse_transform_sampling
           https://en.wikipedia.org/wiki/Bisection_method

    Notes:
    - <a0> and <b0> define the support, i.e. cdf(a0) = 0 and cdf(b0) = 1.
    - it stops if there is not root in the interval (likely due to an incorrect
      definition of the support).
    - returns the best value found (and print a warning) if the maximum number
      of iterations is reached.
    - a root is found if either the tolerance condition on the function or the
      tolerance condition on the interval is satisfied.
    - <m> is actually used instead of 1 when generating random numbers from the
      uniform distribution. This avoids the problems associated with the finite
      step-size used in the numerical integration. For instance, m = 0.99999 for
      the example with function "pdf_random" and n = 100. The higher <n> and the
      closer <m> will be to 1.
    """

    # Generate random numbers in [0, m]
    m = integ(pdf, par, a0, b0, n)
    Y = np.random.uniform(low=0, high=m, size=size)

    X_pdf = np.zeros(size)
    for j in range(size):

        # Initial interval
        a, b = a0, b0

        # Check if <a0> is the solution
        fa = Y[j]
        if (np.abs(fa) < tolF):
            X_pdf[j] = a
            continue

        # Check if <b0> is the solution
        fb = Y[j] - m
        if (np.abs(fb) < tolF):
            X_pdf[j] = b
            continue

        # Check if f(a) and f(b) have same signs (this condition should never
        # be verified if the support is correctly defined)
        if (fa * fb > 0.0):
            print("\nnr = ", j, ", cdf = ", Y[j])
            print("--> Interval does not contain a zero.\n")
            sys.exit(1)

        # Search for the zero using the bisection method
        i = 0
        while (i < maxIt):

            # Function value in the middle-point
            c = (a + b) / 2.0
            fc = Y[j] - integ(pdf, par, a0, c, n)

            # Check if one of the tolerances is satisfied
            if ((np.abs(fc) < tolF) or ((b-a)/2 < tolX)):
                X_pdf[j] = c
                break

            # Reduce the interval and iterate again
            else:
                i += 1
                if (fa * fc > 0.0):
                    a = c
                else:
                    b = c

        # Reached the max. number of iteration --> return best value found
        else:
            print("\nnr = ", j, ", cdf = ", Y[j])
            print("--> Reached max. number of iterations.\n")
            X_pdf[j] = c

    return X_pdf


def prior_dist(prior, x):
    """
    Returns the pdf(x) for the specified prior.
    
    Notes:
    - <prior> is a list with structure [name, par1, par2, ...].
    - <x> is a scalar.
    - in some of the priors the log has been used to improve numerical
      stability during the computation.
    """
    name = prior[0]
    pdf = 0.0

    # Uniform distribution, [a, b]
    if (name == 'unif'):
        a = prior[1]
        b = prior[2]
        if ((x >= a) and (x <= b)):
            pdf = 1.0 / (b - a)

    # Normal distribution, [-inf, +inf]
    elif (name == 'norm'):
        mu = prior[1]
        sigma = prior[2]
        d = (x - mu) / sigma
        pdf = np.exp(-d * d / 2.0) / (sigma * np.sqrt(2.0 * np.pi))

    # Beta distribution, [0, 1]
    elif (name == 'beta'):
        alpha = prior[1]
        beta = prior[2]
        if ((x >= 0.0) and (x <= 1.0)):
            logpdf = (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x) \
                     + sp.gammaln(alpha + beta) - sp.gammaln(alpha) \
                     - sp.gammaln(beta)
            pdf = np.exp(logpdf)

    # Gamma distribution, [0, +inf]
    elif (name == 'gamma'):
        alpha = prior[1]
        beta = prior[2]
        if (x >= 0.0):
            logpdf = alpha * np.log(beta) - sp.gammaln(alpha) \
                     + (alpha - 1.0) * np.log(x) - beta * x
            pdf = np.exp(logpdf)

    # Exponential distribution, [0, +inf]
    elif (name == 'expon'):
        if (x >= 0.0):
            lam = prior[1]
            pdf = lam * np.exp(-lam * x)

    # Pareto distribution, [1, +inf]
    elif (name == 'pareto'):
        xm = prior[1]
        alpha = prior[2]
        if (x >= 1.0):
            logpdf = alpha + alpha * np.log(xm) - (alpha + 1.0) * np.log(x)
            pdf = np.exp(logpdf)

    # Generic distribution defined by arrays X and Y, [min(X), max(X)]
    elif (name == 'generic'):
        X = prior[1]
        Y = prior[2]
        Xmin, Xmax = np.min(X), np.max(X)
        if ((x > Xmin) and (x < Xmax)):
            i = np.searchsorted(X, x)
            # pdf(x) is determined as linear interpolation between the two
            # closest values
            pdf = Y[i-1] + (x - X[i-1]) * (Y[i] - Y[i-1]) / (X[i] - X[i-1])

    # Default option
    else:
        print("\n", name)
        print("--> Prior distribution not recognized.\n")
        sys.exit(1)

    return pdf


def metropolis(data, likelihood, priors, samples=1000, par_init=None,
               width_prop=.5):
    """
    Returns the posterior function of the parameters given the likelihood and
    the prior functions. Returns also the number of the accepted jumps in the
    Metropolis-Hastings algorithm.

    Notes:
    - <width_prop> should be chosen so to result in about 50% accepted jumps.
    - <posterior> has shape (samples, n_par).
    - priors must be from function "prior_dist".
    - for numerical stability the computation is carried out using logarithms.
    """
    # Current parameters
    n_par = len(priors)
    par_curr = np.zeros(n_par) if (par_init is None) else np.asarray(par_init)

    # Init quantities
    jumps = 0
    par_prop = np.zeros(n_par)
    posterior = np.zeros((samples, n_par))
    posterior[0, :] = par_curr

    # Current priors
    bb = 0.0
    for i in range(n_par):
        bb += np.log(prior_dist(priors[i], par_curr[i]))
    prior_curr = np.exp(bb)

    # Current likelihood
    bb = np.log(likelihood(data, par_curr)).sum()
    likelihood_curr = np.exp(bb)

    # Current posterior probability
    p_curr = likelihood_curr * prior_curr

    # Loop <samples> times
    for sample in range(samples):

        # Randomnly pick the proposed parameters
        for i in range(n_par):
            par_prop[i] = stats.norm(par_curr[i], width_prop).rvs()

        # Evaluate priors with the proposed parameters
        bb = 0.0
        for i in range(n_par):
            bb += np.log(prior_dist(priors[i], par_prop[i]))
        prior_prop = np.exp(bb)

        # Evaluate likelihood with the proposed parameters
        bb = np.log(likelihood(data, par_prop)).sum()
        likelihood_prop = np.exp(bb)

        # Proposed posterior probability
        p_prop = likelihood_prop * prior_prop

        # Randomly accept or reject the jump
        p_accept = p_prop / p_curr
        if ((np.random.uniform() < p_accept)):

            # Update quantities if jump accepted
            jumps += 1
            par_curr = par_prop.copy()
            prior_curr = prior_prop
            likelihood_curr = likelihood_prop
            p_curr = p_prop

        # Save (accepted and rejected) parameters
        posterior[sample, :] = par_curr

    return posterior, jumps
