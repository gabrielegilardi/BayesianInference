/*
Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC)
Sampling and the Metropolis-Hastings Algorithm.

Copyright (c) 2021 Gabriele Gilardi
*/

#include <random>

using namespace std;

/* Structure used to define the priors */
struct Prior_type {
    string name;                    // Name
    double par[5];                  // Up to 5 parameters
};

/* Structure used to return the results */
struct Results {
    double **posterior;             // Shape (samples, n_par)
    int jumps;
};


/* Generates random numbers using a uniform or a normal distribution */
double rnd(string type, double par1=0.0, double par2=1.0)
{
    int seed;
    double x;

    // The first call should be to seed the generator with the value in <par1>
    seed = int(floor(par1));
    static mt19937 generator(seed);

    // Seed the generator and return the seed value
    if (type == "seed") {
        x = double(seed);
    }

    // Uniform real distribution between <par1> and <par2> (excluded)
    else if (type == "unif") {
        uniform_real_distribution<double> unif(par1, par2);
        x = unif(generator);
    }

    // Normal distribution with mean <par1> and std <par2>
    else if (type == "norm") {
        normal_distribution<double> norm(par1, par2);
        x = norm(generator);
    }

    // Default case
    else {
        printf("\n%s", type.c_str());
        printf("\n--> Random distribution not recognized.\n");
        exit(EXIT_FAILURE);
    }

    return x;
}


/*
Returns the integral of function <func> in the interval [a, b] using
Simpson's rule with <n> steps.

Ref.: https://en.wikipedia.org/wiki/Newton-Cotes_formulas
*/
double integ(double (*func)(double, double[]), double par[], double a,
             double b, int n=50)
{
    int nh = 2 * n, idx = 0;
    double h, x, f = 0.0;
    double *y;

    // Step size
    h = (b - a) / double(nh);

    // Evaluate the function in all points
    y = new double [nh+1];
    for (int i=0; i<nh+1; i++) {
        x = a + h * double(i);
        y[i] = func(x, par);
    }

    // Add contributions using Simpson's rule
    for (int i=0; i<n; i++) {
        f += (y[idx] + 4.0 * y[idx+1] + y[idx+2]);
        idx += 2;
    }
    f = f * h / 3.0;

    return f;
}


/*
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
*/
double random_number(double (*pdf)(double, double[]), double par[], double a0,
                     double b0, int maxIt=1000, int n=50, double tolF=1.e-7,
                     double tolX=1.e-7)
{
    int i;
    double y, x_pdf, fa, fb, fc, a = a0, b = b0, c, m;

    // Generate a random number in [0, m]
    m = integ(pdf, par, a0, b0, n);
    y = rnd("unif", 0.0, m);

    // Check if <a0> is the solution
    fa = y;
    if (abs(fa) < tolF) {
        return a;
    }

    // Check if <b0> is the solution
    fb = y - m;
    if (abs(fb) < tolF) {
        return b;
    }

    // Check if f(a) and f(b) have same signs (this condition should never
    // be verified if the support is correctly defined)
    if (fa * fb > 0.0) {
        printf("\ncdf = %g", y);
        printf("\n--> Interval does not contain a zero.\n");
        exit(EXIT_FAILURE);
    }

    // Search for the zero using the bisection method
    i = 0;
    while (i < maxIt) {

        // Function value in the middle-point
        c = (a + b) / 2.0;
        fc = y - integ(pdf, par, a0, c, n);

        // Check if one of the tolerances is satisfied
        if ((abs(fc) < tolF) || ((b-a)/2.0 < tolX)) {
            return c;
        }

        // Reduce the interval and iterate again
        else {
            i += 1;
            if (fa * fc > 0.0) {
                a = c;
            }
            else {
                b = c;
            }
        }
    }

    // Reached the max. number of iteration  --> return best value found
    printf("\ncdf = %g", y);
    printf("\n--> Reached max. number of iterations.\n");
    return c;
}


/* Returns the pdf(x) for the specified prior (with up to 5 parameters) */
double prior_dist(Prior_type prior, double x)
{
    double pdf = 0.0, a, b, mu, sigma, alpha, beta, lambda, d, xm, logpdf;

    // Uniform distribution, [a, b]
    if (prior.name == "unif") {
        a = prior.par[0];
        b = prior.par[1];
        if (x >= a && x <= b) {
            pdf = 1.0 / (b - a);
        }
    }

    // Normal distribution, [-inf, +inf]
    else if (prior.name == "norm") {
        mu = prior.par[0];
        sigma = prior.par[1];
        d = (x - mu) / sigma;
        pdf = exp(-d * d / 2.0) / (sigma * sqrt(2.0 * 3.14159265358979323846));
    }

    // Beta distribution, [0, 1]
    else if (prior.name == "beta") {
        alpha = prior.par[0];
        beta = prior.par[1];
        if (x >= 0.0 && x <= 1.0) {
            logpdf = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta) +
                     (alpha - 1.0) * log(x) + (beta - 1.0) * log(1.0 - x);
            pdf = exp(logpdf);
        }
    }

    // Gamma distribution, [0, +inf]
    else if (prior.name == "gamma") {
        alpha = prior.par[0];
        beta = prior.par[1];
        if (x >= 0.0) {
            logpdf = alpha * log(beta) - lgamma(alpha) +
                     (alpha - 1.0) * log(x) - beta * x;
            pdf = exp(logpdf);
        }
    }

    // Exponential distribution, [0, +inf]
    else if (prior.name == "expon") {
        lambda = prior.par[0];
        if (x >= 0.0) {
            pdf = lambda * exp(-lambda * x);
        }
    }

    // Pareto distribution, [1, +inf]
    else if (prior.name == "pareto") {
        xm = prior.par[0];
        alpha = prior.par[1];
        if (x >= 1.0) {
            logpdf = alpha + alpha * log(xm) - (alpha + 1.0) * log(x);
            pdf = exp(logpdf);
        }
    }

    // Default case
    else {
        printf("\n%s", prior.name.c_str());
        printf("\n--> Prior distribution not recognized.\n");
        exit(EXIT_FAILURE);
    }

    return pdf;
}


/*
Returns the posterior function of the parameters given the likelihood and
the prior functions. Returns also the number of the accepted jumps in the
Metropolis-Hastings algorithm.

Notes:
- <width_prop> should be chosen so to result in about 50% accepted jumps.
- <posterior> has shape (samples, n_par).
- priors must be from function "prior_dist".
- for numerical stability the computation is carried out using logarithms.
*/
Results metropolis(double data[], int n_data, double (*func)(double, double[]),
                   Prior_type priors[], int n_par, double par_init[],
                   int samples=10000, double width_prop=0.5)
{
    int jumps = 0;
    double prior_curr, prior_prop, likelihood_curr, likelihood_prop, p_curr,
           p_prop, p_accept, sum;
    double *par_curr, *par_prop, **posterior;
    Results res;

    // Init quantities
    par_curr = new double [n_par];
    par_prop = new double [n_par];
    posterior = new double *[n_par];
    res.posterior = new double *[n_par];
    for (int i=0; i<n_par; i++) {
        posterior[i] = new double[samples];
        res.posterior[i] = new double[samples];
    }

    // Set current parameters
    for (int i=0; i<n_par; i++) {
        par_curr[i] = par_init[i];
    }
    for (int i=0; i<n_par; i++) {
        posterior[i][0] = par_curr[i];
    }

    // Current priors
    sum = 0.0;
    for (int i=0; i<n_par; i++) {
        sum += log(prior_dist(priors[i], par_curr[i]));
    }
    prior_curr = exp(sum);

    // Current likelihood
    sum = 0.0;
    for (int i=0; i<n_data; i++) {
        sum += log(func(data[i], par_curr));
    }
    likelihood_curr = exp(sum);

    // Current posterior probability
    p_curr = likelihood_curr * prior_curr;

    // Loop <samples> times
    for (int sample=0; sample<samples; sample++) {

        // Randomnly pick the proposed parameters
        for (int i=0; i<n_par; i++) {
            par_prop[i] = rnd("norm", par_curr[i], width_prop);
        }

        // Evaluate priors with the proposed parameters
        sum = 0.0;
        for (int i=0; i<n_par; i++) {
            sum += log(prior_dist(priors[i], par_prop[i]));
        }
        prior_prop = exp(sum);

        // Evaluate likelihood with the proposed parameters
        sum = 0.0;
        for (int i=0; i<n_data; i++) {
            sum += log(func(data[i], par_prop));
        }
        likelihood_prop = exp(sum);

        // Proposed posterior probability */
        p_prop = likelihood_prop * prior_prop;

        // Randomly accept or reject the move
        p_accept = p_prop / p_curr;
        if (rnd("unif") < p_accept) {

            // Update quantities if jump accepted
            jumps += 1;
            for (int i=0; i<n_par; i++) {
                par_curr[i] = par_prop[i];
            }
            prior_curr = prior_prop;
            likelihood_curr = likelihood_prop;
            p_curr = p_prop;
        }

        // Save (accepted and rejected) parameters
        for (int i=0; i<n_par; i++) {
            posterior[i][sample] = par_curr[i];
        }
    }

    // Copy the solution to the result structure
    for (int i=0; i<n_par; i++) {
        for (int j=0; j<samples; j++) {
            res.posterior[i][j] = posterior[i][j];
        }
    }
    res.jumps = jumps;

    return res;
}
