/*
Multi-Parameter Bayesian Inference Using Markov Chain Monte Carlo (MCMC)
Sampling and the Metropolis-Hastings Algorithm.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written in plain vanilla C++ and tested using g++ 8.1.0
  (MinGW-W64).
- Likelihood (pdf) can be defined as an arbitrary function with any number
  of independent parameters.
- Prior functions are defined using an array of structures, and can be any
  pdf from function "prior_dist" in file "Metropolis.cpp" (other priors can
  be easily added).
- Jumps in the Metropolis-Hastings algorithm are proposed using a normal
  distribution of the parameters.
- Function "random_number" in file "Metropolis.cpp" can be used to generate
  random numbers from any arbitrary pdf.
- Results are passed using a structure.
- Usage: test.exe <example>.

Main Parameters
---------------
example = Random, Coin, Normal
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
    Array of structures with the priors. Each prior is assigned to one of
    the likelihood parameter following the same order as in <par>.
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
res
    Structure with the results returned by the Metropolis-Hastings algorithm.
save_res
    Name of the file where to save the results. If empty, results are not
    saved.

Examples
--------
There are three examples: Random, Coin, and Normal (see the code for parameters
and results).

- Random: generation of random numbers from a generic pdf.

- Coin: one parameter (theta), Bernoulli distribution as likelihood, beta
        distribution as prior, admit an analytical solution.

- Normal: two parameters (mean and standard deviation), normal distribution
          as likelihood, normal distribution as prior for the mean, gamma
          distribution as prior for the standard deviation.

References
----------
- Metropolis-Hastings algorithm @
  https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm
- Markov chain Monte Carlo @
  https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
- Halls-Moore, 2016, "Bayesian statistics", chapter II in "Advanced Algorithmic
  Trading", @ https://www.quantstart.com/advanced-algorithmic-trading-ebook/
*/

#include <fstream>
#include <random>

using namespace std;

/* Structure used to define the priors */
struct Prior_type {
    string name="none";                             // Name
    double par[5]= {0.0, 0.0, 0.0, 0.0, 0.0};       // Up to 5 parameters
};

/* Structure used to return the results */
struct Results {
    double **posterior;                             // Shape (samples, n_par)
    int jumps;
};

/* External function prototypes */
Results metropolis(double data[], int n_data, double (*func)(double, double[]),
                   Prior_type priors[], int n_par, double par_init[],
                   int samples=10000, double width_prop=0.5);
double random_number(double (*func)(double, double[]), double par[], double a0,
                     double b0, int maxIt=1000, int n=50, double tolF=1.e-7,
                     double tolX=1.e-7);
double rnd(string type, double par1=0.0, double par2=1.0);


/* Returns the mean of an array between <start> and <end-1> */
double mean(double a[], int start, int end)
{
    double sum = 0.0;

    for (int i=start; i<end; i++) {
        sum += a[i];
    }

    return sum / double(end - start);
}


/* Returns the standard deviation of an array between <start> and <end-1> */
double stdev(double a[], int start, int end)
{
    double sum = 0.0, mu;

    mu = mean(a, start, end);

    for (int i=start; i<end; i++) {
        sum += pow(a[i]-mu, 2);
    }

    return (sqrt(sum / double(end - start)));
}


/* Piece-wise pdf (example "Random") */
double pdf_random(double x, double par[])
{
    double pdf = 0.0;

    if (x > 0.0 && x <= 1.0) {
        pdf = 0.3 * x;
    }
    else if (x > 1.0 && x <= 2.0) {
        pdf = -0.2 * x + 0.5;
    }
    else if (x > 2.0 && x <= 3.0) {
        pdf = 0.1;
    }
    else if (x > 3.0 && x <= 4.0) {
        pdf = 0.1 * x - 0.2;
    }
    else if (x > 4.0 && x <= 5.0) {
        pdf = 0.2;
    }
    else if (x > 5.0 && x <= 7.0) {
        pdf = -0.1 * x + 0.7;
    }

    return pdf;
}


/* Bernoulli distribution (example "Coin") */
double pdf_coin(double x, double par[])
{
    double theta = par[0], pdf;

    if (x == 1.0) {
        pdf = theta;
    }
    else {
        pdf = 1.0 - theta;
    }

    return pdf;
}


/* Normal distribution (example "Normal") */
double pdf_normal(double x, double par[])
{
    const double fact = sqrt(2.0 * 3.14159265358979323846);
    double mu = par[0], sigma = par[1], d, pdf;

    d = (x - mu) / sigma;
    pdf = exp(-d * d / 2.0) / (sigma * fact);

    return pdf;
}


/* Main function */
int main(int argc, char **argv) 
{
    int n_data, n_par, samples, n_heads = 0, i0;
    double a0, b0, width_prop, alpha, beta, alpha_post, beta_post, tmp,
           mu_post, sigma_post;
    double (*likelihood)(double, double[]), *par, *par_init, *data;
    string example, save_res = "";
    Prior_type *priors;
    Results res;
    ofstream idf;

    // Read example to run
    if (argc != 2) {
        printf("\nUsage: test <example>\n");
        exit(EXIT_FAILURE);
    }
    example = argv[1];

    // Seed the random generator
    rnd("seed", double(123));

    // Generation of random numbers given the pdf
    if (example == "Random") {

        // Parameters
        par = NULL;
        a0 = -1.0;
        b0 = +8.0;

        // Randomly approximated pdf
        n_data = 50000;
        data = new double [n_data];
        for (int i=0; i<n_data; i++) {
            data[i] = random_number(pdf_random, par, a0, b0, 100, 50);
        }

        // Save results
        save_res = "res_random.txt";
    }

    // Coin flip example:
    // - one parameter (the probability tail comes up)
    // - Bernoulli distribution as likelihood
    // - beta distribution as prior
    // - admit an analytical solution
    else if (example == "Coin") {

        // Generate data (0 = tail, 1 = head)
        likelihood = pdf_coin;
        n_data = 50;                        // Number of coin flip
        data = new double [n_data];
        n_par = 1;
        par = new double [n_par];
        par[0] = 0.63;                      // Probability tail comes up
        for (int i=0; i<n_data; i++) {
            if (rnd("unif") > par[0]) {
                data[i] = 1.0;              // Head
                n_heads += 1;
            }
            else {
                data[i] = 0.0;              // Tail
            }
        }

        // Priors (coefficients are just arbitrary values used as example)
        alpha = 4.0;
        beta = 12.0;
        priors = new Prior_type [n_par];
        priors[0] = {"beta", alpha, beta};

        // Solve
        samples = 10000;
        par_init = new double [n_par];
        par_init[0] = 0.5;
        width_prop = 0.1;
        i0 = int(floor(0.2 * double(samples)));         // Burn-in period
        save_res = "res_coin.txt";
        res = metropolis(data, n_data, likelihood, priors, n_par, par_init,
                         samples, width_prop);

        // Results:
        // - accepted jumps = 53.9%
        // - <theta> mean and std = 0.332, 0.057
        printf("\nAccepted jumps = %3.1f %%", 100.0 * res.jumps / samples);
        printf("\n<theta> mean and std = %4.3f, %4.3f",
                mean(res.posterior[0], i0, samples),
                stdev(res.posterior[0], i0, samples));

        // Analytical solution
        // - Bernoulli likelihood with beta prior results in a beta posterior
        // - posterior (alpha, beta) are converted to equivalent (mu, sigma)
        //   for comparison purpose
        // - Results: <theta> mean and std (anal.) = 0.333, 0.058
        // - Ref.: https://en.wikipedia.org/wiki/Conjugate_prior
        alpha_post = alpha + double(n_heads);
        beta_post = beta + double(n_data - n_heads);
        tmp = alpha_post + beta_post;
        mu_post = alpha_post / tmp;
        sigma_post = sqrt(alpha_post * beta_post / (tmp + 1)) / tmp;
        printf("\n<theta> mean and std (anal.) = %4.3f, %4.3f",
                mu_post, sigma_post);
        printf("\n");
    }

    // Example with normally distributed likelihood:
    // - two parameters (mean and standard deviation)
    // - Normal distribution as likelihood
    // - Normal distribution as prior for the mean <mu >and gamma distribution
    //   as prior for the standard deviation <sigma>
    else if (example == "Normal") {

        // Generate data
        likelihood = pdf_normal;
        n_data = 20;
        data = new double [n_data];
        n_par = 2;
        par = new double [n_par];
        par[0] = -1.3;
        par[1] = 1.0;
        a0 = -10.0;
        b0 = 10.0;
        for (int i=0; i<n_data; i++) {
            data[i] = random_number(likelihood, par, a0, b0);
        }

        // Priors (coefficients are just arbitrary values used as example)
        priors = new Prior_type [n_par];
        priors[0] = {"norm", 2.0, 1.0};             // Mean
        priors[1] = {"gamma", 6.0, 1.0};            // Standard deviation

        // Solve
        samples = 20000;
        par_init = new double [n_par];
        par_init[0] = 2.0;
        par_init[1] = 5.0;
        width_prop = 0.15;
        i0 = int(floor(0.2 * double(samples)));     // Burn-in period
        save_res = "res_normal.txt";
        res = metropolis(data, n_data, likelihood, priors, n_par, par_init,
                         samples, width_prop);

        // Results:
        // - accepted jumps = 52.7%
        // - <mu> mean and std = -1.207, 0.170
        // - <sigma> mean and std = 0.728, 0.150
        printf("\nAccepted jumps = %3.1f %%", 100.0 * res.jumps / samples);
        printf("\n<mu> mean and std = %5.3f, %5.3f",
                mean(res.posterior[0], i0, samples),
                stdev(res.posterior[0], i0, samples));
        printf("\n<sigma> mean and std = %5.3f, %5.3f",
                mean(res.posterior[1], i0, samples),
                stdev(res.posterior[1], i0, samples));
        printf("\n");

        // Results using pymc3 (loading the 20 random data generated here in
        // the python version):
        // - <mu> mean and std = -1.210, 0.171
        // - <sigma> mean and std = 0.730, 0.156
    }

    else {
        printf("\n%s", example.c_str());
        printf("\n--> Example not found.\n");
        exit(EXIT_FAILURE);
    }

    // Save results to file
    if (save_res.length() > 0);

        // Open file
        idf.open(save_res);

        // Save random data
        idf << n_data << endl;
        for (int i=0; i<n_data; i++) {
            idf << data[i] << endl;
        }

        // Save posterior and number of jumps
        if (example != "Random") {
            idf << samples << endl;
            for (int i=0; i<samples; i++) {
                for (int j=0; j<n_par; j++) {
                    idf << res.posterior[j][i] << " ";
                }
                idf << endl;
            }
            idf << res.jumps << endl;
        }
        
        // Close file
        idf.close();

        printf("\nResults saved in %s\n", save_res.c_str());

    return 0;
}