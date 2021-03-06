# LDA-SCVB0

This repository contains code for latent Dirichlet allocation (LDA) using the stochastic collapsed variational Bayesian (SCVB0) inference. The original implementation of this algorithm was in Julia and is available [here](https://github.com/jrfoulds/Stochastic-CVB0).

Reference the paper by
```text
@inproceedings{foulds2013stochastic,
  title={Stochastic collapsed variational Bayesian inference for latent Dirichlet allocation},
  author={Foulds, James and Boyles, Levi and DuBois, Christopher and Smyth, Padhraic and Welling, Max},
  booktitle={Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={446--454},
  year={2013},
  organization={ACM}
}
```

## Project Details
This project is done as part of the CS5234: Advanced Parallel Computation course in Virginia Tech in Spring 2018. The project involves re-implementation of the algorithm in C, parallelization using OpenMP, and acceleration using OpenACC.

## Setup
The project requires currently requires gcc and OpenMP to be installed on the machine. Invoke the Makefile to create the `lda` binary.

```bash
$ make clean && make lda
```

## Usage
The `lda` binary created can be run with the following options.

```text
usage: ./lda [-c filename] [-i iterations] [-k num_topics] [-t num_threads]
```

Supply the following:
* `[-c filename]`: The corpus file. Sample corpus files can be found inside the `data/` directory. More corpus files can be found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bag+of+words).
* `[-i iterations]`: The number of iterations to run
* `[-k num_topics]`: The number of topics in the corpus
* `[-t num_threads]`: The number of threads to parallelize the inference over

## Outcomes of the Project
We parallelized latent dirichlet allocation using stochastic collapsed variational Bayesian inference using OpenMP. The OpenMP version performs roughly 30x to 80x faster than LDA-C and gets around 3x speedup with 20 cores against a single core implementation.
