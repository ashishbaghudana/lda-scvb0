#include "scvb0.h"

// inference macros
#define N_theta_d_k(d, k) N_theta_d_k[(d)*num_topics + (k)]
#define N_phi_w_k(w, k) N_phi_w_k[(w)*num_topics + (k)]
#define N_z_k(k) N_z_k[(k)]
#define N_hat_phi_t_w_k(t, w, k) N_hat_phi_t_w_k[(t)][(w)*num_topics + (k)]
#define N_hat_z_t_k(t, k) N_hat_z_t_k[(t)][(k)]

// recover hidden variables
#define theta_d_k(d, k) theta_d_k[(d)*num_topics + (k)]
#define phi_w_k(w, k) phi_w_k[(w)*num_topics + (k)]

#ifndef REPORT_PERPLEXITY
#define REPORT_PERPLEXITY
#endif

// constants
#ifndef ALPHA
#define ALPHA 0.1
#endif

#ifndef ETA
#define ETA 0.08
#endif

#ifndef NUM_BURN_IN
#define NUM_BURN_IN 1
#endif

#define BATCH_SIZE 500
#define MIN_NUM_THREADS 1
#define NUM_TERMS_REPORTED_PER_TOPIC 100

// input
long iterations;
long num_topics;
long num_threads;

// for calculations
double *N_theta_d_k, *N_phi_w_k, *N_z_k;
long *C_t;
double *rho_phi_times_C_over_C_t;
double *one_over_N_z_k_plus_W_times_ETA;
double *factor, *one_minus_factor;
double **N_hat_phi_t_w_k, **N_hat_z_t_k;
double *theta_d_k, *phi_w_k;

// for output
struct _word_probability **topic;

void calculate_theta_phi();
void calculate_perplexity();
void inference(long iteration_idx);
void calculate_topic();
void serial_lda();
void parallel_lda();
void output();
void print_usage();
long *allocate_memory_long(long size);
double *allocate_memory_double(long size);
double **allocate_memory_double_2d(long size);

int main(int argc, char **argv) {
  bool cflag = false, iflag = false, kflag = false, tflag = false;
  int c;

  char *corpus = NULL;

  while ((c = getopt(argc, argv, ":c:i:k:t")) != -1) {
    switch (c) {
    case 'c':
      cflag = true;
      corpus = optarg;
      break;
    case 'i':
      iflag = true;
      iterations = atoi(optarg);
      break;
    case 'k':
      kflag = true;
      num_topics = atoi(optarg);
      break;
    case 't':
      tflag = true;
      num_threads = atoi(optarg);
      break;
    }
  }

  if (cflag == false || iflag == false || kflag == false) {
    print_usage();
    exit(0);
  }

  printf("Corpus = %s\n", corpus);
  printf("Iterations = %ld\n", iterations);
  printf("Num Topics = %ld\n", num_topics);

  if (tflag == false || num_threads < 0) {
    // Currently setting number of threads equal to 1
    num_threads = 1;
  }

  // Initialize pseudo-random number generator
  srand(time(NULL));

  // Read dataset into memory
#ifdef DEBUG
  start_timer();
#endif
  read_sparse_dataset(corpus);
#ifdef DEBUG
  stop_timer("reading file took %.3f seconds\n");
  printf("\n");
#endif

// Allocate matrices needed for calculation
#ifdef DEBUG
  start_timer();
#endif
  N_theta_d_k = allocate_memory_double((D + 1) * num_topics);
  N_phi_w_k = allocate_memory_double((W + 1) * num_topics);
  N_z_k = N_phi_w_k;
  C_t = allocate_memory_long(num_threads);
  rho_phi_times_C_over_C_t = allocate_memory_double(num_threads);
  one_over_N_z_k_plus_W_times_ETA = allocate_memory_double(num_topics);
  factor = allocate_memory_double(count_max + 1);
  one_minus_factor = allocate_memory_double(count_max + 1);
  N_hat_phi_t_w_k = allocate_memory_double_2d(num_threads);
  for (long t = 0; t < num_threads; t++) {
    N_hat_phi_t_w_k[t] = allocate_memory_double((W + 1) * num_topics);
  }
  N_hat_z_t_k = N_hat_phi_t_w_k;
  theta_d_k = allocate_memory_double((D + 1) * num_topics);
  phi_w_k = allocate_memory_double((W + 1) * num_topics);
#ifdef DEBUG
  stop_timer("memory allocation took %.3f seconds\n");
#endif

  serial_lda();
  return 0;
}

void serial_lda() {
  double start_time = omp_get_wtime();

  // randomly initialize N_theta_d_k, N_phi_w_k, N_z_k
#ifdef DEBUG
  start_timer();
#endif

  for (long t = 0; t < num_threads; ++t)
    memset(N_hat_phi_t_w_k[t], 0, (W + 1) * num_topics * sizeof(double));

  for (long d = 1; d <= D; d++) {
    long thread_id = omp_get_thread_num();
    unsigned long z;
    for (long i = 0; i < num_unique_d[d]; i++) {
      for (long c = 0; c < count_d_i[d][i]; ++c) {
        z = rand();
        long k = z % num_topics;
        N_theta_d_k(d, k) += 1;
        N_hat_phi_t_w_k(thread_id, word_d_i[d][i], k) += 1;
        N_hat_z_t_k(thread_id, k) += 1;
      }
    }
  }

  for (long k = 0; k < num_topics; k++) {
    for (long w = 1; w <= W; ++w) {
      double sum_N_phi = 0;
      for (long t = 0; t < num_threads; ++t) {
        sum_N_phi += N_hat_phi_t_w_k(t, w, k);
      }
      N_phi_w_k(w, k) = sum_N_phi;
    }
    double sum_N_z = 0;
    for (long t = 0; t < num_threads; ++t) {
      sum_N_z += N_hat_z_t_k(t, k);
    }
    N_z_k(k) = sum_N_z;
  }

#ifdef DEBUG
  stop_timer("random initialization took %.3f seconds\n");
  printf("\n");
#endif

#ifdef REPORT_PERPLEXITY
  printf("\ncalculate initial perplexity:\n");

#ifdef DEBUG
  start_timer();
#endif
  calculate_theta_phi();
#ifdef DEBUG
  stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  calculate_perplexity();
#ifdef DEBUG
  stop_timer("perplexity calculation took %.3f seconds\n");
#endif
  printf("\n");
#endif

  // for each iteration
  for (long iteration_idx = 1; iteration_idx <= iterations; iteration_idx++) {
    printf("iteration %ld:\n", iteration_idx);

#ifdef DEBUG
    start_timer();
#endif
    inference(iteration_idx);
#ifdef DEBUG
    stop_timer("inference took %.3f seconds\n");
#endif

#ifdef REPORT_PERPLEXITY
#ifdef DEBUG
    start_timer();
#endif
    calculate_theta_phi();
#ifdef DEBUG
    stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
    start_timer();
#endif
    calculate_perplexity();
#ifdef DEBUG
    stop_timer("perplexity calculation took %.3f seconds\n");
#endif
#endif

    printf("\n");
  }

#ifdef DEBUG
  start_timer();
#endif
  calculate_theta_phi();
#ifdef DEBUG
  stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  calculate_topic();
#ifdef DEBUG
  stop_timer("topic calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  output();
#ifdef DEBUG
  stop_timer("writing files took %.3f seconds\n");
#endif

  double end_time = omp_get_wtime();

  printf("Time taken: %.3f seconds\n\n", end_time - start_time);
}

void parallel_lda() {
  double start_time = omp_get_wtime();

  // randomly initialize N_theta_d_k, N_phi_w_k, N_z_k
#ifdef DEBUG
  start_timer();
#endif

  for (long t = 0; t < num_threads; ++t)
    memset(N_hat_phi_t_w_k[t], 0, (W + 1) * num_topics * sizeof(double));

  for (long d = 1; d <= D; d++) {
    long thread_id = omp_get_thread_num();
    unsigned long z;
    for (long i = 0; i < num_unique_d[d]; i++) {
      for (long c = 0; c < count_d_i[d][i]; ++c) {
        z = rand();
        long k = z % num_topics;
        N_theta_d_k(d, k) += 1;
        N_hat_phi_t_w_k(thread_id, word_d_i[d][i], k) += 1;
        N_hat_z_t_k(thread_id, k) += 1;
      }
    }
  }

  for (long k = 0; k < num_topics; k++) {
    for (long w = 1; w <= W; ++w) {
      double sum_N_phi = 0;
      for (long t = 0; t < num_threads; ++t) {
        sum_N_phi += N_hat_phi_t_w_k(t, w, k);
      }
      N_phi_w_k(w, k) = sum_N_phi;
    }
    double sum_N_z = 0;
    for (long t = 0; t < num_threads; ++t) {
      sum_N_z += N_hat_z_t_k(t, k);
    }
    N_z_k(k) = sum_N_z;
  }

#ifdef DEBUG
  stop_timer("random initialization took %.3f seconds\n");
  printf("\n");
#endif

#ifdef REPORT_PERPLEXITY
  printf("\ncalculate initial perplexity:\n");

#ifdef DEBUG
  start_timer();
#endif
  calculate_theta_phi();
#ifdef DEBUG
  stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  calculate_perplexity();
#ifdef DEBUG
  stop_timer("perplexity calculation took %.3f seconds\n");
#endif
  printf("\n");
#endif

  // for each iteration
  for (long iteration_idx = 1; iteration_idx <= iterations; iteration_idx++) {
    printf("iteration %ld:\n", iteration_idx);

#ifdef DEBUG
    start_timer();
#endif
    inference(iteration_idx);
#ifdef DEBUG
    stop_timer("inference took %.3f seconds\n");
#endif

#ifdef REPORT_PERPLEXITY
#ifdef DEBUG
    start_timer();
#endif
    calculate_theta_phi();
#ifdef DEBUG
    stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
    start_timer();
#endif
    calculate_perplexity();
#ifdef DEBUG
    stop_timer("perplexity calculation took %.3f seconds\n");
#endif
#endif

    printf("\n");
  }

#ifdef DEBUG
  start_timer();
#endif
  calculate_theta_phi();
#ifdef DEBUG
  stop_timer("theta and phi calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  calculate_topic();
#ifdef DEBUG
  stop_timer("topic calculation took %.3f seconds\n");
#endif

#ifdef DEBUG
  start_timer();
#endif
  output();
#ifdef DEBUG
  stop_timer("writing files took %.3f seconds\n");
#endif

  double end_time = omp_get_wtime();

  printf("Time taken: %.3f seconds\n\n", end_time - start_time);
}

void print_usage() {
  printf("usage: ./lda [-c filename] [-i iterations] [-k num_topics] [-t "
         "num_threads]\n");
}

void calculate_theta_phi() {
  for (long d = 1; d <= D; d++) {
    double one_over_N_theta_d_plus_K_times_ALPHA = 0;
    for (long k = 0; k < num_topics; k++) {
      one_over_N_theta_d_plus_K_times_ALPHA += N_theta_d_k(d, k);
    }
    one_over_N_theta_d_plus_K_times_ALPHA =
        1 / (one_over_N_theta_d_plus_K_times_ALPHA + num_topics * ALPHA);
    for (long k = 0; k < num_topics; k++) {
      theta_d_k(d, k) = (double)(N_theta_d_k(d, k) + ALPHA) *
                        one_over_N_theta_d_plus_K_times_ALPHA;
    }
  }
  for (long k = 0; k < num_topics; k++) {
    one_over_N_z_k_plus_W_times_ETA[k] = 1 / (N_z_k(k) + W * ETA);
  }
  for (long w = 1; w <= W; ++w) {
    for (long k = 0; k < num_topics; k++) {
      phi_w_k(w, k) =
          (double)(N_phi_w_k(w, k) + ETA) * one_over_N_z_k_plus_W_times_ETA[k];
    }
  }
}

void calculate_perplexity() {
  double entropy = 0;
  for (long d = 1; d <= D; d++) {
    for (long i = 0; i < num_unique_d[d]; i++) {
      double P_d_i = 0;
      for (long k = 0; k < num_topics; k++) {
        P_d_i += theta_d_k(d, k) * phi_w_k(word_d_i[d][i], k);
      }
      entropy += count_d_i[d][i] * log2(P_d_i);
    }
  }
  entropy = -entropy;
  printf("(per word) entropy, perplexity: %.2f, %.2f\n", entropy / C,
         exp2(entropy / C));
}

void inference(long iteration_idx) {
  double rho_theta = pow(100 + 10 * iteration_idx, -0.9);
  double rho_phi = pow(100 + 10 * iteration_idx, -0.9);
  double one_minus_rho_phi = 1 - rho_phi;

  long num_batches = ceil((double)D / BATCH_SIZE);
  long num_epochs = ceil((double)num_batches / num_threads);

  for (long c = 1; c <= count_max; ++c) {
    factor[c] = pow(1 - rho_theta, c);
    one_minus_factor[c] = 1 - factor[c];
  }

  for (long epoch_id = 0; epoch_id < num_epochs; ++epoch_id) {
    long first_batch_this_epoch = epoch_id * num_threads;
    long first_batch_next_epoch = (epoch_id + 1) * num_threads;
    if (first_batch_next_epoch > num_batches) {
      first_batch_next_epoch = num_batches;
    }
    long num_batches_this_epoch =
        first_batch_next_epoch - first_batch_this_epoch;

    for (long k = 0; k < num_topics; k++) {
      one_over_N_z_k_plus_W_times_ETA[k] = 1 / (N_z_k(k) + W * ETA);
    }

    memset(C_t, 0, num_batches_this_epoch * sizeof(long));

    // for each batch in epoch
    {
      long thread_id = omp_get_thread_num();
      long batch_id = thread_id + epoch_id * num_threads;
      long first_doc_this_batch = batch_id * BATCH_SIZE + 1;
      long first_doc_next_batch = (batch_id + 1) * BATCH_SIZE + 1;
      if (first_doc_next_batch > D + 1) {
        first_doc_next_batch = D + 1;
      }

      // set N_hat_phi_t_w_k, N_hat_z_t_k to zero
      memset(N_hat_phi_t_w_k[thread_id], 0,
             (W + 1) * num_topics * sizeof(double));

      double *_gamma_k, *_N_theta_k;
      if ((_gamma_k = (double *)malloc(num_topics * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(-1);
      }
      if ((_N_theta_k = (double *)malloc(num_topics * sizeof(double))) ==
          NULL) {
        printf("Out of memory\n");
        exit(-1);
      }

      // for each document d in batch
      for (long d = first_doc_this_batch; d < first_doc_next_batch; d++) {
        long _C = C_d[d];
        long _num_unique = num_unique_d[d];
        C_t[thread_id] += _C;
        memcpy(_N_theta_k, &N_theta_d_k(d, 0), num_topics * sizeof(double));

        // for zero or more burn-in passes
        for (long b = 0; b < NUM_BURN_IN; ++b) {
          // for each token i
          for (long i = 0; i < _num_unique; i++) {
            long _word = word_d_i[d][i];
            long _count = count_d_i[d][i];
            // update gamma
            double normalizer = 0;
            for (long k = 0; k < num_topics; k++) {
              _gamma_k[k] = (_N_theta_k[k] + ALPHA) *
                            (N_phi_w_k(_word, k) + ETA) *
                            one_over_N_z_k_plus_W_times_ETA[k];
              normalizer += _gamma_k[k];
            }
            normalizer = 1 / normalizer;
            // update N_theta_d_k
            for (long k = 0; k < num_topics; k++) {
              _N_theta_k[k] =
                  factor[_count] * _N_theta_k[k] +
                  one_minus_factor[_count] * _C * _gamma_k[k] * normalizer;
            }
          }
        }

        // done with burn-in
        // for each token i
        for (long i = 0; i < _num_unique; i++) {
          long _word = word_d_i[d][i];
          long _count = count_d_i[d][i];
          // update gamma
          double normalizer = 0;
          for (long k = 0; k < num_topics; k++) {
            _gamma_k[k] = (_N_theta_k[k] + ALPHA) *
                          (N_phi_w_k(_word, k) + ETA) *
                          one_over_N_z_k_plus_W_times_ETA[k];
            normalizer += _gamma_k[k];
          }
          normalizer = 1 / normalizer;
          // update N_theta_d_k, N_hat_phi_t_w_k, N_hat_z_t_k
          for (long k = 0; k < num_topics; k++) {
            _N_theta_k[k] =
                factor[_count] * _N_theta_k[k] +
                one_minus_factor[_count] * _C * _gamma_k[k] * normalizer;
            double temp = _count * _gamma_k[k] * normalizer;
            N_hat_phi_t_w_k(thread_id, _word, k) += temp;
            N_hat_z_t_k(thread_id, k) += temp;
          }
        }

        memcpy(&N_theta_d_k(d, 0), _N_theta_k, num_topics * sizeof(double));
      }

      free(_gamma_k);
      free(_N_theta_k);
    } // end omp parallel

    // compute rho_phi * C / C_t[t]
    for (long t = 0; t < num_batches_this_epoch; ++t) {
      rho_phi_times_C_over_C_t[t] = rho_phi * C / C_t[t];
    }

// update N_phi_w_k
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long w = 1; w <= W; ++w) {
      double *_N_phi_k;
      if ((_N_phi_k = (double *)malloc(num_topics * sizeof(double))) == NULL) {
        printf("Out of memory\n");
        exit(-1);
      }
      memcpy(_N_phi_k, &N_phi_w_k(w, 0), num_topics * sizeof(double));
      for (long t = 0; t < num_batches_this_epoch; ++t) {
        for (long k = 0; k < num_topics; k++) {
          _N_phi_k[k] = rho_phi_times_C_over_C_t[t] * N_hat_phi_t_w_k(t, w, k) +
                        one_minus_rho_phi * _N_phi_k[k];
        }
      }
      memcpy(&N_phi_w_k(w, 0), _N_phi_k, num_topics * sizeof(double));
      free(_N_phi_k);
    }

// update N_z_k
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (long k = 0; k < num_topics; k++) {
      double _N_z = N_z_k(k);
      for (long t = 0; t < num_batches_this_epoch; ++t) {
        _N_z = rho_phi_times_C_over_C_t[t] * N_hat_z_t_k(t, k) +
               one_minus_rho_phi * _N_z;
      }
      N_z_k(k) = _N_z;
    }
  }
}

void calculate_topic() {
  if ((topic = (struct _word_probability **)malloc(
           num_topics * sizeof(struct _word_probability *))) == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  for (long k = 0; k < num_topics; k++) {
    if ((topic[k] = (struct _word_probability *)malloc(
             W * sizeof(struct _word_probability))) == NULL) {
      printf("Out of memory\n");
      exit(-1);
    }
    for (long i = 0; i < W; i++) {
      topic[k][i].word = i + 1;
      topic[k][i].probability = phi_w_k(i + 1, k);
    }
  }

  // sort topic probabilities over words
  for (long k = 0; k < num_topics; k++) {
    merge_sort(topic[k], W);
  }
}

void output() {
  FILE *output_file;

  // topics.txt
  if ((output_file = fopen("topics.txt", "w")) == NULL) {
    printf("Can't open topics.txt to write\n");
    exit(-1);
  }
  for (long k = 0; k < num_topics; k++) {
    for (long i = 0; i < NUM_TERMS_REPORTED_PER_TOPIC; i++) {
      fprintf(output_file, "%ld:%8.6f", topic[k][i].word,
              topic[k][i].probability);
      if (i == NUM_TERMS_REPORTED_PER_TOPIC - 1) {
        fprintf(output_file, "\n");
      } else {
        fprintf(output_file, ", ");
      }
    }
  }
  if (0 != fclose(output_file)) {
    printf("Can't close topics.txt\n");
    exit(-1);
  }

  // doctopic.txt
  if ((output_file = fopen("doctopic.txt", "w")) == NULL) {
    printf("Can't open doctopic.txt to write\n");
    exit(-1);
  }
  for (long d = 1; d <= D; d++) {
    for (long k = 0; k < num_topics; k++) {
      fprintf(output_file, "%8.6f", theta_d_k(d, k));
      if (k == num_topics - 1) {
        fprintf(output_file, "\n");
      } else {
        fprintf(output_file, ", ");
      }
    }
  }
  if (0 != fclose(output_file)) {
    printf("Can't close doctopic.txt\n");
    exit(-1);
  }
}

long *allocate_memory_long(long size) {
  long *array = (long *)malloc(size * sizeof(long));
  if (array == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  return array;
}

double *allocate_memory_double(long size) {
  double *array = (double *)malloc(size * sizeof(double));
  if (array == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  return array;
}

double **allocate_memory_double_2d(long size) {
  double **matrix = (double **)malloc(size * sizeof(double *));
  if (matrix == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  return matrix;
}
