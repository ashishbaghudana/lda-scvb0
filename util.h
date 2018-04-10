#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void read_sparse_dataset(char input_file_name[]);

void start_timer();
void stop_timer(char message[]);

struct _word_probability {
  long word;
  double probability;
};

void merge_sort(struct _word_probability topic_dist[], long n);
