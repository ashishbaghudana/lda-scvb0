#include "dataset.h"

long * word_temp, * count_temp;

long * allocate_memory(long size) {
  long * array = (long *) malloc(size * sizeof(long));
  if (array == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  return array;
}

long** allocate_memory_2d(long size) {
  long ** matrix = (long **) malloc(size * sizeof(long *));
  if (matrix == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  return matrix;
}

void read_metadata(FILE *input_file) {
  if (3 != fscanf(input_file, "%ld\n%ld\n%ld\n", &D, &W, &NNZ)) {
    printf("There is something wrong with input file format\n");
    exit(-1);
  }
}

void read_sparse_dataset(char input_file_name[]) {
  FILE *input_file = fopen(input_file_name, "r");
  if (input_file == NULL) {
    printf("Can't open the corpus file to read\n");
    exit(-1);
  }

  // Get number of documents, number of words, and number of non-zero counts
  read_metadata(input_file);
  printf("D = %ld; W = %ld; NNZ = %ld;\n", D, W, NNZ);

  C = 0;
  count_max = 0;

  word_temp = allocate_memory(W);
  count_temp = allocate_memory(W);
  num_unique_d = allocate_memory(D + 1);
  C_d = allocate_memory(D + 1);
  memset(C_d, 0, (D + 1) * sizeof(long));
  word_d_i = allocate_memory_2d(D+1);
  count_d_i = allocate_memory_2d(D+1);

  printf("lines done reading: ");
  long next_temp, this_doc, last_doc = -1;
  for (long i = 0; i < NNZ; ++i) {
    // Counter of lines read - by million
    if (i % 1000000 == 0 && i != 0) {
      printf("%ldM ", i/1000000);
      fflush(stdout);
    }

    if (1 != fscanf(input_file, "%ld ", &this_doc)) {
      printf("There is something wrong with input file format\n");
      exit(-1);
    }

    // new document
    if (this_doc != last_doc) {
      if (last_doc != -1) {
        // malloc and copy over
        num_unique_d[last_doc] = next_temp;
        long copy_size = next_temp * sizeof(long);

        if ((word_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
          printf("Out of memory\n");
          exit(-1);
        }
        memcpy(word_d_i[last_doc], word_temp, copy_size);

        if ((count_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
          printf("Out of memory\n");
          exit(-1);
        }
        memcpy(count_d_i[last_doc], count_temp, copy_size);
      }
      last_doc = this_doc;

      // reset temp array
      next_temp = 0;
    }

    if (2 != fscanf(input_file, "%ld %ld\n", &word_temp[next_temp], &count_temp[next_temp])) {
        printf("There is something wrong with input file format\n");
        exit(-1);
    }

    long ctnt = count_temp[next_temp];
    C_d[this_doc] += ctnt;
    C += ctnt;
    count_max = (ctnt > count_max)? ctnt: count_max;
    next_temp++;
  }
  num_unique_d[last_doc] = next_temp;
  long copy_size = next_temp * sizeof(long);
  if ((word_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
      printf("Out of memory\n");
      exit(-1);
  }
  memcpy(word_d_i[last_doc], word_temp, copy_size);
  if ((count_d_i[last_doc] = (long *) malloc(copy_size)) == NULL) {
      printf("Out of memory\n");
      exit(-1);
  }
  memcpy(count_d_i[last_doc], count_temp, copy_size);
  printf("\n");

  if (0 != fclose(input_file)) {
      printf("Can't close docword.txt\n");
      exit(-1);
  }

  printf("C = %ld;\n", C);
}
