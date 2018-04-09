#include "util.h"

struct timeval tic, toc, diff;

// Borrowed directly from
// http://www.gnu.org/software/libc/manual/html_node/Elapsed-Time.html
int timeval_subtract (result, x, y)
struct timeval *result, *x, *y; {
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }
  /* Compute the time remaining to wait. tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;
  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

void start_timer() {
  gettimeofday(&tic, NULL);
}

void stop_timer(char message[]) {
  gettimeofday(&toc, NULL);
  timeval_subtract(&diff, &toc, &tic);
  printf(message, diff.tv_sec + (double) diff.tv_usec / 1000000);
}

// void bubble_sort(struct _word_probability topic_dist[], long n) {
//   struct _word_probability * temp;
//
//   if ((temp = (struct _word_probability *) malloc(n * sizeof(struct _word_probability))) == NULL) {
//     printf("Out of memory\n");
//     exit(-1);
//   }
//   memcpy(temp, topic_dist, n * sizeof(struct _word_probability));
//   for (long i = 0; i < n - 1; ++i) {
//     long max_idx = i;
//     for (long j = i + 1; j < n; ++j) {
//       if (temp[max_idx].probability < temp[j].probability) {
//           max_idx = j;
//       }
//     }
//     if (max_idx != i) {
//       struct _word_probability t;
//       t = temp[i];
//       temp[i] = temp[max_idx];
//       temp[max_idx] = t;
//     }
//   }
//   memcpy(topic_dist, temp, n * sizeof(struct _word_probability));
//   free(temp);
// }

void merge(struct _word_probability topic_dist[], long n) {
  struct _word_probability * temp;
  long l = 0, r = (n/2), i = 0;

  if ((temp = (struct _word_probability *) malloc(n * sizeof(struct _word_probability))) == NULL) {
    printf("Out of memory\n");
    exit(-1);
  }
  while (l < (n/2) && r < n) {
    if (topic_dist[l].probability > topic_dist[r].probability) {
        temp[i++] = topic_dist[l++];
    } else {
        temp[i++] = topic_dist[r++];
    }
  }
  while (l < (n/2)) {
    temp[i++] = topic_dist[l++];
  }
  while (r < n) {
    temp[i++] = topic_dist[r++];
  }
  memcpy(topic_dist, temp, n * sizeof(struct _word_probability));
  free(temp);
}

void merge_sort(struct _word_probability topic_dist[], long n) {
  merge_sort(topic_dist, n/2);
  merge_sort(topic_dist + (n/2), n - (n/2));
  merge(topic_dist, n);
}
