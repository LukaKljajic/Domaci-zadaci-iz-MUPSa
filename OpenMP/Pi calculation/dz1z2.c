#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void Usage(char* prog_name) {
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(0);
}

const double ACCURACY = 0.01;

double sequential(long long n, double* time) {
   double start = omp_get_wtime(), end;
   long long i;
   double factor;
   double sum = 0.0;

   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }

   sum = 4.0*sum;
   end = omp_get_wtime();
   *time = end - start;
   
   return sum;
}

double parallel(long long n, double* time) {
   double start = omp_get_wtime(), end;
   long long i; 
   double factor;
   double sum = 0.0;

#pragma omp parallel for default(none) private(i, factor) shared(n) reduction(+:sum)
   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }

   sum = 4.0*sum;
   end = omp_get_wtime();
   *time = end - start;

   return sum;
}

int main(int argc, char* argv[]) {
   long long n;
   double time_sequential, time_parallel;

   if (argc != 2) Usage(argv[0]);
   n = strtoll(argv[1], NULL, 10);
   if (n < 1) Usage(argv[0]);

   double sequential_res = sequential(n, &time_sequential);
   double parallel_res = parallel(n, &time_parallel);
   printf("Time for sequential processing is: %.14f\n", time_sequential);
   printf("Time for parallel processing is: %.14f\n", time_parallel);
   printf("Speedup is: %.14f\n", time_sequential/time_parallel);

   printf("With n = %lld terms\n", n);
   printf("   Our sequential estimate of pi = %.14f\n", sequential_res);
   printf("   Our parallel estimate of pi = %.14f\n", parallel_res);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));

   double diff = sequential_res - parallel_res;
   if(diff < 0) diff = -diff;
   if(diff <= ACCURACY) printf("TEST PASSED\n");
   else printf("TEST FAILED\n");
}
