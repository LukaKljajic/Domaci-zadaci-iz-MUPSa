#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define START_TIMER \
   cudaEvent_t start = cudaEvent_t(); \
   cudaEvent_t stop = cudaEvent_t(); \
   cudaEventCreate(&start); \
   cudaEventCreate(&stop); \
   cudaEventRecord(start, 0);

#define END_TIMER \
   cudaEventRecord(stop, 0); \
   cudaEventSynchronize(stop); \
   cudaEventElapsedTime(time, start, stop); \
   cudaEventDestroy(start); \
   cudaEventDestroy(stop);

const int BLOCK_SIZE = 128;
const int GRID_SIZE = 192;
const float ACC = 0.01; 

void Usage(char* prog_name);
float sequential(long long n, float* time);
__global__ void kernel(long long n, float* sums);
float parallel(long long n, float* time);
bool compare(float first, float second);
 
int main(int argc, char* argv[]) 
{
   long long n;
   float sequential_res, parallel_res, sequential_time, parallel_time;

   if (argc != 2) Usage(argv[0]);
   n = strtoll(argv[1], NULL, 10);
   if (n < 1) Usage(argv[0]);

   sequential_res = sequential(n, &sequential_time);
   parallel_res = parallel(n, &parallel_time);

   printf("With n = %lld terms\n", n);
   printf("   Our sequential estimate of pi = %.14f\n", sequential_res);
   printf("   Our parallel estimate of pi = %.14f\n", parallel_res);
   printf("   Ref estimate of pi = %.14f\n", 4.0*atan(1.0));
   printf("Sequential time: %.15f ms\n", sequential_time);
   printf("Parallel time: %.15f ms\n", parallel_time);
   printf("Speedup: %.15f\n", sequential_time / parallel_time);

   if(compare(sequential_res, parallel_res))
      printf("TEST PASSED\n");
   else
      printf("TEST FAILED\n");

   return 0;
}

float sequential(long long n, float* time)
{
   long long i;
   float factor;
   float sum = 0.0;

   START_TIMER
   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }   
   END_TIMER

   return 4.0 * sum;
}

__global__ void kernel(long long n, float* sums)
{
   long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x, i;
   float factor;
   for(i = idx; i < n; i += gridDim.x * blockDim.x)
   {
      factor = (i % 2 == 0) ? 1.0 : -1.0;
      sums[idx] += factor / (2.0 * i + 1.0);
   }
}

float parallel(long long n, float* time)
{
   long long i;
   float sum = 0;
   float* sumsHost;
   float* sumsDev;
   size_t size = GRID_SIZE * BLOCK_SIZE;

   START_TIMER
   sumsHost = (float*) malloc(size * sizeof(float));
   cudaMalloc(&sumsDev, size * sizeof(float));
   cudaMemset(sumsDev, 0, size * sizeof(float));

   kernel<<<GRID_SIZE, BLOCK_SIZE>>>(n, sumsDev);

   cudaMemcpy(sumsHost, sumsDev, size * sizeof(float), cudaMemcpyDeviceToHost);
   for(i = 0; i < size; i++)
      sum += sumsHost[i];

   END_TIMER

   free(sumsHost);
   cudaFree(sumsDev);

   return 4.0 * sum;
}

void Usage(char* prog_name) 
{
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   exit(0);
}

bool compare(float first, float second)
{
   float diff = fabs(first - second);
   return diff < ACC;
}
