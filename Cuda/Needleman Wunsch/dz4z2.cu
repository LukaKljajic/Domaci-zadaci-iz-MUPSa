#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

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

// #define TESTING

const int BLOCK_SIZE = 128;

void sequential(int max, int penalty, int* reference, float* time);
void parallel(int max, int penalty, int* reference, float* time);
void traceback(const char* filename, int* input_itemsets, int* reference, int max, int penalty);
__device__ __host__ int maximum(int a, int b, int c);
bool compare_files(const char *filename1, const char *filename2);

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

int main(int argc, char** argv) 
{
	int max, penalty, *reference, *rand_array_1, *rand_array_2;
	float sequential_time, parallel_time;
	if (argc == 3)
	{
		max = atoi(argv[1]) + 1;
		penalty = atoi(argv[2]);
	}
    else
		fprintf(stderr, "Bad arguments\n");

	printf("Start Needleman-Wunsch\n");
	reference = (int*)malloc(max * max * sizeof(int));
	rand_array_1 = (int*)malloc(max * sizeof(int));
	rand_array_2 = (int*)malloc(max * sizeof(int));
	if (!reference || !rand_array_1 || !rand_array_2)
		fprintf(stderr, "error: can not allocate memory\n");
	srand (time(NULL));
	for(int i = 1; i < max; i++){     
		rand_array_1[i] = rand() % 10 + 1;
		rand_array_2[i] = rand() % 10 + 1;
	}
	for (int i = 1; i < max; i++)
		for (int j = 1; j < max; j++)
			reference[i*max+j] = blosum62[rand_array_1[i]][rand_array_2[j]];
	free(rand_array_1);
	free(rand_array_2);

	sequential(max, penalty, reference, &sequential_time);
	parallel(max, penalty, reference, &parallel_time);
	
	printf("Sequential time: %.15f ms\n", sequential_time);
	printf("Parallel time: %.15f ms\n", parallel_time);
	printf("Speedup: %.15f\n", sequential_time / parallel_time);

	if (compare_files("result_parallel", "result_sequential"))
		printf("TEST PASSED\n");
	else
		printf("TEST FAILED\n");

	free(reference);
    return 0;
}

void sequential(int max, int penalty, int* reference, float* time) 
{
    int idx, index;
    int *input_itemsets;

	START_TIMER
    input_itemsets = (int*)calloc(max * max, sizeof(int));
	
	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory\n");

    for(int i = 1; i< max; i++)
       input_itemsets[i*max] = -i * penalty;
	for(int j = 1; j< max; j++)
       input_itemsets[j] = -j * penalty;

    for(int i = 0; i < max-2; i++){
		for(idx = 0; idx <= i; idx++){
			index = (idx + 1) * max + (i + 1 - idx);
			input_itemsets[index] = maximum(input_itemsets[index-1-max] + reference[index], input_itemsets[index-1] - penalty, input_itemsets[index-max]  - penalty);
		}
	}
 	for(int i = max - 4; i >= 0; i--){
        for( idx = 0 ; idx <= i ; idx++){
			index =  ( max - idx - 2 ) * max + idx + max - i - 2 ;
			input_itemsets[index] = maximum(input_itemsets[index-1-max] + reference[index], input_itemsets[index-1] - penalty, input_itemsets[index-max]  - penalty);
	    }
	}

#ifdef TESTING
	printf("\nSequential\n");
	for(int i = 0; i < max; i++)
	{
		for(int j = 0; j < max; j++)
			printf("%5d ", input_itemsets[i * max + j]);
		putchar('\n');
	}
#endif

	traceback("result_sequential", input_itemsets, reference, max, penalty);
	free(input_itemsets);
	END_TIMER
}

__global__ void kernel_1(int n, int* gdata, int max, int penalty, int* reference)
{
	// extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int index;
	
	if(idx < n)
	{
		index = (idx + 1) * max + (n - idx);
		// sdata[2 * tid] = gdata[index - max];
		// sdata[2 * tid + 1] = gdata[index - max - 1];
		// if(tid == blockDim.x - 1 || (blockIdx.x == gridDim.x - 1 && idx == n - 1))
		// 	sdata[2 * tid + 2] = gdata[index - 1];
		// __syncthreads();

		// gdata[index] = maximum(sdata[2 * tid] - penalty, sdata[2 * tid + 1] + reference[index], sdata[2 * tid + 2] - penalty);
		gdata[index] = maximum(gdata[index-1-max] + reference[index], gdata[index-1] - penalty, gdata[index-max] - penalty);
	}
}

__global__ void kernel_2(int n, int* gdata, int max, int penalty, int* reference)
{
	// extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int index;

	if(idx < n)
	{
		index = (max - idx - 2) * max + idx + max - n - 1;
		// sdata[2 * tid] = gdata[index - 1];
		// sdata[2 * tid + 1] = gdata[index - max - 1];
		// if(tid == blockDim.x - 1 || (blockIdx.x == gridDim.x - 1 && idx == n - 1))
		// 	sdata[2 * tid + 2] = gdata[index - max];
		// __syncthreads();

		// gdata[index] = maximum(sdata[2 * tid] - penalty, sdata[2 * tid + 1] + reference[index], sdata[2 * tid + 2] - penalty);
		gdata[index] = maximum(gdata[index-1-max] + reference[index], gdata[index-1] - penalty, gdata[index-max] - penalty);
	}
}

void parallel(int max, int penalty, int* reference_host, float* time) 
{
    int idx, index, grid_size;
    int *input_host, *input_device, *reference_device;

	START_TIMER
	input_host = (int*)calloc(max * max, sizeof(int));
	cudaMalloc(&input_device, max * max * sizeof(int));
	cudaMalloc(&reference_device, max * max * sizeof(int));
	
	if (!input_host || !input_device)
		fprintf(stderr, "error: can not allocate memory\n");

    for(int i = 1; i < max; i++)
       input_host[i * max] = -i * penalty;
	for(int j = 1; j < max; j++)
       input_host[j] = -j * penalty;

	cudaMemcpy(input_device, input_host, max * max * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(reference_device, reference_host, max * max * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < max - 2; i++){
		grid_size = (i + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
		kernel_1<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE + 1>>>(i + 1, input_device, max, penalty, reference_device);
	}
 	for(int i = max - 4; i >= 0; i--){
		grid_size = (i + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
		kernel_2<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE + 1>>>(i + 1, input_device, max, penalty, reference_device);
	}

	cudaMemcpy(input_host, input_device, max * max * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef TESTING
	printf("\nParallel\n");
	for(int i = 0; i < max; i++)
	{
		for(int j = 0; j < max; j++)
			printf("%5d ", input_host[i * max + j]);
		putchar('\n');
	}
#endif

	traceback("result_parallel", input_host, reference_host, max, penalty);
	free(input_host);
	cudaFree(input_device);
	cudaFree(reference_device);
	END_TIMER
}

void traceback(const char* filename, int* input_itemsets, int* reference, int max, int penalty)
{
	FILE *fpo = fopen(filename, "w");
    
	for (int i = max - 2, j = max - 2; i>=0, j>=0;){
		int nw, n, w, traceback;
		if ( i == max - 2 && j == max - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max + j]);
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max + j - 1];
		    w  = input_itemsets[ i * max + j - 1 ];
            n  = input_itemsets[(i - 1) * max + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = input_itemsets[ i * max + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max + j];
		}

		int new_nw, new_w, new_n;
		new_nw = nw + reference[i * max + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}
	}
	
	fclose(fpo);
}

int maximum(int a, int b, int c)
{
	int k;
	if(a <= b)
		k = b;
	else 
		k = a;
	if(k <= c)
		return c;
	else
		return k;
}

bool compare_files(const char *filename1, const char *filename2)
{
	FILE *file1 = fopen(filename1, "r");
	FILE *file2 = fopen(filename2, "r");

	bool ret = true;
	char c1, c2;
	do
	{
		c1 = fgetc(file1);
		c2 = fgetc(file2);
		if (c1 != c2)
		{
			ret = false;
			break;
		}
	} while (c1 != EOF && c2 != EOF);
	if (c1 != EOF || c2 != EOF)
		ret = false;

	fclose(file1);
	fclose(file2);
	return ret;
}



