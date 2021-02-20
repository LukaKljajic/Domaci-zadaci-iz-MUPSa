#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#define TRACEBACK

int maximum(int a, int b, int c)
{
	int k;
	if (a <= b)
		k = b;
	else
		k = a;
	if (k <= c)
		return (c);
	else
		return (k);
}

int blosum62[24][24] = {
	{4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4},
	{-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4},
	{-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4},
	{-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4},
	{0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4},
	{-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
	{0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4},
	{-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4},
	{-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4},
	{0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4},
	{0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4},
	{-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4},
	{-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4},
	{0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1}};

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv[0]);
	fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
	fprintf(stderr, "\t<penalty>        - penalty(positive integer)\n");
	exit(1);
}

void runTest(int max, int penalty, FILE *fpo, int *reference)
{
	int max_rows = max, max_cols = max, idx, index;
	int *input_itemsets;
	int size;
	int omp_num_threads;

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

	for (int i = 0; i < max_cols; i++)
	{
		for (int j = 0; j < max_rows; j++)
		{
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	for (int i = 1; i < max_rows; i++)
		input_itemsets[i * max_cols] = -i * penalty;
	for (int j = 1; j < max_cols; j++)
		input_itemsets[j] = -j * penalty;

	for (int i = 0; i < max_cols - 2; i++)
	{
		for (idx = 0; idx <= i; idx++)
		{
			index = (idx + 1) * max_cols + (i + 1 - idx);
			input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + reference[index], input_itemsets[index - 1] - penalty, input_itemsets[index - max_cols] - penalty);
		}
	}
	for (int i = max_cols - 4; i >= 0; i--)
	{
		for (idx = 0; idx <= i; idx++)
		{
			index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
			input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + reference[index], input_itemsets[index - 1] - penalty, input_itemsets[index - max_cols] - penalty);
		}
	}

#ifdef TRACEBACK

	for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
	{
		int nw, n, w, traceback;
		if (i == max_rows - 2 && j == max_rows - 2)
			fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
		if (i == 0 && j == 0)
			break;
		if (i > 0 && j > 0)
		{
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
			w = input_itemsets[i * max_cols + j - 1];
			n = input_itemsets[(i - 1) * max_cols + j];
		}
		else if (i == 0)
		{
			nw = n = LIMIT;
			w = input_itemsets[i * max_cols + j - 1];
		}
		else if (j == 0)
		{
			nw = w = LIMIT;
			n = input_itemsets[(i - 1) * max_cols + j];
		}

		int new_nw, new_w, new_n;
		new_nw = nw + reference[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;

		traceback = maximum(new_nw, new_w, new_n);
		if (traceback == new_nw)
			traceback = nw;
		if (traceback == new_w)
			traceback = w;
		if (traceback == new_n)
			traceback = n;

		fprintf(fpo, "%d ", traceback);

		if (traceback == nw)
		{
			i--;
			j--;
			continue;
		}

		else if (traceback == w)
		{
			j--;
			continue;
		}

		else if (traceback == n)
		{
			i--;
			continue;
		}
	}

#endif

	free(input_itemsets);
}

void runTestParallel(int max, int penalty, FILE *fpo, int *reference)
{
	int max_rows = max, max_cols = max, idx, index;
	int *input_itemsets = NULL;
	int size;
	int omp_num_threads;

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	input_itemsets = (int *)malloc(max_rows * max_cols * sizeof(int));

	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

#pragma omp parallel for
	for (int i = 0; i < max_cols; i++)
	{
		for (int j = 0; j < max_rows; j++)
		{
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	printf("Start Needleman-Wunsch\n");

	for (int i = 1; i < max_rows; i++)
		input_itemsets[i * max_cols] = -i * penalty;
	for (int j = 1; j < max_cols; j++)
		input_itemsets[j] = -j * penalty;

	for (int i = 0; i < max_cols - 2; i++)
	{
#pragma omp parallel for private(idx, index) 
		for (idx = 0; idx <= i; idx++)
		{
			index = (idx + 1) * max_cols + (i + 1 - idx);
			input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + reference[index], input_itemsets[index - 1] - penalty, input_itemsets[index - max_cols] - penalty);
		}
	}
	for (int i = max_cols - 4; i >= 0; i--)
	{
#pragma omp parallel for private(idx, index) 
		for (idx = 0; idx <= i; idx++)
		{
			index = (max_cols - idx - 2) * max_cols + idx + max_cols - i - 2;
			input_itemsets[index] = maximum(input_itemsets[index - 1 - max_cols] + reference[index], input_itemsets[index - 1] - penalty, input_itemsets[index - max_cols] - penalty);
		}
	}

#ifdef TRACEBACK

	for (int i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;)
	{
		int nw, n, w, traceback;
		if (i == max_rows - 2 && j == max_rows - 2)
			fprintf(fpo, "%d ", input_itemsets[i * max_cols + j]);
		if (i == 0 && j == 0)
			break;
		if (i > 0 && j > 0)
		{
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
			w = input_itemsets[i * max_cols + j - 1];
			n = input_itemsets[(i - 1) * max_cols + j];
		}
		else if (i == 0)
		{
			nw = n = LIMIT;
			w = input_itemsets[i * max_cols + j - 1];
		}
		else if (j == 0)
		{
			nw = w = LIMIT;
			n = input_itemsets[(i - 1) * max_cols + j];
		}

		int new_nw, new_w, new_n;
		new_nw = nw + reference[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;

		traceback = maximum(new_nw, new_w, new_n);
		if (traceback == new_nw)
			traceback = nw;
		if (traceback == new_w)
			traceback = w;
		if (traceback == new_n)
			traceback = n;

		fprintf(fpo, "%d ", traceback);

		if (traceback == nw)
		{
			i--;
			j--;
			continue;
		}

		else if (traceback == w)
		{
			j--;
			continue;
		}

		else if (traceback == n)
		{
			i--;
			continue;
		}
	}

#endif

	free(input_itemsets);
}

int compare_files(FILE *file1, FILE *file2)
{
	char c1, c2;
	do
	{
		c1 = fgetc(file1);
		c2 = fgetc(file2);
		if (c1 != c2)
			return 0;
	} while (c1 != EOF && c2 != EOF);
	if (c1 == EOF && c2 == EOF)
		return 1;
	else
		return 0;
}

int main(int argc, char **argv)
{
	int *reference = NULL, *array1, *array2;
	int max, penalty;
	double start, time_sequential, time_parallel;
	FILE *file_sequential = fopen("result_sequential", "w"), *file_parallel = fopen("result_parallel", "w");

	if (argc == 3)
	{
		max = atoi(argv[1]);
		penalty = atoi(argv[2]);
	}
	else
		usage(argc, argv);

	srand(time(NULL));
	reference = (int *)malloc((max + 1) * (max + 1) * sizeof(int));
	array1 = (int *)malloc(max * sizeof(int));
	array2 = (int *)malloc(max * sizeof(int));
	if (!reference || !array1 || !array2)
		fprintf(stderr, "error: can not allocate memory");
	for (int i = 0; i < max; i++)
	{
		array1[i] = rand() % 10 + 1;
		array2[i] = rand() % 10 + 1;
	}
#pragma omp parallel for
	for (int i = 1; i < max + 1; i++)
	{
		for (int j = 1; j < max + 1; j++)
		{
			reference[i * (max + 1) + j] = blosum62[array1[i]][array2[j]];
		}
	}

	start = omp_get_wtime();
	runTest(max, penalty, file_sequential, reference);
	time_sequential = omp_get_wtime() - start;
	printf("Sequential time = %.14f\n", time_sequential);

	start = omp_get_wtime();
	runTestParallel(max, penalty, file_parallel, reference);
	time_parallel = omp_get_wtime() - start;
	printf("Parallel time = %.14f\n", time_parallel);

	printf("Speedup = %.14f\n", time_sequential / time_parallel);

	fclose(file_sequential);
	fclose(file_parallel);

	file_sequential = fopen("result_sequential", "r");
	file_parallel = fopen("result_parallel", "r");

	if (compare_files(file_parallel, file_sequential))
		printf("TEST PASSED\n");
	else
		printf("TEST FAILED\n");

	fclose(file_sequential);
	fclose(file_parallel);
	free(reference);
	return EXIT_SUCCESS;
}
