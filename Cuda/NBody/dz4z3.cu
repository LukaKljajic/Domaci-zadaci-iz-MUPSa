#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */
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

const int GRID_SIZE = 192;
const int BLOCK_SIZE = 128;

const double G = 6.673e-11;
const double ACCURACY = 0.01;

typedef double vect_t[DIM]; /* Vector type for position, etc. */

struct particle_s
{
   double m; /* Mass     */
   vect_t s; /* Position */
   vect_t v; /* Velocity */
};

void Get_args(int, char *[], int *, int *, double *, int *, char *);
void Get_init_cond(struct particle_s*, int);
void Gen_init_cond(struct particle_s[], int);
void Compute_force(int, vect_t*, struct particle_s*, int);
void Update_part(int, vect_t*, struct particle_s*, int, double);
void Compute_energy(struct particle_s*, int, double *, double *);
void sequential(struct particle_s *, double *, double *, int, int, double, float *);
void parallel_compute_forces(vect_t*, struct particle_s*, int);
void parallel_update_parts(vect_t*, struct particle_s*, int, double);
void parallel_compute_energy(struct particle_s*, int, double *, double *);
void parallel(struct particle_s *, double *, double *, int n, int, double, float *);
int verify_solution(int, struct particle_s*, struct particle_s*, double, double, double, double);
int is_close(double, double, double);
__device__ void indexes(unsigned, unsigned, unsigned*, unsigned*);

int main(int argc, char *argv[])
{
   int n;                   /* Number of particles          */
   int n_steps;             /* Number of timesteps          */
   int output_freq;         /* Frequency of output          */
   double delta_t;           /* Size of timestep             */
   struct particle_s *sequential_curr, *parallel_curr;
   char g_i;                /* Generate or input init conds */
   double sequential_kinetic_energy, sequential_potential_energy;
   double parallel_kinetic_energy, parallel_potential_energy;
   float sequential_time, parallel_time;

   Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
   sequential_curr = (struct particle_s*)malloc(n * sizeof(struct particle_s));
   parallel_curr = (struct particle_s*)malloc(n * sizeof(struct particle_s));
   if (g_i == 'i')
      Get_init_cond(sequential_curr, n);
   else
      Gen_init_cond(sequential_curr, n);
   memcpy(parallel_curr, sequential_curr, n * sizeof(struct particle_s));

   sequential(sequential_curr, &sequential_kinetic_energy, &sequential_potential_energy, n, n_steps, delta_t, &sequential_time);
   parallel(parallel_curr, &parallel_kinetic_energy, &parallel_potential_energy, n, n_steps, delta_t, &parallel_time);
   printf("Sequential time: %.15f ms\n", sequential_time);
   printf("Parallel time: %.15f ms\n", parallel_time);
   printf("Speedup: %.15f\n", sequential_time / parallel_time);

   if(verify_solution(n, sequential_curr, parallel_curr, sequential_kinetic_energy, parallel_kinetic_energy, sequential_potential_energy, parallel_potential_energy))
      printf("TEST PASSED\n");
   else
      printf("TEST FAILED\n");

   free(sequential_curr);
   free(parallel_curr);
   return 0;
} /* main */

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p, double *delta_t_p, int *output_freq_p, char *g_i_p)
{
   if (argc != 6)
      printf("Bad arguments\n");
   *n_p = strtol(argv[1], NULL, 10);
   *n_steps_p = strtol(argv[2], NULL, 10);
   *delta_t_p = strtod(argv[3], NULL);
   *output_freq_p = strtol(argv[4], NULL, 10);
   *g_i_p = argv[5][0];

   if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
      printf("Bad arguments\n");
   if (*g_i_p != 'g' && *g_i_p != 'i')
      printf("Bad arguments\n");

} /* Get_args */

void Get_init_cond(struct particle_s* curr, int n)
{
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++)
   {
      scanf("%lf", &curr[part].m);
      scanf("%lf", &curr[part].s[X]);
      scanf("%lf", &curr[part].s[Y]);
      scanf("%lf", &curr[part].v[X]);
      scanf("%lf", &curr[part].v[Y]);
   }
} /* Get_init_cond */

void Gen_init_cond(struct particle_s* curr, int n)
{
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   for (part = 0; part < n; part++)
   {
      curr[part].m = mass;
      curr[part].s[X] = part * gap;
      curr[part].s[Y] = 0.0;
      curr[part].v[X] = 0.0;
      if (part % 2 == 0)
         curr[part].v[Y] = speed;
      else
         curr[part].v[Y] = -speed;
   }
} /* Gen_init_cond */

void Compute_force(int part, vect_t* forces, struct particle_s* curr, int n)
{
   int k;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

   for (k = part + 1; k < n; k++)
   {
      f_part_k[X] = curr[part].s[X] - curr[k].s[X];
      f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
      len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
      len_3 = len * len * len;
      mg = -G * curr[part].m * curr[k].m;
      fact = mg / len_3;
      f_part_k[X] *= fact;
      f_part_k[Y] *= fact;

      forces[part][X] += f_part_k[X];
      forces[part][Y] += f_part_k[Y];
      forces[k][X] -= f_part_k[X];
      forces[k][Y] -= f_part_k[Y];
   }
} /* Compute_force */

void Update_part(int part, vect_t* forces, struct particle_s* curr, int n, double delta_t)
{
   double fact = delta_t / curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
} /* Update_part */

void Compute_energy(struct particle_s* curr, int n, double *kin_en_p, double *pot_en_p)
{
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

   for (i = 0; i < n; i++)
   {
      speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
      ke += curr[i].m * speed_sqr;
   }
   ke *= 0.5;

   for (i = 0; i < n - 1; i++)
   {
      for (j = i + 1; j < n; j++)
      {
         diff[X] = curr[i].s[X] - curr[j].s[X];
         diff[Y] = curr[i].s[Y] - curr[j].s[Y];
         dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
         pe += -G * curr[i].m * curr[j].m / dist;
      }
   }

   *kin_en_p = ke;
   *pot_en_p = pe;
} /* Compute_energy */

void sequential(struct particle_s *curr, double* kinetic_energy, double* potential_energy, int n, int n_steps, double delta_t, float* time)
{
   START_TIMER
   vect_t *forces = (vect_t*)malloc(n * sizeof(vect_t));   
   Compute_energy(curr, n, kinetic_energy, potential_energy);
   printf("   PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   
   for (int step = 1; step <= n_steps; step++)
   {
      memset(forces, 0, n * sizeof(vect_t));
      for (int part = 0; part < n - 1; part++)
         Compute_force(part, forces, curr, n);
      for (int part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
   }

   Compute_energy(curr, n, kinetic_energy, potential_energy);
   printf("   PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   free(forces);
   END_TIMER
}

__global__ void kernel_forces(vect_t* forces, struct particle_s* curr, int n, vect_t* partial_forces)
{
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   vect_t to_add = {0, 0};
   struct particle_s my_curr = curr[idx];
   
   if(idx < n - 1)
   {
      for(int k = idx + 1; k < n; k++)
      {
         f_part_k[X] = my_curr.s[X] - curr[k].s[X];
         f_part_k[Y] = my_curr.s[Y] - curr[k].s[Y];
         len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
         len_3 = len * len * len;
         mg = -G * my_curr.m * curr[k].m;
         fact = mg / len_3;
         f_part_k[X] *= fact;
         f_part_k[Y] *= fact;
   
         to_add[X] += f_part_k[X];
         to_add[Y] += f_part_k[Y];
         partial_forces[k * n + idx][X] = -f_part_k[X];
         partial_forces[k * n + idx][Y] = -f_part_k[Y];
      }
      forces[idx][X] += to_add[X];
      forces[idx][Y] += to_add[Y];
   }
}

__global__ void kernel_add_partials(int n, vect_t* forces, vect_t* partial_forces)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if(idx < n)
   {
      for(int j = 0; j < n; j++)
      {
         if(j != idx)
         {
            forces[idx][X] += partial_forces[idx * n + j][X];
            forces[idx][Y] += partial_forces[idx * n + j][Y];
         }
      }
   }
}

void parallel_compute_forces(vect_t* forces, struct particle_s* curr, int n, vect_t* partial_forces)
{
   int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
   
   // cudaMemset(partial_forces, 0, n * n * sizeof(vect_t));

   kernel_forces<<<grid_size, BLOCK_SIZE>>>(forces, curr, n, partial_forces);
   kernel_add_partials<<<grid_size, BLOCK_SIZE>>>(n, forces, partial_forces);
}

__global__ void kernel_update(vect_t* forces, struct particle_s* curr, int n, double delta_t)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < n)
   {
      double fact = delta_t / curr[idx].m;
      curr[idx].s[X] += delta_t * curr[idx].v[X];
      curr[idx].s[Y] += delta_t * curr[idx].v[Y];
      curr[idx].v[X] += fact * forces[idx][X];
      curr[idx].v[Y] += fact * forces[idx][Y];
   }
}

void parallel_update_parts(vect_t* forces, struct particle_s* curr, int n, double delta_t)
{ // deluje da radi
   int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
   kernel_update<<<grid_size, BLOCK_SIZE>>>(forces, curr, n, delta_t);
}

__global__ void kernel_kinetic_energy(double* partial_sums, int n, struct particle_s* curr)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   for(int i = idx; i < n; i += gridDim.x * blockDim.x)
      partial_sums[idx] += curr[i].m * (curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y]);
}

__global__ void kernel_potential_energy(double* partial_sums, int n, struct particle_s* curr)
{
   unsigned idx = blockIdx.x * blockDim.x + threadIdx.x, i, j;
   unsigned size = n * (n - 1) / 2;
   vect_t diff;
   double dist;
   for(unsigned k = idx; k < size; k += gridDim.x * blockDim.x) 
   {
      indexes(k, n - 1, &i, &j);
      j++;
      diff[X] = curr[i].s[X] - curr[j].s[X];
      diff[Y] = curr[i].s[Y] - curr[j].s[Y];
      dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
      partial_sums[idx] += -G * curr[i].m * curr[j].m / dist;
   }
}

void parallel_compute_energy(struct particle_s* curr, int n, double *kin_en_p, double *pot_en_p)
{ // deluje da radi
   int i;
   double pe = 0.0, ke = 0.0;
   double* partial_sums_device, *partial_sums_host;

   partial_sums_host = (double*)malloc(GRID_SIZE * BLOCK_SIZE * sizeof(double));
   cudaMalloc(&partial_sums_device, GRID_SIZE * BLOCK_SIZE * sizeof(double));
   
   cudaMemset(partial_sums_device, 0, GRID_SIZE * BLOCK_SIZE * sizeof(double));
   kernel_kinetic_energy<<<GRID_SIZE, BLOCK_SIZE>>>(partial_sums_device, n, curr);
   cudaMemcpy(partial_sums_host, partial_sums_device, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
   // razmisliti o redukciji na gpu
   for(i = 0; i < GRID_SIZE * BLOCK_SIZE; i++)
      ke += partial_sums_host[i];
   ke *= 0.5;

   cudaMemset(partial_sums_device, 0, GRID_SIZE * BLOCK_SIZE * sizeof(double));
   kernel_potential_energy<<<GRID_SIZE, BLOCK_SIZE>>>(partial_sums_device, n, curr);
   cudaMemcpy(partial_sums_host, partial_sums_device, GRID_SIZE * BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
   // razmisliti o redukciji na gpu
   for(i = 0; i < GRID_SIZE * BLOCK_SIZE; i++)
      pe += partial_sums_host[i];

   *kin_en_p = ke;
   *pot_en_p = pe;
   free(partial_sums_host);
   cudaFree(partial_sums_device);
}

void parallel(struct particle_s *curr, double* kinetic_energy, double* potential_energy, int n, int n_steps, double delta_t, float* time)
{
   START_TIMER
   vect_t *forces, *forces_device, *partial_forces;  
   struct particle_s* curr_device;

   forces = (vect_t*)malloc(n * sizeof(vect_t));
   cudaMalloc(&forces_device, n * sizeof(vect_t));
   cudaMalloc(&curr_device, n * sizeof(struct particle_s));
   cudaMalloc(&partial_forces, n * n * sizeof(vect_t));
   cudaMemcpy(forces_device, forces, n * sizeof(vect_t), cudaMemcpyHostToDevice);
   cudaMemcpy(curr_device, curr, n * sizeof(struct particle_s), cudaMemcpyHostToDevice);
   
   parallel_compute_energy(curr_device, n, kinetic_energy, potential_energy);
   printf("   PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   
   for (int step = 1; step <= n_steps; step++)
   {
      cudaMemset(forces_device, 0, n * sizeof(vect_t));
      parallel_compute_forces(forces_device, curr_device, n, partial_forces);
      parallel_update_parts(forces_device, curr_device, n, delta_t);
   }
   
   parallel_compute_energy(curr_device, n, kinetic_energy, potential_energy);
   printf("   PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   
   cudaMemcpy(curr, curr_device, n * sizeof(struct particle_s), cudaMemcpyDeviceToHost);
   cudaFree(forces_device);
   cudaFree(curr_device);
   cudaFree(partial_forces);
   free(forces);
   END_TIMER
}

int verify_solution(int n, struct particle_s curr_sequential[], struct particle_s curr_parallel[], double kinetic_energy_sequential, double kinetic_energy_parallel, double potential_energy_sequential, double potential_energy_parallel)
{
   if (!is_close(kinetic_energy_sequential, kinetic_energy_parallel, ACCURACY))
      return 0;
   if (!is_close(potential_energy_sequential, potential_energy_parallel, ACCURACY))
      return 0;
   for (int i = 0; i < n; i++)
   {
      if (!is_close(curr_sequential[i].m, curr_parallel[i].m, ACCURACY))
         return 0;
      for (int j = 0; j < DIM; j++)
      {
         if(!is_close(curr_sequential[i].s[j], curr_parallel[i].s[j], ACCURACY))
            return 0;
         if(!is_close(curr_sequential[i].v[j], curr_parallel[i].v[j], ACCURACY))
            return 0;
      }
   }
   return 1;
}

inline int is_close(double first, double second, double accuracy)
{
   return (fabs(1 - second / first) < accuracy);
}

__device__ void indexes(unsigned k, unsigned N, unsigned* row_ptr, unsigned* col_ptr)
{
    double n = N;
    double row = (-2 * n - 1 + sqrt((4 * n * (n + 1) - 8 * (double)k - 7))) / -2;
    if (row == (double)(int)row)
        row -= 1;
    *row_ptr = (unsigned)row;
    *col_ptr = k - N * *row_ptr + *row_ptr * (*row_ptr + 1) / 2;
}