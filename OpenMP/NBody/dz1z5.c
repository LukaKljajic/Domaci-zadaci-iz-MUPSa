#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define DIM 2 /* Two-dimensional system */
#define X 0   /* x-coordinate subscript */
#define Y 1   /* y-coordinate subscript */

const double G = 6.673e-11;
const double ACCURACY = 0.01;

typedef double vect_t[DIM]; /* Vector type for position, etc. */

struct particle_s
{
   double m; /* Mass     */
   vect_t s; /* Position */
   vect_t v; /* Velocity */
};

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p, double *delta_t_p, char *g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], int n);
void parallel_Compute_forces(vect_t forces[], struct particle_s curr[], int n, omp_lock_t *lockvars);
void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p);
void parallel_Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p);
void sequential_code(int n, int n_steps, double delta_t, struct particle_s *curr, double *time, double *kinetic_energy, double *potential_energy);
void parallel_code(int n, int n_steps, double delta_t, struct particle_s *curr, double *time, double *kinetic_energy, double *potential_energy);
int verify_solution(int n, struct particle_s curr_sequential[], struct particle_s curr_parallel[], double kinetic_energy_sequential, double kinetic_energy_parallel, double potential_energy_sequential, double potential_energy_parallel);
inline int check_doubles_with_accuracy(double first, double second, double accuracy);

int main(int argc, char *argv[])
{
   int n;                              /* Number of particles        */
   int n_steps;                        /* Number of timesteps        */
   double delta_t;                     /* Size of timestep           */
   char g_i;                           /*_G_en or _i_nput init conds */
   struct particle_s *curr_sequential; /* Current state in sequential processing */
   struct particle_s *curr_parallel;   /* Current state in parallel processing */
   double time_parallel, time_sequential, kinetic_energy_sequential, kinetic_energy_parallel, potential_energy_sequential, potential_energy_parallel;

   Get_args(argc, argv, &n, &n_steps, &delta_t, &g_i);
   curr_sequential = malloc(n * sizeof(struct particle_s));
   curr_parallel = malloc(n * sizeof(struct particle_s));

   if (g_i == 'i')
      Get_init_cond(curr_sequential, n);
   else
      Gen_init_cond(curr_sequential, n);

   memcpy(curr_parallel, curr_sequential, n * sizeof(struct particle_s));

   sequential_code(n, n_steps, delta_t, curr_sequential, &time_sequential, &kinetic_energy_sequential, &potential_energy_sequential);
   parallel_code(n, n_steps, delta_t, curr_parallel, &time_parallel, &kinetic_energy_parallel, &potential_energy_parallel);
   printf("Sequential elapsed time = %.14f seconds\n", time_sequential);
   printf("Parallel elapsed time = %.14f seconds\n", time_parallel);
   printf("Speedup is %.14f\n", time_sequential / time_parallel);

   if (verify_solution(n, curr_sequential, curr_parallel, kinetic_energy_sequential, kinetic_energy_parallel, potential_energy_sequential, potential_energy_parallel))
      printf("TEST PASSED\n");
   else
      printf("TEST FAILED\n");

   free(curr_sequential);
   free(curr_parallel);
   return 0;
} /* main */

void Usage(char *prog_name)
{
   fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
           prog_name);
   fprintf(stderr, "   <size of timestep> <output frequency>\n");
   fprintf(stderr, "   <g|i>\n");
   fprintf(stderr, "   'g': program should generate init conds\n");
   fprintf(stderr, "   'i': program should get init conds from stdin\n");

   exit(0);
} /* Usage */

void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p, double *delta_t_p, char *g_i_p)
{
   if (argc != 6)
      Usage(argv[0]);
   *n_p = strtol(argv[1], NULL, 10);
   *n_steps_p = strtol(argv[2], NULL, 10);
   *delta_t_p = strtod(argv[3], NULL);
   *g_i_p = argv[5][0];

   if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
      Usage(argv[0]);
   if (*g_i_p != 'g' && *g_i_p != 'i')
      Usage(argv[0]);

} /* Get_args */

void Get_init_cond(struct particle_s curr[], int n)
{
   int part;

   printf("For each particle, enter (in order):\n");
   printf("   its mass, its x-coord, its y-coord, ");
   printf("its x-velocity, its y-velocity\n");
   for (part = 0; part < n; part++)
   {
      if (!scanf("%lf", &curr[part].m))
         fprintf(stderr, "Los unos\n");
      if (!scanf("%lf", &curr[part].s[X]))
         fprintf(stderr, "Los unos\n");
      if (!scanf("%lf", &curr[part].s[Y]))
         fprintf(stderr, "Los unos\n");
      if (!scanf("%lf", &curr[part].v[X]))
         fprintf(stderr, "Los unos\n");
      if (!scanf("%lf", &curr[part].v[Y]))
         fprintf(stderr, "Los unos\n");
   }
} /* Get_init_cond */

void Gen_init_cond(struct particle_s curr[], int n)
{
   int part;
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   srandom(1);
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

void Compute_force(int part, vect_t forces[], struct particle_s curr[], int n)
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

void parallel_Compute_forces(vect_t forces[], struct particle_s curr[], int n, omp_lock_t *lockvars)
{
   int k, part;
   double mg;
   vect_t f_part_k;
   double len, len_3, fact;

#pragma omp parallel for private(part, k, f_part_k, len, len_3, fact, mg) schedule(static, 8)
   for (part = 0; part < n - 1; part++)
   {
      vect_t part_add_task = {0};
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

         part_add_task[X] += f_part_k[X];
         part_add_task[Y] += f_part_k[Y];
         omp_set_lock(&lockvars[k]);
         forces[k][X] -= f_part_k[X];
         forces[k][Y] -= f_part_k[Y];
         omp_unset_lock(&lockvars[k]);
      }
      omp_set_lock(&lockvars[part]);
      forces[part][X] += part_add_task[X];
      forces[part][Y] += part_add_task[Y];
      omp_unset_lock(&lockvars[part]);
   }
} /* Compute_force */

void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t)
{
   double fact = delta_t / curr[part].m;

   curr[part].s[X] += delta_t * curr[part].v[X];
   curr[part].s[Y] += delta_t * curr[part].v[Y];
   curr[part].v[X] += fact * forces[part][X];
   curr[part].v[Y] += fact * forces[part][Y];
} /* Update_part */

void Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p)
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

void parallel_Compute_energy(struct particle_s curr[], int n, double *kin_en_p, double *pot_en_p)
{
   int i, j;
   vect_t diff;
   double pe = 0.0, ke = 0.0;
   double dist, speed_sqr;

#pragma omp parallel for private(i, j) reduction(+: ke) reduction(+: pe) schedule(static, 8)
   for (i = 0; i < n; i++)
   {
      speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
      ke += curr[i].m * speed_sqr;
      if (i != n - 1)
         for (j = i + 1; j < n; j++)
         {
            diff[X] = curr[i].s[X] - curr[j].s[X];
            diff[Y] = curr[i].s[Y] - curr[j].s[Y];
            dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
            pe += -G * curr[i].m * curr[j].m / dist;
         }
   }
   ke *= 0.5;

   *kin_en_p = ke;
   *pot_en_p = pe;
} /* Compute_energy */

void sequential_code(int n, int n_steps, double delta_t, struct particle_s *curr, double *time, double *kinetic_energy, double *potential_energy)
{
   int step;       /* Current step               */
   int part;       /* Current particle           */
   vect_t *forces; /* Forces on each particle    */
   double start = omp_get_wtime(), finish;

   forces = malloc(n * sizeof(vect_t));

   Compute_energy(curr, n, kinetic_energy, potential_energy);
   printf(" PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   for (step = 1; step <= n_steps; step++)
   {
      memset(forces, 0, n * sizeof(vect_t));
      for (part = 0; part < n - 1; part++)
         Compute_force(part, forces, curr, n);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      Compute_energy(curr, n, kinetic_energy, potential_energy);
   }

   printf(" PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);

   finish = omp_get_wtime();
   *time = finish - start;

   free(forces);
}

void parallel_code(int n, int n_steps, double delta_t, struct particle_s *curr, double *time, double *kinetic_energy, double *potential_energy)
{
   int step;       /* Current step               */
   int part;       /* Current particle           */
   vect_t *forces; /* Forces on each particle    */
   double start = omp_get_wtime(), finish;
   omp_lock_t *lockvars;

   lockvars = malloc(n * sizeof(omp_lock_t));
   forces = malloc(n * sizeof(vect_t));
   for (int i = 0; i < n; i++)
      omp_init_lock(&lockvars[i]);

   Compute_energy(curr, n, kinetic_energy, potential_energy);
   printf(" PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);
   for (step = 1; step <= n_steps; step++)
   {
      memset(forces, 0, n * sizeof(vect_t));
      parallel_Compute_forces(forces, curr, n, lockvars);
      for (part = 0; part < n; part++)
         Update_part(part, forces, curr, n, delta_t);
      parallel_Compute_energy(curr, n, kinetic_energy, potential_energy);
   }

   printf(" PE = %e, KE = %e, Total Energy = %e\n", *potential_energy, *kinetic_energy, *kinetic_energy + *potential_energy);

   finish = omp_get_wtime();
   *time = finish - start;

   free(forces);
   free(lockvars);
}

int verify_solution(int n, struct particle_s curr_sequential[], struct particle_s curr_parallel[], double kinetic_energy_sequential, double kinetic_energy_parallel, double potential_energy_sequential, double potential_energy_parallel)
{
   if (!check_doubles_with_accuracy(kinetic_energy_sequential, kinetic_energy_parallel, ACCURACY))
      return 0;
   if (!check_doubles_with_accuracy(potential_energy_sequential, potential_energy_parallel, ACCURACY))
      return 0;
   for (int i = 0; i < n; i++)
   {
      if (!check_doubles_with_accuracy(curr_sequential[i].m, curr_parallel[i].m, ACCURACY))
         return 0;
      for (int j = 0; j < DIM; j++)
      {
         if(!check_doubles_with_accuracy(curr_sequential[i].s[j], curr_parallel[i].s[j], ACCURACY))
            return 0;
         if(!check_doubles_with_accuracy(curr_sequential[i].v[j], curr_parallel[i].v[j], ACCURACY))
            return 0;
      }
   }
   return 1;
}

inline int check_doubles_with_accuracy(double first, double second, double accuracy)
{
   return (fabs(1 - second / first) < accuracy);
}