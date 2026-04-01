#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {
    int nx_configs[] = {250, 500, 1000};
    int ny_configs[] = {100, 200, 400};
    int thread_counts[] = {1, 2, 4, 8, 16};
    
    NUM_Points = 14000000;
    Maxiter = 10;

    printf("=== Experiment 02: Parallel Scalability (Approach 2: Immediate) ===\n");

    for (int g = 0; g < 3; g++) {
        NX = nx_configs[g];
        NY = ny_configs[g];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int t = 0; t < 5; t++) {
            int num_threads = thread_counts[t];
            omp_set_num_threads(num_threads);

            double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *)calloc(NUM_Points, sizeof(Points));
            
            if (!mesh_value || !points) {
                printf("Memory allocation failed for Grid %dx%d, Threads %d\n", NX, NY, num_threads);
                if (mesh_value) free(mesh_value);
                if (points) free(points);
                continue;
            }
            
            initializepoints(points);

            char filename[150];
            sprintf(filename, "parallel_grid%d_imm_%dthreads.txt", g + 1, num_threads);
            FILE *out = fopen(filename, "w");

            fprintf(out, "=== Experiment 02: Parallel Scalability (Immediate) ===\n");
            fprintf(out, "Grid: %dx%d, Particles: %d, Threads: %d\n", NX, NY, NUM_Points, num_threads);
            fprintf(out, "Iter\tInterp(s)\tMover(s)\tTotal(s)\tDeleted\n");

            printf("\n------------------------------------------------\n");
            printf("Grid: %dx%d, Particles: %d, Threads: %d, Approach: Immediate\n", 
                   NX, NY, NUM_Points, num_threads);
            printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\tDeleted\n");

            double t_interp_acc = 0, t_mover_acc = 0;
            int total_del = 0;

            for (int iter = 0; iter < Maxiter; iter++) {
                double s_int = omp_get_wtime();
                interpolation(mesh_value, points);
                double e_int = omp_get_wtime();

                int del_count = 0;
                double s_mov = omp_get_wtime();
                if (num_threads == 1) {
                    mover_serial_immediate(points, dx, dy, &del_count);
                } else {
                    mover_parallel_immediate(points, dx, dy, &del_count);
                }
                double e_mov = omp_get_wtime();

                double interp_time = e_int - s_int;
                double mover_time = e_mov - s_mov;
                double total = interp_time + mover_time;

                t_interp_acc += interp_time;
                t_mover_acc += mover_time;
                total_del += del_count;

                fprintf(out, "%d\t%lf\t%lf\t%lf\t%d\n", iter + 1, interp_time, mover_time, total, del_count);
                printf("%d\t%lf\t%lf\t%lf\t%d\n", iter + 1, interp_time, mover_time, total, del_count);
            }

            fprintf(out, "\nSummary:\n");
            fprintf(out, "Total interpolation time: %lf seconds\n", t_interp_acc);
            fprintf(out, "Total mover time: %lf seconds\n", t_mover_acc);
            fprintf(out, "Total execution time: %lf seconds\n", t_interp_acc + t_mover_acc);
            fprintf(out, "Total particles deleted: %d\n", total_del);
            fclose(out);

            free(mesh_value);
            free(points);
        }
    }

    return 0;
}
