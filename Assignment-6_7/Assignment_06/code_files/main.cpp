#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h> 

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
                                                                                                                                                                                                  
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    // Read metadata from binary file
    if (fread(&NX, sizeof(int), 1, file) != 1) return 1;
    if (fread(&NY, sizeof(int), 1, file) != 1) return 1;
    if (fread(&NUM_Points, sizeof(int), 1, file) != 1) return 1;
    if (fread(&Maxiter, sizeof(int), 1, file) != 1) return 1;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;
    const long data_offset = 4L * sizeof(int);

    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    
    PointsSoA points;
    allocate_points(&points, NUM_Points);

    const int thread_counts[] = {2, 4, 8, 16};
    const int num_configs = sizeof(thread_counts) / sizeof(thread_counts[0]);

    omp_set_dynamic(0);

    for (int cfg = 0; cfg < num_configs; cfg++) {
        int num_threads = thread_counts[cfg];
        omp_set_num_threads(num_threads);

        // Reset file pointer to start of points data
        if (fseek(file, data_offset, SEEK_SET) != 0) {
            printf("Error resetting input file\n");
            free(mesh_value);
            free_points(&points);
            fclose(file);
            return 1;
        }

        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        printf("Threads: %d, Grid: %dx%d, Particles: %d, Iterations: %d\n", num_threads, NX, NY, NUM_Points, Maxiter);

        double total_time = 0.0;
        for (int iter = 0; iter < Maxiter; iter++) {
            read_points(file, &points);
            double start = omp_get_wtime();
            interpolation_hybrid(mesh_value, &points);
            double end = omp_get_wtime();

            double iter_time = end - start;
            total_time += iter_time;
            printf("Iter %d: %.6lf s\n", iter + 1, iter_time);
        }

        printf("Total: %.6lf s, Average: %.6lf s\n\n", total_time, total_time / Maxiter);

        if (cfg == num_configs - 1) {
            save_mesh(mesh_value);
        }
    }

    free(mesh_value);
    free_points(&points);
    fclose(file);

    return 0;
}
