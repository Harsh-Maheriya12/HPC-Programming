#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {
    int nx_configs[] = {250, 500, 1000};
    int ny_configs[] = {100, 200, 400};
    int point_counts[] = {100, 10000, 1000000, 100000000};
    int num_point_configs = 4;

    Maxiter = 10;

    printf("=== EXPERIMENT 1: Scaling with Number of Particles ===\n");
    printf("Config, NX, NY, NumPoints, TotalInterpTime(s)\n");

    for (int c = 0; c < 3; c++) {
        NX = nx_configs[c];
        NY = ny_configs[c];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int pc = 0; pc < num_point_configs; pc++) {
            NUM_Points = point_counts[pc];

            Points *points = (Points *)malloc(NUM_Points * sizeof(Points));
            double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));

            if (!points || !mesh_value) {
                printf("Memory allocation failed for NX=%d, NY=%d, Points=%d\n",
                       NX, NY, NUM_Points);
                free(points);
                free(mesh_value);
                continue;
            }

            double total_interp_time = 0.0;
            srand(42);

            for (int iter = 0; iter < Maxiter; iter++) {
                initializepoints(points);
                memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

                struct timespec t1, t2;
                clock_gettime(CLOCK_MONOTONIC, &t1);

                interpolation(mesh_value, points);

                clock_gettime(CLOCK_MONOTONIC, &t2);

                double elapsed = (t2.tv_sec - t1.tv_sec) +
                                 (t2.tv_nsec - t1.tv_nsec) / 1e9;
                total_interp_time += elapsed;
            }

            printf("%d, %d, %d, %d, %lf\n", c + 1, NX, NY, NUM_Points, total_interp_time);

            free(points);
            free(mesh_value);
        }
    }

    return 0;
}
