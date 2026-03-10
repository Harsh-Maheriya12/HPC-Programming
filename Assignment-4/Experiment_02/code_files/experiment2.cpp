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

    NUM_Points = 100000000;
    Maxiter = 10;

    printf("=== EXPERIMENT 2: Consistency Across Configurations ===\n");
    printf("Config, NX, NY, NumPoints, TotalInterpTime(s)\n");

    Points *points = (Points *)malloc(NUM_Points * sizeof(Points));
    if (!points) {
        printf("Memory allocation failed for points\n");
        return 1;
    }

    srand(42);
    initializepoints(points);

    for (int c = 0; c < 3; c++) {
        NX = nx_configs[c];
        NY = ny_configs[c];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
        if (!mesh_value) {
            printf("Memory allocation failed for mesh\n");
            free(points);
            return 1;
        }

        double total_interp_time = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {
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

        free(mesh_value);
    }

    free(points);
    return 0;
}
