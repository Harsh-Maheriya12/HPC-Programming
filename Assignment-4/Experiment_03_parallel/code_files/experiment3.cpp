#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {
    NX = 1000;
    NY = 400;
    NUM_Points = 14000000;
    Maxiter = 10;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    Points *points = (Points *)malloc(NUM_Points * sizeof(Points));
    double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    unsigned int *seeds = (unsigned int *)malloc(NUM_Points * sizeof(unsigned int));

    if (!points || !mesh_value || !seeds) {
        printf("Memory allocation failed\n");
        return 1;
    }

    srand(42);
    initializepoints(points);

    for (int i = 0; i < NUM_Points; i++) {
        seeds[i] = (unsigned int)(rand());
    }

    printf("=== EXPERIMENT 3: Mover Operation ===\n\n");

    printf("--- SERIAL ---\n");
    printf("Iter, InterpTime(s), MoverTime(s), TotalTime(s)\n");

    double serial_mover_total = 0.0;

    srand(42);
    initializepoints(points);
    for (int i = 0; i < NUM_Points; i++)
        seeds[i] = (unsigned int)(rand());

    for (int iter = 0; iter < Maxiter; iter++) {
        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        struct timespec t1, t2, t3;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        interpolation(mesh_value, points);
        clock_gettime(CLOCK_MONOTONIC, &t2);

        mover_serial(points, seeds);
        clock_gettime(CLOCK_MONOTONIC, &t3);

        double interp_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        double mover_time = (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) / 1e9;
        double total_time = interp_time + mover_time;

        serial_mover_total += mover_time;

        printf("%d, %lf, %lf, %lf\n", iter + 1, interp_time, mover_time, total_time);
    }

    printf("Total serial mover time: %lf\n\n", serial_mover_total);

    printf("--- PARALLEL (4 threads) ---\n");
    printf("Iter, InterpTime(s), MoverTime(s), TotalTime(s)\n");

    double parallel_mover_total = 0.0;

    srand(42);
    initializepoints(points);
    for (int i = 0; i < NUM_Points; i++)
        seeds[i] = (unsigned int)(rand());

    for (int iter = 0; iter < Maxiter; iter++) {
        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

        struct timespec t1, t2, t3;

        clock_gettime(CLOCK_MONOTONIC, &t1);
        interpolation(mesh_value, points);
        clock_gettime(CLOCK_MONOTONIC, &t2);

        mover_parallel(points, seeds);
        clock_gettime(CLOCK_MONOTONIC, &t3);

        double interp_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        double mover_time = (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) / 1e9;
        double total_time = interp_time + mover_time;

        parallel_mover_total += mover_time;

        printf("%d, %lf, %lf, %lf\n", iter + 1, interp_time, mover_time, total_time);
    }

    printf("Total parallel mover time: %lf\n", parallel_mover_total);
    printf("Speedup: %lf\n", serial_mover_total / parallel_mover_total);

    save_mesh(mesh_value);

    free(points);
    free(mesh_value);
    free(seeds);

    return 0;
}
