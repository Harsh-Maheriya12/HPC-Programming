#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>
#include "utils.h"

// Persistent buffers to avoid allocation overhead during benchmarking
static double **global_thread_meshes = NULL;
static int cached_grid_size = 0;
static int cached_max_threads = 0;

static void cleanup_buffers() {
    if (global_thread_meshes) {
        for (int i = 0; i < cached_max_threads; i++) free(global_thread_meshes[i]);
        free(global_thread_meshes);
        global_thread_meshes = NULL;
    }
}

// Internal: Atomic implementation used by Optimized Hybrid
static void internal_interpolation_atomic(double *mesh_value, PointsSoA *points) {
    const double inv_dx = 1.0 / dx; const double inv_dy = 1.0 / dy; const double area = dx * dy;
    #pragma omp parallel for schedule(guided, 1024)
    for (int p = 0; p < NUM_Points; p++) {
        if (p + 16 < NUM_Points) {
            __builtin_prefetch(&points->x[p+16], 0, 3);
            __builtin_prefetch(&points->y[p+16], 0, 3);
        }
        int i = (int)(points->x[p] * inv_dx); int j = (int)(points->y[p] * inv_dy);
        if (i >= NX) i = NX - 1; if (j >= NY) j = NY - 1;
        double rx = (points->x[p] - i * dx) * inv_dx; double ry = (points->y[p] - j * dy) * inv_dy;
        int base_idx = j * GRID_X + i;
        #pragma omp atomic
        mesh_value[base_idx] += (1.0 - rx) * (1.0 - ry) * area;
        #pragma omp atomic
        mesh_value[base_idx + 1] += rx * (1.0 - ry) * area;
        #pragma omp atomic
        mesh_value[base_idx + GRID_X] += (1.0 - rx) * ry * area;
        #pragma omp atomic
        mesh_value[base_idx + GRID_X + 1] += rx * ry * area;
    }
}

// Internal: Privatization implementation used by Optimized Hybrid
static void internal_interpolation_privatization(double *mesh_value, PointsSoA *points) {
    const double inv_dx = 1.0 / dx; const double inv_dy = 1.0 / dy; const double area = dx * dy;
    int grid_size = GRID_X * GRID_Y;
    int max_threads = omp_get_max_threads();

    if (!global_thread_meshes || cached_grid_size != grid_size || cached_max_threads != max_threads) {
        cleanup_buffers();
        global_thread_meshes = (double **)malloc(max_threads * sizeof(double *));
        for (int i = 0; i < max_threads; i++) global_thread_meshes[i] = (double *)calloc(grid_size, sizeof(double));
        cached_grid_size = grid_size; cached_max_threads = max_threads;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        double *local_mesh = global_thread_meshes[tid];
        memset(local_mesh, 0, grid_size * sizeof(double));

        #pragma omp for nowait schedule(static)
        for (int p = 0; p < NUM_Points; p++) {
            int i = (int)(points->x[p] * inv_dx); int j = (int)(points->y[p] * inv_dy);
            if (i >= NX) i = NX - 1; if (j >= NY) j = NY - 1;
            double rx = (points->x[p] - i * dx) * inv_dx; double ry = (points->y[p] - j * dy) * inv_dy;
            int base_idx = j * GRID_X + i;
            local_mesh[base_idx] += (1.0 - rx) * (1.0 - ry) * area;
            local_mesh[base_idx + 1] += rx * (1.0 - ry) * area;
            local_mesh[base_idx + GRID_X] += (1.0 - rx) * ry * area;
            local_mesh[base_idx + GRID_X + 1] += rx * ry * area;
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int k = 0; k < grid_size; k++) {
            double sum = 0;
            for (int t = 0; t < nthreads; t++) sum += global_thread_meshes[t][k];
            mesh_value[k] += sum;
        }
    }
}

// 1. SERIAL VERSION
void interpolation_serial(double *mesh_value, PointsSoA *points) {
    const double inv_dx = 1.0 / dx; const double inv_dy = 1.0 / dy; const double area = dx * dy;
    for (int p = 0; p < NUM_Points; p++) {
        double px = points->x[p]; double py = points->y[p];
        int i = (int)(px * inv_dx); int j = (int)(py * inv_dy);
        if (i >= NX) i = NX - 1; if (j >= NY) j = NY - 1;
        double rx = (px - i * dx) * inv_dx; double ry = (py - j * dy) * inv_dy;
        int base_idx = j * GRID_X + i;
        mesh_value[base_idx] += (1.0 - rx) * (1.0 - ry) * area;
        mesh_value[base_idx + 1] += rx * (1.0 - ry) * area;
        mesh_value[base_idx + GRID_X] += (1.0 - rx) * ry * area;
        mesh_value[base_idx + GRID_X + 1] += rx * ry * area;
    }
}

// 2. ADAPTIVE HYBRID VERSION
void interpolation_hybrid(double *mesh_value, PointsSoA *points) {
    int grid_size = GRID_X * GRID_Y;
    // Threshold: If grid is larger than 12MB (L3 cache limit), atomics win.
    if (grid_size * sizeof(double) > 12 * 1024 * 1024) {
        internal_interpolation_atomic(mesh_value, points);
    } else {
        internal_interpolation_privatization(mesh_value, points);
    }
}

void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) return;
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
}
