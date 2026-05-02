#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

// Persistent buffer to avoid expensive calloc/free in every iteration
static double* all_local_meshes = NULL;
static size_t current_buf_size = 0;

// Optimized interpolation using thread privatization to eliminate atomic contention
void interpolation(double * __restrict mesh_value, Points * __restrict points) {
    int grid_size = GRID_X * GRID_Y;
    int num_threads = omp_get_max_threads();
    size_t required_size = (size_t)num_threads * grid_size;

    // Allocate or resize buffer once
    if (all_local_meshes == NULL || current_buf_size < required_size) {
        if (all_local_meshes) free(all_local_meshes);
        all_local_meshes = (double *)malloc(required_size * sizeof(double));
        current_buf_size = required_size;
    }

    // Re-zeroing the persistent buffer is much faster than re-allocating
    memset(all_local_meshes, 0, required_size * sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double * __restrict local_mesh = all_local_meshes + (size_t)tid * grid_size;

        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++) {
            if (points[i].is_void) continue;

            double px = points[i].x;
            double py = points[i].y;

            // Fast index calculation assuming domain [0,1]
            int ix = (int)(px * NX);
            int iy = (int)(py * NY);

            if (ix >= NX) ix = NX - 1; if (iy >= NY) iy = NY - 1;
            if (ix < 0) ix = 0; if (iy < 0) iy = 0;

            double lx = px - ix * dx;
            double ly = py - iy * dy;
            double dxlx = dx - lx;
            double dyly = dy - ly;

            int idx = iy * GRID_X + ix;
            
            local_mesh[idx]           += dxlx * dyly;
            local_mesh[idx + 1]       += lx   * dyly;
            local_mesh[idx + GRID_X]   += dxlx * ly;
            local_mesh[idx + GRID_X + 1] += lx   * ly;
        }
    }

    // High-performance parallel reduction
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grid_size; i++) {
        double sum = 0.0;
        for (int t = 0; t < num_threads; t++) {
            sum += all_local_meshes[(size_t)t * grid_size + i];
        }
        mesh_value[i] = sum;
    }
}

void normalization(double * __restrict mesh_value) {
    int sz = GRID_X * GRID_Y;
    double lmin = 1e30, lmax = -1e30;

    #pragma omp parallel for reduction(min:lmin) reduction(max:lmax)
    for (int i = 0; i < sz; i++) {
        if (mesh_value[i] < lmin) lmin = mesh_value[i];
        if (mesh_value[i] > lmax) lmax = mesh_value[i];
    }

    min_val = lmin;
    max_val = lmax;
    double range = max_val - min_val;
    double inv = (range > 1e-15) ? (2.0 / range) : 0.0;

    #pragma omp parallel for
    for (int i = 0; i < sz; i++) {
        mesh_value[i] = (mesh_value[i] - min_val) * inv - 1.0;
    }
}

void mover(double * __restrict mesh_value, Points * __restrict points) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_Points; i++) {
        if (points[i].is_void) continue;

        double px = points[i].x;
        double py = points[i].y;

        int ix = (int)(px * NX);
        int iy = (int)(py * NY);

        if (ix >= NX) ix = NX - 1; if (iy >= NY) iy = NY - 1;
        if (ix < 0) ix = 0; if (iy < 0) iy = 0;

        double lx = px - (ix * dx);
        double ly = py - (iy * dy);
        double dxlx = dx - lx;
        double dyly = dy - ly;

        int idx = iy * GRID_X + ix;
        
        // Cache neighboring mesh values for better performance
        double m00 = mesh_value[idx];
        double m10 = mesh_value[idx + 1];
        double m01 = mesh_value[idx + GRID_X];
        double m11 = mesh_value[idx + GRID_X + 1];

        double Fi = dxlx * (dyly * m00 + ly * m01) + 
                    lx   * (dyly * m10 + ly * m11);

        points[i].x += Fi * dx;
        points[i].y += Fi * dy;

        if (points[i].x < 0.0 || points[i].x > 1.0 || 
            points[i].y < 0.0 || points[i].y > 1.0) {
            points[i].is_void = true;
        }
    }
}

void denormalization(double * __restrict mesh_value) {
    int sz = GRID_X * GRID_Y;
    double hr = (max_val - min_val) * 0.5;

    #pragma omp parallel for
    for (int i = 0; i < sz; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * hr + min_val;
    }
}

long long int void_count(Points * __restrict points) {
    long long int v = 0;
    #pragma omp parallel for reduction(+:v)
    for (int i = 0; i < NUM_Points; i++) {
        v += (long long int)points[i].is_void;
    }
    return v;
}

void save_mesh(double * __restrict mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) exit(1);
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
