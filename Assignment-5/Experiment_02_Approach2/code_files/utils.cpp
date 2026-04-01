#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "utils.h"

// Fast XORShift PRNG (from Exp 1)
static inline double fast_rand(unsigned long *state) {  
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    return (double)(*state & 0x7FFFFFFF) / 2147483647.0;
}

// Bilinear Interpolation (CIC) - (from Exp 1)
void interpolation(double * __restrict__ mesh_value, Points * __restrict__ points) {
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
    
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    for (int i = 0; i < NUM_Points; i++) {
        double px = points[i].x * inv_dx;
        double py = points[i].y * inv_dy;

        int i_idx = (int)px;
        int j_idx = (int)py;

        if (i_idx >= NX) i_idx = NX - 1;
        if (j_idx >= NY) j_idx = NY - 1;

        double wx = px - i_idx;
        double wy = py - j_idx;

        // Compute weights
        double o_wx = 1.0 - wx;
        double o_wy = 1.0 - wy;

        int base = j_idx * GRID_X + i_idx;
        
        // Fast memory access
        mesh_value[base] += o_wx * o_wy;
        mesh_value[base + 1] += wx * o_wy;
        mesh_value[base + GRID_X] += o_wx * wy;
        mesh_value[base + GRID_X + 1] += wx * wy;
    }
}

// Serial Mover (Immediate) - (from Exp 1)
void mover_serial_immediate(Points *points, double deltaX, double deltaY, int *deleted_count) {
    int del = 0;
    unsigned long state = 123456789UL; 

    for (int i = 0; i < NUM_Points; i++) {
        double rx = (2.0 * fast_rand(&state) - 1.0) * deltaX;
        double ry = (2.0 * fast_rand(&state) - 1.0) * deltaY;
        double px = points[i].x + rx;
        double py = points[i].y + ry;

        if (px < 0.0 || py < 0.0 || px > 1.0 || py > 1.0) {
            points[i].x = fast_rand(&state);
            points[i].y = fast_rand(&state);
            del++;
        } else {
            points[i].x = px;
            points[i].y = py;
        }
    }
    *deleted_count = del;
}

// Serial Mover (Deferred) - (from Exp 1)
void mover_serial_deferred(Points *points, double deltaX, double deltaY, int *deleted_count) {
    *deleted_count = 0;
    int active = NUM_Points;
    unsigned long state = 987654321UL;
    
    for (int i = 0; i < active; i++) {
        double rx = (2.0 * fast_rand(&state) - 1.0) * deltaX;
        double ry = (2.0 * fast_rand(&state) - 1.0) * deltaY;
        double px = points[i].x + rx;
        double py = points[i].y + ry;
        if (px < 0.0 || py < 0.0 || px > 1.0 || py > 1.0) {
            points[i] = points[active - 1];
            active--;
            (*deleted_count)++;
            i--;
        } else {
            points[i].x = px;
            points[i].y = py;
        }
    }
    for (int i = active; i < NUM_Points; i++) {
        points[i].x = fast_rand(&state);
        points[i].y = fast_rand(&state);
    }
}

// Parallel Mover (Immediate) - (logic from Exp 1 parallelized)
void mover_parallel_immediate(Points *points, double deltaX, double deltaY, int *deleted_count) {
    int total_del = 0;
    #pragma omp parallel reduction(+:total_del)
    {
        unsigned long state = (unsigned long)(time(NULL) ^ omp_get_thread_num());
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++) {
            double rx = (2.0 * fast_rand(&state) - 1.0) * deltaX;
            double ry = (2.0 * fast_rand(&state) - 1.0) * deltaY;
            double px = points[i].x + rx;
            double py = points[i].y + ry;

            if (px < 0.0 || py < 0.0 || px > 1.0 || py > 1.0) {
                points[i].x = fast_rand(&state);
                points[i].y = fast_rand(&state);
                total_del++;
            } else {
                points[i].x = px;
                points[i].y = py;
            }
        }
    }
    *deleted_count = total_del;
}

// Parallel Mover (Deferred) - (logic from Exp 1 parallelized)
void mover_parallel_deferred(Points * __restrict__ points, double deltaX, double deltaY, int *deleted_count) {
    int total_del = 0;
    #pragma omp parallel reduction(+:total_del)
    {
        unsigned long state = (unsigned long)(time(NULL) ^ omp_get_thread_num());
        #pragma omp for schedule(static)
        for (int i = 0; i < NUM_Points; i++) {
            double rx = (2.0 * fast_rand(&state) - 1.0) * deltaX;
            double ry = (2.0 * fast_rand(&state) - 1.0) * deltaY;
            double px = points[i].x + rx; double py = points[i].y + ry;
            if (px < 0.0 || py < 0.0 || px > 1.0 || py > 1.0) {
                points[i].x = -1.0; total_del++;
            } else { points[i].x = px; points[i].y = py; }
        }
    }

    int active = 0;
    for (int i = 0; i < NUM_Points; i++) {
        if (points[i].x >= 0.0) {
            if (i != active) points[active] = points[i];
            active++;
        }
    }
    #pragma omp parallel
    {
        unsigned long state = (unsigned long)(time(NULL) ^ omp_get_thread_num() ^ 0x55555UL);
        #pragma omp for schedule(static)
        for (int i = active; i < NUM_Points; i++) {
            points[i].x = fast_rand(&state); points[i].y = fast_rand(&state);
        }
    }
    *deleted_count = total_del;
}
