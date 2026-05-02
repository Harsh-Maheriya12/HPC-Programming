#ifndef INIT_H
#define INIT_H
#include <stdio.h>

// Structure of Arrays (SoA) for better SIMD vectorization
typedef struct {
    double *x;
    double *y;
} PointsSoA;

extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

void allocate_points(PointsSoA *p, int count);
void free_points(PointsSoA *p);
void read_points(FILE *file, PointsSoA *p);
#endif
