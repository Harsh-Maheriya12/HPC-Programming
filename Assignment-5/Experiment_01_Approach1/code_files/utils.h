#ifndef UTILS_H  
#define UTILS_H  
  
#include "init.h"  
  
// Global variables shared across the simulation  
extern int GRID_X, GRID_Y, NX, NY;  
extern int NUM_Points, Maxiter;  
extern double dx, dy;  
  
// Function declarations  
void interpolation(double *mesh_value, Points *points);  
void mover_serial_deferred(Points *points, double deltaX, double deltaY, int *deleted_count);  
void mover_serial_immediate(Points *points, double deltaX, double deltaY, int *deleted_count);  
void mover_parallel_immediate(Points *points, double deltaX, double deltaY, int *deleted_count);  
void save_mesh(double *mesh_value);  
  
#endif