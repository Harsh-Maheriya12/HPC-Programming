#ifndef UTILS_H
#define UTILS_H

#include "init.h"

void interpolation(double *mesh_value, Points *points);
void save_mesh(double *mesh_value);
void mover_serial(Points *points, unsigned int *seeds);
void mover_parallel(Points *points, unsigned int *seeds);

#endif
