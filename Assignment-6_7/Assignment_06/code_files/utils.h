#ifndef UTILS_H
#define UTILS_H
#include "init.h"

void interpolation_serial(double *mesh_value, PointsSoA *points);
void interpolation_hybrid(double *mesh_value, PointsSoA *points);
void save_mesh(double *mesh_value);

#endif
