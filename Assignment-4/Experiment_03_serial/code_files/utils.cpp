#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

void interpolation(double *mesh_value, Points *points) {

    for( int i=0;i<GRID_X*GRID_Y;i++) 
        mesh_value[i]=0.0;
	
    for (int p = 0; p < NUM_Points; p++) {
        double px = points[p].x;
        double py = points[p].y;

        int i = (int)(px / dx);
        int j = (int)(py / dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        double rx = (px - i * dx) / dx;
        double ry = (py - j * dy) / dy;

        double w00 = (1.0 - rx) * (1.0 - ry);
        double w10 = rx * (1.0 - ry);
        double w01 = (1.0 - rx) * ry;
        double w11 = rx * ry;

        double area = dx * dy;
        
		if(i < NX-1 && j < NY-1){
			mesh_value[j * GRID_X + i]             += w00 * area;
			mesh_value[j * GRID_X + (i + 1)]       += w10 * area;
			mesh_value[(j + 1) * GRID_X + i]       += w01 * area;
			mesh_value[(j + 1) * GRID_X + (i + 1)] += w11 * area;
		}
    }
}

void mover_serial(Points *points, unsigned int *seeds) {
    for (int i = 0; i < NUM_Points; i++) {
        double new_x, new_y;
        do {
            double rand_dx = ((double)rand_r(&seeds[i]) / RAND_MAX * 2.0 - 1.0) * dx;
            double rand_dy = ((double)rand_r(&seeds[i]) / RAND_MAX * 2.0 - 1.0) * dy;
            new_x = points[i].x + rand_dx;
            new_y = points[i].y + rand_dy;
        } while (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0);
        points[i].x = new_x;
        points[i].y = new_y;
    }
}

void mover_parallel(Points *points, unsigned int *seeds) {
    #pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < NUM_Points; i++) {
        double new_x, new_y;
        do {
            double rand_dx = ((double)rand_r(&seeds[i]) / RAND_MAX * 2.0 - 1.0) * dx;
            double rand_dy = ((double)rand_r(&seeds[i]) / RAND_MAX * 2.0 - 1.0) * dy;
            new_x = points[i].x + rand_dx;
            new_y = points[i].y + rand_dy;
        } while (new_x < 0.0 || new_x > 1.0 || new_y < 0.0 || new_y > 1.0);
        points[i].x = new_x;
        points[i].y = new_y;
    }
}

void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}
