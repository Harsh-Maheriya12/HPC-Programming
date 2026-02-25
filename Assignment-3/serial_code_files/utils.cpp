#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

// Serial interpolation 
void interpolation(double *mesh_value, Points *points) {
    for (int p = 0; p < NUM_Points; p++) {
        double px = points[p].x;
        double py = points[p].y;

        // 1. Identify cell index (i, j)
        int i = (int)(px / dx);
        int j = (int)(py / dy);

        // Guard against points exactly at 1.0 or slightly outside due to precision
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        // 2. Compute relative distances (rx, ry) within the cell
        double rx = (px - i * dx) / dx;
        double ry = (py - j * dy) / dy;

        // 3. Compute bilinear weights
        double w00 = (1.0 - rx) * (1.0 - ry);
        double w10 = rx * (1.0 - ry);
        double w01 = (1.0 - rx) * ry;
        double w11 = rx * ry;

        // 4. Distribute value (fi=1) to the 4 nearest grid points
        // Scale by dx * dy to match reference test output
        double area = dx * dy;
        mesh_value[j * GRID_X + i] += w00 * area;
        mesh_value[j * GRID_X + (i + 1)] += w10 * area;
        mesh_value[(j + 1) * GRID_X + i] += w01 * area;
        mesh_value[(j + 1) * GRID_X + (i + 1)] += w11 * area;
    }
}

// Write mesh to file
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
