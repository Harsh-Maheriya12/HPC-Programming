#include <stdio.h>
#include <stdlib.h>
#include "init.h"

void allocate_points(PointsSoA *p, int count) {
    // Align to 64 bytes for AVX-512 and cache line friendliness
    if (posix_memalign((void**)&p->x, 64, count * sizeof(double)) != 0) {
        perror("posix_memalign x");
        exit(1);
    }
    if (posix_memalign((void**)&p->y, 64, count * sizeof(double)) != 0) {
        perror("posix_memalign y");
        exit(1);
    }
}

void free_points(PointsSoA *p) {
    free(p->x);
    free(p->y);
}   

void read_points(FILE *file, PointsSoA *p) {
    for (int i = 0; i < NUM_Points; i++) {
        if (fread(&p->x[i], sizeof(double), 1, file) != 1) return;
        if (fread(&p->y[i], sizeof(double), 1, file) != 1) return;
    }
}
