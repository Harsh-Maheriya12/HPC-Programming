#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <time.h>  
#include <omp.h>  
  
#include "init.h"  
#include "utils.h"  
  
// Global variables  
int GRID_X, GRID_Y, NX, NY;  
int NUM_Points, Maxiter;  
double dx, dy;  
  
int main() {    
    int nx_configs[] = {250, 500, 1000};
    int ny_configs[] = {100, 200, 400};
    long point_counts[] = {100, 10000, 1000000, 100000000, 1000000000};
    int num_grid_configs = 3;
    int num_point_configs = 5;

    Maxiter = 10;
    int num_threads = 1;
    omp_set_num_threads(num_threads);

    printf("=== Experiment 01: Scaling with Particle Count (Approach 1: Deferred) ===\n");

    for (int g = 0; g < num_grid_configs; g++) {
        NX = nx_configs[g];
        NY = ny_configs[g];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int pc = 0; pc < num_point_configs; pc++) {
            NUM_Points = (int)point_counts[pc];

            // Allocate memory for grid and Points  
            double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));  
            Points *points = (Points *) calloc(NUM_Points, sizeof(Points));  
            
            if (!mesh_value || !points) {
                printf("\nMemory allocation failed for NX=%d, NY=%d, Particles=%d\n", NX, NY, NUM_Points);
                if (mesh_value) free(mesh_value);
                if (points) free(points);
                continue;
            }

            initializepoints(points);  
          
            printf("\n------------------------------------------------\n");
            printf("Grid: %dx%d, Particles: %d, Threads: %d, Approach: Deferred\n", 
                   NX, NY, NUM_Points, num_threads);  
            printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\tDeleted\n");  
              
            double total_interp_time = 0.0;  
            double total_move_time = 0.0;  
            int total_deleted_all = 0;  
              
            for (int iter = 0; iter < Maxiter; iter++) {  
                // Interpolation timing  
                double start_interp = omp_get_wtime();  
                interpolation(mesh_value, points);  
                double end_interp = omp_get_wtime();  
          
                // Mover timing with deletion tracking  
                int deleted_count = 0;    
                double start_move = omp_get_wtime();    
                
                mover_serial_deferred(points, dx, dy, &deleted_count);    
                
                double end_move = omp_get_wtime();    
                  
                double interp_time = end_interp - start_interp;  
                double move_time = end_move - start_move;  
                double total = interp_time + move_time;  
                  
                total_interp_time += interp_time;  
                total_move_time += move_time;  
                total_deleted_all += deleted_count;  
          
                printf("%d\t%lf\t%lf\t%lf\t%d\n", iter+1, interp_time, move_time, total, deleted_count);  
            }   
              
            printf("\nSummary for Particles=%d:\n", NUM_Points);  
            printf("Total interpolation time: %lf seconds\n", total_interp_time);  
            printf("Total mover time: %lf seconds\n", total_move_time);  
            printf("Total execution time: %lf seconds\n", total_interp_time + total_move_time);  
            printf("Total particles deleted: %d\n", total_deleted_all);  
            printf("Average deletions per iteration: %.2f\n", (double)total_deleted_all / Maxiter);  
          
            // Free memory  
            free(mesh_value);  
            free(points);  
        }
    }
  
    return 0;  
}