#include <math.h>
#include "utils.h"

// Problem 01
void matrix_multiplication_ijk(double** m1, double** m2, double** result, int N) {

    // i -> j -> k
    for(int i = 0; i<N;i++){
        for(int j = 0; j<N;j++){
            double sum = 0;
            for(int k = 0; k<N;k++){
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum ;
        }
    }

}

void matrix_multiplication_ikj(double** m1, double** m2, double** result, int N) {

    // i -> k -> j
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double r = m1[i][k];
            for (int j = 0; j < N; ++j) {
                result[i][j] += r * m2[k][j];
            }
        }
    }

}

void matrix_multiplication_jik(double** m1, double** m2, double** result, int N) {

    // j -> i -> k
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0 ;
            for (int k = 0; k < N; ++k) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] = sum;
        }
    }

}

void matrix_multiplication_jki(double** m1, double** m2, double** result, int N) {

    // j -> k -> i
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            double r = m2[k][j];
            for (int i = 0; i < N; ++i) {
                result[i][j] += m1[i][k] * r;
            }
        }
    }

}


void matrix_multiplication_kij(double** m1, double** m2, double** result, int N) {

    // k -> i -> j
    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            double r = m1[i][k];
            for (int j = 0; j < N; ++j) {
                result[i][j] += r * m2[k][j];
            }
        }
    }

}


void matrix_multiplication_kji(double** m1, double** m2, double** result, int N) {

    // k -> j -> i
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            double r = m2[k][j];
            for (int i = 0; i < N; ++i) {
                result[i][j] += m1[i][k] * r;
            }
        }
    }

}

// Problem 02
void transpose(double** m, double** mt, int N) {
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            mt[i][j]=m[j][i];
        }
    }
}

void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N) {
    // Allocate memory
    double **m2_T = (double**) malloc(N*sizeof(double*));
    for(int i = 0; i<N; i++) m2_T[i] = (double*)malloc(N*sizeof(double));
    
    // Transpose B
    transpose(m2,m2_T,N);

    // Multiplication
    for(int i = 0; i<N; i++){
        for(int j = 0;j<N;j++){
            double sum = 0.0;
            for(int k = 0; k<N; k++){
                sum += m1[i][k] * m2_T[j][k];
            }
            result[i][j] = sum;
        }
    }
    for(int i = 0;i<N;i++) free(m2_T[i]);
    free(m2_T);

}

// Problem 03
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N) {
    // init 
    for(int i=0; i<N; i++) 
        for(int j=0; j<N; j++) result[i][j] = 0.0;

    for (int ii = 0; ii < N; ii += B) {
        for (int jj = 0; jj < N; jj += B) {
            for (int kk = 0; kk < N; kk += B) {
                // Regular mult within block
                for (int i = ii; i < std::min(ii + B, N); i++) {
                    for (int k = kk; k < std::min(kk + B, N); k++) {
                         double r = m1[i][k];
                         for (int j = jj; j < std::min(jj + B, N); j++) {
                             result[i][j] += r * m2[k][j];
                         }
                    }
                }
            }
        }
    }
}