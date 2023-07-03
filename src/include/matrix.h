#ifndef MATRIX
#define MATRIX

#include <stdint.h>

typedef struct MATRIX_STRUCT
{
  double **entries;
} matrix_t;

void nullCatchAndDie(void *ptr, char *msg);

matrix_t *matrixInit(double coefficient, uint32_t rowNum, uint32_t colNum);

void freeMatrix(matrix_t *matrix, uint32_t rowNum);

double randn(double mu, double sigma);

#endif
