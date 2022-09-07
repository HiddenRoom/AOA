#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct MATRIX_STRUCT
{
  double **entries;

  uint64_t rows, cols;
} matrix_t;

/* Will return a dynamically allocated matrix with leftMatrix->rows rows and
 * rightMatix->cols columns if the given matrices can be multiplied otherwise,
 * it will return NULL */
matrix_t *matrix_mul(matrix_t *leftMatrix, matrix_t *rightMatrix);

#endif
