#ifndef MATRIX
#define MATRIX

#include <stdint.h>

typedef struct MATRIX_STRUCT
{
  uint32_t rowNum, colNum;
  double **entries;
} matrix_t;

matrix_t *matrix_init(double coefficient, uint32_t rowNum, uint32_t colNum);

#endif
