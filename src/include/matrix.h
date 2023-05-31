#ifndef MATRIX
#define MATRIX

#include <stdint.h>

typedef struct MATRIX_STRUCT
{
  uint16_t rowNum, colNum;
  double **entries;
} matrix_t;

matrix_t *matrix_init(double coefficient, uint16_t rowNum, uint16_t colNum);

#endif
