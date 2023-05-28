#ifndef MATRIX
#define MATRIX

#include <stdint.h>

typedef struct MATRIX_STRUCT
{
  uint8_t rowNum, colNum;
  double **entries;
} matrix_t;

matrix_t *matrix_init(double coefficient, uint8_t rowNum, uint8_t colNum);

#endif
