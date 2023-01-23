#include <stdlib.h>

#include "include/matrix.h"

matrix_t *matrix_init(uint8_t rowNum, uint8_t colNum)
{
  uint8_t i, j;

  matrix_t *result = malloc(sizeof(matrix_t));
  result->rowNum = rowNum;
  result->colNum = colNum;
  result->entries = malloc(sizeof(double *) * rowNum);

  /* randomize weights */
  for(i = 0; i < rowNum; i++)
  {
    result->entries[i] = malloc(sizeof(double) * colNum);
    for(j = 0; j < colNum; j++)
    {
      result->entries[i][j] = (double)((double)rand() / (double)RAND_MAX);
    }
  }


  return result;
}
