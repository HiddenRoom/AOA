#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "include/matrix.h"

#define ACCURACY 5000

double randn(double mu, double sigma)
{
  double s;
  double x, y;
  double coefficient;
  double xUni;
  static double yUni;
  static bool xOrY = false;

  if(xOrY)
  {
    xOrY = !xOrY;
    return mu + sigma * (double)yUni;
  }

  do
  {
    x = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
    y = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;

    s = pow(x, 2.0) + pow(y, 2.0);
  }
  while(s >= 1.0 || s == 0);

  coefficient = sqrt(-2.0 * log(s) / s);

  xUni = x * coefficient;
  yUni = y * coefficient;

  xOrY = !xOrY;

  return mu + sigma * (double)xUni;
}

matrix_t *matrix_init(double coefficient, uint32_t rowNum, uint32_t colNum)
{
  uint32_t i, j;

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
      result->entries[i][j] = coefficient * randn(0, 1);
    }
  }


  return result;
}

void freeMatrix(matrix_t *matrix, uint32_t rowNum)
{
  uint32_t i, j;

  for(i = 0; i < rowNum; i++)
  {
    free(matrix->entries[i]);
  }
  free(matrix->entries);

  free(matrix)
}
