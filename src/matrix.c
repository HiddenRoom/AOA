#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "include/matrix.h"

#define ACCURACY 5000

double randn(double mu, double sigma)
{
  double s;
  double x, y;
  double mult;
  static double xUni, yUni;
  static bool xOrY = false;

  if(xOrY)
  {
    xOrY = !xOrY;
    return mu + sigma * yUni;
  }

  do
  {
    x = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;
    y = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;

    s = pow(x, 2.0) + pow(y, 2.0);
  }
  while(s >= 1.0);

  mult = sqrt(-2.0 * log(s) / s);

  xUni = x * mult;
  yUni = y * mult;

  xOrY = !xOrY;

  return mu + sigma * xUni;
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
