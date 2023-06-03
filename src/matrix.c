#include <stdlib.h>
#include <math.h>

#include "include/matrix.h"

double randn(double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
  {
    call = !call;
    return (mu + sigma * (double) X2);
  }

  do
  {
    U1 = -1 + ((double) rand () / RAND_MAX) * 2;
    U2 = -1 + ((double) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  } 
  while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
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
