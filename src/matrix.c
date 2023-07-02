#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "include/matrix.h"

#define ACCURACY 5000

void nullCatchAndDie(void *ptr, char *msg)
{
  if(ptr == NULL)
  {
    printf("%s", msg);
    exit(1);
  }
}

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

matrix_t *matrixInit(double coefficient, uint32_t rowNum, uint32_t colNum)
{
  uint32_t i, j;

  matrix_t *result = malloc(sizeof(matrix_t));
  nullCatchAndDie(result, "malloc returned NULL in matrixInit when allocating matrix_t *result\n");
  result->entries = malloc(sizeof(double *) * rowNum);
  nullCatchAndDie(result->entries, "malloc returned NULL in matrixInit when allocating double **result->entries\n");

  /* He init */
  for(i = 0; i < rowNum; i++)
  {
    result->entries[i] = malloc(sizeof(double) * colNum);
    nullCatchAndDie(result->entries[i], "malloc returned NULL in matrixInit when allocating double *result->entries[i]\n");
    for(j = 0; j < colNum; j++)
    {
      result->entries[i][j] = coefficient * randn(0, 1);
    }
  }


  return result;
}

void freeMatrix(matrix_t *matrix, uint32_t rowNum)
{
  uint32_t i;

  for(i = 0; i < rowNum; i++)
  {
    free(matrix->entries[i]);
  }
  free(matrix->entries);

  free(matrix);
}
