#include <stdlib.h>

#include "include/neuralNetwork.h"

neuralNet_t *neuralNet_init(uint8_t layerNum, uint8_t *layerSizes)
{
  uint8_t i, j;

  neuralNet_t *result = malloc(sizeof(neuralNet_t));
  result->layerNum = layerNum;
  result->layerSizes = layerSizes;
  result->weights = malloc(sizeof(matrix_t *) * (layerNum - 1)); /* no weights on the last layer */
  result->biases = malloc(sizeof(double *) * (layerNum)); 

  for(i = 0; i < layerNum - 1; i++) /* -1 for no weights on last layer */
  {
    result->weights[i] = matrix_init(layerSizes[i], layerSizes[i + 1]);
    result->biases[i] = malloc(sizeof(double) * layerSizes[i]);
    result->biases[i + 1] = malloc(sizeof(double) * layerSizes[i + 1]);
  }

  /* randomize biases */
  for(i = 0; i < layerNum; i++)
  {
    for(j = 0; j < layerSizes[i]; j++)
      result->biases[i][j] = (double)((double)rand() / (double)RAND_MAX);
  }

  return result;
}
