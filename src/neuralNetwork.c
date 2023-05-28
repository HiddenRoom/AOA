#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LEARNING_RATE 0.01

double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

double dSigmoid(double x)
{
  return x * (1 - x);
}



neuralNet_t *neuralNet_init(double learningRate, uint16_t epochLen, uint8_t layerNum, uint8_t *layerSizes) /* generate randomly seeded neural net */
{
  uint8_t i, j;

  neuralNet_t *result = malloc(sizeof(neuralNet_t));
  result->learningRate = learningRate;
  result->epochLen = epochLen; /* number of examples per training cycle */

  result->layerNum = layerNum;
  result->layerSizes = layerSizes;
  result->neurons = malloc(sizeof(double *) * (layerNum)); 
  result->weights = malloc(sizeof(matrix_t *) * (layerNum - 1)); /* no weights on the last layer and no biases on first layer */
  result->biases = malloc(sizeof(double *) * (layerNum - 1)); 
  result->weightsTmp = malloc(sizeof(matrix_t *) * (layerNum - 1)); /* no weights on the last layer and no biases on first layer */
  result->biasesTmp = malloc(sizeof(double *) * (layerNum - 1)); 

  for(i = 0; i < layerNum - 1; i++) /* -1 for no weights on last layer */
  {
    result->weights[i] = matrix_init(1.0, layerSizes[i], layerSizes[i + 1]);
    result->weightsTmp[i] = matrix_init(0.0, layerSizes[i], layerSizes[i + 1]);
    result->biases[i] = malloc(sizeof(double) * layerSizes[i + 1]);
    result->biasesTmp[i] = malloc(sizeof(double) * layerSizes[i + 1]);
  }

  for(i = 0; i < layerNum; i++)
  {
    result->neurons[i] = malloc(sizeof(double) * layerSizes[i]);
  }

  /* randomize biases */
  for(i = 0; i < layerNum - 1; i++)
  {
    for(j = 0; j < layerSizes[i + 1]; j++)
    {
      result->biases[i][j] = (double)((double)rand() / (double)RAND_MAX);
      result->biasesTmp[i][j] = 0.0;
    }
  }

  return result;
}

void forwardPass(neuralNet_t *network) /* network should have the input/first layer neurons loaded with input vals */
{
  uint8_t i, j, k;

  for(i = 0; i < network->layerNum - 1; i++)
  {
    for(j = 0; j < network->weights[i]->colNum; j++)
    {
      network->neurons[i + 1][j] = network->biases[i][j];

      for(k = 0; k < network->weights[i]->rowNum; k++)
      {
        network->neurons[i + 1][j] += network->weights[i]->entries[k][j] * network->neurons[i][k];
      }
      
      network->neurons[i + 1][j] = sigmoid(network->neurons[i + 1][j]);
    }
  }
} 

void backPropagation(double *input, double *desired, neuralNet_t *network) /* changes will be placed into tmp weights and biases of network*/
{
  uint8_t i, j, k;

  double dError;
  double *deltaCurrentLayer;
  double *deltaLastLayer;

  /* input input values into first network layer */
  for(i = 0; i < network->layerSizes[0]; i++)
  {
    network->neurons[0][i] = input[i];
  }

  forwardPass(network);

  for(i = 0; i < network->layerSizes[network->layerNum - 1]; i++)
  {
    dError = 2.0 * (network->neurons[network->layerNum - 1][i] - desired[i]);

    deltaLastLayer[i] = dError * dSigmoid(network->neurons[network->layerNum - 1][i]);
  }

  for(i = network->layerNum - 2; i > 0; i--)
  {
    deltaCurrentLayer = malloc(sizeof(double) * network->layerSizes[i]);

    for(j = 0; j < network->layerSizes[i]; j++)
    {
      for(k = 0; k < network->layerSizes[i - 1]; k++)
      {
      }
    }

    free(deltaLastLayer);
    deltaCurrentLayer = deltaLastLayer;
  }
}

double *dCost(double *actualOutput, double *desiredOutput, uint8_t size)
{
  uint8_t i;

  double *cost = malloc(sizeof(double) * size);

  for(i = 0; i < size; i++)
  {
    cost[i] += 2.0 * (actualOutput[i] - desiredOutput[i]);
  }

  return cost;
}
