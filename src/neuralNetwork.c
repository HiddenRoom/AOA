#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/neuralNetwork.h"

double activation(double x)
{
  return tanh(x);
}

double derivativeActivation(double x)
{
  return pow(sinh(x) / cosh(x), 2.0);
}

neuralNet_t *neuralNet_init(uint8_t layerNum, uint8_t *layerSizes)
{
  uint8_t i, j;

  neuralNet_t *result = malloc(sizeof(neuralNet_t));
  result->layerNum = layerNum;
  result->layerSizes = layerSizes;
  result->neurons = malloc(sizeof(double *) * (layerNum)); 
  result->weights = malloc(sizeof(matrix_t *) * (layerNum - 1)); /* no weights on the last layer and no biases on first layer */
  result->biases = malloc(sizeof(double *) * (layerNum - 1)); 

  for(i = 0; i < layerNum - 1; i++) /* -1 for no weights on last layer */
  {
    result->weights[i] = matrix_init(layerSizes[i], layerSizes[i + 1]);
    result->biases[i] = malloc(sizeof(double) * layerSizes[i + 1]);
  }

  for(i = 0; i < layerNum; i++)
  {
    result->neurons[i] = malloc(sizeof(double) * layerSizes[i]);
  }

  /* randomize biases */
  for(i = 0; i < layerNum - 1; i++)
  {
    for(j = 0; j < layerSizes[i + 1]; j++)
      result->biases[i][j] = (double)((double)rand() / (double)RAND_MAX);
  }

  return result;
}

void neuralNet_cpy(neuralNet_t *result, neuralNet_t *network)
{
  uint8_t i, j; 

  for(i = 0; i < network->layerNum; i++)
  {
    memcpy(result->neurons[i], network->neurons[i], sizeof(double) * network->layerSizes[i]);
  }

  for(i = 0; i < network->layerNum - 1; i++)
  {
    memcpy(result->biases[i], network->biases[i], sizeof(double) * network->layerSizes[i + 1]);
    result->weights[i]->rowNum = network->weights[i]->rowNum;
    result->weights[i]->colNum = network->weights[i]->colNum;
    for(j = 0; j < network->weights[i]->rowNum; j++)
    {
      memcpy(result->weights[i]->entries[j], network->weights[i]->entries[j], sizeof(double) * network->weights[i]->colNum);
    }
  }
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
      
      network->neurons[i + 1][j] = activation(network->neurons[i + 1][j]);
    }
  }
} 

void backPropagation(double *input, double *desiredOutput, neuralNet_t *network, neuralNet_t *tmp) /* desired changes for network will be put in tmp */
{
  uint8_t i, j, k;

  double totalError;

  /* input input values into first network layer */
  for(i = 0; i < network->layerSizes[0]; i++)
  {
    network->neurons[0][i] = input[i];
  }

  totalError = cost(network->neurons[network->layerNum - 1], desiredOutput, network->layerNum);
}

double cost(double *actualOutput, double *desiredOutput, uint8_t size)
{
  uint8_t i;

  double cost = 0;

  for(i = 0; i < size; i++)
  {
    cost += pow(actualOutput[i] - desiredOutput[i], 2.0);
  }

  return cost;
}
