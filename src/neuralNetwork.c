#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

double activation(double x)
{
  return fmax(0, x); 
  //return tanh(x);
}

double dActivation(double activationOfX)
{
  return activationOfX <= 0 ? 0 : 1; 
  //return 1 - activationOfX * activationOfX;
}

neuralNet_t *neuralNet_init(double learningRate, uint32_t epochLen, uint32_t layerNum, uint32_t *layerSizes) /* generate randomly seeded neural net */
{
  uint32_t i, j;

  neuralNet_t *result = malloc(sizeof(neuralNet_t));
  result->learningRate = learningRate;
  result->epochLen = epochLen; /* number of examples per training cycle */

  result->layerNum = layerNum;
  result->layerSizes = layerSizes;
  result->neurons = malloc(sizeof(double *) * (layerNum)); 
  result->weights = malloc(sizeof(matrix_t *) * (layerNum - 1)); /* no weights on the last layer and no biases on first layer */
  result->biases = malloc(sizeof(double *) * (layerNum - 1)); 
  result->weightsTmp = malloc(sizeof(matrix_t *) * (layerNum - 1));
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
  uint32_t i, j, k;

  for(i = 0; i < network->layerNum - 1; i++)
  {
    for(j = 0; j < network->layerSizes[i + 1]; j++)
    {
      network->neurons[i + 1][j] = network->biases[i][j];

      for(k = 0; k < network->layerSizes[i]; k++)
      {
        network->neurons[i + 1][j] += network->weights[i]->entries[k][j] * network->neurons[i][k];
      }
      
      network->neurons[i + 1][j] = activation(network->neurons[i + 1][j]);
    }
  }
} 

void backPropagation(double *input, double *desired, neuralNet_t *network) /* changes will be placed into tmp weights and biases of network*/
{
  uint32_t i, j, k;

  double dError;
  double *deltaCurrentLayer;
  double *deltaLastLayer;

  /* input input values into first network layer */
  for(i = 0; i < network->layerSizes[0]; i++)
  {
    network->neurons[0][i] = input[i];
  }

  forwardPass(network);

  deltaLastLayer = malloc(sizeof(double) * network->layerSizes[network->layerNum - 1]);

  for(i = 0; i < network->layerSizes[network->layerNum - 1]; i++)
  {
    dError = 2.0 * (network->neurons[network->layerNum - 1][i] - desired[i]);
    deltaLastLayer[i] = dError * dActivation(network->neurons[network->layerNum - 1][i]);

    network->biasesTmp[network->layerNum - 2][i] += deltaLastLayer[i] * network->learningRate;

    for(j = 0; j < network->layerSizes[network->layerNum - 2]; j++)
    {
      network->weightsTmp[network->layerNum - 2]->entries[j][i] += network->neurons[network->layerNum - 2][j] * deltaLastLayer[i] * network->learningRate;
    }
  }

  for(i = network->layerNum - 2; i > 0; i--)
  {
    deltaCurrentLayer = malloc(sizeof(double) * network->layerSizes[i]);

    for(j = 0; j < network->layerSizes[i]; j++)
    {
      dError = 0.0;
      for(k = 0; k < network->layerSizes[i + 1]; k++)
      {
        dError += network->weights[i]->entries[j][k] * deltaLastLayer[k];
      }

      deltaCurrentLayer[j] = dError * dActivation(network->neurons[i][j]);
      network->biasesTmp[i - 1][j] += deltaCurrentLayer[j] * network->learningRate;

      for(k = 0; k < network->layerSizes[i - 1]; k++)
      {
        network->weightsTmp[i - 1]->entries[k][j] += network->neurons[i - 1][k] * deltaCurrentLayer[j] * network->learningRate;
      }
    }

    free(deltaLastLayer);
    deltaLastLayer = deltaCurrentLayer;
  }

  free(deltaCurrentLayer);
}

void train(bool stochastic, uint32_t exampleNum, double **input, double **desired, neuralNet_t *network)
{
  uint32_t i, j, k;

  uint32_t trainingCycles;

  if(stochastic)
  {
    exampleShuffle(exampleNum, input, desired);

    for(i = 0; i < network->epochLen && i < exampleNum; i++)
    {
      backPropagation(input[i], desired[i], network);
    }

    /* i is 1 more than the final index at loop exit making up for indices being 1 less than the number of training cycles */
    trainingCycles = i;
  }
  else
  {
    for(i = 0; i < exampleNum; i++)
    {
      backPropagation(input[i], desired[i], network);
    }

    trainingCycles = i;
  }

  for(i = 0; i < network->layerNum - 1; i++)
  {
    for(j = 0; j < network->layerSizes[i + 1]; j++)
    {
      network->biases[i][j] -= network->biasesTmp[i][j] / trainingCycles;
      network->biasesTmp[i][j] = 0.0;
    }
  }

  for(i = 0; i < network->layerNum - 1; i++)
  {
    for(j = 0; j < network->layerSizes[i]; j++)
    {
      for(k = 0; k < network->layerSizes[i + 1]; k++)
      {
        network->weights[i]->entries[j][k] -= network->weightsTmp[i]->entries[j][k] / trainingCycles;
        network->weightsTmp[i]->entries[j][k] = 0.0;
      }
    }
  }
}

void exampleShuffle(uint32_t exampleNum, double **input, double **desired)
{
  double *tmpIn, *tmpDesi;

  uint32_t i;
  uint32_t randIndex;

  for(i = 0; i < exampleNum; i++)
  {
    randIndex = rand() % exampleNum;

    tmpIn = input[randIndex];
    tmpDesi = desired[randIndex];

    input[randIndex] = input[i];
    desired[randIndex] = desired[i];
    input[i] = tmpIn;
    desired[i] = tmpDesi;
  }
}
