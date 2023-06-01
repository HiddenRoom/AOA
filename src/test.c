#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 4
#define LEARNING_RATE 0.01
#define EPOCH_LEN 10
#define NUM_TRAIN 500
#define NUM_EXAMPLES 5000
#define FUNC_TEST_RANGE 3.0

double funcToApprox(double x)
{
  return fabs(x);
}

/* I stole this code */
double randDouble(double max)
{
  return (rand() > RAND_MAX / 2 ? -1 : 1) *(max / RAND_MAX) * rand();
}

void printNetwork(neuralNet_t *network)
{
  uint32_t i, j, k;

  for(i = 0; i < network->layerNum - 1; i++)
  {
    printf("biases in layer %d:\n", i + 2);

    for(j = 0; j < network->layerSizes[i + 1]; j++)
    {
      printf("%lf ", network->biases[i][j]);
    }

    printf("\n");
  }

  printf("\n");

  for(i = 0; i < network->layerNum - 1; i++)
  {
    printf("weights from layer %d to %d\n", i + 1, i + 2);

    for(j = 0; j < network->layerSizes[i]; j++)
    {
      for(k = 0; k < network->layerSizes[i + 1]; k++)
      {
        printf("%lf ", network->weights[i]->entries[j][k]);
      }

      printf("\n");
    }
  }

  printf("\n");
}

int main(void)
{
  srand(time(NULL));

  uint32_t i, j;

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = 1;
  layerSizes[1] = 8;
  layerSizes[2] = 8;
  layerSizes[3] = 1;

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  double **trainingInputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingInputs[i] = malloc(sizeof(double) * network->layerSizes[0]);
    for(j = 0; j < network->layerSizes[0]; j++)
    {
      trainingInputs[i][j] = randDouble(FUNC_TEST_RANGE);
    }
  }
  double **trainingOutputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingOutputs[i] = malloc(sizeof(double) * network->layerSizes[network->layerNum - 1]);
    for(j = 0; j < network->layerSizes[network->layerNum - 1]; j++)
    {
      trainingOutputs[i][j] = funcToApprox(trainingInputs[i][j]);
    }
  }

  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    printf("input %lf\noutput %lf\n", trainingInputs[i][0], trainingOutputs[i][0]);
  }

  printf("neuralNet_init(%lf, %d, %d, {1, 8, 8, 2}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  double testExample = randDouble(FUNC_TEST_RANGE);

  network->neurons[0][0] = testExample;
  forwardPass(network);
  printf("feedForward should give the value %lf for input %lf\n\nin reality it gives\n", funcToApprox(testExample), testExample);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  bool isStochastic = false;

  for(int x = 0; x < NUM_TRAIN; x++)
  {
    train(isStochastic, NUM_EXAMPLES, trainingInputs, trainingOutputs, network);
  }

  printf("%d rounds of %s backpropagation completed\n\n", NUM_TRAIN, isStochastic ? "stochastic" : "batch");

  testExample = randDouble(FUNC_TEST_RANGE);

  network->neurons[0][0] = testExample;
  forwardPass(network);
  printf("feedForward should give the value %lf for input %lf\n\nin reality it gives\n", funcToApprox(testExample), testExample);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  return 0;
}
