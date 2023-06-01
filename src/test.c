#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 4
#define LEARNING_RATE 0.1
#define EPOCH_LEN 20
#define NUM_TRAIN 2000
#define NUM_EXAMPLES 1000
#define FUNC_RANGE 4.0

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

double randDouble(double max)
{
    srand((unsigned) time(0));
    return (rand() > RAND_MAX / 2 ? -1 : 1) *(max / RAND_MAX) * rand();
}

double funcToApprox(double x)
{
  return sin(x);
}

int main(void)
{
  uint32_t i;

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = 1;
  layerSizes[1] = 8;
  layerSizes[2] = 8;
  layerSizes[3] = 1;

  double **trainingInputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    trainingInputs[i][0] = randDouble(FUNC_RANGE);
  }
  double **trainingOutputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    trainingOutputs[i][0] = funcToApprox(trainingInputs[i][0]);
  }

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNet_init(%lf, %d, %d, {1, 8, 8, 2}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  double testExample = randDouble(FUNC_RANGE);

  printf("feedForward gives the following output(s) with inputs: %lf\n", testExample);
  network->neurons[0][0] = testExample;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  printf("it should give: %lf\n", funcToApprox(testExample));

  for(int x = 0; x < NUM_TRAIN; x++)
  {
    train(false, NUM_EXAMPLES, trainingInputs, trainingOutputs, network);
  }

  printNetwork(network);

  printf("feedForward gives the following output(s) with inputs: %lf\n", testExample);
  network->neurons[0][0] = testExample;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  printf("it should give: %lf\n", funcToApprox(testExample));

  return 0;
}
