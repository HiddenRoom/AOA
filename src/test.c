#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 3
#define TEST_VAL1 0.5
#define TEST_VAL2 0.5
#define LEARNING_RATE 0.01
#define EPOCH_LEN 1
#define NUM_TRAIN 200
#define NUM_EXAMPLES 1

void printNetwork(neuralNet_t *network)
{
  uint16_t i, j, k;

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
  uint16_t i;

  uint16_t *layerSizes = malloc(sizeof(uint16_t) * LAYER_NUM);
  layerSizes[0] = 1;
  layerSizes[1] = 2;
  layerSizes[2] = 2;

  double **trainingInputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  trainingInputs[0] = malloc(sizeof(double) * layerSizes[0]);
  trainingInputs[0][0] = 0.5;
  double **trainingOutputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  trainingOutputs[0] = malloc(sizeof(double) * layerSizes[LAYER_NUM - 1]);
  trainingOutputs[0][0] = 0.2;
  trainingOutputs[0][1] = 0.8;

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNet_init(%lf, %d, %d, {1, 8, 8, 2}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  printf("feedForward gives the following output(s) with inputs: %lf\n", trainingInputs[0][0]);
  network->neurons[0][0] = trainingInputs[0][0];
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  for(int x = 0; x < NUM_TRAIN; x++)
  {
    train(false, NUM_EXAMPLES, trainingInputs, trainingOutputs, network);

    printNetwork(network);
  }

  printf("after %d backprop rounds with targets %lf %lf feedForward gives the following output(s) with inputs: %lf\n", NUM_TRAIN, trainingOutputs[0][0], trainingOutputs[0][1], trainingInputs[0][0]);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  return 0;
}
