#include <stdio.h>
#include <stdlib.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 4
#define TEST_VAL1 0.5
#define TEST_VAL2 0.5
#define LEARNING_RATE 0.001
#define EPOCH_LEN 1
#define NUM_TRAIN 500

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
}

int main(void)
{
  uint16_t i;

  uint16_t *layerSizes = malloc(sizeof(uint16_t) * LAYER_NUM);
  layerSizes[0] = 1;
  layerSizes[1] = 20;
  layerSizes[2] = 5;
  layerSizes[3] = 2;

  double input = 0.5;

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNet_init(%lf, %d, %d, {1, 8, 8, 2}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  printf("feedForward gives the following output(s) with inputs: %lf\n", input);
  network->neurons[0][0] = input;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  double desirdOutput[] = {0.8, 0.2};

  for(int x = 0; x < NUM_TRAIN; x++)
  {
    backPropagation(&input, desirdOutput, network);

    //printNetwork(network);
  }

  printf("after %d backprop rounds with targets %lf %lf feedForward gives the following output(s) with inputs: %lf\n", NUM_TRAIN, desirdOutput[0], desirdOutput[1], input);
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  return 0;
}
