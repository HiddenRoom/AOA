#include <stdio.h>
#include <stdlib.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 3
#define TEST_VAL1 0.5
#define TEST_VAL2 0.5

int main(void)
{
  uint8_t i, j, k;

  uint8_t *layerSizes = malloc(sizeof(uint8_t) * LAYER_NUM);
  layerSizes[0] = 2;
  layerSizes[1] = 2;
  layerSizes[2] = 1;

  neuralNet_t *network = neuralNet_init(LAYER_NUM, layerSizes);

  printf("neuralNet_init(%d, {2, 2, 1}) yielded a staring network with the following weights and biases\n", LAYER_NUM);

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

    for(j = 0; j < network->weights[i]->rowNum; j++)
    {
      for(k = 0; k < network->weights[i]->colNum; k++)
      {
        printf("%lf ", network->weights[i]->entries[j][k]);
      }

      printf("\n");
    }
  }

  printf("\n");

  printf("feedForward gives the following output(s) with inputs: %lf, %lf\n", TEST_VAL1, TEST_VAL2);
  network->neurons[0][0] = TEST_VAL1;
  network->neurons[0][1] = TEST_VAL1;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  neuralNet_t *networkCpy = neuralNet_dup(network);

  printf("neuralNet_dup; output should be identical to above network\n");

  for(i = 0; i < networkCpy->layerNum - 1; i++)
  {
    printf("biases in layer %d:\n", i + 2);

    for(j = 0; j < networkCpy->layerSizes[i + 1]; j++)
    {
      printf("%lf ", networkCpy->biases[i][j]);
    }

    printf("\n");
  }

  printf("\n");

  for(i = 0; i < networkCpy->layerNum - 1; i++)
  {
    printf("weights from layer %d to %d\n", i + 1, i + 2);

    for(j = 0; j < networkCpy->weights[i]->rowNum; j++)
    {
      for(k = 0; k < networkCpy->weights[i]->colNum; k++)
      {
        printf("%lf ", networkCpy->weights[i]->entries[j][k]);
      }

      printf("\n");
    }
  }

  printf("\n\n");

  printf("\n");

  return 0;
}
