#include <stdio.h>
#include <stdlib.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 3

int main(void)
{
  uint8_t i, j, k;

  uint8_t *layerSizes = malloc(sizeof(uint8_t) * LAYER_NUM);
  layerSizes[0] = 2;
  layerSizes[1] = 4;
  layerSizes[2] = 1;

  neuralNet_t *network = neuralNet_init(LAYER_NUM, layerSizes);

  printf("neuralNet_init(%d, {2, 4, 1}) yielded a staring network with the following weights and biases\n", LAYER_NUM);

  for(i = 0; i < network->layerNum; i++)
  {
    printf("biases in layer %d:\n", i + 1);

    for(j = 0; j < network->layerSizes[i]; j++)
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

  return 0;
}
