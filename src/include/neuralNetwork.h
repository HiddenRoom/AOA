#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stdint.h>

typedef struct NEURAL_NETWORK_STRUCT
{
  uint8_t layerNum;
  uint8_t *layerSizes;

  double *weights; /* could be matrix but will be parsed from 1D array to avoid cache annoyingness */
  double *biases;
} neuralNet_t;

neuralNet_t *neuralNet_init(uint8_t layerNum, uint8_t *layerSizes) /* generate randomly seeded neural net */

#endif
