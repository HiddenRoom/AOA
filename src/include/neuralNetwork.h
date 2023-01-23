#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stdint.h>

#include "matrix.h"

typedef struct NEURAL_NETWORK_STRUCT
{
  uint8_t layerNum;
  uint8_t *layerSizes;

  matrix_t **weights; 
  double **biases;
} neuralNet_t;

neuralNet_t *neuralNet_init(uint8_t layerNum, uint8_t *layerSizes); /* generate randomly seeded neural net */

#endif
