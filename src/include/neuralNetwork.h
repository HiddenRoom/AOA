#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stdint.h>

#include "matrix.h"

typedef struct NEURAL_NETWORK_STRUCT
{
  double learningRate;
  uint16_t epochLen;

  uint16_t layerNum;
  uint16_t *layerSizes;
  double **neurons;

  matrix_t **weights; 
  double **biases;

  matrix_t **weightsTmp; 
  double **biasesTmp;
} neuralNet_t;

neuralNet_t *neuralNet_init(double learningRate, uint16_t epochLen, uint16_t layerNum, uint16_t *layerSizes); /* generate randomly seeded neural net */

void neuralNet_cpy(neuralNet_t *result, neuralNet_t *network);

void forwardPass(neuralNet_t *network); /* network should have the input/first layer neurons loaded with input vals */

void backPropagation(double *input, double *desired, neuralNet_t *network); /* desired changes for network will be put in tmp weights and biases */

#endif
