#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <stdint.h>
#include <stdbool.h>

#include "matrix.h"

typedef struct NEURAL_NETWORK_STRUCT
{
  double learningRate;
  uint32_t epochLen;

  uint32_t layerNum;
  uint32_t *layerSizes;
  double **neurons;

  matrix_t **weights; 
  double **biases;

  matrix_t **weightsTmp; 
  double **biasesTmp;
} neuralNet_t;

double activation(double x);

double dActivation(double activationOfX);

neuralNet_t *neuralNetInit(double learningRate, uint32_t epochLen, uint32_t layerNum, uint32_t *layerSizes); /* generate randomly seeded neural net */

void freeNeuralNet(neuralNet_t *network);

void forwardPass(neuralNet_t *network); /* network should have the input/first layer neurons loaded with input vals */

void backPropagation(double *input, double *desired, neuralNet_t *network); /* desired changes for network will be put in tmp weights and biases */

void train(bool stochastic, uint32_t exampleNum, double **input, double **desired, neuralNet_t *network);

void exampleShuffle(uint32_t epochLen, uint32_t exampleNum, double **input, double **desired);

#endif
