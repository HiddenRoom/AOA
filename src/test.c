#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 4
#define LEARNING_RATE 0.0032
#define EPOCH_LEN 100
#define NUM_TRAIN 14000
#define NUM_EXAMPLES 60000
#define FUNC_RANGE 5.0
#define STOCHASTIC true

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
  return /* (rand() % 2 == 0 ? 1.0 : -1.0) * */ (max * (double)((double)rand() / (double)RAND_MAX));
}

double funcToApprox(double x, double y)
{
  return fmax(x, y);
}

int main(void)
{
  srand(time(NULL));

  uint32_t i;

  for(i = 0; i < 200; i++)
  {
    printf("%lf\n", randn(0, 1));
  }

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = 2;
  layerSizes[1] = 30;
  layerSizes[2] = 30;
  layerSizes[3] = 1;

  double **trainingInputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    trainingInputs[i][0] = randDouble(FUNC_RANGE);
    trainingInputs[i][1] = randDouble(FUNC_RANGE);
  }
  double **trainingOutputs = malloc(sizeof(double *) * NUM_EXAMPLES);
  for(i = 0; i < NUM_EXAMPLES; i++)
  {
    trainingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    trainingOutputs[i][0] = funcToApprox(trainingInputs[i][0], trainingInputs[i][1]);

    if(i % (NUM_EXAMPLES / 20) == 0)
    {
      printf("input %lf %lf\toutput %lf\n", trainingInputs[i][0], trainingInputs[i][1], trainingOutputs[i][0]);
    }
  }

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNet_init(%lf, %d, %d, {2, 8, 8, 2}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  double testExample = randDouble(FUNC_RANGE);
  double testExample2 = randDouble(FUNC_RANGE);

  printf("feedForward gives the following output(s) with inputs: %lf %lf\n", testExample, testExample2);
  network->neurons[0][0] = testExample;
  network->neurons[0][1] = testExample2;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  printf("it should give: %lf\n", funcToApprox(testExample, testExample2));

  for(i = 0; i < NUM_TRAIN; i++)
  {
    train(STOCHASTIC, NUM_EXAMPLES, trainingInputs, trainingOutputs, network);
    if(i % 100 == 0)
    {
      printf("%d\n", i);
    }
  }

  printNetwork(network);

  testExample = randDouble(FUNC_RANGE);
  testExample2 = randDouble(FUNC_RANGE);

  printf("feedForward gives the following output(s) with inputs: %lf %lf\n", testExample, testExample2);
  network->neurons[0][0] = testExample;
  network->neurons[0][1] = testExample2;
  forwardPass(network);

  for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
  {
    printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
  }

  printf("\n\n");

  printf("it should give: %lf\n", funcToApprox(testExample, testExample2));

  while(true)
  {
    scanf("%lf", &testExample);
    scanf("%lf", &testExample2);

    printf("feedForward gives the following output(s) with inputs: %lf, %lf\n", testExample, testExample2);
    network->neurons[0][0] = testExample;
    network->neurons[0][1] = testExample2;
    forwardPass(network);

    for(i = 0; i < layerSizes[LAYER_NUM - 1]; i++)
    {
      printf("%lf ", network->neurons[LAYER_NUM - 1][i]);
    }

    printf("\n\n");

    printf("it should give: %lf\n", funcToApprox(testExample, testExample2));
  }

  return 0;
}
