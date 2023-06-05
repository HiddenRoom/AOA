#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 4
#define LEARNING_RATE 0.01
#define EPOCH_LEN 100
#define TRAINING_ROUNDS 250
#define TRAINING_EXAMPLES 60000
#define TESTING_EXAMPLES 10000
#define FUNC_RANGE 5.0
#define STOCHASTIC true

typedef struct IMAGE_STRUCT
{
    double label;
    double imgData[784];
} img_t;

img_t* parseCSV(const char* fileName, int exampleNum)
{
  FILE* file = fopen(fileName, "r");
  if (file == NULL) 
  {
    printf("Error opening file: %s\n", fileName);
    return NULL;
  }

  // Skip the first line
  char header[16384];
  fgets(header, sizeof(header), file);

  img_t* images = (img_t*)malloc(exampleNum * sizeof(img_t));
  if (images == NULL)
  {
    printf("Memory allocation failed.\n");
    fclose(file);
    return NULL;
  }

  char line[16384];  // Assuming maximum line length of 16,384 characters

  int count = 0;
  while (fgets(line, sizeof(line), file) != NULL && count < exampleNum) 
  {
    char* token = strtok(line, ",");
    int tokenCount = 0;

    images[count].label = atof(token);

    while (token != NULL && tokenCount < 784) 
    {
        token = strtok(NULL, ",");
        if (token != NULL) 
        {
          images[count].imgData[tokenCount] = atof(token) / 255.0;
        }
        tokenCount++;
    }

    count++;
  }

  fclose(file);

  return images;
}

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

/*
double randDouble(double max)
{
  return (max * (double)((double)rand() / (double)RAND_MAX));
}

double funcToApprox(double x, double y)
{
  return fmax(x, y);
}
*/

int main(int argc, char **argv)
{
  srand(time(NULL));

  uint32_t i, j;

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = 784;
  layerSizes[1] = 30;
  layerSizes[2] = 30;
  layerSizes[3] = 10;

  img_t *trainingImgs = parseCSV(argv[1], TRAINING_EXAMPLES);
  img_t *testingImgs = parseCSV(argv[2], TESTING_EXAMPLES);

  double **trainingInputs = malloc(sizeof(double *) * TRAINING_EXAMPLES);
  for(i = 0; i < TRAINING_EXAMPLES; i++)
  {
    trainingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[0]; j++)
    {
      trainingInputs[i][j] = trainingImgs[i].imgData[j];
    }
  }
  double **trainingOutputs = malloc(sizeof(double *) * TRAINING_EXAMPLES);
  for(i = 0; i < TRAINING_EXAMPLES; i++)
  {
    trainingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[LAYER_NUM - 1]; j++)
    {
      if(j == (uint32_t)round(trainingImgs[i].label))
      {
        trainingOutputs[i][j] = 1.0;
      }
      else
      {
        trainingOutputs[i][j] = 0.0;
      }
    }
  }

  double **testingInputs = malloc(sizeof(double *) * TESTING_EXAMPLES);
  for(i = 0; i < TESTING_EXAMPLES; i++)
  {
    testingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[0]; j++)
    {
      testingInputs[i][j] = testingImgs[i].imgData[j];
    }
  }
  double **testingOutputs = malloc(sizeof(double *) * TESTING_EXAMPLES);
  for(i = 0; i < TESTING_EXAMPLES; i++)
  {
    testingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[LAYER_NUM - 1]; j++)
    {
      if(j == (uint32_t)round(testingImgs[i].label))
      {
        testingOutputs[i][j] = 1.0;
      }
      else
      {
        testingOutputs[i][j] = 0.0;
      }
    }
  }

  neuralNet_t *network = neuralNet_init(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNet_init(%lf, %d, %d, {784, 8, 8, 10}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  for(i = 0; i < TRAINING_ROUNDS; i++)
  {
    train(STOCHASTIC, TRAINING_EXAMPLES, trainingInputs, trainingOutputs, network);
    if(i % 100 == 0)
    {
      printf("%d\n", i);
    }
  }

  printf("%d %s training rounds completed\n", TRAINING_ROUNDS, STOCHASTIC ? "stochastic" : "batch");

  for(i = 0; i < TESTING_EXAMPLES; i++)
  {
    for(j = 0; j < network->layerSizes[0]; j++)
    {
      network->neurons[0][j] = testingInputs[i][j];
    }

    forwardPass(network);

    printf("expected ");
    for(j = 0; j < network->layerSizes[network->layerNum - 1]; j++)
    {
      printf("%lf ", testingOutputs[i][j]);
    }
    printf("\n\n");

    printf("actual ");
    for(j = 0; j < network->layerSizes[network->layerNum - 1]; j++)
    {
      printf("%lf ", network->neurons[network->layerNum - 1][j]);
    }
    printf("\n------\n");
  }

  return 0;
}
