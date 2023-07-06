#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define LAYER_NUM 5
#define LEARNING_RATE 0.01
#define EPOCH_LEN 20
#define TRAINING_ROUNDS 7500
#define STOCHASTIC true
#define DATA_SIZE 784
#define LABEL_OFFSET (0.0)

typedef struct IMAGE_STRUCT
{
    double label;
    double data[DATA_SIZE];
} img_t;

typedef struct IMAGE_SET
{
  img_t *imgs;
  uint32_t imgNum;
} imgSet_t;

imgSet_t parseCSV(const char* fileName)
{
  uint32_t i;

  FILE *file = fopen(fileName, "r");

  nullCatchAndDie(file, "fopen returned NULL in parseCSV when assigning FILE *file")

  img_t *imgs = malloc(sizeof(img_t));

  uint32_t cnt = 0;

  char *tok;

  char lineBuf[16384];

  while(fgets(lineBuf, 16384, file))
  {
    tok = strtok(lineBuf, ",");

    imgs[cnt].label = strtod(tok, NULL);

    for(i = 0; i < 784; i++)
    {
      tok = strtok(NULL, ",");

      imgs[cnt].data[i] = strtod(tok, NULL) / 255.0;
    }

    cnt++;

    imgs = realloc(imgs, sizeof(img_t) * (cnt + 1));
  }

  fclose(file);

  imgSet_t result;
  result.imgs = imgs;
  result.imgNum = cnt;

  return result;
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

int main(int argc, char **argv)
{
  if(argc < 3)
  {
    printf("USAGE: %s training.csv testing.csv\n", argv[0]);
    return 1;
  }

  srand(time(NULL));

  uint32_t i, j;

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = 784;
  layerSizes[1] = 30;
  layerSizes[2] = 30;
  layerSizes[3] = 30;
  layerSizes[4] = 26;

  imgSet_t trainingSet = parseCSV(argv[1]);
  imgSet_t testingSet = parseCSV(argv[2]);

  img_t *trainingImgs = trainingSet.imgs;
  img_t *testingImgs = testingSet.imgs;

  double **trainingInputs = malloc(sizeof(double *) * trainingSet.imgNum);
  for(i = 0; i < trainingSet.imgNum; i++)
  {
    trainingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[0]; j++)
    {
      trainingInputs[i][j] = trainingImgs[i].data[j];
    }
  }
  double **trainingOutputs = malloc(sizeof(double *) * trainingSet.imgNum);
  for(i = 0; i < trainingSet.imgNum; i++)
  {
    trainingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[LAYER_NUM - 1]; j++)
    {
      if(j == (uint32_t)round(trainingImgs[i].label + LABEL_OFFSET))
      {
        trainingOutputs[i][j] = 1.0;
      }
      else
      {
        trainingOutputs[i][j] = 0.0;
      }
    }
  }

  double **testingInputs = malloc(sizeof(double *) * testingSet.imgNum);
  for(i = 0; i < testingSet.imgNum; i++)
  {
    testingInputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[0]; j++)
    {
      testingInputs[i][j] = testingImgs[i].data[j];
    }
  }
  double **testingOutputs = malloc(sizeof(double *) * testingSet.imgNum);
  for(i = 0; i < testingSet.imgNum; i++)
  {
    testingOutputs[i] = malloc(sizeof(double) * layerSizes[0]);
    for(j = 0; j < layerSizes[LAYER_NUM - 1]; j++)
    {
      if(j == (uint32_t)round(testingImgs[i].label + LABEL_OFFSET))
      {
        testingOutputs[i][j] = 1.0;
      }
      else
      {
        testingOutputs[i][j] = 0.0;
      }
    }
  }

  neuralNet_t *network = neuralNetInit(LEARNING_RATE, EPOCH_LEN, LAYER_NUM, layerSizes);

  printf("neuralNetInit(%lf, %d, %d, {784, 8, 8, 10}) yielded a staring network with the following weights and biases\n", LEARNING_RATE, EPOCH_LEN, LAYER_NUM);

  printNetwork(network);

  printf("\n");

  for(i = 0; i < TRAINING_ROUNDS; i++)
  {
    train(STOCHASTIC, trainingSet.imgNum, trainingInputs, trainingOutputs, network);
    if(i % 100 == 0)
    {
      printf("%d\n", i);
    }
  }

  printf("%d %s training rounds completed\n", TRAINING_ROUNDS, STOCHASTIC ? "stochastic" : "batch");

  for(i = 0; i < testingSet.imgNum; i++)
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
