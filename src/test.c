#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "include/neuralNetwork.h"
#include "include/matrix.h"

#define NET_NUM 1
#define LAYER_NUM 4
#define LEARNING_RATE 0.025
#define BATCH_SIZE 4
#define TRAINING_ROUNDS 20000
#define DATA_SIZE 784
#define OUTPUT_NUM 10
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

  nullCatchAndDie(file, "fopen returned NULL in parseCSV when assigning FILE *file");

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

int main(int argc, char **argv)
{
  if(argc < 3)
  {
    printf("USAGE: %s training.csv testing.csv\n", argv[0]);
    return 1;
  }

  srand(time(NULL));

  uint32_t i, j, k;

  uint32_t *layerSizes = malloc(sizeof(uint32_t) * LAYER_NUM);
  layerSizes[0] = DATA_SIZE;
  layerSizes[1] = 200;
  layerSizes[2] = 100;
  layerSizes[LAYER_NUM - 1] = OUTPUT_NUM;

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

  neuralNet_t *network[NET_NUM];

  for(i = 0; i < NET_NUM; i++)
  {
    network[i] = neuralNetInit(LEARNING_RATE, BATCH_SIZE, LAYER_NUM, layerSizes);
  }

  for(i = 0; i < TRAINING_ROUNDS; i++)
  {
    for(j = 0; j < NET_NUM; j++)
    {
      train(trainingSet.imgNum, trainingInputs, trainingOutputs, network[j]);
    }

    if(i % 100 == 0)
    {
      printf("%d\n", i);
    }
  }
  
  printf("%d training rounds completed\n", TRAINING_ROUNDS);

  uint8_t outCorrect, outPredictedConsensus;
  uint8_t outPredicted;
  uint32_t predictions[OUTPUT_NUM];
  uint32_t maxPredictionInt;
  double maxPrediction;

  uint32_t numCorrect = 0;

  uint32_t groupSavings = 0;

  for(i = 0; i < testingSet.imgNum; i++)
  {
    for(j = 0; j < OUTPUT_NUM; j++)
    {
      predictions[j] = 0;
    }

    for(j = 0; j < NET_NUM; j++)
    {
      for(k = 0; k < DATA_SIZE; k++)
      {
        network[j]->neurons[0][k] = testingInputs[i][k];
      }

      forwardPass(network[j]);
    }

    outCorrect = 0;

    printf("expected ");
    for(j = 0; j < OUTPUT_NUM; j++)
    {
      printf("%lf ", testingOutputs[i][j]);

      if(testingOutputs[i][j] > 0.01)
      {
        outCorrect = j;
      }
    }
    printf("\n\n");
    
    outPredicted = 0;
    maxPrediction = 0.0;

    printf("actual ");


    for(j = 0; j < NET_NUM; j++)
    {
      printf("    network %d    ", j);
     
      for(k = 0; k < OUTPUT_NUM; k++)
      {
        printf("%lf ", network[j]->neurons[LAYER_NUM - 1][k]);

        if(network[j]->neurons[LAYER_NUM - 1][k] >= maxPrediction)
        {
          outPredicted = k;
          maxPrediction = network[j]->neurons[LAYER_NUM - 1][k];
        }
      }

      predictions[outPredicted]++;
    }

    outPredictedConsensus = 0;
    maxPredictionInt = 0;

    for(j = 0; j < OUTPUT_NUM; j++)
    {
      printf("%d ", predictions[j]);
      if(predictions[j] > maxPredictionInt)
      {
        outPredictedConsensus = j;
        maxPredictionInt = predictions[j];
      }
    }
    printf("\n");

    if(maxPredictionInt != NET_NUM && outPredictedConsensus == outCorrect)
    {
      groupSavings++;
    }

    if(outPredictedConsensus == outCorrect)
    {
      numCorrect++;
      printf("Correct on test example #%d\n", i);
    }
    else
    {
      printf("Incorrect on test example #%d\n", i);
    }

    printf("\n------\n");
  }

  printf("network accuracy: %lf\ngroupSavings: %d\n", (double)((double)numCorrect / (double)(i + 1)), groupSavings);

  return 0;
}
