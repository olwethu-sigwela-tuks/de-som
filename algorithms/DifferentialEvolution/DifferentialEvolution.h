#ifndef DIFF_H
#define DIFF_H

#include <random>
#include <vector>

struct TrainTestError{
    double trainingError;
    double testError;

};

struct CrossValError{
    double meanTrainingError;
    double stdTrainingError;
    double meanTestError;
    double stdTestError;
};


TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR);

std::vector<std::vector<double>> loadDataSet(char* dataFileName, char* namesFileName);

std::vector<std::vector<std::vector<double>>> kFoldSplit(std::vector<std::vector<double>>& dataset, int numFolds, int seed);

CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator);

#endif