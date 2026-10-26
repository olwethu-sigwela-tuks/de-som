#ifndef DIFF_H
#define DIFF_H

#include <random>
#include <vector>
#include <string>
#include <gsl/gsl_rng.h>

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


struct TrainTestSplit {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> test;
};

struct FoldPartition {
    int numFolds;
    std::vector<std::vector<size_t>> foldIndices;
};

FoldPartition createFoldPartition(size_t datasetSize, int numFolds, unsigned int seed = 42);
void saveFoldPartition(const std::string& filename, const FoldPartition& partition);
FoldPartition loadOrCreateFoldPartition(const std::string& filename, size_t datasetSize, int numFolds, unsigned int seed = 42);
TrainTestSplit getTrainTestFromPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int testFoldIdx);

TrainTestSplit trainTestSplit(std::vector<std::vector<double>>& dataset, double testRatio, std::mt19937& g);

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons, std::vector<std::vector<double>>& dataset, gsl_rng* generator);
std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP, int nGenerations, double CR, double F, gsl_rng* gen);
std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP, int nGenerations, double CR, double F, std::mt19937& gen);

double quantizationError(const std::vector<double>& weights, const std::vector<std::vector<double>>& dataset);

std::vector<double> loadMapFile(const std::string& mapFileName);


TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator);
TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator);

std::vector<std::vector<double>> loadDataSet(char* dataFileName, char* namesFileName);

std::vector<std::vector<std::vector<double>>> kFoldSplit(std::vector<std::vector<double>>& dataset, int numFolds, std::mt19937& g);

CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator);
CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator);
CrossValError crossValidationWithPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator);
CrossValError crossValidationWithPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator);

void minMaxScaleTrainTest(std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet);
double topographicError(const std::vector<double>& weights, int width, int height, bool isHexa, const std::vector<std::vector<double>>& dataset);
void saveMapFile(const std::string& mapFileName, const std::vector<double>& weights, int width, int height, bool isHexa, int neuronSize);

#endif