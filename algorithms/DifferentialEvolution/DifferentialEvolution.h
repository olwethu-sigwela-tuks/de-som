#ifndef DIFF_H
#define DIFF_H

#include <vector>

std::vector<double> runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, unsigned long seed = 42);

std::vector<std::vector<double>> loadDataSet(char* dataFileName, char* namesFileName);

double quantizationError(const std::vector<double>& weights, const std::vector<std::vector<double>>& dataset);

double topographicError(const std::vector<double>& weights, int width, int height, bool isHexa, const std::vector<std::vector<double>>& dataset);

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP, int nGenerations, double CR, double F, unsigned long seed = 42);

#endif