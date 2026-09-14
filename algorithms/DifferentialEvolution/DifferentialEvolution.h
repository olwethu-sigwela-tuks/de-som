#ifndef DIFF_H
#define DIFF_H

#include <vector>

std::vector<double> runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR);

std::vector<std::vector<double>> loadDataSet(char* dataFileName, char* namesFileName);

#endif