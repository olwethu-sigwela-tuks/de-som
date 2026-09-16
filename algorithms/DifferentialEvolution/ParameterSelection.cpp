#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <unistd.h>
#include "DifferentialEvolution.h"

struct Fold {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> test;
};

std::vector<Fold> create5Folds(const std::vector<std::vector<double>>& dataset) {
    std::vector<Fold> folds(5);
    size_t n = dataset.size();
    size_t foldSize = n / 5;
    for (size_t k = 0; k < 5; ++k) {
        size_t testStart = k * foldSize;
        size_t testEnd = (k == 4) ? n : (k + 1) * foldSize;
        for (size_t i = 0; i < n; ++i) {
            if (i >= testStart && i < testEnd) {
                folds[k].test.push_back(dataset[i]);
            } else {
                folds[k].train.push_back(dataset[i]);
            }
        }
    }
    return folds;
}

void selectParameters(char* dataFile, char* namesFile, char* outFile) {
    auto dataset = loadDataSet(dataFile, namesFile);
    auto folds = create5Folds(dataset);

    std::vector<int> npCandidates = {20};
    std::vector<int> genCandidates = {2000, 3000, 4000};
    std::vector<int> neuronCandidates = {48};
    std::vector<double> crCandidates = {0.002, 0.005, 0.008, 0.010, 0.015};
    std::vector<double> fCandidates = {0.15, 0.20, 0.25, 0.30};

    double bestCvError = DBL_MAX;
    int bestNP = 0;
    int bestGen = 0;
    int bestNeurons = 0;
    double bestCR = 0.0;
    double bestF = 0.0;

    std::ofstream out(outFile);

    for (int NP : npCandidates) {
        for (int numGenerations : genCandidates) {
            for (int numNeurons : neuronCandidates) {
                for (double CR : crCandidates) {
                    for (double F : fCandidates) {
                        double totalTestError = 0.0;
                        double totalTrainError = 0.0;

                        for (size_t k = 0; k < 5; ++k) {
                            auto errors = runDE(folds[k].train, folds[k].test, numGenerations, numNeurons, NP, F, CR);
                            totalTrainError += errors[0];
                            totalTestError += errors[1];
                        }

                        double avgTrainError = totalTrainError / 5.0;
                        double avgTestError = totalTestError / 5.0;

                        std::cout << "NP: " << NP 
                                  << " | Gen: " << numGenerations 
                                  << " | Neurons: " << numNeurons 
                                  << " | CR: " << CR 
                                  << " | F: " << F 
                                  << " | Train Error: " << avgTrainError 
                                  << " | 5-Fold Test Error: " << avgTestError 
                                  << std::endl;

                        if (out.is_open()) {
                            out << "NP: " << NP 
                                << " | Gen: " << numGenerations 
                                << " | Neurons: " << numNeurons 
                                << " | CR: " << CR 
                                << " | F: " << F 
                                << " | Train Error: " << avgTrainError 
                                << " | 5-Fold Test Error: " << avgTestError 
                                << std::endl;
                        }

                        if (avgTestError < bestCvError) {
                            bestCvError = avgTestError;
                            bestNP = NP;
                            bestGen = numGenerations;
                            bestNeurons = numNeurons;
                            bestCR = CR;
                            bestF = F;
                        }
                    }
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << "=== Best Hyperparameters from 5-Fold Cross-Validation ===" << std::endl;
    std::cout << "Best NP: " << bestNP << std::endl;
    std::cout << "Best Generations: " << bestGen << std::endl;
    std::cout << "Best Neurons: " << bestNeurons << std::endl;
    std::cout << "Best CR: " << bestCR << std::endl;
    std::cout << "Best F: " << bestF << std::endl;
    std::cout << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;

    if (out.is_open()) {
        out << std::endl << std::endl;
        out << "=== Best Hyperparameters from 5-Fold Cross-Validation ===" << std::endl;
        out << "Best NP: " << bestNP << std::endl;
        out << "Best Generations: " << bestGen << std::endl;
        out << "Best Neurons: " << bestNeurons << std::endl;
        out << "Best CR: " << bestCR << std::endl;
        out << "Best F: " << bestF << std::endl;
        out << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;
        out.close();
    }
}

int main(int argc, char** argv) {
    char defaultData[] = "iris.dat";
    char defaultNames[] = "iris.som.names";
    char defaultOut[] = "best_parameters.txt";
    char* dataFile = defaultData;
    char* namesFile = defaultNames;
    char* outFile = defaultOut;

    if (argc >= 3) {
        dataFile = argv[1];
        namesFile = argv[2];
    }
    if (argc >= 4) {
        outFile = argv[3];
    }

    selectParameters(dataFile, namesFile, outFile);

    return 0;
}