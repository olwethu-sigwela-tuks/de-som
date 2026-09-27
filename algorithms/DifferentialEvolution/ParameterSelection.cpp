#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
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

std::string getTimestampString() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", &tm);
    return std::string(buf);
}

std::string resolveFile(const std::string& filename) {
    if (access(filename.c_str(), F_OK) == 0) return filename;
    std::string parentPath = "../../" + filename;
    if (access(parentPath.c_str(), F_OK) == 0) return parentPath;
    return filename;
}

std::string resolveResultsDir(const std::string& datasetName) {
    if (access("../../iris.dat", F_OK) == 0) {
        mkdir("../../results", 0777);
        std::string dir = "../../results/" + datasetName;
        mkdir(dir.c_str(), 0777);
        return dir;
    }
    mkdir("results", 0777);
    std::string dir = "results/" + datasetName;
    mkdir(dir.c_str(), 0777);
    return dir;
}

void selectParameters(const std::string& dataFile, const std::string& namesFile, const std::string& datasetName) {
    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);

    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());
    if (datasetName == "glass") {
        dataset = minMaxNormalize(dataset);
    }
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

    std::string outDir = resolveResultsDir(datasetName);
    std::string timestamp = getTimestampString();
    std::string timestampedFile = outDir + "/best_params_de_" + timestamp + ".txt";
    std::string activeFile = outDir + "/best_params_de.txt";

    std::ofstream outTime(timestampedFile);
    std::ofstream outActive(activeFile);

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

                        if (outTime.is_open()) {
                            outTime << "NP: " << NP 
                                    << " | Gen: " << numGenerations 
                                    << " | Neurons: " << numNeurons 
                                    << " | CR: " << CR 
                                    << " | F: " << F 
                                    << " | Train Error: " << avgTrainError 
                                    << " | 5-Fold Test Error: " << avgTestError 
                                    << std::endl;
                        }

                        if (outActive.is_open()) {
                            outActive << "NP: " << NP 
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

    if (outTime.is_open()) {
        outTime << std::endl << std::endl;
        outTime << "=== Best Hyperparameters from 5-Fold Cross-Validation ===" << std::endl;
        outTime << "Best NP: " << bestNP << std::endl;
        outTime << "Best Generations: " << bestGen << std::endl;
        outTime << "Best Neurons: " << bestNeurons << std::endl;
        outTime << "Best CR: " << bestCR << std::endl;
        outTime << "Best F: " << bestF << std::endl;
        outTime << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;
        outTime.close();
    }

    if (outActive.is_open()) {
        outActive << std::endl << std::endl;
        outActive << "=== Best Hyperparameters from 5-Fold Cross-Validation ===" << std::endl;
        outActive << "Best NP: " << bestNP << std::endl;
        outActive << "Best Generations: " << bestGen << std::endl;
        outActive << "Best Neurons: " << bestNeurons << std::endl;
        outActive << "Best CR: " << bestCR << std::endl;
        outActive << "Best F: " << bestF << std::endl;
        outActive << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;
        outActive.close();
    }
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string dataFile = "iris.dat";
    std::string namesFile = "iris.som.names";

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "iris" || arg1 == "glass" || arg1 == "ionosphere") {
            datasetName = arg1;
            dataFile = arg1 + ".dat";
            namesFile = arg1 + ".som.names";
        } else {
            dataFile = arg1;
            datasetName = arg1;
            size_t slash = datasetName.find_last_of("/\\");
            if (slash != std::string::npos) datasetName = datasetName.substr(slash + 1);
            size_t dot = datasetName.find_last_of(".");
            if (dot != std::string::npos) datasetName = datasetName.substr(0, dot);
            if (argc >= 3) {
                namesFile = argv[2];
            } else {
                namesFile = datasetName + ".som.names";
            }
        }
    }

    selectParameters(dataFile, namesFile, datasetName);

    return 0;
}