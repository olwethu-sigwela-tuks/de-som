#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
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

struct SOMGrid {
    int width;
    int height;
    int dimension;
    std::vector<std::vector<double>> neurons;
};

SOMGrid initSOMGrid(int width, int height, const std::vector<std::vector<double>>& dataset, unsigned int seed = 42) {
    int dimension = dataset[0].size();
    int numNeurons = width * height;
    std::mt19937 gen(seed);

    std::vector<double> minVal(dimension, DBL_MAX);
    std::vector<double> maxVal(dimension, -DBL_MAX);
    for (const auto& sample : dataset) {
        for (int k = 0; k < dimension; ++k) {
            if (sample[k] < minVal[k]) minVal[k] = sample[k];
            if (sample[k] > maxVal[k]) maxVal[k] = sample[k];
        }
    }

    std::vector<std::uniform_real_distribution<double>> distribs;
    for (int k = 0; k < dimension; ++k) {
        distribs.emplace_back(minVal[k], maxVal[k]);
    }

    SOMGrid grid;
    grid.width = width;
    grid.height = height;
    grid.dimension = dimension;
    grid.neurons.resize(numNeurons, std::vector<double>(dimension));

    for (int i = 0; i < numNeurons; ++i) {
        for (int k = 0; k < dimension; ++k) {
            grid.neurons[i][k] = distribs[k](gen);
        }
    }
    return grid;
}

double getGridDistance(int i, int j, int width, bool isHexa) {
    int y1 = i / width;
    int x1 = i % width;
    int y2 = j / width;
    int x2 = j % width;
    double dx = x1 - x2;
    double dy = y1 - y2;
    if (isHexa && (y1 % 2 != y2 % 2)) {
        if (y1 % 2 == 1) dx += 0.5;
        else dx -= 0.5;
    }
    return std::sqrt(dx * dx + dy * dy * (isHexa ? 0.75 : 1.0));
}

void trainConventionalSOM(SOMGrid& grid, const std::vector<std::vector<double>>& dataset, int epochs, double initialLR, double lrDecay, double initialRadius, double radiusDecay, bool isHexa, unsigned int seed = 42) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;
    std::mt19937 gen(seed);
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    double lr = initialLR;
    double radius = initialRadius;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);
        double radSq2 = 2.0 * radius * radius;

        for (size_t idx : indices) {
            const auto& sample = dataset[idx];
            int bmu = 0;
            double minDist = DBL_MAX;

            for (int i = 0; i < numNeurons; ++i) {
                double dist = 0.0;
                for (int k = 0; k < dimension; ++k) {
                    double diff = sample[k] - grid.neurons[i][k];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bmu = i;
                }
            }

            for (int i = 0; i < numNeurons; ++i) {
                double gDist = getGridDistance(bmu, i, grid.width, isHexa);
                if (gDist <= radius * 3.0) {
                    double h = std::exp(-(gDist * gDist) / radSq2);
                    double factor = lr * h;
                    for (int k = 0; k < dimension; ++k) {
                        grid.neurons[i][k] += factor * (sample[k] - grid.neurons[i][k]);
                    }
                }
            }
        }

        lr *= lrDecay;
        radius *= radiusDecay;
        if (radius < 0.5) radius = 0.5;
    }
}

double evaluateSOMQuantizationError(const SOMGrid& grid, const std::vector<std::vector<double>>& dataset) {
    double totalDist = 0.0;
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;

    for (const auto& sample : dataset) {
        double minDist = DBL_MAX;
        for (int i = 0; i < numNeurons; ++i) {
            double sumSq = 0.0;
            for (int k = 0; k < dimension; ++k) {
                double diff = sample[k] - grid.neurons[i][k];
                sumSq += diff * diff;
            }
            double dist = std::sqrt(sumSq);
            if (dist < minDist) {
                minDist = dist;
            }
        }
        totalDist += minDist;
    }
    return totalDist / dataset.size();
}

struct SOMParams {
    int width;
    int height;
    int epochs;
    double lrate;
    double lrateDecay;
    double radius;
    double radiusDecay;
    bool isHexa;
};

void selectParametersSOM(char* dataFile, char* namesFile, char* outFile) {
    auto dataset = loadDataSet(dataFile, namesFile);
    auto folds = create5Folds(dataset);

    std::vector<std::pair<int, int>> gridSizes = {{8, 6}, {6, 4}, {10, 7}};
    std::vector<int> epochCandidates = {100, 200, 500};
    std::vector<double> lrCandidates = {0.05, 0.1, 0.5};
    std::vector<double> lrDecayCandidates = {0.95, 0.99};
    std::vector<double> radiusCandidates = {2.0, 4.0, 6.0};
    std::vector<double> radiusDecayCandidates = {0.95, 0.99};
    std::vector<bool> topologyCandidates = {true};

    double bestCvError = DBL_MAX;
    SOMParams bestParams = {0, 0, 0, 0.0, 0.0, 0.0, 0.0, true};

    for (auto grid : gridSizes) {
        for (int epochs : epochCandidates) {
            for (double lrate : lrCandidates) {
                for (double lrateDecay : lrDecayCandidates) {
                    for (double radius : radiusCandidates) {
                        for (double radiusDecay : radiusDecayCandidates) {
                            for (bool isHexa : topologyCandidates) {
                                double totalTestError = 0.0;
                                double totalTrainError = 0.0;

                                for (size_t k = 0; k < 5; ++k) {
                                    SOMGrid som = initSOMGrid(grid.first, grid.second, folds[k].train);
                                    trainConventionalSOM(som, folds[k].train, epochs, lrate, lrateDecay, radius, radiusDecay, isHexa);
                                    totalTrainError += evaluateSOMQuantizationError(som, folds[k].train);
                                    totalTestError += evaluateSOMQuantizationError(som, folds[k].test);
                                }

                                double avgTrainError = totalTrainError / 5.0;
                                double avgTestError = totalTestError / 5.0;

                                std::cout << "Grid: " << grid.first << "x" << grid.second 
                                          << " | Epochs: " << epochs 
                                          << " | LRate: " << lrate 
                                          << " | LRDecay: " << lrateDecay 
                                          << " | Radius: " << radius 
                                          << " | RadiusDecay: " << radiusDecay 
                                          << " | Train Error: " << avgTrainError 
                                          << " | 5-Fold Test Error: " << avgTestError 
                                          << std::endl;

                                if (avgTestError < bestCvError) {
                                    bestCvError = avgTestError;
                                    bestParams = {grid.first, grid.second, epochs, lrate, lrateDecay, radius, radiusDecay, isHexa};
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::cout << std::endl;
    std::cout << "=== Best Hyperparameters for Conventional SOM from 5-Fold Cross-Validation ===" << std::endl;
    std::cout << "Best Grid Width: " << bestParams.width << std::endl;
    std::cout << "Best Grid Height: " << bestParams.height << std::endl;
    std::cout << "Best Epochs: " << bestParams.epochs << std::endl;
    std::cout << "Best Learning Rate: " << bestParams.lrate << std::endl;
    std::cout << "Best Learning Rate Decay: " << bestParams.lrateDecay << std::endl;
    std::cout << "Best Radius: " << bestParams.radius << std::endl;
    std::cout << "Best Radius Decay: " << bestParams.radiusDecay << std::endl;
    std::cout << "Best Topology: " << (bestParams.isHexa ? "hexa" : "rect") << std::endl;
    std::cout << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;

    std::ofstream out(outFile);
    if (out.is_open()) {
        out << "GridWidth: " << bestParams.width << std::endl;
        out << "GridHeight: " << bestParams.height << std::endl;
        out << "Epochs: " << bestParams.epochs << std::endl;
        out << "LearningRate: " << bestParams.lrate << std::endl;
        out << "LearningRateDecay: " << bestParams.lrateDecay << std::endl;
        out << "Radius: " << bestParams.radius << std::endl;
        out << "RadiusDecay: " << bestParams.radiusDecay << std::endl;
        out << "Topology: " << (bestParams.isHexa ? "hexa" : "rect") << std::endl;
        out << "BestTestQuantizationError: " << bestCvError << std::endl;
        out.close();
    }
}

int main(int argc, char** argv) {
    char defaultData[] = "iris.dat";
    char defaultNames[] = "iris.som.names";
    char defaultOut[] = "best_parameters_som.txt";
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

    selectParametersSOM(dataFile, namesFile, outFile);

    return 0;
}
