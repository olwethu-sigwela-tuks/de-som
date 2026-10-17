#include <iostream>
#include <fstream>
#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include "DifferentialEvolution.h"

struct Fold {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> test;
};

std::vector<Fold> create5Folds(const std::vector<std::vector<double>>& dataset, unsigned long seed = 42) {
    std::vector<Fold> folds(5);
    size_t n = dataset.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = gsl_rng_uniform_int(rng, i + 1);
        std::swap(indices[i], indices[j]);
    }
    gsl_rng_free(rng);
    size_t foldSize = n / 5;
    for (size_t k = 0; k < 5; ++k) {
        size_t testStart = k * foldSize;
        size_t testEnd = (k == 4) ? n : (k + 1) * foldSize;
        for (size_t i = 0; i < n; ++i) {
            if (i >= testStart && i < testEnd) {
                folds[k].test.push_back(dataset[indices[i]]);
            } else {
                folds[k].train.push_back(dataset[indices[i]]);
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

SOMGrid initSOMGrid(int width, int height, const std::vector<std::vector<double>>& dataset) {
    int dimension = dataset[0].size();
    int numNeurons = width * height;
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42);

    std::vector<double> minVal(dimension, DBL_MAX);
    std::vector<double> maxVal(dimension, -DBL_MAX);
    for (const auto& sample : dataset) {
        for (int k = 0; k < dimension; ++k) {
            if (sample[k] < minVal[k]) minVal[k] = sample[k];
            if (sample[k] > maxVal[k]) maxVal[k] = sample[k];
        }
    }

    SOMGrid grid;
    grid.width = width;
    grid.height = height;
    grid.dimension = dimension;
    grid.neurons.resize(numNeurons, std::vector<double>(dimension));

    for (int i = 0; i < numNeurons; ++i) {
        for (int k = 0; k < dimension; ++k) {
            grid.neurons[i][k] = minVal[k] + (maxVal[k] - minVal[k]) * gsl_rng_uniform(rng);
        }
    }

    gsl_rng_free(rng);
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

void trainConventionalSOM(SOMGrid& grid, const std::vector<std::vector<double>>& dataset, int epochs, double initialLR, double lrDecay, double initialRadius, double radiusDecay, bool isHexa) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42);

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    double lr = initialLR;
    double radius = initialRadius;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = indices.size() - 1; i > 0; --i) {
            size_t j = gsl_rng_uniform_int(rng, i + 1);
            std::swap(indices[i], indices[j]);
        }

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

    gsl_rng_free(rng);
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

void selectParametersSOM(const std::string& dataFile, const std::string& namesFile, const std::string& datasetName) {
    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);

    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());
    auto folds = create5Folds(dataset);
    if (datasetName == "glass") {
        for (size_t k = 0; k < 5; ++k) {
            normalizeTrainTest(folds[k].train, folds[k].test);
        }
    }

    std::vector<std::pair<int, int>> gridSizes = {{6, 4}, {8, 6}};
    std::vector<int> epochCandidates = {1000, 2000};
    std::vector<double> lrCandidates = {0.05, 0.1, 0.5};
    std::vector<double> lrDecayCandidates = {0.99, 0.995};
    std::vector<double> radiusCandidates = {2.0, 4.0, 6.0};
    std::vector<double> radiusDecayCandidates = {0.99, 0.995};
    std::vector<bool> topologyCandidates = {true};

    double bestCvError = DBL_MAX;
    SOMParams bestParams = {0, 0, 0, 0.0, 0.0, 0.0, 0.0, true};

    std::string outDir = resolveResultsDir(datasetName);
    std::string timestamp = getTimestampString();
    std::string timestampedFile = outDir + "/best_params_som_" + timestamp + ".txt";
    std::string activeFile = outDir + "/best_params_som.txt";

    std::ofstream outTime(timestampedFile);
    std::ofstream outActive(activeFile);

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

                                if (outTime.is_open()) {
                                    outTime << "Grid: " << grid.first << "x" << grid.second 
                                            << " | Epochs: " << epochs 
                                            << " | LRate: " << lrate 
                                            << " | LRDecay: " << lrateDecay 
                                            << " | Radius: " << radius 
                                            << " | RadiusDecay: " << radiusDecay 
                                            << " | Train Error: " << avgTrainError 
                                            << " | 5-Fold Test Error: " << avgTestError 
                                            << std::endl;
                                }

                                if (outActive.is_open()) {
                                    outActive << "Grid: " << grid.first << "x" << grid.second 
                                              << " | Epochs: " << epochs 
                                              << " | LRate: " << lrate 
                                              << " | LRDecay: " << lrateDecay 
                                              << " | Radius: " << radius 
                                              << " | RadiusDecay: " << radiusDecay 
                                              << " | Train Error: " << avgTrainError 
                                              << " | 5-Fold Test Error: " << avgTestError 
                                              << std::endl;
                                }

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

    if (outTime.is_open()) {
        outTime << std::endl << std::endl;
        outTime << "=== Best Hyperparameters for Conventional SOM from 5-Fold Cross-Validation ===" << std::endl;
        outTime << "Best Grid Width: " << bestParams.width << std::endl;
        outTime << "Best Grid Height: " << bestParams.height << std::endl;
        outTime << "Best Epochs: " << bestParams.epochs << std::endl;
        outTime << "Best Learning Rate: " << bestParams.lrate << std::endl;
        outTime << "Best Learning Rate Decay: " << bestParams.lrateDecay << std::endl;
        outTime << "Best Radius: " << bestParams.radius << std::endl;
        outTime << "Best Radius Decay: " << bestParams.radiusDecay << std::endl;
        outTime << "Best Topology: " << (bestParams.isHexa ? "hexa" : "rect") << std::endl;
        outTime << "Best 5-Fold Test Quantization Error: " << bestCvError << std::endl;
        outTime.close();
    }

    if (outActive.is_open()) {
        outActive << std::endl << std::endl;
        outActive << "=== Best Hyperparameters for Conventional SOM from 5-Fold Cross-Validation ===" << std::endl;
        outActive << "Best Grid Width: " << bestParams.width << std::endl;
        outActive << "Best Grid Height: " << bestParams.height << std::endl;
        outActive << "Best Epochs: " << bestParams.epochs << std::endl;
        outActive << "Best Learning Rate: " << bestParams.lrate << std::endl;
        outActive << "Best Learning Rate Decay: " << bestParams.lrateDecay << std::endl;
        outActive << "Best Radius: " << bestParams.radius << std::endl;
        outActive << "Best Radius Decay: " << bestParams.radiusDecay << std::endl;
        outActive << "Best Topology: " << (bestParams.isHexa ? "hexa" : "rect") << std::endl;
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

    selectParametersSOM(dataFile, namesFile, datasetName);

    return 0;
}
