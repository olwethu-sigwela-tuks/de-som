#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cfloat>
#include <gsl/gsl_rng.h>
#include "DifferentialEvolution.h"

struct Stats {
    double mean;
    double stddev;
    double min;
    double max;
};

Stats computeStats(const std::vector<double>& values) {
    Stats s = {0.0, 0.0, 0.0, 0.0};
    if (values.empty()) return s;

    double sum = 0.0;
    s.min = values[0];
    s.max = values[0];

    for (double v : values) {
        sum += v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.mean = sum / values.size();

    if (values.size() > 1) {
        double varSum = 0.0;
        for (double v : values) {
            double diff = v - s.mean;
            varSum += diff * diff;
        }
        s.stddev = std::sqrt(varSum / (values.size() - 1));
    }
    return s;
}

struct SOMGrid {
    int width;
    int height;
    int dimension;
    std::vector<std::vector<double>> neurons;
};

SOMGrid initSOMGrid(int width, int height, const std::vector<std::vector<double>>& dataset, unsigned long seed) {
    int dimension = dataset[0].size();
    int numNeurons = width * height;
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);

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

void trainConventionalSOM(SOMGrid& grid, const std::vector<std::vector<double>>& dataset, int epochs, double initialLR, double lrDecay, double initialRadius, double radiusDecay, bool isHexa, unsigned long seed) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);

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

double evaluateSOMTopographicError(const SOMGrid& grid, const std::vector<std::vector<double>>& dataset, bool isHexa) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;
    int errorCount = 0;

    for (const auto& sample : dataset) {
        int bmu1 = -1;
        int bmu2 = -1;
        double minDist1 = DBL_MAX;
        double minDist2 = DBL_MAX;

        for (int i = 0; i < numNeurons; ++i) {
            double sumSq = 0.0;
            for (int k = 0; k < dimension; ++k) {
                double diff = sample[k] - grid.neurons[i][k];
                sumSq += diff * diff;
            }
            double dist = std::sqrt(sumSq);
            if (dist < minDist1) {
                minDist2 = minDist1;
                bmu2 = bmu1;
                minDist1 = dist;
                bmu1 = i;
            } else if (dist < minDist2) {
                minDist2 = dist;
                bmu2 = i;
            }
        }

        double gDist = getGridDistance(bmu1, bmu2, grid.width, isHexa);
        if (gDist > 1.05) {
            errorCount++;
        }
    }
    return (double)errorCount / dataset.size();
}

int main(int argc, char** argv) {
    char defaultData[] = "iris.dat";
    char defaultNames[] = "iris.som.names";
    char defaultOut[] = "results_30_runs_som.txt";
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

    auto dataset = loadDataSet(dataFile, namesFile);
    int width = 8;
    int height = 6;
    int epochs = 1000;
    double lrate = 0.5;
    double lrateDecay = 0.99;
    double radius = 2.0;
    double radiusDecay = 0.99;
    bool isHexa = true;

    std::vector<double> trainQEs;
    std::vector<double> testQEs;
    std::vector<double> trainTEs;
    std::vector<double> testTEs;

    std::ofstream out(outFile);

    for (unsigned long run = 1; run <= 30; ++run) {
        unsigned long seed = run;

        gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng, seed);

        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t i = indices.size() - 1; i > 0; --i) {
            size_t j = gsl_rng_uniform_int(rng, i + 1);
            std::swap(indices[i], indices[j]);
        }
        gsl_rng_free(rng);

        size_t trainSize = dataset.size() * 4 / 5;
        std::vector<std::vector<double>> trainSet;
        std::vector<std::vector<double>> testSet;
        trainSet.reserve(trainSize);
        testSet.reserve(dataset.size() - trainSize);

        for (size_t i = 0; i < dataset.size(); ++i) {
            if (i < trainSize) {
                trainSet.push_back(dataset[indices[i]]);
            } else {
                testSet.push_back(dataset[indices[i]]);
            }
        }

        SOMGrid som = initSOMGrid(width, height, trainSet, seed);
        trainConventionalSOM(som, trainSet, epochs, lrate, lrateDecay, radius, radiusDecay, isHexa, seed);

        double trainQE = evaluateSOMQuantizationError(som, trainSet);
        double testQE = evaluateSOMQuantizationError(som, testSet);
        double trainTE = evaluateSOMTopographicError(som, trainSet, isHexa);
        double testTE = evaluateSOMTopographicError(som, testSet, isHexa);

        trainQEs.push_back(trainQE);
        testQEs.push_back(testQE);
        trainTEs.push_back(trainTE);
        testTEs.push_back(testTE);

        std::cout << "Run: " << std::setw(2) << run 
                  << " | Seed: " << std::setw(2) << seed 
                  << " | Train QE: " << std::fixed << std::setprecision(6) << trainQE 
                  << " | Test QE: " << testQE 
                  << " | Train TE: " << trainTE 
                  << " | Test TE: " << testTE 
                  << std::endl;

        if (out.is_open()) {
            out << "Run: " << std::setw(2) << run 
                << " | Seed: " << std::setw(2) << seed 
                << " | Train QE: " << std::fixed << std::setprecision(6) << trainQE 
                << " | Test QE: " << testQE 
                << " | Train TE: " << trainTE 
                << " | Test TE: " << testTE 
                << std::endl;
        }
    }

    Stats trainQEStats = computeStats(trainQEs);
    Stats testQEStats = computeStats(testQEs);
    Stats trainTEStats = computeStats(trainTEs);
    Stats testTEStats = computeStats(testTEs);

    std::cout << std::endl;
    std::cout << "=== 30-Run Summary for Conventional SOM ===" << std::endl;
    std::cout << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
    std::cout << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
    std::cout << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
    std::cout << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;

    if (out.is_open()) {
        out << std::endl << std::endl;
        out << "=== 30-Run Summary for Conventional SOM ===" << std::endl;
        out << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
        out << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
        out << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
        out << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;
        out.close();
    }

    return 0;
}
