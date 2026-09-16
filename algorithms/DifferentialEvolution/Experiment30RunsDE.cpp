#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
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

int main(int argc, char** argv) {
    char defaultData[] = "iris.dat";
    char defaultNames[] = "iris.som.names";
    char defaultOut[] = "results_30_runs_de.txt";
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
    int neuronSize = dataset[0].size();
    int width = 8;
    int height = 6;
    int numNeurons = width * height;
    int NP = 20;
    int numGenerations = 2000;
    double CR = 0.01;
    double F = 0.3;
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

        auto bestWeights = differentialEvolution(trainSet, neuronSize, numNeurons, NP, numGenerations, CR, F, seed);

        double trainQE = quantizationError(bestWeights, trainSet);
        double testQE = quantizationError(bestWeights, testSet);
        double trainTE = topographicError(bestWeights, width, height, isHexa, trainSet);
        double testTE = topographicError(bestWeights, width, height, isHexa, testSet);

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
    std::cout << "=== 30-Run Summary for Differential Evolution SOM ===" << std::endl;
    std::cout << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
    std::cout << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
    std::cout << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
    std::cout << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;

    if (out.is_open()) {
        out << std::endl << std::endl;
        out << "=== 30-Run Summary for Differential Evolution SOM ===" << std::endl;
        out << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
        out << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
        out << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
        out << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;
        out.close();
    }

    return 0;
}
