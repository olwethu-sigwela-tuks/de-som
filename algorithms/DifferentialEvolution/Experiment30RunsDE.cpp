#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
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

bool loadBestDEParams(const std::string& filename, int& NP, int& numGenerations, int& numNeurons, double& CR, double& F) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Best NP:") != std::string::npos) {
            NP = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best Generations:") != std::string::npos) {
            numGenerations = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best Neurons:") != std::string::npos) {
            numNeurons = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best CR:") != std::string::npos) {
            CR = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best F:") != std::string::npos) {
            F = std::stod(line.substr(line.find(":") + 1));
        }
    }
    return true;
}

const unsigned long RUN_SEEDS[30] = {
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020,
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030
};

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

    int NP = 20;
    int numGenerations = 4000;
    int numNeurons = 48;
    double CR = 0.005;
    double F = 0.15;

    std::string outDir = resolveResultsDir(datasetName);
    std::string paramsFile = outDir + "/best_params_de.txt";

    if (loadBestDEParams(paramsFile, NP, numGenerations, numNeurons, CR, F)) {
        std::cout << "Loaded optimal parameters from: " << paramsFile << std::endl;
    } else {
        std::cout << "Using default parameters for DE." << std::endl;
    }

    if (argc >= 6) {
        NP = std::stoi(argv[2]);
        numGenerations = std::stoi(argv[3]);
        CR = std::stod(argv[4]);
        F = std::stod(argv[5]);
    }

    std::cout << "Running DE 30 runs with:" << std::endl;
    std::cout << "  Dataset: " << datasetName << " (" << dataFile << ", " << namesFile << ")" << std::endl;
    std::cout << "  NP: " << NP << " | Gen: " << numGenerations << " | Neurons: " << numNeurons << " | CR: " << CR << " | F: " << F << std::endl;

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());
    int neuronSize = dataset[0].size();
    int width = 8;
    int height = 6;
    if (numNeurons == 24) {
        width = 6;
        height = 4;
    }
    bool isHexa = true;

    std::vector<double> trainQEs;
    std::vector<double> testQEs;
    std::vector<double> trainTEs;
    std::vector<double> testTEs;

    std::string timestamp = getTimestampString();
    std::string timestampedFile = outDir + "/runs_30_de_" + timestamp + ".txt";
    std::string activeFile = outDir + "/runs_30_de.txt";

    std::ofstream outTime(timestampedFile);
    std::ofstream outActive(activeFile);

    for (unsigned long run = 1; run <= 30; ++run) {
        unsigned long seed = RUN_SEEDS[run - 1];

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

        if (datasetName == "glass") {
            normalizeTrainTest(trainSet, testSet);
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
                  << " | Seed: " << seed 
                  << " | Train QE: " << std::fixed << std::setprecision(6) << trainQE 
                  << " | Test QE: " << testQE 
                  << " | Train TE: " << trainTE 
                  << " | Test TE: " << testTE 
                  << std::endl;

        if (outTime.is_open()) {
            outTime << "Run: " << std::setw(2) << run 
                    << " | Seed: " << seed 
                    << " | Train QE: " << std::fixed << std::setprecision(6) << trainQE 
                    << " | Test QE: " << testQE 
                    << " | Train TE: " << trainTE 
                    << " | Test TE: " << testTE 
                    << std::endl;
        }

        if (outActive.is_open()) {
            outActive << "Run: " << std::setw(2) << run 
                      << " | Seed: " << seed 
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
    std::cout << "=== 30-Run Summary for Differential Evolution SOM (" << datasetName << ") ===" << std::endl;
    std::cout << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
    std::cout << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
    std::cout << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
    std::cout << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;

    if (outTime.is_open()) {
        outTime << std::endl << std::endl;
        outTime << "=== 30-Run Summary for Differential Evolution SOM (" << datasetName << ") ===" << std::endl;
        outTime << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
        outTime << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
        outTime << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
        outTime << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;
        outTime.close();
    }

    if (outActive.is_open()) {
        outActive << std::endl << std::endl;
        outActive << "=== 30-Run Summary for Differential Evolution SOM (" << datasetName << ") ===" << std::endl;
        outActive << "Train Quantization Error: Mean = " << trainQEStats.mean << " | StdDev = " << trainQEStats.stddev << " | Min = " << trainQEStats.min << " | Max = " << trainQEStats.max << std::endl;
        outActive << "Test Quantization Error:  Mean = " << testQEStats.mean  << " | StdDev = " << testQEStats.stddev  << " | Min = " << testQEStats.min  << " | Max = " << testQEStats.max  << std::endl;
        outActive << "Train Topographic Error:  Mean = " << trainTEStats.mean << " | StdDev = " << trainTEStats.stddev << " | Min = " << trainTEStats.min << " | Max = " << trainTEStats.max << std::endl;
        outActive << "Test Topographic Error:   Mean = " << testTEStats.mean  << " | StdDev = " << testTEStats.stddev  << " | Min = " << testTEStats.min  << " | Max = " << testTEStats.max  << std::endl;
        outActive.close();
    }

    return 0;
}
