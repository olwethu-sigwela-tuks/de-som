#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <random>
#include "DifferentialEvolution.h"

std::string resolveFile(const std::string& filename) {
    if (access(filename.c_str(), F_OK) == 0) return filename;
    std::string parentPath = "../../" + filename;
    if (access(parentPath.c_str(), F_OK) == 0) return parentPath;
    std::string inside = "somlib/" + filename;
    if (access(inside.c_str(), F_OK) == 0) return inside;
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

bool loadBestDEParams(const std::string& filename, int& np, int& gen, int& neurons, double& cr, double& f) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Best NP:") != std::string::npos) {
            np = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best Gen:") != std::string::npos) {
            gen = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best Neurons:") != std::string::npos) {
            neurons = std::stoi(line.substr(line.find(":") + 1));
        } else if (line.find("Best CR:") != std::string::npos) {
            cr = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best F:") != std::string::npos) {
            f = std::stod(line.substr(line.find(":") + 1));
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string dataFile = "iris.dat";
    std::string namesFile = "iris.som.names";

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "iris" || arg1 == "glass" || arg1 == "ionosphere" || arg1 == "seeds" || arg1 == "haberman" || arg1 == "vertebral") {
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

    std::string outDir = resolveResultsDir(datasetName);
    std::string paramsFile = outDir + "/random_params_de.txt";

    int NP = 50;
    int numGenerations = 2000;
    int numNeurons = 48;
    double CR = 0.05;
    double F = 0.5;

    if (!loadBestDEParams(paramsFile, NP, numGenerations, numNeurons, CR, F)) {
        std::cerr << "Could not read " << paramsFile << ", using defaults." << std::endl;
    }

    std::cout << "DE Parameters: NP=" << NP << ", Gen=" << numGenerations << ", Neurons=" << numNeurons << ", CR=" << CR << ", F=" << F << std::endl;

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());

    int neuronSize = dataset[0].size();
    int width = 8;
    int height = 6;
    bool isHexa = true;

    std::string activeFile = outDir + "/runs_30_de.txt";
    std::ofstream out(activeFile);
    out << "Run\tSeed\tTrain_QE\tTest_QE\tTrain_TE\tTest_TE\tTime_s\n";
    out.flush();

    for (unsigned long run = 1; run <= 30; ++run) {
        sleep(1);
        unsigned long seed = (unsigned long)std::time(nullptr);

        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 shuffleRng(seed);
        std::shuffle(indices.begin(), indices.end(), shuffleRng);

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

        minMaxScaleTrainTest(trainSet, testSet);

        auto start = std::chrono::high_resolution_clock::now();
        std::mt19937 deRng(seed);
        auto bestWeights = differentialEvolution(trainSet, neuronSize, numNeurons, NP, numGenerations, CR, F, deRng);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        double trainQE = quantizationError(bestWeights, trainSet);
        double testQE = quantizationError(bestWeights, testSet);
        double trainTE = topographicError(bestWeights, width, height, isHexa, trainSet);
        double testTE = topographicError(bestWeights, width, height, isHexa, testSet);

        out << run << "\t" << seed << "\t" << trainQE << "\t" << testQE << "\t" << trainTE << "\t" << testTE << "\t" << elapsed << "\n";
        out.flush();

        std::cout << "Run " << std::setw(2) << run << "/30 | Seed: " << seed << " | Test QE: " << testQE << " | Test TE: " << testTE << " | Time: " << elapsed << "s" << std::endl;
    }

    out.close();
    std::cout << "Finished 30 runs of DE-SOM. Saved to: " << activeFile << std::endl;
    return 0;
}