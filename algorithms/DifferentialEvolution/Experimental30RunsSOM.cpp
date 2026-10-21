#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdlib>
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

std::string resolveSOMPAK(const std::string& binary) {
    if (access(("algorithms/SOM_PAK/" + binary).c_str(), X_OK) == 0) return "algorithms/SOM_PAK/" + binary;
    if (access(("../SOM_PAK/" + binary).c_str(), X_OK) == 0) return "../SOM_PAK/" + binary;
    if (access(("../../algorithms/SOM_PAK/" + binary).c_str(), X_OK) == 0) return "../../algorithms/SOM_PAK/" + binary;
    return binary;
}

void saveSOMDataFile(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream out(filename);
    out << data[0].size() << "\n";
    for (const auto& row : data) {
        for (size_t k = 0; k < row.size(); ++k) {
            if (k > 0) out << " ";
            out << std::setprecision(15) << row[k];
        }
        out << "\n";
    }
    out.close();
}

bool loadBestSOMParams(const std::string& filename, double& lr, double& lrDecay, double& radius, double& radiusDecay) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Best Initial LR:") != std::string::npos) {
            lr = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best LR Decay:") != std::string::npos) {
            lrDecay = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best Initial Radius:") != std::string::npos) {
            radius = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best Radius Decay:") != std::string::npos) {
            radiusDecay = std::stod(line.substr(line.find(":") + 1));
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

    std::string outDir = resolveResultsDir(datasetName);
    std::string paramsFile = outDir + "/random_params_som.txt";

    double lr = 0.5;
    double lrDecay = 100.0;
    double radius = 6.0;
    double radiusDecay = 10.0;

    if (!loadBestSOMParams(paramsFile, lr, lrDecay, radius, radiusDecay)) {
        std::cerr << "Could not read " << paramsFile << ", using defaults." << std::endl;
    }

    std::cout << "SOM_PAK Parameters: LR=" << lr << ", LRDecay=" << lrDecay << ", Radius=" << radius << ", RadiusDecay=" << radiusDecay << std::endl;

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());

    int width = 8;
    int height = 6;
    bool isHexa = true;

    std::string randinitBin = resolveSOMPAK("randinit");
    std::string vsomBin = resolveSOMPAK("vsom");
    pid_t myPid = getpid();

    std::string activeFile = outDir + "/runs_30_som.txt";
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

        std::string tmpPrefix = "/tmp/sompak_run_" + std::to_string(myPid) + "_r" + std::to_string(run);
        std::string trainFile = tmpPrefix + "_train.som";
        std::string initMap = tmpPrefix + "_init.map";
        std::string outMap = tmpPrefix + "_out.map";

        saveSOMDataFile(trainFile, trainSet);

        auto start = std::chrono::high_resolution_clock::now();

        std::string cmdInit = randinitBin + " -din " + trainFile + " -cout " + initMap + " -topol hexa -neigh gaussian -xdim " + std::to_string(width) + " -ydim " + std::to_string(height) + " -rand " + std::to_string(seed) + " > /dev/null 2>&1";
        std::system(cmdInit.c_str());

        std::string cmdTrain = vsomBin + " -din " + trainFile + " -vin " + trainFile + " -cin " + initMap + " -cout " + outMap + " -lrate " + std::to_string(lr) + " -lratedecay " + std::to_string(lrDecay) + " -radius " + std::to_string(radius) + " -radiusdecay " + std::to_string(radiusDecay) + " -rand " + std::to_string(seed) + " > /dev/null 2>&1";
        std::system(cmdTrain.c_str());

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        auto trainedWeights = loadMapFile(outMap);

        std::remove(trainFile.c_str());
        std::remove(initMap.c_str());
        std::remove(outMap.c_str());

        double trainQE = quantizationError(trainedWeights, trainSet);
        double testQE = quantizationError(trainedWeights, testSet);
        double trainTE = topographicError(trainedWeights, width, height, isHexa, trainSet);
        double testTE = topographicError(trainedWeights, width, height, isHexa, testSet);

        out << run << "\t" << seed << "\t" << trainQE << "\t" << testQE << "\t" << trainTE << "\t" << testTE << "\t" << elapsed << "\n";
        out.flush();

        std::cout << "Run " << std::setw(2) << run << "/30 | Seed: " << seed << " | Test QE: " << testQE << " | Test TE: " << testTE << " | Time: " << elapsed << "s" << std::endl;
    }

    out.close();
    std::cout << "Finished 30 runs of Conventional SOM. Saved to: " << activeFile << std::endl;
    return 0;
}