#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "DifferentialEvolution.h"

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

struct SOMCandidate {
    double lr;
    double lrDecay;
    double radius;
    double radiusDecay;
    double meanTestQE;
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

    std::cout << "Starting SOM_PAK Random Search for: " << datasetName << std::endl;
    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());

    unsigned long mainSeed = 42;
    std::mt19937 rng(mainSeed);

    int numFolds = 5;
    int width = 8;
    int height = 6;
    int numNeurons = width * height;
    int numCandidates = 20;

    std::uniform_real_distribution<double> distLR(0.1, 10.0);
    std::uniform_real_distribution<double> distLRDecay(50.0, 1500.0);
    std::uniform_real_distribution<double> distRadius(1.0, 8.0);
    std::uniform_real_distribution<double> distRadiusDecay(5.0, 100.0);

    double bestTestQE = DBL_MAX;
    SOMCandidate bestCandidate = {0.0, 0.0, 0.0, 0.0, DBL_MAX};

    std::string randinitBin = resolveSOMPAK("randinit");
    std::string vsomBin = resolveSOMPAK("vsom");
    pid_t myPid = getpid();

    unsigned int evalSeed = 42;
    std::mt19937 foldRng(evalSeed);
    auto baseFolds = kFoldSplit(dataset, numFolds, foldRng);
    std::vector<unsigned long> foldSeeds(numFolds);
    for (int f = 0; f < numFolds; ++f) {
        foldSeeds[f] = foldRng();
    }

    for (int c = 1; c <= numCandidates; ++c) {
        double candLR = distLR(rng);
        double candLRDecay = distLRDecay(rng);
        double candRadius = distRadius(rng);
        double candRadiusDecay = distRadiusDecay(rng);

        std::vector<double> foldTestQEs;

        for (int f = 0; f < numFolds; ++f) {
            auto testFold = baseFolds[f];
            std::vector<std::vector<double>> trainFold;
            for (int j = 0; j < numFolds; ++j) {
                if (j != f) {
                    trainFold.insert(trainFold.end(), baseFolds[j].begin(), baseFolds[j].end());
                }
            }

            minMaxScaleTrainTest(trainFold, testFold);

            std::string tmpPrefix = "/tmp/sompak_param_" + std::to_string(myPid) + "_f" + std::to_string(f);
            std::string trainFile = tmpPrefix + "_train.som";
            std::string initMap = tmpPrefix + "_init.map";
            std::string outMap = tmpPrefix + "_out.map";

            saveSOMDataFile(trainFile, trainFold);

            unsigned long foldSeed = foldSeeds[f];
            std::string cmdInit = randinitBin + " -din " + trainFile + " -cout " + initMap + " -topol hexa -neigh gaussian -xdim " + std::to_string(width) + " -ydim " + std::to_string(height) + " -rand " + std::to_string(foldSeed) + " > /dev/null 2>&1";
            std::system(cmdInit.c_str());

            std::string cmdTrain = vsomBin + " -din " + trainFile + " -vin " + trainFile + " -cin " + initMap + " -cout " + outMap + " -lrate " + std::to_string(candLR) + " -lratedecay " + std::to_string(candLRDecay) + " -radius " + std::to_string(candRadius) + " -radiusdecay " + std::to_string(candRadiusDecay) + " -rand " + std::to_string(foldSeed) + " > /dev/null 2>&1";
            std::system(cmdTrain.c_str());

            auto trainedWeights = loadMapFile(outMap);

            std::remove(trainFile.c_str());
            std::remove(initMap.c_str());
            std::remove(outMap.c_str());

            if (!trainedWeights.empty()) {
                double qe = quantizationError(trainedWeights, testFold);
                foldTestQEs.push_back(qe);
            }
        }

        if (foldTestQEs.size() == (size_t)numFolds) {
            double avgTestQE = std::accumulate(foldTestQEs.begin(), foldTestQEs.end(), 0.0) / numFolds;
            std::cout << "Candidate " << std::setw(2) << c << "/" << numCandidates 
                      << " | LR: " << std::fixed << std::setprecision(4) << candLR 
                      << " | LRDecay: " << std::setprecision(2) << candLRDecay 
                      << " | Rad: " << std::setprecision(3) << candRadius 
                      << " | RadDecay: " << std::setprecision(2) << candRadiusDecay 
                      << " | 5-Fold Test QE: " << std::setprecision(6) << avgTestQE << std::endl;

            if (avgTestQE < bestTestQE) {
                bestTestQE = avgTestQE;
                bestCandidate = {candLR, candLRDecay, candRadius, candRadiusDecay, avgTestQE};
                std::cout << "  --> New Best SOM_PAK Parameter Found!" << std::endl;
            }
        }
    }

    std::cout << "\nSOM_PAK Best Parameters for " << datasetName << ":" << std::endl;
    std::cout << "  Grid: " << width << "x" << height << " (" << numNeurons << " neurons)" << std::endl;
    std::cout << "  Initial LR: " << bestCandidate.lr << std::endl;
    std::cout << "  LR Decay: " << bestCandidate.lrDecay << std::endl;
    std::cout << "  Initial Radius: " << bestCandidate.radius << std::endl;
    std::cout << "  Radius Decay: " << bestCandidate.radiusDecay << std::endl;
    std::cout << "  5-Fold Test QE: " << bestCandidate.meanTestQE << std::endl;

    std::string outDir = resolveResultsDir(datasetName);
    std::string timestamp = getTimestampString();
    std::string activeFile = outDir + "/random_params_som.txt";
    std::string timestampedFile = outDir + "/random_params_som_" + timestamp + ".txt";

    std::ofstream outActive(activeFile);
    outActive << "Best Grid: " << width << "x" << height << "\n";
    outActive << "Best Neurons: " << numNeurons << "\n";
    outActive << "Best Initial LR: " << bestCandidate.lr << "\n";
    outActive << "Best LR Decay: " << bestCandidate.lrDecay << "\n";
    outActive << "Best Initial Radius: " << bestCandidate.radius << "\n";
    outActive << "Best Radius Decay: " << bestCandidate.radiusDecay << "\n";
    outActive << "Best 5-Fold Test QE: " << bestCandidate.meanTestQE << "\n";
    outActive.close();

    std::ofstream outTime(timestampedFile);
    outTime << "Best Grid: " << width << "x" << height << "\n";
    outTime << "Best Neurons: " << numNeurons << "\n";
    outTime << "Best Initial LR: " << bestCandidate.lr << "\n";
    outTime << "Best LR Decay: " << bestCandidate.lrDecay << "\n";
    outTime << "Best Initial Radius: " << bestCandidate.radius << "\n";
    outTime << "Best Radius Decay: " << bestCandidate.radiusDecay << "\n";
    outTime << "Best 5-Fold Test QE: " << bestCandidate.meanTestQE << "\n";
    outTime.close();

    std::cout << "Saved optimal parameters to: " << activeFile << std::endl;
    return 0;
}