#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cfloat>
#include <cmath>
#include <random>
#include <string>
#include <numeric>
#include <algorithm>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "DifferentialEvolution.h"

struct SOMGrid {
    int width;
    int height;
    int dimension;
    std::vector<std::vector<double>> neurons;
};

SOMGrid initSOMGrid(int width, int height, const std::vector<std::vector<double>>& dataset, std::mt19937& rng) {
    int dimension = dataset[0].size();
    int numNeurons = width * height;

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
            std::uniform_real_distribution<double> dist(minVal[k], maxVal[k]);
            grid.neurons[i][k] = dist(rng);
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

void trainConventionalSOM(SOMGrid& grid, const std::vector<std::vector<double>>& dataset, int epochs, double initialLR, double lrDecay, double initialRadius, double radiusDecay, bool isHexa, std::mt19937& rng) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);

    double lr = initialLR;
    double radius = initialRadius;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        double radSq2 = 2.0 * radius * radius;

        for (size_t idx : indices) {
            const auto& sample = dataset[idx];

            int bmu = -1;
            double minDist = DBL_MAX;
            for (int i = 0; i < numNeurons; ++i) {
                double dist = 0.0;
                for (int k = 0; k < dimension; ++k) {
                    double d = sample[k] - grid.neurons[i][k];
                    dist += d * d;
                }
                if (dist < minDist) {
                    minDist = dist;
                    bmu = i;
                }
            }

            for (int i = 0; i < numNeurons; ++i) {
                double gDist = getGridDistance(bmu, i, grid.width, isHexa);
                if (gDist <= radius) {
                    double h = std::exp(-(gDist * gDist) / radSq2);
                    double effLR = lr * h;
                    for (int k = 0; k < dimension; ++k) {
                        grid.neurons[i][k] += effLR * (sample[k] - grid.neurons[i][k]);
                    }
                }
            }
        }

        lr *= lrDecay;
        radius *= radiusDecay;
        if (radius < 0.1) radius = 0.1;
    }
}

double computeSOMQuantizationError(const SOMGrid& grid, const std::vector<std::vector<double>>& dataset) {
    int numNeurons = grid.width * grid.height;
    int dimension = grid.dimension;
    double totalDist = 0.0;

    for (const auto& sample : dataset) {
        double minDist = DBL_MAX;
        for (int i = 0; i < numNeurons; ++i) {
            double dSum = 0.0;
            for (int k = 0; k < dimension; ++k) {
                double diff = sample[k] - grid.neurons[i][k];
                dSum += diff * diff;
            }
            if (dSum < minDist) {
                minDist = dSum;
            }
        }
        totalDist += std::sqrt(minDist);
    }

    return totalDist / dataset.size();
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
    std::string parent = "../../" + filename;
    if (access(parent.c_str(), F_OK) == 0) return parent;
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

int countCompletedTrials(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return 0;
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.rfind("Trial:", 0) == 0) {
            count++;
        }
    }
    return count;
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string dataFile = "iris.dat";
    std::string namesFile = "iris.som.names";
    int totalTrials = 30;
    unsigned int masterSeed = 42;

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
            if (argc >= 3) namesFile = argv[2];
            else namesFile = datasetName + ".som.names";
        }
    }

    if (argc >= 3) {
        try {
            int t = std::stoi(argv[2]);
            if (t > 0) totalTrials = t;
        } catch (...) {}
    }

    if (argc >= 4) {
        try {
            masterSeed = std::stoul(argv[3]);
        } catch (...) {}
    }

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());

    std::string outDir = resolveResultsDir(datasetName);
    std::string activeFile = outDir + "/random_params_som.txt";
    std::string timestamp = getTimestampString();
    std::string timeFile = outDir + "/random_params_som_" + timestamp + ".txt";

    int completedTrials = countCompletedTrials(activeFile);
    if (completedTrials >= totalTrials) {
        std::cout << "Already evaluated " << completedTrials << " trials in " << activeFile << ". Exiting." << std::endl;
        return 0;
    }

    std::ofstream outActive(activeFile, std::ios::app);
    std::ofstream outTime(timeFile, std::ios::app);

    if (completedTrials == 0) {
        outActive << "# Random Search Hyperparameter Tuning for Conventional SOM (" << datasetName << ")\n";
        outTime << "# Random Search Hyperparameter Tuning for Conventional SOM (" << datasetName << ")\n";
    }

    std::mt19937 searchRng(masterSeed);
    std::uniform_int_distribution<int> distEpoch(0, 2);
    std::uniform_real_distribution<double> distLR(0.10, 0.80);
    std::uniform_real_distribution<double> distLRDecay(0.990, 0.999);
    std::uniform_real_distribution<double> distRadius(1.5, 6.0);
    std::uniform_real_distribution<double> distRadiusDecay(0.990, 0.999);

    int epochChoices[] = {1000, 1500, 2000};
    int width = 8;
    int height = 6;
    bool isHexa = true;

    double bestTestError = DBL_MAX;
    int bestEpochs = 1000;
    double bestLR = 0.5;
    double bestLRDecay = 0.995;
    double bestRadius = 2.0;
    double bestRadiusDecay = 0.995;

    if (completedTrials > 0) {
        std::ifstream existing(activeFile);
        std::string line;
        while (std::getline(existing, line)) {
            if (line.rfind("Trial:", 0) == 0) {
                int trialEpochs = 1000;
                double trialLR = 0.5, trialLRDec = 0.995, trialRad = 2.0, trialRadDec = 0.995, trialTestErr = DBL_MAX;
                std::stringstream ss(line);
                std::string part;
                while (std::getline(ss, part, '|')) {
                    if (part.find("Epochs:") != std::string::npos) trialEpochs = std::stoi(part.substr(part.find(":") + 1));
                    else if (part.find("LR:") != std::string::npos && part.find("LRDecay") == std::string::npos) trialLR = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("LRDecay:") != std::string::npos) trialLRDec = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("Radius:") != std::string::npos && part.find("RadiusDecay") == std::string::npos) trialRad = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("RadiusDecay:") != std::string::npos) trialRadDec = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("5-Fold Test Error:") != std::string::npos) trialTestErr = std::stod(part.substr(part.find(":") + 1));
                }
                if (trialTestErr < bestTestError) {
                    bestTestError = trialTestErr;
                    bestEpochs = trialEpochs;
                    bestLR = trialLR;
                    bestLRDecay = trialLRDec;
                    bestRadius = trialRad;
                    bestRadiusDecay = trialRadDec;
                }
            }
        }
    }

    for (int t = 0; t < completedTrials; ++t) {
        distEpoch(searchRng);
        distLR(searchRng);
        distLRDecay(searchRng);
        distRadius(searchRng);
        distRadiusDecay(searchRng);
    }

    std::cout << "Starting Random Search Conventional SOM for " << datasetName << " (Total Trials: " << totalTrials << ", Resuming from: " << (completedTrials + 1) << ")" << std::endl;

    for (int trial = completedTrials + 1; trial <= totalTrials; ++trial) {
        int epochs = epochChoices[distEpoch(searchRng)];
        double initialLR = distLR(searchRng);
        double lrDecay = distLRDecay(searchRng);
        double initialRadius = distRadius(searchRng);
        double radiusDecay = distRadiusDecay(searchRng);

        std::mt19937 cvRng(masterSeed + trial * 100);
        auto folds = kFoldSplit(dataset, 5, cvRng);

        double foldTrainSum = 0.0;
        double foldTestSum = 0.0;

        for (int k = 0; k < 5; ++k) {
            auto testSet = folds[k];
            std::vector<std::vector<double>> trainSet;
            for (int j = 0; j < 5; ++j) {
                if (j != k) {
                    trainSet.insert(trainSet.end(), folds[j].begin(), folds[j].end());
                }
            }

            if (datasetName == "glass") {
                normalizeTrainTest(trainSet, testSet);
            }

            std::mt19937 trainRng(masterSeed + trial * 100 + k * 10);
            SOMGrid grid = initSOMGrid(width, height, trainSet, trainRng);
            trainConventionalSOM(grid, trainSet, epochs, initialLR, lrDecay, initialRadius, radiusDecay, isHexa, trainRng);

            foldTrainSum += computeSOMQuantizationError(grid, trainSet);
            foldTestSum += computeSOMQuantizationError(grid, testSet);
        }

        double meanTrainErr = foldTrainSum / 5.0;
        double meanTestErr = foldTestSum / 5.0;

        std::cout << "Trial: " << std::setw(2) << trial 
                  << " | Grid: 8x6 | Epochs: " << epochs 
                  << " | LR: " << std::fixed << std::setprecision(4) << initialLR 
                  << " | LRDecay: " << std::setprecision(4) << lrDecay 
                  << " | Radius: " << std::setprecision(3) << initialRadius 
                  << " | RadiusDecay: " << std::setprecision(4) << radiusDecay 
                  << " | Train Error: " << std::setprecision(6) << meanTrainErr 
                  << " | 5-Fold Test Error: " << meanTestErr << std::endl;

        outActive << "Trial: " << std::setw(2) << trial 
                  << " | Grid: 8x6 | Epochs: " << epochs 
                  << " | LR: " << std::fixed << std::setprecision(4) << initialLR 
                  << " | LRDecay: " << std::setprecision(4) << lrDecay 
                  << " | Radius: " << std::setprecision(3) << initialRadius 
                  << " | RadiusDecay: " << std::setprecision(4) << radiusDecay 
                  << " | Train Error: " << std::setprecision(6) << meanTrainErr 
                  << " | 5-Fold Test Error: " << meanTestErr << std::endl;
        outActive.flush();

        outTime << "Trial: " << std::setw(2) << trial 
                << " | Grid: 8x6 | Epochs: " << epochs 
                << " | LR: " << std::fixed << std::setprecision(4) << initialLR 
                << " | LRDecay: " << std::setprecision(4) << lrDecay 
                << " | Radius: " << std::setprecision(3) << initialRadius 
                << " | RadiusDecay: " << std::setprecision(4) << radiusDecay 
                << " | Train Error: " << std::setprecision(6) << meanTrainErr 
                << " | 5-Fold Test Error: " << meanTestErr << std::endl;
        outTime.flush();

        if (meanTestErr < bestTestError) {
            bestTestError = meanTestErr;
            bestEpochs = epochs;
            bestLR = initialLR;
            bestLRDecay = lrDecay;
            bestRadius = initialRadius;
            bestRadiusDecay = radiusDecay;
        }
    }

    std::cout << "\n=== Best Hyperparameters from Random Search ===" << std::endl;
    std::cout << "Best Grid: 8x6" << std::endl;
    std::cout << "Best Epochs: " << bestEpochs << std::endl;
    std::cout << "Best Initial LR: " << bestLR << std::endl;
    std::cout << "Best LR Decay: " << bestLRDecay << std::endl;
    std::cout << "Best Initial Radius: " << bestRadius << std::endl;
    std::cout << "Best Radius Decay: " << bestRadiusDecay << std::endl;
    std::cout << "Best 5-Fold Test Quantization Error: " << bestTestError << std::endl;

    outActive << "\n=== Best Hyperparameters from Random Search ===\n";
    outActive << "Best Grid: 8x6\n";
    outActive << "Best Epochs: " << bestEpochs << "\n";
    outActive << "Best Initial LR: " << bestLR << "\n";
    outActive << "Best LR Decay: " << bestLRDecay << "\n";
    outActive << "Best Initial Radius: " << bestRadius << "\n";
    outActive << "Best Radius Decay: " << bestRadiusDecay << "\n";
    outActive << "Best 5-Fold Test Quantization Error: " << bestTestError << "\n";
    outActive.close();

    outTime << "\n=== Best Hyperparameters from Random Search ===\n";
    outTime << "Best Grid: 8x6\n";
    outTime << "Best Epochs: " << bestEpochs << "\n";
    outTime << "Best Initial LR: " << bestLR << "\n";
    outTime << "Best LR Decay: " << bestLRDecay << "\n";
    outTime << "Best Initial Radius: " << bestRadius << "\n";
    outTime << "Best Radius Decay: " << bestRadiusDecay << "\n";
    outTime << "Best 5-Fold Test Quantization Error: " << bestTestError << "\n";
    outTime.close();

    return 0;
}
